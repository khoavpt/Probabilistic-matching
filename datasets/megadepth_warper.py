"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    MegaDepth data handling was adapted from 
    LoFTR official code: https://github.com/zju3dv/LoFTR/blob/master/src/datasets/megadepth.py
"""

import torch
from kornia.utils import create_meshgrid
import matplotlib.pyplot as plt
import pdb
import cv2
import torch.nn.functional as F

@torch.no_grad()
def get_gt_warp(data, scale = 8, depth_interpolation_mode = 'bilinear', relative_depth_error_threshold = 0.05, mutual_dist = 1.5):
    """
        Supervise corresp with dense depth & camera poses
    """
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    #scale = 8
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    with torch.no_grad():
        grid_pt1 = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
        grid_pt1 = scale * grid_pt1
        mask10, grid_pt10 = warp_kpts_v2(
            grid_pt1.double(),
            data['depth1'].double(),
            data['depth0'].double(),
            data['T_1to0'].double(),
            data['K1'].double(),
            data['K0'].double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        grid_pt10[~mask10, :] = 0
        mask01, grid_pt01 = warp_kpts_v2(
            grid_pt10.double(),
            data['depth0'].double(),
            data['depth1'].double(),
            data['T_0to1'].double(),
            data['K0'].double(),
            data['K1'].double(),
            depth_interpolation_mode = depth_interpolation_mode,
            relative_depth_error_threshold = relative_depth_error_threshold,
        )
        #check reproj error
        dist = torch.linalg.norm(grid_pt1 - grid_pt01, dim=-1)
        mask_mutual = (dist < mutual_dist)
        mask = mask_mutual & mask01 & mask10
        #hardcord
        batched_corrs = [ 
            torch.cat(
                [
                    grid_pt10[i, mask[i]],
                    grid_pt1[i, mask[i]]
                ], dim=-1
            ) for i in range(len(mask))
        ]
        #Remove repeated correspondences - this is important for network convergence
        corrs = []
        for pts in batched_corrs:
            lut_mat12 = torch.ones((h1, w1, 4), device = device, dtype = pts.dtype) * -1
            lut_mat21 = torch.clone(lut_mat12)
            src_pts = pts[:, :2] / scale
            tgt_pts = pts[:, 2:] / scale
            try:
                lut_mat12[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
                mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
                points = lut_mat12[mask_valid12]

                #Target-src check
                src_pts, tgt_pts = points[:, :2], points[:, 2:]
                lut_mat21[tgt_pts[:,1].long(), tgt_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
                mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
                points = lut_mat21[mask_valid21]

                corrs.append(points)
            except:
                pdb.set_trace()
                print('..')
        # prob = mask.float().reshape(N, h0, w0)
        # x1 = x1.reshape(N, h0, w0, 2)
        
    # for i in range(len(corrs)):
        
    #     plot_corrs(data['image0'][i], data['image1'][i], corrs[i][:, :2].cpu()*8, corrs[i][:, 2:].cpu()*8, dist[i][mask[i]].cpu().numpy())
    #     breakpoint()
    return corrs

@torch.no_grad()
def warp_kpts_v2(kpts0, depth0, depth1, T_0to1, K0, K1, smooth_mask = False, return_relative_depth_error = False, depth_interpolation_mode = "bilinear", relative_depth_error_threshold = 0.2):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    (
        n,
        h,
        w,
    ) = depth0.shape
    if depth_interpolation_mode == "combined":
        # Inspired by approach in inloc, try to fill holes from bilinear interpolation by nearest neighbour interpolation
        if smooth_mask:
            raise NotImplementedError("Combined bilinear and NN warp not implemented")
        valid_bilinear, warp_bilinear = warp_kpts_v2(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "bilinear",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        valid_nearest, warp_nearest = warp_kpts_v2(kpts0, depth0, depth1, T_0to1, K0, K1, 
                  smooth_mask = smooth_mask, 
                  return_relative_depth_error = return_relative_depth_error, 
                  depth_interpolation_mode = "nearest-exact",
                  relative_depth_error_threshold = relative_depth_error_threshold)
        nearest_valid_bilinear_invalid = (~valid_bilinear).logical_and(valid_nearest) 
        warp = warp_bilinear.clone()
        warp[nearest_valid_bilinear_invalid] = warp_nearest[nearest_valid_bilinear_invalid]
        valid = valid_bilinear | valid_nearest
        return valid, warp
    kpts0_g = torch.stack(
        (2 * (kpts0[..., 0] + 0.5) / w - 1, 2 * (kpts0[..., 1] + 0.5) / h - 1), dim=-1
    )  # from [0, h-1] -> [-1+1/w, 1+1/h]
    kpts0_long = kpts0.round().long()
    kpts0_long[:,:,0] = kpts0_long[:,:,0].clip(0, w-1)
    kpts0_long[:,:,1] = kpts0_long[:,:,1].clip(0, h-1)
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    # kpts0_depth = F.grid_sample(depth0[:, None], kpts0_g[:, :, None], mode = depth_interpolation_mode, align_corners=False)[
    #     :, 0, :, 0
    # ]

    nonzero_mask = kpts0_depth != 0
    # for b in range(len(kpts0)):
    #     fig, ax = plt.subplots(1,2)
    #     depth_np = depth0.detach().cpu().numpy()[b]
    #     depth_np_plot = depth_np.copy()
    #     for x, y in kpts0.round().long()[b, nonzero_mask[b], :].detach().cpu().numpy():
    #         cv2.drawMarker(depth_np_plot, (x, y), (255), cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #     ax[0].imshow(depth_np)
    #     ax[1].imshow(depth_np_plot)
    # breakpoint()

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
         * kpts0_depth[..., None]
    )# (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1) # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-5
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )

    # for b in range(len(kpts0)):
    #     fig, ax = plt.subplots(1,2)
    #     depth_np = depth1.detach().cpu().numpy()[b]
    #     depth_np_plot = depth_np.copy()
    #     for x, y in w_kpts0.round().long()[b, covisible_mask[b], :].detach().cpu().numpy():
    #         cv2.drawMarker(depth_np_plot, (x, y), (255), cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #     ax[0].imshow(depth_np)
    #     ax[1].imshow(depth_np_plot)
    #     fig.savefig('vis/bla.png')

    w_kpts0_g = torch.stack(
        (2 * (w_kpts0[..., 0] + 0.5) / w - 1, 2 * (w_kpts0[..., 1] + 0.5) / h - 1), dim=-1
    ) # from [0,h-1] -> [-1+1/h, 1-1/h]

    # w_kpts0[~covisible_mask, :] = -5 # xd
    w_kpts0_long = w_kpts0.round().long()
    w_kpts0_long[:,:,0] = w_kpts0_long[:,:,0].clip(0, w-1)
    w_kpts0_long[:,:,1] = w_kpts0_long[:,:,1].clip(0, h-1)
    w_kpts0_depth  = torch.stack(
        [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    )
    # w_kpts0_depth = F.grid_sample(
    #     depth1[:, None], w_kpts0_g[:, :, None], mode=depth_interpolation_mode, align_corners=False
    # )[:, 0, :, 0]
    relative_depth_error = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs()
    if not smooth_mask:
        consistent_mask = relative_depth_error < relative_depth_error_threshold
    else:
        consistent_mask = (-relative_depth_error/smooth_mask).exp()

    valid_mask = nonzero_mask & covisible_mask & consistent_mask
    if return_relative_depth_error:
        return relative_depth_error, w_kpts0
    else:
        return valid_mask, w_kpts0

@torch.no_grad()
def get_correspondences(pts2, data, idx):
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape

    pts2 = pts2[None, ...]

    scale0 = data['scale0'][idx, None][None, ...] if 'scale0' in data else 1
    scale1 = data['scale1'][idx, None][None, ...] if 'scale1' in data else 1

    pts2 = scale1 * pts2 * 8

    # warp kpts bi-directionally and check reproj error
    nonzero_m1, pts1  = warp_kpts(pts2, data['depth1'][idx][None, ...], data['depth0'][idx][None, ...], data['T_1to0'][idx][None, ...], 
                                                                        data['K1'][idx][None, ...], data['K0'][idx][None, ...]) 

    corrs = torch.cat([pts1[0, :] / data['scale0'][idx],
                       pts2[0, :] / data['scale1'][idx]],dim=-1)

    #plot_corrs(data['image0'][idx], data['image1'][idx], corrs[:, :2], corrs[:, 2:])

    return corrs

@torch.no_grad()
def convert_points_to_homogeneous(points):
    """
    Convert Cartesian points to homogeneous coordinates.
    
    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, D) representing N points in D-dimensional space.
    
    Returns
    -------
    np.ndarray
        Array of shape (N, D+1) representing points in homogeneous coordinates.
    """
    if points.ndim != 2:
        raise ValueError("Input must be a 2D array of shape (N, D).")
    
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    return torch.hstack([points, ones])



@torch.no_grad()
def spvs_coarse(data, scale = 8):
    """
        Supervise corresp with dense depth & camera poses
    """

    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    #scale = 8
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt1_i = scale1 * grid_pt1_c
    # warp kpts bi-directionally and check reproj error
    nonzero_m1, w_pt1_i  =  warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0']) 
    nonzero_m2, w_pt1_og =  warp_kpts(   w_pt1_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1']) 


    dist = torch.linalg.norm( grid_pt1_i - w_pt1_og, dim=-1)
    mask_mutual = (dist < 1.5) & nonzero_m1 & nonzero_m2

    #_, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    batched_corrs = [ torch.cat([w_pt1_i[i, mask_mutual[i]] / data['scale0'][i],grid_pt1_i[i, mask_mutual[i]] / data['scale1'][i]],dim=-1) for i in range(len(mask_mutual))]


    #Remove repeated correspondences - this is important for network convergence
    corrs = []
    for pts in batched_corrs:
        lut_mat12 = torch.ones((h1, w1, 4), device = device, dtype = torch.float32) * -1
        lut_mat21 = torch.clone(lut_mat12)
        src_pts = pts[:, :2] / scale
        tgt_pts = pts[:, 2:] / scale
        try:
            lut_mat12[src_pts[:,1].long(), src_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
            mask_valid12 = torch.all(lut_mat12 >= 0, dim=-1)
            points = lut_mat12[mask_valid12]

            #Target-src check
            src_pts, tgt_pts = points[:, :2], points[:, 2:]
            lut_mat21[tgt_pts[:,1].long(), tgt_pts[:,0].long()] = torch.cat([src_pts, tgt_pts], dim=1)
            mask_valid21 = torch.all(lut_mat21 >= 0, dim=-1)
            points = lut_mat21[mask_valid21]

            corrs.append(points)
        except:
            pdb.set_trace()
            print('..')

    #Plot for debug purposes    
    # for i in range(len(corrs)):
    #     plot_corrs(data['image0'][i], data['image1'][i], corrs[i][:, :2]*8, corrs[i][:, 2:]*8)

    return corrs


@torch.no_grad()
def warp_kpts(kpts0, depth0, depth1, T_0to1, K0, K1):
    """ Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).
    
    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long().clip(0, 2000-1)

    depth0[:, 0, :] = 0 ; depth1[:, 0, :] = 0 
    depth0[:, :, 0] = 0 ; depth1[:, :, 0] = 0 

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]] for i in range(kpts0.shape[0])], dim=0
    )  # (N, L)
    nonzero_mask = kpts0_depth > 0

    # Draw cross marks on the image for each keypoint
    # for b in range(len(kpts0)):
    #     fig, ax = plt.subplots(1,2)
    #     depth_np = depth0.numpy()[b]
    #     depth_np_plot = depth_np.copy()
    #     for x, y in kpts0_long[b, nonzero_mask[b], :].numpy():
    #         cv2.drawMarker(depth_np_plot, (x, y), (255), cv2.MARKER_CROSS, markerSize=10, thickness=2)
    #     ax[0].imshow(depth_np)
    #     ax[1].imshow(depth_np_plot)

    # Unproject
    kpts0_h = torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1) * kpts0_depth[..., None]  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]    # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (w_kpts0_h[:, :, [2]] + 1e-5)  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    # h, w = depth1.shape[1:3]
    # covisible_mask = (w_kpts0[:, :, 0] > 0) * (w_kpts0[:, :, 0] < w-1) * \
    #     (w_kpts0[:, :, 1] > 0) * (w_kpts0[:, :, 1] < h-1)
    # w_kpts0_long = w_kpts0.long()
    # w_kpts0_long[~covisible_mask, :] = 0

    # w_kpts0_depth = torch.stack(
    #     [depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]] for i in range(w_kpts0_long.shape[0])], dim=0
    # )  # (N, L)
    # consistent_mask = ((w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth).abs() < 0.2


    valid_mask = nonzero_mask #* consistent_mask* covisible_mask 

    return valid_mask, w_kpts0
