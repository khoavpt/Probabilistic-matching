import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid

from loguru import logger


def _start_uncertainty_timer(data, device):
    if not data.get("measure_uncertainty_time", False):
        return None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _stop_uncertainty_timer(data, start, device):
    if start is None:
        return
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    data["uncertainty_time"] = data.get("uncertainty_time", 0.0) + time.perf_counter() - start

class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local_regress_temperature = config['match_fine']['local_regress_temperature']
        self.local_regress_slicedim = config['match_fine']['local_regress_slicedim']
        self.fp16 = config['half']
        self.validate = False
        self.calibration = config['match_coarse'].get('calibration', False)

    def forward(self, feat_0, feat_1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts0_f' (torch.Tensor): [M, 2],
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always > 0 while training, see coarse_matching.py"
            update = {
                'conf_matrix_f': torch.empty(0, WW, WW, device=feat_0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            }
            if self.calibration:
                update['std_fine'] = torch.empty(0, 2, device=feat_0.device)
            data.update(update)
            return

        # compute pixel-level confidence matrix
        with torch.autocast(enabled=True if not (self.training or self.validate) else False, device_type='cuda'):
            feat_f0, feat_f1 = feat_0[...,:-self.local_regress_slicedim], feat_1[...,:-self.local_regress_slicedim] # (M, WW, C-slicedim)
            feat_ff0, feat_ff1 = feat_0[...,-self.local_regress_slicedim:], feat_1[...,-self.local_regress_slicedim:] # (M, WW, slicedim)
            feat_f0, feat_f1 = feat_f0 / C**.5, feat_f1 / C**.5
            conf_matrix_f = torch.einsum('mlc,mrc->mlr', feat_f0, feat_f1) # 
            conf_matrix_ff = torch.einsum('mlc,mrc->mlr', feat_ff0, feat_ff1 / (self.local_regress_slicedim)**.5)
        softmax_matrix_f = F.softmax(conf_matrix_f, 1) * F.softmax(conf_matrix_f, 2)
        softmax_matrix_f = softmax_matrix_f.reshape(M, self.WW, self.W+2, self.W+2)
        softmax_matrix_f = softmax_matrix_f[...,1:-1,1:-1].reshape(M, self.WW, self.WW)

        # for fine-level supervision
        if self.training or self.validate:
            data.update({'sim_matrix_ff': conf_matrix_ff})
            data.update({'conf_matrix_f': softmax_matrix_f})
        # compute pixel-level absolute kpt coords
        self.get_fine_ds_match(softmax_matrix_f, data)

        # generate seconde-stage 3x3 grid
        idx_l, idx_r = data['idx_l'], data['idx_r']
        m_ids = torch.arange(M, device=idx_l.device, dtype=torch.long).unsqueeze(-1)
        m_ids = m_ids[:len(data['mconf'])]
        idx_r_iids, idx_r_jids = idx_r // W, idx_r % W

        m_ids, idx_l, idx_r_iids, idx_r_jids = m_ids.reshape(-1), idx_l.reshape(-1), idx_r_iids.reshape(-1), idx_r_jids.reshape(-1)
        delta = create_meshgrid(3, 3, True, conf_matrix_ff.device).to(torch.long) # [1, 3, 3, 2]

        m_ids = m_ids[...,None,None].expand(-1, 3, 3)
        idx_l = idx_l[...,None,None].expand(-1, 3, 3) # [m, k, 3, 3]

        idx_r_iids = idx_r_iids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 1]
        idx_r_jids = idx_r_jids[...,None,None].expand(-1, 3, 3) + delta[None, ..., 0]

        if idx_l.numel() == 0:
            update = {
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            }
            if self.calibration:
                update['std_fine'] = torch.empty(0, 2, device=feat_0.device)
            data.update(update)
            return

        # compute second-stage heatmap
        conf_matrix_ff = conf_matrix_ff.reshape(M, self.WW, self.W+2, self.W+2)
        conf_matrix_ff = conf_matrix_ff[m_ids, idx_l, idx_r_iids, idx_r_jids]
        conf_matrix_ff = conf_matrix_ff.reshape(-1, 9)
        conf_matrix_ff = F.softmax(conf_matrix_ff / self.local_regress_temperature, -1)
        heatmap = conf_matrix_ff.reshape(-1, 3, 3)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]
        std_local = None
        if self.calibration:
            uncertainty_start = _start_uncertainty_timer(data, heatmap.device)
            std_local = self.compute_heatmap_std(
                heatmap, coords_normalized, normalized=True
            )
            _stop_uncertainty_timer(data, uncertainty_start, heatmap.device)

        if data['bs'] == 1:
            scale1 = scale * data['scale1'] if 'scale0' in data else scale
        else:
            scale1 = scale * data['scale1'][data['b_ids']][:len(data['mconf']), ...][:,None,:].expand(-1, -1, 2).reshape(-1, 2) if 'scale0' in data else scale

        # compute subpixel-level absolute kpt coords
        self.get_fine_match_local(coords_normalized, std_local, data, scale1)

    def compute_heatmap_std(self, heatmap, coords, normalized):
        h, w = heatmap.shape[-2:]
        grid = create_meshgrid(h, w, normalized, heatmap.device)[0].to(dtype=heatmap.dtype)
        diff = grid[None] - coords[:, None, None, :]
        var = (heatmap[..., None] * diff.square()).sum(dim=(1, 2))
        return torch.sqrt(var.clamp_min(1e-12))

    def compute_heatmap_expectation_std(self, heatmap, normalized):
        h, w = heatmap.shape[-2:]
        grid = create_meshgrid(h, w, normalized, heatmap.device)[0].to(dtype=heatmap.dtype)
        mean = (heatmap[..., None] * grid[None]).sum(dim=(1, 2))
        return mean, self.compute_heatmap_std(heatmap, mean, normalized)

    def get_fine_match_local(self, coords_normed, std_normed, data, scale1):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        mkpts0_c, mkpts1_c = data['mkpts0_c'], data['mkpts1_c']

        # mkpts0_f and mkpts1_f
        mkpts0_f = mkpts0_c
        mkpts1_f = mkpts1_c + (coords_normed * (3 // 2) * scale1)

        update = {
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
        }
        if self.calibration:
            std_local = std_normed * (3 // 2) * scale1
            std_patch = data.get('std_fine_8x8')
            update["std_fine"] = std_local if std_patch is None else torch.sqrt(std_patch.square() + std_local.square())
        data.update(update)

    @torch.no_grad()
    def get_fine_ds_match(self, conf_matrix, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        m, _, _ = conf_matrix.shape

        conf_matrix = conf_matrix.reshape(m, -1)[:len(data['mconf']),...]
        val, idx = torch.max(conf_matrix, dim = -1)
        idx = idx[:,None]
        idx_l, idx_r = idx // WW, idx % WW

        data.update({'idx_l': idx_l, 'idx_r': idx_r})

        if self.fp16:
            grid = create_meshgrid(W, W, False, conf_matrix.device).to(dtype=torch.float16) - W // 2 + 0.5 # kornia >= 0.5.1
        else:
            grid = create_meshgrid(W, W, False, conf_matrix.device) - W // 2 + 0.5
        grid = grid.reshape(1, -1, 2).expand(m, -1, -1)
        delta_l = torch.gather(grid, 1, idx_l.unsqueeze(-1).expand(-1, -1, 2))
        delta_r = torch.gather(grid, 1, idx_r.unsqueeze(-1).expand(-1, -1, 2))

        scale0 = scale * data['scale0'][data['b_ids']] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        if self.calibration:
            uncertainty_start = _start_uncertainty_timer(data, conf_matrix.device)
            prob_r = conf_matrix.reshape(-1, WW, WW).sum(dim=1)
            prob_r = prob_r / prob_r.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            _, std_r = self.compute_heatmap_expectation_std(
                prob_r.reshape(-1, W, W), normalized=False
            )
            _stop_uncertainty_timer(data, uncertainty_start, conf_matrix.device)

        if torch.is_tensor(scale0) and scale0.numel() > 1: # scale0 is a tensor
            mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1[:len(data['mconf']),...][:,None,:])).reshape(-1, 2)
            if self.calibration:
                std_fine_8x8 = std_r * scale1[:len(data['mconf']), ...]
        else: # scale0 is a float
            mkpts0_f = (data['mkpts0_c'][:,None,:] + (delta_l * scale0)).reshape(-1, 2)
            mkpts1_f = (data['mkpts1_c'][:,None,:] + (delta_r * scale1)).reshape(-1, 2)
            if self.calibration:
                std_fine_8x8 = std_r * scale1
        
        update = {
            "mkpts0_c": mkpts0_f,
            "mkpts1_c": mkpts1_f,
        }
        if self.calibration:
            uncertainty_start = _start_uncertainty_timer(data, std_r.device)
            update["std_fine_8x8"] = std_fine_8x8
            _stop_uncertainty_timer(data, uncertainty_start, std_r.device)
        data.update(update)
