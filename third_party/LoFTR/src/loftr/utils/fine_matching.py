import math
import torch
import torch.nn as nn
import time

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


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

    def __init__(self, calibration=False):
        super().__init__()
        self.calibration = calibration

    def forward(self, feat_f0, feat_f1, data):
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
        M, WW, C = feat_f0.shape
        W = int(math.sqrt(WW))
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
            })
            if self.calibration:
                data.update({'std_fine': torch.empty(0, 2, device=feat_f0.device)})
            return

        feat_f0_picked = feat_f0_picked = feat_f0[:, WW//2, :]
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        std_xy = None
        if self.calibration:
            uncertainty_start = _start_uncertainty_timer(data, heatmap.device)
            grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

            # compute std over <x, y>
            var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
            std_xy = torch.sqrt(torch.clamp(var, min=1e-10))  # [M, 2] clamp needed for numerical stability
            _stop_uncertainty_timer(data, uncertainty_start, heatmap.device)
            std = torch.sum(std_xy, -1)  # [M]
        else:
            std = torch.zeros(M, device=feat_f0.device, dtype=coords_normalized.dtype)
        
        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, std_xy, data)

    @torch.no_grad()
    def get_fine_match(self, coords_normed, std_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]

        update = {
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f,
        }
        if self.calibration:
            uncertainty_start = _start_uncertainty_timer(data, std_normed.device)
            update["std_fine"] = (std_normed * (W // 2) * scale1)[:len(data['mconf'])]
            _stop_uncertainty_timer(data, uncertainty_start, std_normed.device)
        data.update(update)
