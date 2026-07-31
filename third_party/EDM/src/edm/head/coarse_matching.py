import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from src.edm.utils.supervision import compute_supervision_fine

INF = 1e9  # -1e4 for fp16matmul


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


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window_size = config["local_resolution"]
        self.thr = config["coarse"]["mconf_thr"]
        self.temperature = config["coarse"]["dsmax_temperature"]
        self.ds_opt = config["coarse"]["ds_opt"]
        self.pad_num = config["coarse"]["train_pad_num"]
        self.topk = config["coarse"]["topk"]
        self.deploy = config["deploy"]
        self.calibration = config["coarse"]["calibration"]


    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            data (dict)
            mask_c0 (torch.Tensor): [N, L] (optional)
            mask_c1 (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        # normalize
        feat_c0, feat_c1 = map(
            lambda feat: feat / feat.shape[-1] ** 0.5, [feat_c0, feat_c1]
        )

        with torch.autocast(enabled=False, device_type="cuda"):
            sim_matrix = (
                torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) /
                self.temperature
            )
            del feat_c0, feat_c1
            if mask_c0 is not None:
                sim_matrix = sim_matrix.float().masked_fill(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF,
                )

        if not self.training and self.ds_opt:
            # Alternative implementation of daul-softmax operator for efficient inference
            sim_matrix = torch.exp(sim_matrix)
            P_row = F.normalize(sim_matrix, p=1, dim=2)
            P_col = F.normalize(sim_matrix, p=1, dim=1)
            conf_matrix = P_row * P_col
            if self.calibration:
                uncertainty_start = _start_uncertainty_timer(data, sim_matrix.device)
                Hc1, Wc1 = data["hw1_c"]
                # stride = data["hw1_i"][0] / Hc1

                ys = (torch.arange(Hc1, device=sim_matrix.device) + 0.5) * self.window_size
                xs = (torch.arange(Wc1, device=sim_matrix.device) + 0.5) * self.window_size

                grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
                coords1 = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (M, 2)

                mean = torch.einsum('bnm,md->bnd', P_row, coords1)
                mean_sq = torch.einsum('bnm,md->bnd', P_row, coords1**2)
                var = mean_sq - mean**2
                std_map = torch.sqrt(var.clamp(min=1e-8))  # (B, N, 2)
                data["std_map_coarse"] = std_map # (B, H*W, 2)
                _stop_uncertainty_timer(data, uncertainty_start, sim_matrix.device)
        else:
            # Native daul-softmax operator
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        
        data.update(
            {
                "conf_matrix": conf_matrix,
            }
        )

        if not self.deploy:
            # predict coarse matches from conf_matrix
            self.coarse_matching_selection(data)

        return conf_matrix # Returning the sim_matrix can be faster, but it may reduce the accuracy. 

    # Static tensor shape for mini-batch inference.
    @torch.no_grad()
    def coarse_matching_selection(self, data):
        """
        Args:
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        conf_matrix = data["conf_matrix"]

        # mutual nearest
        # mask = (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
        #     * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        # conf_matrix[~mask]=0

        k = self.topk
        row_max_val, row_max_idx = torch.max(conf_matrix, dim=2)

        # prevent out of range
        if k == -1 or k > row_max_val.shape[-1]:
            k = row_max_val.shape[-1]

        topk_val, topk_idx = torch.topk(row_max_val, k)
        b_ids = (
            torch.arange(conf_matrix.shape[0], device=conf_matrix.device)
            .unsqueeze(1)
            .repeat(1, k)
            .flatten()
        )
        i_ids = topk_idx.flatten()
        j_ids = row_max_idx[b_ids, i_ids].flatten()
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        scale = data["hw0_i"][0] / data["hw0_c"][0]
        scale0 = scale * data["scale0"][b_ids] if "scale0" in data else scale
        scale1 = scale * data["scale1"][b_ids] if "scale1" in data else scale
        mkpts0_c = (
            torch.stack(
                [
                    i_ids % data["hw0_c"][1],
                    torch.div(i_ids, data["hw0_c"][1], rounding_mode="floor"),
                ],
                dim=1,
            )
            * scale0
        )
        mkpts1_c = (
            torch.stack(
                [
                    j_ids % data["hw1_c"][1],
                    torch.div(j_ids, data["hw1_c"][1], rounding_mode="floor"),
                ],
                dim=1,
            )
            * scale1
        ) # (B*k, 2)
        if "std_map_coarse" in data:
            uncertainty_start = _start_uncertainty_timer(data, data["std_map_coarse"].device)
            std_coarse = data["std_map_coarse"][b_ids, i_ids]  # (K, 2)
            data["std_coarse"] = std_coarse
            del data["std_map_coarse"]
            _stop_uncertainty_timer(data, uncertainty_start, data["std_coarse"].device)
        data.update(
            {
                "mconf": mconf,
                "mkpts0_c": mkpts0_c,
                "mkpts1_c": mkpts1_c,
                "b_ids": b_ids,
                "i_ids": i_ids,
                "j_ids": j_ids,
            }
        )

        # Fine matching supervision during the training process
        if data.get("spv_w_pt0_i", None) is not None:
            pad_num = self.pad_num if self.training else 0
            with torch.autocast(enabled=False, device_type="cuda"):
                compute_supervision_fine(data, pad_num, self.window_size)
