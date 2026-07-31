import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch import distributions


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


class Conv1d_BN_Act(nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
        act=None,
        drop=None,
    ):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module(
            "c", nn.Conv1d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = nn.BatchNorm1d(b)
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)
        if act != None:
            self.add_module("a", act)
        if drop != None:
            self.add_module("d", nn.Dropout(drop))


class RealNVP(nn.Module):
    """RealNVP: a flow-based generative model

    `Density estimation using Real NVP
    arXiv: <https://arxiv.org/abs/1605.08803>`_.

    Code is modified from `the mmpose implementation of RLE
    <https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/utils/realnvp.py>`_.
    """

    @staticmethod
    def get_scale_net(channel):
        """Get the scale model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, channel),
            nn.GELU(),
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.Linear(channel, 2),
            nn.Tanh(),
        )

    @staticmethod
    def get_trans_net(channel):
        """Get the translation model in a single invertable mapping."""
        return nn.Sequential(
            nn.Linear(2, channel),
            nn.GELU(),
            nn.Linear(channel, channel),
            nn.GELU(),
            nn.Linear(channel, 2),
        )

    @property
    def prior(self):
        """The prior distribution."""
        return distributions.MultivariateNormal(self.loc, self.cov)

    def __init__(self, channel=64):
        super(RealNVP, self).__init__()
        self.channel = channel
        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer(
            "mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32)
        )

        self.s = torch.nn.ModuleList(
            [self.get_scale_net(self.channel) for _ in range(len(self.mask))]
        )
        self.t = torch.nn.ModuleList(
            [self.get_trans_net(self.channel) for _ in range(len(self.mask))]
        )
        self.init_weights()

    def init_weights(self):
        """Initialization model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """Apply mapping form the data space to the latent space and calculate
        the log determinant of the Jacobian matrix."""

        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])  # torch.exp(s): betas
            t = self.t[i](z_) * (1 - self.mask[i])  # gammas
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """Calculate the log probability of given sample in data space."""

        z, log_det = self.backward_p(x)
        return self.prior.log_prob(z) + log_det


def soft_argmax(x, temperature=1.0):
    L = x.shape[1]
    assert L % 2  # L is odd to ensure symmetry
    idx = torch.arange(0, L, 1, device=x.device).repeat(x.shape[0], 1)
    scale_x = x / temperature
    out = F.softmax(scale_x, dim=1) * idx
    out = out.sum(dim=1, keepdim=True)

    return out


class FineMatching(nn.Module):
    def __init__(self, config, act_layer=nn.GELU):
        super(FineMatching, self).__init__()
        self.config = config

        self.block_dims = self.config["backbone"]["block_dims"]
        self.local_resolution = self.config["local_resolution"]
        self.drop = self.config["fine"]["droprate"]
        self.coord_length = self.config["fine"]["coord_length"]
        self.bi_directional_refine = self.config["fine"]["bi_directional_refine"]
        self.sigma_selection = self.config["fine"]["sigma_selection"]
        self.mconf_thr = self.config["coarse"]["mconf_thr"]
        self.sigma_thr = self.config["fine"]["sigma_thr"]
        self.border_rm = self.config["coarse"]["border_rm"] * \
            self.local_resolution
        self.deploy = self.config["deploy"]
        self.calibration = config["coarse"]["calibration"]

        # network
        self.query_encoder = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                act=act_layer(),
                drop=self.drop,
            ),
            Conv1d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                act=act_layer(),
                drop=self.drop,
            ),
        )

        self.reference_encoder = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                act=act_layer(),
                drop=self.drop,
            ),
            Conv1d_BN_Act(
                self.block_dims[-1],
                self.block_dims[-1],
                act=act_layer(),
                drop=self.drop,
            ),
        )

        self.merge_qr = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1] * 2,
                self.block_dims[-1] * 2,
                act=act_layer(),
                drop=self.drop,
            ),
            Conv1d_BN_Act(
                self.block_dims[-1] * 2,
                self.block_dims[-1] * 2,
                act=act_layer(),
                drop=self.drop,
            ),
        )

        self.x_head = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1] * 2,
                self.block_dims[-1] * 2,
                act=act_layer(),
                drop=self.drop,
            ),
            nn.Conv1d(self.block_dims[-1] * 2,
                      self.coord_length + 2, kernel_size=1),
        )

        self.y_head = nn.Sequential(
            Conv1d_BN_Act(
                self.block_dims[-1] * 2,
                self.block_dims[-1] * 2,
                act=act_layer(),
                drop=self.drop,
            ),
            nn.Conv1d(self.block_dims[-1] * 2,
                      self.coord_length + 2, kernel_size=1),
        )

        self.flow = RealNVP()
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data={}):
        q = self.query_encoder(
            feat_f0.permute(0, 2, 1).contiguous()
            + feat_c0.permute(0, 2, 1).contiguous()
        )
        r = self.reference_encoder(
            feat_f1.permute(0, 2, 1).contiguous()
            + feat_c1.permute(0, 2, 1).contiguous()
        )
        out = self.merge_qr(torch.cat([q, r], dim=1))

        x = self.x_head(out).permute(0, 2, 1).contiguous()
        y = self.y_head(out).permute(0, 2, 1).contiguous()

        if self.bi_directional_refine:
            x01, x10 = x.chunk(2, dim=1)
            x01 = x01.reshape(-1, self.coord_length + 2)
            x10 = x10.reshape(-1, self.coord_length + 2)
            x_out = torch.cat([x01, x10])

            y01, y10 = y.chunk(2, dim=1)
            y01 = y01.reshape(-1, self.coord_length + 2)
            y10 = y10.reshape(-1, self.coord_length + 2)
            y_out = torch.cat([y01, y10])
        else:
            x_out = x.reshape(-1, self.coord_length + 2)
            y_out = y.reshape(-1, self.coord_length + 2)

        x_cls = x_out[:, : self.coord_length + 1]
        coord_x = soft_argmax(x_cls) / self.coord_length - \
            0.5  # range [-0.5, +0.5]
        grid = (
            torch.arange(self.coord_length + 1, device=x_cls.device, dtype=x_cls.dtype)
            / self.coord_length
            - 0.5
        )
        prob_x = F.softmax(x_cls, dim=1)
        std_x = torch.sqrt(
            (prob_x * (grid[None, :] - coord_x) ** 2).sum(dim=1, keepdim=True).clamp(min=1e-12)
        )
        x_sigma = x_out[:, -1:].sigmoid()

        y_cls = y_out[:, : self.coord_length + 1]
        coord_y = soft_argmax(y_cls) / self.coord_length - 0.5
        prob_y = F.softmax(y_cls, dim=1)
        std_y = torch.sqrt(
            (prob_y * (grid[None, :] - coord_y) ** 2).sum(dim=1, keepdim=True).clamp(min=1e-12)
        )
        y_sigma = y_out[:, -1:].sigmoid()

        coord = torch.cat([coord_x, coord_y], dim=1)
        sigma = torch.cat([x_sigma, y_sigma], dim=1)
        std_heatmap = torch.cat([std_x, std_y], dim=1)
        if self.calibration:
            uncertainty_start = _start_uncertainty_timer(data, std_heatmap.device)
            data["std_fine"] = std_heatmap * self.local_resolution
            _stop_uncertainty_timer(data, uncertainty_start, std_heatmap.device)

        if data.get("target_uv", None) is not None:
            gt_uv = data["target_uv"]
            mask = data["target_uv_weight"].clone()

            if mask.sum() == 0:
                mask[0] = True
            mask_coord = coord[mask]
            mask_gt_uv = gt_uv[mask]
            mask_sigma = sigma[mask]

            mask_sigma = torch.clamp(mask_sigma, 1e-6, 1 - 1e-6)
            bar_mu = (mask_coord - mask_gt_uv) / mask_sigma

            log_phi = self.flow.log_prob(bar_mu).unsqueeze(-1)
            nf_loss = torch.log(mask_sigma) - log_phi

            data.update(
                {
                    "pred_coord": coord,
                    "pred_score": 1.0 - torch.mean(sigma, dim=-1).flatten(),
                    "mask_coord": mask_coord,
                    "mask_sigma": mask_sigma,
                    "nf_loss": nf_loss,
                }
            )
        else:
            data.update(
                {
                    "pred_coord": coord,
                    "pred_score": 1.0 - torch.mean(sigma, dim=-1).flatten(),
                }
            )

        if not self.deploy:
            self.final_matching_selection(data)

        return data["pred_coord"], data["pred_score"]

    @torch.no_grad()
    def final_matching_selection(self, data):
        offset = data["pred_coord"] * self.local_resolution

        if self.bi_directional_refine:
            fine_offset01, fine_offset10 = torch.clamp(
                offset, -self.local_resolution / 2, self.local_resolution / 2
            ).chunk(2)
        else:
            fine_offset01 = torch.clamp(
                offset, -self.local_resolution / 2, self.local_resolution / 2
            )

        h0, w0 = data["hw0_i"]
        h1, w1 = data["hw1_i"]
        scale0 = data["scale0"][data["b_ids"]] if "scale0" in data else 1.0
        scale1 = data["scale1"][data["b_ids"]] if "scale1" in data else 1.0
        scale0_w = scale0[:, 0] if "scale0" in data else 1.0
        scale0_h = scale0[:, 1] if "scale0" in data else 1.0
        scale1_w = scale1[:, 0] if "scale1" in data else 1.0
        scale1_h = scale1[:, 1] if "scale1" in data else 1.0

        # Filter by mconf and border
        mkpts0_f = data["mkpts0_c"]
        mkpts1_f = data["mkpts1_c"] + fine_offset01 * scale1
        mask = (
            (data["mconf"] > self.mconf_thr)
            & (mkpts0_f[:, 0] >= self.border_rm)
            & (mkpts0_f[:, 0] <= w0 * scale0_w - self.border_rm)
            & (mkpts0_f[:, 1] >= self.border_rm)
            & (mkpts0_f[:, 1] <= h0 * scale0_h - self.border_rm)
            & (mkpts1_f[:, 0] >= self.border_rm)
            & (mkpts1_f[:, 0] <= w1 * scale1_w - self.border_rm)
            & (mkpts1_f[:, 1] >= self.border_rm)
            & (mkpts1_f[:, 1] <= h1 * scale1_h - self.border_rm)
        )
        if self.bi_directional_refine:
            mkpts0_f_ = data["mkpts0_c"] + fine_offset10 * scale0
            mkpts1_f_ = data["mkpts1_c"]
            mask_ = (
                (data["mconf"] > self.mconf_thr)
                & (mkpts0_f_[:, 0] >= self.border_rm)
                & (mkpts0_f_[:, 0] <= w0 * scale0_w - self.border_rm)
                & (mkpts0_f_[:, 1] >= self.border_rm)
                & (mkpts0_f_[:, 1] <= h0 * scale0_h - self.border_rm)
                & (mkpts1_f_[:, 0] >= self.border_rm)
                & (mkpts1_f_[:, 0] <= w1 * scale1_w - self.border_rm)
                & (mkpts1_f_[:, 1] >= self.border_rm)
                & (mkpts1_f_[:, 1] <= h1 * scale1_h - self.border_rm)
            )

        if self.bi_directional_refine:
            mkpts0_f = torch.cat([mkpts0_f, mkpts0_f_])
            mkpts1_f = torch.cat([mkpts1_f, mkpts1_f_])
            mask = torch.cat([mask, mask_])
            data["mconf"] = torch.cat([data["mconf"], data["mconf"]])
            data["b_ids"] = torch.cat([data["b_ids"], data["b_ids"]])

        # Filter by sigma
        if self.bi_directional_refine and self.sigma_selection:
            # Retain the more confident matching pair with a smaller sigma (more significant) in the bi-directional matching pairs
            pred_score01, pred_score10 = data["pred_score"].chunk(2)
            pred_score_mask = pred_score01 > pred_score10
            pred_score_mask = torch.cat([pred_score_mask, ~pred_score_mask])
            pred_score_mask &= data["pred_score"] > self.sigma_thr
            mask &= pred_score_mask

        data.update(
            {
                # "gt_mask": data["mconf"] == 0,
                "m_bids": data["b_ids"][mask],
                "mkpts0_f": mkpts0_f[mask],
                "mkpts1_f": mkpts1_f[mask],
                "mconf": data["mconf"][mask],
            }
        )

        if self.calibration:
            uncertainty_start = _start_uncertainty_timer(data, data["std_coarse"].device)
            data.update(
                {
                    "std_coarse": data["std_coarse"][mask],
                    "std_fine": data["std_fine"][mask]
                }
            )
            _stop_uncertainty_timer(data, uncertainty_start, data["std_coarse"].device)
