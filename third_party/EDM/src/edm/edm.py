from .head.fine_matching import FineMatching
from .head.coarse_matching import CoarseMatching
from .neck.neck import CIM
from .backbone.resnet import ResNet18
from einops.einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch
torch.set_float32_matmul_precision("highest")  # highest (defualt) high medium


def detect_NaN(feat_0, feat_1):
    """Inference-only replacement for the upstream training utility."""
    feat_0.nan_to_num_(nan=0.0)
    feat_1.nan_to_num_(nan=0.0)


class EDM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.local_resolution = self.config["local_resolution"]
        self.bi_directional_refine = self.config["fine"]["bi_directional_refine"]
        self.deploy = self.config["deploy"]
        self.topk = config["coarse"]["topk"]

        # Modules
        self.backbone = ResNet18(config)
        self.neck = CIM(config)
        self.coarse_matching = CoarseMatching(config)
        self.fine_matching = FineMatching(config)

    def forward(self, data):
        """
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        if self.deploy:
            image0, image1 = data.split(1, 1)
            data = {"image0": image0, "image1": image1}

        data.update(
            {
                "bs": data["image0"].size(0),
                "hw0_i": data["image0"].shape[2:],
                "hw1_i": data["image1"].shape[2:],
            }
        )

        # 1. Feature Extraction
        if data["hw0_i"] == data["hw1_i"]:
            # faster & better BN convergence
            feats = self.backbone(
                torch.cat([data["image0"], data["image1"]], dim=0))
            f8, f16, f32, f8_fine = feats
            ms_feats = f8, f16, f32
            feat_f0, feat_f1 = f8_fine.chunk(2)
        else:
            # handle different input shapes
            # raise ValueError("image0 and image1 should have the same shape.")
            feats0, feats1 = self.backbone(data["image0"]), self.backbone(
                data["image1"]
            )
            f8_0, f16_0, f32_0, feat_f0 = feats0
            f8_1, f16_1, f32_1, feat_f1 = feats1
            ms_feats = f8_0, f16_0, f32_0, f8_1, f16_1, f32_1

        mask_c0 = mask_c1 = None  # mask is useful in training
        if "mask0" in data:
            mask_c0, mask_c1 = data["mask0"], data["mask1"]

        # 2.  Feature Interaction & Multi-Scale Fusion
        feat_c0, feat_c1 = self.neck(ms_feats, mask_c0, mask_c1)

        data.update(
            {
                "hw0_c": feat_c0.shape[2:],
                "hw1_c": feat_c1.shape[2:],
                "hw0_f": feat_c0.shape[2:] * self.config["local_resolution"],
                "hw1_f": feat_c1.shape[2:] * self.config["local_resolution"],
            }
        )
        feat_c0 = rearrange(feat_c0, "n c h w -> n (h w) c")
        feat_c1 = rearrange(feat_c1, "n c h w -> n (h w) c")
        feat_f0 = rearrange(feat_f0, "n c h w -> n (h w) c")
        feat_f1 = rearrange(feat_f1, "n c h w -> n (h w) c")

        # detect NaN during mixed precision training
        if self.config["mp"] and (
            torch.any(torch.isnan(feat_c0)) or torch.any(torch.isnan(feat_c1))
        ):
            detect_NaN(feat_c0, feat_c1)

        # 3. Coarse-Level Matching
        conf_matrix = self.coarse_matching(
            feat_c0,
            feat_c1,
            data,
            mask_c0=(
                mask_c0.view(mask_c0.size(0), -
                             1) if mask_c0 is not None else mask_c0
            ),
            mask_c1=(
                mask_c1.view(mask_c1.size(0), -
                             1) if mask_c1 is not None else mask_c1
            ),
        )

        if self.deploy:
            k = self.topk
            row_max_val, row_max_idx = torch.max(conf_matrix, dim=2)
            topk_val, topk_idx = torch.topk(row_max_val, k, dim=1)

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
            scale0 = scale * \
                data["scale0"][b_ids] if "scale0" in data else scale
            scale1 = scale * \
                data["scale1"][b_ids] if "scale1" in data else scale
            mkpts0_c = (
                torch.stack(
                    [
                        i_ids % data["hw0_c"][1],
                        torch.div(i_ids, data["hw0_c"][1],
                                  rounding_mode="floor"),
                    ],
                    dim=1,
                )
                * scale0
            )
            mkpts1_c = (
                torch.stack(
                    [
                        j_ids % data["hw1_c"][1],
                        torch.div(j_ids, data["hw1_c"][1],
                                  rounding_mode="floor"),
                    ],
                    dim=1,
                )
                * scale1
            )

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

        # 4. Fine-Level Matching
        K0 = data["i_ids"].shape[0] // data["bs"]
        K1 = data["j_ids"].shape[0] // data["bs"]
        feat_f0 = feat_f0[data["b_ids"], data["i_ids"]
                          ].reshape(data["bs"], K0, -1)
        feat_f1 = feat_f1[data["b_ids"], data["j_ids"]
                          ].reshape(data["bs"], K1, -1)
        feat_c0 = feat_c0[data["b_ids"], data["i_ids"]
                          ].reshape(data["bs"], K0, -1)
        feat_c1 = feat_c1[data["b_ids"], data["j_ids"]
                          ].reshape(data["bs"], K1, -1)

        if self.bi_directional_refine:
            # Bidirectional Refinement
            offset, score = self.fine_matching(
                torch.cat([feat_f0, feat_f1], dim=1),
                torch.cat([feat_f1, feat_f0], dim=1),
                torch.cat([feat_c0, feat_c1], dim=1),
                torch.cat([feat_c1, feat_c0], dim=1),
                data,
            )
        else:
            offset, score = self.fine_matching(
                feat_f0, feat_f1, feat_c0, feat_c1, data)

        if self.deploy:
            if self.bi_directional_refine:
                fine_offset01, fine_offset10 = offset.chunk(2)
                fine_score01, fine_score10 = score.unsqueeze(dim=1).chunk(2)
                output = torch.cat(
                    [mkpts0_c, mkpts1_c, fine_offset01, fine_offset10, fine_score01, fine_score10, mconf.unsqueeze(dim=1)], 1) # [K, 11]
            else:
                output = torch.cat(
                    [mkpts0_c, mkpts1_c, offset, score, mconf.unsqueeze(dim=1)], 1)
            return output

    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith("matcher."):
                state_dict[k.replace("matcher.", "", 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
