# Third-party matcher code

This directory contains only the inference code needed by the released matcher
adapters. The coarse/fine matching modules include the uncertainty-cue
extraction used in the paper.

The source is derived from the official EDM, EfficientLoFTR, LoFTR,
GIM-LoFTR, CoMatch, and XFeat repositories. Each subdirectory retains its
original license. Training pipelines, datasets, checkpoints, notebooks,
figures, and nested Git histories are intentionally excluded.

| Matcher | Upstream | Uncertainty extraction |
|---|---|---|
| EDM | [chicleee/EDM](https://github.com/chicleee/EDM) | `src/edm/head/{coarse_matching,fine_matching}.py` |
| EfficientLoFTR | [zju3dv/EfficientLoFTR](https://github.com/zju3dv/EfficientLoFTR) | `src/loftr/utils/{coarse_matching,fine_matching}.py` |
| LoFTR | [zju3dv/LoFTR](https://github.com/zju3dv/LoFTR) | `src/loftr/utils/{coarse_matching,fine_matching}.py` |
| GIM-LoFTR | [xuelunshen/gim](https://github.com/xuelunshen/gim) | `networks/loftr/utils/{coarse_matching,fine_matching}.py` |
| CoMatch | [ZizhuoLi/CoMatch](https://github.com/ZizhuoLi/CoMatch) | `src/loftr/utils/{coarse_matching,fine_matching_epipolar}.py` |
| XFeat | [verlab/accelerated_features](https://github.com/verlab/accelerated_features) | `modules/xfeat.py` |
