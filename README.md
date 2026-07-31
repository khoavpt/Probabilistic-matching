<p align="center">
  <h1 align="center">Semi-Dense Matching Uncertainty Is Not Just Local Confidence 🎯</h1>
  <!-- <p align="center">
    <a href="https://www.linkedin.com/in/philipplindenberger/">Philipp Lindenberger</a>
    ·
    <a href="https://psarlin.com/">Paul-Edouard&nbsp;Sarlin</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/mapoll/">Marc&nbsp;Pollefeys</a>
  </p> -->
  <h2 align="center">
    <!-- <p>WACV 2026?</p> -->
    <a href="https://arxiv.org/pdf/2306.13643.pdf" align="center">Paper</a> | 
    <!-- <a href="https://colab.research.google.com/github/cvg/LightGlue/blob/main/demo.ipynb" align="center">Colab</a> | -->
    <a href="https://huggingface.co/spaces/ETH-CVG/LightGlue" align="center">🤗 Demo </a> | 
  </h2>

</p>
<p align="center">
    <a href="https://arxiv.org/abs/2306.13643"><img src="assets/results_teaser.png" alt="example" width=90%></a>
    <br>
    <em>LightGlue is a deep neural network that matches sparse local features across image pairs.<br>An adaptive mechanism makes it fast for easy pairs (top) and reduces the computational complexity for difficult ones (bottom).</em>
</p>




This repository host the official implementation of **Semi-Dense Matching Uncertainty Is Not Just Local Confidence**. Our works focus on two main things:
- Model the full spatial matching error (in pixels) using uncertainty cues extracted from the matching process.
- Propose **CoRe**, a lightweight add-on which leverages the calibrated error model for robust geometry estimation.

The code currently shows integration and experiments on various Semi-dense models including LoFTR, GIM_{LOFTR}, EfficientLoFTR, CoMatch, EDM, XFeat. Installations, Usage and Experiment recreation details are mentioned below.

## Installation 

Clone the repo and setup environment using conda

```bash
git clone REPOSITORY_URL
cd REPOSITORY_NAME

conda env create -f environment.yml
conda activate core
```

Download the checkpoint(s) you need into `weights/`:

| Matcher | Official checkpoint | Expected path |
|---|---|---|
| EfficientLoFTR | [download](https://drive.google.com/drive/folders/1GOw6iVqsB-f1vmG6rNmdCcgwfB4VZ7_Q?usp=sharing) | `weights/eloftr_outdoor.ckpt` |
| LoFTR | [download](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) | `weights/loftr_outdoor.ckpt` |
| EDM | [download](https://drive.google.com/drive/folders/1PkYNihwgnNwqQeeewBz4OUDrvY8xFdH0?usp=sharing) | `weights/edm_outdoor.ckpt` |
| GIM-LoFTR | [GIM repository](https://github.com/xuelunshen/gim) | `weights/gim_loftr_50h.ckpt` |
| CoMatch | [download](https://drive.google.com/file/d/1DHd7LZtNH1vwhoTUA8a8nVIiIpx-38RX/view?usp=sharing) | `weights/comatch_outdoor.ckpt` |
| XFeat | [download](https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt) | `weights/xfeat.pt` |

Pass `--weights /another/path/checkpoint` to any public script to override the
default location.

## Usage  [![](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ETH-CVG/LightGlue)

### Refinement with CoRe
Run the homography demo on one image pair:

```bash
python demo.py
```

Minimal script to use CoRe 

```python
from core import estimate_homography, load_calibration, refit_homography_core

params = load_calibration("calibration/params.json", "eloftr")
H0, inliers0 = estimate_homography(points0, points1, threshold=2.0)
H, inliers, weights = refit_homography_core(
    H0,
    points0,
    points1,
    std_coarse,
    std_fine,
    confidence,
    params,
    image_size=(1280, 960),
    threshold=2.0,
)
```

Equivalent version for Essential matrix estmation and PnP are provided in `core/essential_ransac.py` and `core/pnp_ransac.py`.

### Calibration

Released calibration parameters are stored in `calibration/params.json`. To fit a
new matcher, run calibration directly on MegaDepth-style depth data:

```bash
MATCHERS=eloftr OUTPUT=calibration/eloftr_custom.json \
./scripts/fit_calibration.sh /path/to/MegaDepth/MegaDepth/train
```

To calibrate several supported matchers with the same wrapper:

```bash
MATCHERS="eloftr loftr xfeat" OUTPUT_DIR=outputs/calibration \
./scripts/fit_calibration.sh /path/to/MegaDepth/MegaDepth/train
```

## Reproduce Resulst

HPatches:

```bash
./scripts/run_hpatches.sh /path/to/hpatches-sequences-release
```

Robust estimators:

```bash
./scripts/run_robust_estimators.sh /path/to/hpatches-sequences-release
```

Aachen Day-Night v1.1:

```bash
./scripts/run_aachen.sh \
  /path/to/AachenDayNight/images/images_upright \
  /path/to/AachenDayNight/3D-models/aachen_v_1_1 \
  /path/to/AachenDayNight/queries/queries_with_intrinsics.txt \
  /path/to/pairs-query-netvlad50.txt
```

All wrappers accept extra CLI flags and support environment overrides. See
`scripts/README.md` for the common variables.

## Qualitative results

<p align="center">
  <img src="assets/qualitative_homography.png" alt="Qualitative homography refit comparison" width="950">
</p>

## Citation

<!-- TODO(release): replace with the final WACV BibTeX entry. -->

If you find this work useful, please cite the paper. The final BibTeX entry will
be added after publication.

## Acknowledgements and license

This work builds on the official matcher and robust-estimator
implementations. Third-party code remains under the licenses included in each
subdirectory.

<!-- TODO(release): add the chosen license for the original CoRe code. -->
