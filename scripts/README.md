# Shell wrappers

This folder contains thin wrappers around the released calibration and
experiment entrypoints. Activate your environment first, then run:

- `./scripts/fit_calibration.sh /path/to/MegaDepth/MegaDepth/train`
- `./scripts/run_hpatches.sh /path/to/hpatches-sequences-release`
- `./scripts/run_robust_estimators.sh /path/to/hpatches-sequences-release`
- `./scripts/run_aachen.sh /path/to/images_upright /path/to/model /path/to/queries.txt /path/to/retrieval.txt`

All scripts accept extra Python CLI flags after the required positional
arguments. Common environment overrides include `MATCHERS`, `WEIGHTS`,
`PARAMS_JSON`, `OUTPUT`, `OUTPUT_DIR`, `DEVICE`, `IMAGE_WIDTH`, and
`IMAGE_HEIGHT`.

For calibration, `MATCHERS` may contain one or more matcher names:

```bash
MATCHERS="eloftr loftr xfeat" OUTPUT_DIR=outputs/calibration \
./scripts/fit_calibration.sh /path/to/MegaDepth/MegaDepth/train
```
