#!/usr/bin/env python3
"""Reproduce the released HPatches baseline-versus-CoRe experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.hpatches_common import (  # noqa: E402
    print_table,
    run_evaluation,
    save_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--matcher",
        default="eloftr",
        choices=["edm", "eloftr", "loftr", "gim_loftr", "comatch", "xfeat"],
    )
    parser.add_argument("--weights", default=None)
    parser.add_argument(
        "--params",
        default=str(REPOSITORY_ROOT / "calibration" / "params.json"),
    )
    parser.add_argument(
        "--estimator",
        default="ransac",
        choices=[
            "ransac",
            "lo-ransac",
            "prosac",
            "magsac",
            "usac-fast",
            "usac-accurate",
        ],
    )
    parser.add_argument(
        "--methods",
        default="baseline,core",
        help="Comma-separated subset of baseline,core.",
    )
    parser.add_argument("--resize", nargs=2, type=int, default=(1280, 960), metavar=("W", "H"))
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/hpatches.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = {item.strip() for item in args.methods.split(",") if item.strip()}
    if not methods or not methods <= {"baseline", "core"}:
        raise ValueError("--methods must contain baseline, core, or both.")
    results = run_evaluation(
        data_root=args.data_root,
        matcher_name=args.matcher,
        estimators=[args.estimator],
        methods=methods,
        weights=args.weights,
        params_path=args.params,
        image_size=tuple(args.resize),
        threshold=args.threshold,
        max_pairs=args.max_pairs,
        device=args.device,
    )
    save_results(results, args.output)
    print_table(results)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
