#!/usr/bin/env python3
"""Evaluate the same CoRe refit after several robust homography estimators."""

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
        "--estimators",
        default="ransac,lo-ransac,prosac,magsac",
        help="Comma-separated estimators.",
    )
    parser.add_argument("--resize", nargs=2, type=int, default=(1280, 960), metavar=("W", "H"))
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/robust_estimators.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    estimators = [
        item.strip() for item in args.estimators.split(",") if item.strip()
    ]
    if not estimators:
        raise ValueError("--estimators cannot be empty.")
    results = run_evaluation(
        data_root=args.data_root,
        matcher_name=args.matcher,
        estimators=estimators,
        methods={"baseline", "core"},
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
