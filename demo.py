#!/usr/bin/env python3
"""Minimal homography demo for uncertainty extraction and CoRe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from core import (
    estimate_homography,
    load_calibration,
    refit_homography_core,
)
from core.homography_ransac import project_points
from matchers import canonical_name, load_matcher


REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_PARAMS = REPOSITORY_ROOT / "calibration" / "params.json"
DEFAULT_IMAGE0 = REPOSITORY_ROOT / "assets" / "test_A.jpg"
DEFAULT_IMAGE1 = REPOSITORY_ROOT / "assets" / "test_B.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a supported matcher and CoRe on the bundled demo pair."
    )
    parser.add_argument(
        "--matcher",
        default="eloftr",
        choices=["edm", "eloftr", "loftr", "gim_loftr", "comatch", "xfeat"],
    )
    parser.add_argument("--weights", default=None, help="Override checkpoint path.")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--resize", nargs=2, type=int, default=(1184, 896), metavar=("W", "H"))
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--max-lines", type=int, default=120)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="outputs/demo.png")
    return parser.parse_args()


def read_image(path: str, size: tuple[int, int]) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def draw_homography(
    canvas: np.ndarray,
    homography: np.ndarray | None,
    source_size: tuple[int, int],
    offset: int,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if homography is None:
        return
    width, height = source_size
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )
    warped = project_points(homography, corners)
    if not np.isfinite(warped).all():
        return
    warped[:, 0] += offset
    cv2.polylines(canvas, [np.round(warped).astype(np.int32)], True, color, 2, cv2.LINE_AA)
    anchor = tuple(np.round(warped[0]).astype(int))
    cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    matches: np.ndarray,
    weights: np.ndarray,
    max_lines: int,
) -> np.ndarray:
    height = max(image0.shape[0], image1.shape[0])
    width0 = image0.shape[1]
    canvas = np.zeros((height, width0 + image1.shape[1], 3), dtype=np.uint8)
    canvas[: image0.shape[0], :width0] = image0
    canvas[: image1.shape[0], width0:] = image1

    if len(matches) > max_lines:
        order = np.argsort(weights)
        indices = order[np.linspace(0, len(order) - 1, max_lines).astype(int)]
    else:
        indices = np.arange(len(matches))
    for index in indices:
        weight = float(np.clip(weights[index], 0.0, 1.0))
        color = (32, int(48 + 207 * weight), int(255 * (1.0 - weight)))
        point0 = tuple(np.round(matches[index, :2]).astype(int))
        point1 = tuple(
            np.round(matches[index, 2:4] + np.asarray([width0, 0])).astype(int)
        )
        cv2.line(canvas, point0, point1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, point0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point1, 2, color, -1, cv2.LINE_AA)
    return canvas


def main() -> None:
    args = parse_args()
    matcher_name = canonical_name(args.matcher)
    image_size = tuple(args.resize)
    image0 = read_image(str(DEFAULT_IMAGE0), image_size)
    image1 = read_image(str(DEFAULT_IMAGE1), image_size)
    matcher = load_matcher(
        matcher_name, weights=args.weights, device=args.device
    )
    result = matcher.match(image0, image1)
    matches = np.asarray(result["matches"], dtype=np.float64)
    if len(matches) < 8:
        raise RuntimeError(f"Only {len(matches)} matches were found; at least 8 are required.")
    params = load_calibration(args.params, matcher_name)

    points0 = matches[:, :2]
    points1 = matches[:, 2:4]
    initial_model, initial_inliers = estimate_homography(
        points0,
        points1,
        threshold=args.threshold,
    )
    if initial_model is None:
        raise RuntimeError("Initial homography estimation failed.")
    final_model, final_inliers, weights = refit_homography_core(
        initial_model,
        points0,
        points1,
        result["std_coarse"],
        result["std_fine"],
        result["confidence"],
        params,
        image_size,
        threshold=args.threshold,
    )
    summary = {
        "geometry": "homography",
        "matches": len(matches),
        "initial_inliers": int(initial_inliers.sum()),
        "core_inliers": int(final_inliers.sum()),
        "initial_homography": initial_model.tolist(),
        "core_homography": None if final_model is None else final_model.tolist(),
    }

    canvas = draw_matches(image0, image1, matches, weights, args.max_lines)
    draw_homography(
        canvas,
        initial_model,
        image_size,
        image0.shape[1],
        (180, 180, 180),
        "RANSAC",
    )
    draw_homography(
        canvas,
        final_model,
        image_size,
        image0.shape[1],
        (40, 220, 70),
        "RANSAC + CoRe",
    )
    text = (
        f"{matcher_name} | {len(matches)} matches | "
        f"{summary['initial_inliers']} -> {summary['core_inliers']} inliers"
    )
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 820), 34), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Could not write visualization: {output}")
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
