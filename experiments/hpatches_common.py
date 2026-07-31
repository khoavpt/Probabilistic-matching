"""Shared HPatches evaluation used by the released experiment scripts."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

from core import estimate_homography, load_calibration, refit_homography_core
from core.homography_ransac import project_points
from matchers import canonical_name, load_matcher


EVALUATION_THRESHOLDS = (1.0, 3.0, 5.0, 10.0)


def iter_pairs(root: Path, max_pairs: int | None = None):
    count = 0
    for sequence in sorted(path for path in root.iterdir() if path.is_dir()):
        reference = sequence / "1.ppm"
        if not reference.is_file():
            continue
        for index in range(2, 7):
            target = sequence / f"{index}.ppm"
            homography = sequence / f"H_1_{index}"
            if not target.is_file() or not homography.is_file():
                continue
            yield sequence.name, reference, target, homography
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


def read_resized(path: Path, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    original_size = (image.shape[1], image.shape[0])
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), original_size


def scale_homography(
    homography: np.ndarray,
    source_original: tuple[int, int],
    target_original: tuple[int, int],
    evaluation_size: tuple[int, int],
) -> np.ndarray:
    width, height = evaluation_size
    source_scale = np.diag(
        [source_original[0] / width, source_original[1] / height, 1.0]
    )
    target_scale = np.diag(
        [target_original[0] / width, target_original[1] / height, 1.0]
    )
    return np.linalg.inv(target_scale) @ homography @ source_scale


def corner_error(
    predicted: np.ndarray | None,
    ground_truth: np.ndarray,
    image_size: tuple[int, int],
) -> float:
    if predicted is None or not np.isfinite(predicted).all():
        return float("inf")
    width, height = image_size
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float64,
    )
    predicted_corners = project_points(predicted, corners)
    true_corners = project_points(ground_truth, corners)
    if not np.isfinite(predicted_corners).all():
        return float("inf")
    return float(np.mean(np.linalg.norm(predicted_corners - true_corners, axis=1)))


def error_auc(errors: Iterable[float], threshold: float) -> float:
    errors = np.asarray(list(errors), dtype=np.float64)
    if not len(errors):
        return 0.0
    finite = np.sort(errors[np.isfinite(errors) & (errors <= threshold)])
    x = np.concatenate(([0.0], finite, [threshold]))
    recall = np.concatenate(
        (
            [0.0],
            np.arange(1, len(finite) + 1, dtype=np.float64) / len(errors),
            [len(finite) / len(errors)],
        )
    )
    return float(np.trapz(recall, x) / threshold)


def summarize(errors: list[float], runtimes: list[float]) -> dict:
    array = np.asarray(errors, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "pairs": int(len(array)),
        "failures": int((~np.isfinite(array)).sum()),
        "mean_corner_error": float(np.mean(finite)) if len(finite) else float("inf"),
        "auc": {
            f"auc@{int(threshold)}": error_auc(array, threshold)
            for threshold in EVALUATION_THRESHOLDS
        },
        "correct": {
            f"@{int(threshold)}px": float(np.mean(array <= threshold))
            if len(array)
            else 0.0
            for threshold in EVALUATION_THRESHOLDS
        },
        "mean_geometry_time": float(np.mean(runtimes)) if runtimes else 0.0,
    }


def run_evaluation(
    *,
    data_root: str | Path,
    matcher_name: str,
    estimators: list[str],
    methods: set[str],
    weights: str | Path | None,
    params_path: str | Path,
    image_size: tuple[int, int],
    threshold: float,
    max_pairs: int | None,
    device: str | None,
) -> dict:
    root = Path(data_root)
    pairs = list(iter_pairs(root, max_pairs))
    if not pairs:
        raise FileNotFoundError(
            f"No HPatches sequences found under {root}. Expected <sequence>/1.ppm."
        )
    matcher_name = canonical_name(matcher_name)
    matcher = load_matcher(
        matcher_name, weights=weights, device=device, confidence_threshold=0.01
    )
    params = load_calibration(params_path, matcher_name)
    stats = {
        estimator: {
            method: {"errors": [], "runtimes": []} for method in methods
        }
        for estimator in estimators
    }
    match_times = []
    match_counts = []

    import time

    for _, reference_path, target_path, homography_path in tqdm(
        pairs, desc=f"HPatches / {matcher_name}"
    ):
        image0, original0 = read_resized(reference_path, image_size)
        image1, original1 = read_resized(target_path, image_size)
        ground_truth = scale_homography(
            np.loadtxt(homography_path),
            original0,
            original1,
            image_size,
        )
        result = matcher.match(image0, image1)
        matches = np.asarray(result["matches"], dtype=np.float64)
        match_times.append(float(result["elapsed"]))
        match_counts.append(len(matches))
        points0 = matches[:, :2]
        points1 = matches[:, 2:4]

        for estimator in estimators:
            start = time.perf_counter()
            initial, _ = estimate_homography(
                points0,
                points1,
                estimator=estimator,
                threshold=threshold,
            )
            initial_time = time.perf_counter() - start
            if "baseline" in methods:
                stats[estimator]["baseline"]["errors"].append(
                    corner_error(initial, ground_truth, image_size)
                )
                stats[estimator]["baseline"]["runtimes"].append(initial_time)

            if "core" in methods:
                start = time.perf_counter()
                if initial is None:
                    refined = None
                else:
                    refined, _, _ = refit_homography_core(
                        initial,
                        points0,
                        points1,
                        result["std_coarse"],
                        result["std_fine"],
                        result["confidence"],
                        params,
                        image_size,
                        threshold=threshold,
                    )
                refit_time = time.perf_counter() - start
                stats[estimator]["core"]["errors"].append(
                    corner_error(refined, ground_truth, image_size)
                )
                stats[estimator]["core"]["runtimes"].append(
                    initial_time + refit_time
                )

    output = {
        "matcher": matcher_name,
        "data_root": str(root),
        "image_size": list(image_size),
        "ransac_threshold": threshold,
        "pairs": len(pairs),
        "mean_matches": float(np.mean(match_counts)),
        "mean_matching_time": float(np.mean(match_times)),
        "estimators": {},
    }
    for estimator, method_stats in stats.items():
        output["estimators"][estimator] = {
            method: summarize(values["errors"], values["runtimes"])
            for method, values in method_stats.items()
        }
    return output


def save_results(results: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")


def print_table(results: dict) -> None:
    print("estimator, method, AUC@1, AUC@3, AUC@5, AUC@10")
    for estimator, method_results in results["estimators"].items():
        for method, values in method_results.items():
            auc = values["auc"]
            print(
                f"{estimator}, {method}, "
                f"{100 * auc['auc@1']:.2f}, "
                f"{100 * auc['auc@3']:.2f}, "
                f"{100 * auc['auc@5']:.2f}, "
                f"{100 * auc['auc@10']:.2f}"
            )
