#!/usr/bin/env python3
"""Run baseline PnP and PnP+CoRe on Aachen Day-Night v1.1.

The script uses retrieved database images to transfer COLMAP 3D points to each
query through dense matches. Match direction is database -> query so the
extracted uncertainty and CoRe residuals are both expressed in query pixels.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core import estimate_pnp, load_calibration, refit_pnp_core  # noqa: E402
from matchers import canonical_name, load_matcher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Aachen images_upright root.")
    parser.add_argument("--model", required=True, help="COLMAP reconstruction directory.")
    parser.add_argument(
        "--queries", required=True, help="queries_with_intrinsics.txt from Aachen."
    )
    parser.add_argument(
        "--retrieval",
        required=True,
        help="Text file containing '<query> <database>' pairs, ordered by rank.",
    )
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
    parser.add_argument("--resize", nargs=2, type=int, default=(1184, 896), metavar=("W", "H"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--association-radius",
        type=float,
        default=4.0,
        help="Maximum database-image distance to a registered COLMAP observation.",
    )
    parser.add_argument("--ransac-threshold", type=float, default=12.0)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-prefix", default="outputs/aachen")
    return parser.parse_args()


def parse_queries(path: str | Path) -> dict[str, dict]:
    queries = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields:
                continue
            name, model = fields[:2]
            width, height = (int(fields[2]), int(fields[3]))
            queries[name] = {
                "model": model,
                "width": width,
                "height": height,
                "params": np.asarray(fields[4:], dtype=np.float64),
            }
    return queries


def parse_retrieval(path: str | Path) -> dict[str, list[str]]:
    pairs: dict[str, list[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2:
                pairs[fields[0]].append(fields[1])
    return dict(pairs)


def camera_matrices(
    record: dict, resized: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray | None]:
    model = record["model"].upper()
    params = record["params"]
    if model == "SIMPLE_PINHOLE":
        focal, cx, cy = params
        fx = fy = focal
        distortion = None
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        distortion = None
    elif model == "SIMPLE_RADIAL":
        focal, cx, cy, k1 = params
        fx = fy = focal
        distortion = np.asarray([k1, 0.0, 0.0, 0.0])
    elif model == "RADIAL":
        focal, cx, cy, k1, k2 = params
        fx = fy = focal
        distortion = np.asarray([k1, k2, 0.0, 0.0])
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        distortion = np.asarray([k1, k2, p1, p2])
    else:
        raise ValueError(f"Unsupported query camera model: {model}")

    scale_x = resized[0] / record["width"]
    scale_y = resized[1] / record["height"]
    intrinsics = np.asarray(
        [
            [fx * scale_x, 0.0, cx * scale_x],
            [0.0, fy * scale_y, cy * scale_y],
            [0.0, 0.0, 1.0],
        ]
    )
    return intrinsics, distortion


def resolve_image(root: Path, name: str) -> Path:
    candidates = [root / name]
    if name.startswith("db/"):
        candidates.append(root / name[3:])
    if name.startswith("query/"):
        candidates.append(root / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve '{name}' below {root}.")


def read_resized(path: Path, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    original_size = (image.shape[1], image.shape[0])
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), original_size


def registered_observations(reconstruction, image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points2d = []
    points3d = []
    identifiers = []
    for observation in image.points2D:
        point3d_id = int(observation.point3D_id)
        if point3d_id < 0 or point3d_id not in reconstruction.points3D:
            continue
        points2d.append(np.asarray(observation.xy, dtype=np.float64))
        points3d.append(
            np.asarray(reconstruction.points3D[point3d_id].xyz, dtype=np.float64)
        )
        identifiers.append(point3d_id)
    if not points2d:
        return (
            np.empty((0, 2)),
            np.empty((0, 3)),
            np.empty(0, dtype=np.int64),
        )
    return np.asarray(points2d), np.asarray(points3d), np.asarray(identifiers)


def collect_correspondences(
    query_name: str,
    database_names: list[str],
    images_root: Path,
    reconstruction,
    model_images: dict[str, object],
    matcher,
    resized: tuple[int, int],
    association_radius: float,
) -> dict[str, np.ndarray]:
    query_image, _ = read_resized(resolve_image(images_root, query_name), resized)
    candidates: dict[int, tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = {}

    for database_name in database_names:
        model_image = model_images.get(database_name)
        if model_image is None and database_name.startswith("db/"):
            model_image = model_images.get(database_name[3:])
        if model_image is None:
            continue
        database_image, original_size = read_resized(
            resolve_image(images_root, database_name), resized
        )
        registered2d, registered3d, point_ids = registered_observations(
            reconstruction, model_image
        )
        if not len(registered2d):
            continue

        result = matcher.match(database_image, query_image)
        matches = np.asarray(result["matches"], dtype=np.float64)
        if not len(matches):
            continue
        database_points = matches[:, :2] * np.asarray(
            [original_size[0] / resized[0], original_size[1] / resized[1]]
        )
        distances, indices = cKDTree(registered2d).query(
            database_points, k=1, distance_upper_bound=association_radius
        )
        valid = np.isfinite(distances) & (indices < len(registered2d))
        for match_index in np.flatnonzero(valid):
            registered_index = int(indices[match_index])
            point_id = int(point_ids[registered_index])
            score = float(result["confidence"][match_index])
            previous = candidates.get(point_id)
            if previous is not None and previous[0] >= score:
                continue
            candidates[point_id] = (
                score,
                registered3d[registered_index],
                matches[match_index, 2:4],
                np.asarray(result["std_coarse"][match_index]),
                np.asarray(result["std_fine"][match_index]),
            )

    if not candidates:
        return {
            "points3d": np.empty((0, 3)),
            "points2d": np.empty((0, 2)),
            "std_coarse": np.empty((0, 2)),
            "std_fine": np.empty((0, 2)),
            "confidence": np.empty(0),
        }
    values = list(candidates.values())
    return {
        "confidence": np.asarray([value[0] for value in values]),
        "points3d": np.asarray([value[1] for value in values]),
        "points2d": np.asarray([value[2] for value in values]),
        "std_coarse": np.asarray([value[3] for value in values]),
        "std_fine": np.asarray([value[4] for value in values]),
    }


def pose_line(name: str, rotation_vector: np.ndarray, translation: np.ndarray) -> str:
    rotation = cv2.Rodrigues(np.asarray(rotation_vector).reshape(3, 1))[0]
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    qx, qy, qz, qw = quaternion_xyzw
    tx, ty, tz = np.asarray(translation).reshape(3)
    return (
        f"{name} {qw:.12g} {qx:.12g} {qy:.12g} {qz:.12g} "
        f"{tx:.12g} {ty:.12g} {tz:.12g}"
    )


def main() -> None:
    args = parse_args()
    try:
        import pycolmap
    except ImportError as error:
        raise ImportError(
            "The Aachen experiment requires pycolmap: pip install pycolmap"
        ) from error

    resized = tuple(args.resize)
    queries = parse_queries(args.queries)
    retrieval = parse_retrieval(args.retrieval)
    reconstruction = pycolmap.Reconstruction(args.model)
    model_images = {image.name: image for image in reconstruction.images.values()}
    matcher_name = canonical_name(args.matcher)
    matcher = load_matcher(
        matcher_name,
        weights=args.weights,
        device=args.device,
        confidence_threshold=0.01,
    )
    params = load_calibration(args.params, matcher_name)

    selected_queries = [name for name in queries if name in retrieval]
    if args.max_queries is not None:
        selected_queries = selected_queries[: args.max_queries]
    baseline_lines = []
    core_lines = []
    records = []
    for query_name in tqdm(selected_queries, desc=f"Aachen / {matcher_name}"):
        record = {"query": query_name, "status": "failed"}
        try:
            correspondences = collect_correspondences(
                query_name,
                retrieval[query_name][: args.top_k],
                Path(args.images),
                reconstruction,
                model_images,
                matcher,
                resized,
                args.association_radius,
            )
            record["correspondences"] = int(len(correspondences["points3d"]))
            intrinsics, distortion = camera_matrices(queries[query_name], resized)
            rotation, translation, initial_inliers = estimate_pnp(
                correspondences["points3d"],
                correspondences["points2d"],
                intrinsics,
                distortion,
                threshold=args.ransac_threshold,
            )
            if rotation is None or translation is None:
                record["message"] = "PnP RANSAC failed"
                records.append(record)
                continue
            baseline_lines.append(pose_line(query_name, rotation, translation))
            core_rotation, core_translation, core_inliers, _ = refit_pnp_core(
                rotation,
                translation,
                correspondences["points3d"],
                correspondences["points2d"],
                intrinsics,
                correspondences["std_coarse"],
                correspondences["std_fine"],
                correspondences["confidence"],
                params,
                resized,
                distortion,
                threshold=args.ransac_threshold,
            )
            if core_rotation is None or core_translation is None:
                core_rotation, core_translation = rotation, translation
                core_inliers = initial_inliers
                record["core_fallback"] = True
            core_lines.append(
                pose_line(query_name, core_rotation, core_translation)
            )
            record.update(
                {
                    "status": "ok",
                    "baseline_inliers": int(initial_inliers.sum()),
                    "core_inliers": int(core_inliers.sum()),
                }
            )
        except Exception as error:
            record["message"] = str(error)
        records.append(record)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    baseline_path = prefix.with_name(f"{prefix.name}_baseline.txt")
    core_path = prefix.with_name(f"{prefix.name}_core.txt")
    summary_path = prefix.with_name(f"{prefix.name}_summary.json")
    baseline_path.write_text("\n".join(baseline_lines) + "\n", encoding="utf-8")
    core_path.write_text("\n".join(core_lines) + "\n", encoding="utf-8")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "matcher": matcher_name,
                "queries_requested": len(selected_queries),
                "queries_localized": len(core_lines),
                "records": records,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Localized {len(core_lines)}/{len(selected_queries)} queries.")
    print(f"Baseline poses: {baseline_path}")
    print(f"CoRe poses: {core_path}")


if __name__ == "__main__":
    main()
