"""matching / sift for the formal SIFT + prior + BAE pipeline."""

from collections import defaultdict
from pathlib import Path
import numpy as np


def canonicalize_pair_array(pairs, num_images=None):
    canonical = set()
    array = np.asarray(pairs, dtype=np.int64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    for raw_i, raw_j in array.reshape(-1, 2).tolist():
        i, j = sorted((int(raw_i), int(raw_j)))
        if i == j:
            continue
        if num_images is not None and not (0 <= i < num_images and 0 <= j < num_images):
            raise ValueError(
                f"Pair ({i}, {j}) is outside image range [0, {num_images})"
            )
        canonical.add((i, j))
    return np.asarray(sorted(canonical), dtype=np.int64).reshape(-1, 2)


def build_temporal_pairs(num_images, window):
    num_images = int(num_images)
    window = int(window)
    if num_images < 0:
        raise ValueError("num_images must be >= 0")
    if window < 0:
        raise ValueError("window must be >= 0")
    pairs = [
        (i, j)
        for i in range(num_images)
        for j in range(i + 1, min(num_images, i + window + 1))
    ]
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def build_sift_candidate_pairs(legacy_pose_pairs, num_images, temporal_window):
    legacy = canonicalize_pair_array(legacy_pose_pairs, num_images=num_images)
    temporal = build_temporal_pairs(num_images, temporal_window)
    legacy_set = {tuple(pair) for pair in legacy.tolist()}
    temporal_set = {tuple(pair) for pair in temporal.tolist()}
    combined = np.asarray(
        sorted(legacy_set | temporal_set),
        dtype=np.int64,
    ).reshape(-1, 2)
    return combined, {
        "legacy_pose_pair_count": int(len(legacy_set)),
        "temporal_pair_count": int(len(temporal_set)),
        "temporal_overlap_pair_count": int(len(legacy_set & temporal_set)),
        "temporal_new_pair_count": int(len(temporal_set - legacy_set)),
        "sift_candidate_pair_count": int(combined.shape[0]),
    }


def _grid_coverage_for_matched_keypoints(
    keypoints,
    matched_indices,
    image_size_hw,
    grid_size,
    min_inliers_per_cell,
):
    keypoints = np.asarray(keypoints, dtype=np.float64)
    matched_indices = np.asarray(matched_indices, dtype=np.int64)
    height, width = (int(value) for value in image_size_hw)
    grid_size = int(grid_size)
    min_inliers_per_cell = int(min_inliers_per_cell)
    if grid_size <= 0:
        raise ValueError("grid_size must be >= 1")
    if min_inliers_per_cell <= 0:
        raise ValueError("min_inliers_per_cell must be >= 1")
    if height <= 0 or width <= 0:
        raise ValueError("image_size_hw must contain positive values")
    if matched_indices.size == 0 or keypoints.shape[0] == 0:
        return 0.0, 0
    valid = (matched_indices >= 0) & (matched_indices < keypoints.shape[0])
    xy = keypoints[matched_indices[valid], :2]
    if xy.size == 0:
        return 0.0, 0
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if xy.size == 0:
        return 0.0, 0
    grid_x = np.clip(
        np.floor(xy[:, 0] * grid_size / width).astype(np.int64),
        0,
        grid_size - 1,
    )
    grid_y = np.clip(
        np.floor(xy[:, 1] * grid_size / height).astype(np.int64),
        0,
        grid_size - 1,
    )
    counts = np.bincount(
        grid_y * grid_size + grid_x,
        minlength=grid_size * grid_size,
    )
    occupied = int(np.sum(counts >= min_inliers_per_cell))
    return float(occupied / (grid_size * grid_size)), occupied


def analyze_sift_schedule_graph(
    database_path,
    image_names,
    candidate_pairs,
    image_size_hw,
    *,
    legacy_pose_pairs=None,
    temporal_pairs=None,
    grid_size=8,
    min_inliers_per_cell=2,
    min_pair_inliers=128,
    min_grid_coverage=0.20,
):
    from ffba.reporting.statistics import summarize_distribution
    from ffba.runtime import _lazy_import_pycolmap

    pycolmap = _lazy_import_pycolmap()
    candidate_pairs = canonicalize_pair_array(
        candidate_pairs,
        num_images=len(image_names),
    )
    legacy_set = {
        tuple(pair)
        for pair in canonicalize_pair_array(
            legacy_pose_pairs if legacy_pose_pairs is not None else [],
            num_images=len(image_names),
        ).tolist()
    }
    temporal_set = {
        tuple(pair)
        for pair in canonicalize_pair_array(
            temporal_pairs if temporal_pairs is not None else [],
            num_images=len(image_names),
        ).tolist()
    }
    min_pair_inliers = int(min_pair_inliers)
    min_grid_coverage = float(min_grid_coverage)
    if min_pair_inliers < 0:
        raise ValueError("min_pair_inliers must be >= 0")
    if not 0.0 <= min_grid_coverage <= 1.0:
        raise ValueError("min_grid_coverage must be in [0, 1]")

    database = pycolmap.Database.open(str(database_path))
    records = []
    verified_pairs = 0
    valid_edges = []
    source_verified = defaultdict(int)
    source_valid = defaultdict(int)
    try:
        database_images = {
            str(Path(image.name)): image for image in database.read_all_images()
        }
        image_ids = []
        keypoints = []
        for name in image_names:
            normalized_name = str(Path(name))
            image = database_images.get(normalized_name)
            if image is None:
                raise ValueError(f"Image {normalized_name!r} is missing from SIFT DB")
            image_ids.append(int(image.image_id))
            current = database.read_keypoints(image.image_id)
            if current is None or len(current) == 0:
                keypoints.append(np.empty((0, 2), dtype=np.float64))
            else:
                keypoints.append(np.asarray(current[:, :2], dtype=np.float64))

        for i, j in candidate_pairs.tolist():
            pair = (int(i), int(j))
            sources = []
            if pair in legacy_set:
                sources.append("pose")
            if pair in temporal_set:
                sources.append("temporal")
            geometry = database.read_two_view_geometry(image_ids[i], image_ids[j])
            inlier_matches = getattr(geometry, "inlier_matches", None)
            if inlier_matches is None:
                inlier_matches = np.empty((0, 2), dtype=np.int64)
            else:
                inlier_matches = np.asarray(inlier_matches, dtype=np.int64).reshape(
                    -1, 2
                )
            inlier_count = int(inlier_matches.shape[0])
            if inlier_count > 0:
                verified_pairs += 1
                for source in sources:
                    source_verified[source] += 1

            source_coverage, source_cells = _grid_coverage_for_matched_keypoints(
                keypoints[i],
                inlier_matches[:, 0],
                image_size_hw,
                grid_size,
                min_inliers_per_cell,
            )
            target_coverage, target_cells = _grid_coverage_for_matched_keypoints(
                keypoints[j],
                inlier_matches[:, 1],
                image_size_hw,
                grid_size,
                min_inliers_per_cell,
            )
            valid = bool(
                inlier_count >= min_pair_inliers
                and source_coverage >= min_grid_coverage
                and target_coverage >= min_grid_coverage
            )
            if valid:
                valid_edges.append(pair)
                for source in sources:
                    source_valid[source] += 1
            records.append(
                {
                    "pair": [i, j],
                    "sources": sources,
                    "inlier_count": inlier_count,
                    "source_grid_coverage": source_coverage,
                    "target_grid_coverage": target_coverage,
                    "source_occupied_cells": source_cells,
                    "target_occupied_cells": target_cells,
                    "valid_schedule_edge": valid,
                }
            )
    finally:
        database.close()

    source_totals = {
        "pose": int(
            sum(tuple(pair) in legacy_set for pair in candidate_pairs.tolist())
        ),
        "temporal": int(
            sum(tuple(pair) in temporal_set for pair in candidate_pairs.tolist())
        ),
    }
    source_stats = {}
    for source, total in source_totals.items():
        verified = int(source_verified[source])
        valid = int(source_valid[source])
        source_stats[source] = {
            "candidate_pairs": total,
            "verified_pairs": verified,
            "valid_schedule_pairs": valid,
            "verified_rate": float(verified / total) if total else 0.0,
            "valid_rate": float(valid / total) if total else 0.0,
        }
    return {
        "config": {
            "grid_size": int(grid_size),
            "min_inliers_per_cell": int(min_inliers_per_cell),
            "min_pair_inliers": min_pair_inliers,
            "min_grid_coverage": min_grid_coverage,
        },
        "candidate_pair_count": int(candidate_pairs.shape[0]),
        "verified_pair_count": int(verified_pairs),
        "valid_schedule_pair_count": int(len(valid_edges)),
        "valid_edges": [list(edge) for edge in valid_edges],
        "inlier_count": summarize_distribution(
            [record["inlier_count"] for record in records]
        ),
        "source_grid_coverage": summarize_distribution(
            [record["source_grid_coverage"] for record in records]
        ),
        "target_grid_coverage": summarize_distribution(
            [record["target_grid_coverage"] for record in records]
        ),
        "source_stats": source_stats,
        "pairs": records,
    }
