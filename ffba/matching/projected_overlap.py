"""matching / projected_overlap for the formal SIFT + prior + BAE pipeline."""

import time
from collections import defaultdict
import numpy as np


def _regular_grid_samples(height, width, max_samples):
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("projected-overlap max_samples must be >= 1")
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid depth image size: {(height, width)}")

    grid_height = max(1, int(np.sqrt(max_samples * height / width)))
    grid_width = max(1, max_samples // grid_height)
    grid_height = min(grid_height, height)
    grid_width = min(grid_width, width)
    ys = np.linspace(0, height - 1, grid_height).round().astype(np.int64)
    xs = np.linspace(0, width - 1, grid_width).round().astype(np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    samples = np.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
    return np.unique(samples, axis=0)


def _bilinear_sample_numpy(value_map, xy):
    value_map = np.asarray(value_map)
    xy = np.asarray(xy, dtype=np.float64)
    height, width = value_map.shape
    output = np.full(xy.shape[0], np.nan, dtype=np.float64)
    valid = (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= height - 1)
    )
    if not np.any(valid):
        return output

    valid_indices = np.flatnonzero(valid)
    x = xy[valid_indices, 0]
    y = xy[valid_indices, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    output[valid_indices] = (
        value_map[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + value_map[y0, x1] * wx * (1.0 - wy)
        + value_map[y1, x0] * (1.0 - wx) * wy
        + value_map[y1, x1] * wx * wy
    )
    return output


def _depth_confidence_thresholds(depth, depth_conf, confidence_quantile):
    num_images = depth.shape[0]
    if depth_conf is None:
        return np.full(num_images, -np.inf, dtype=np.float64)
    if not 0.0 <= confidence_quantile < 1.0:
        raise ValueError("confidence_quantile must be in [0, 1)")

    thresholds = np.full(num_images, -np.inf, dtype=np.float64)
    for image_idx in range(num_images):
        depth_map = depth[image_idx]
        confidence_map = depth_conf[image_idx]
        valid = (
            np.isfinite(depth_map) & (depth_map > 1e-6) & np.isfinite(confidence_map)
        )
        values = confidence_map[valid]
        if values.size:
            thresholds[image_idx] = float(np.quantile(values, confidence_quantile))
    return thresholds


def _prepare_projected_overlap_source(
    center,
    sample_xy,
    depth,
    depth_conf,
    confidence_thresholds,
    extrinsic,
    intrinsics,
    grid_size=8,
):
    x = sample_xy[:, 0]
    y = sample_xy[:, 1]
    source_depth = depth[center, y, x]
    valid = np.isfinite(source_depth) & (source_depth > 1e-6)
    if depth_conf is not None:
        source_conf = depth_conf[center, y, x]
        valid &= np.isfinite(source_conf) & (
            source_conf >= confidence_thresholds[center]
        )

    source_xy = sample_xy[valid].astype(np.float64, copy=False)
    source_depth = source_depth[valid].astype(np.float64, copy=False)
    if source_xy.shape[0] == 0:
        return {
            "xy": source_xy,
            "world_points": np.empty((0, 3), dtype=np.float64),
            "grid_cells": np.empty((0,), dtype=np.int64),
            "num_grid_cells": 0,
        }

    intrinsic = intrinsics[center]
    rays = np.stack(
        [
            (source_xy[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
            (source_xy[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1],
            np.ones(source_xy.shape[0], dtype=np.float64),
        ],
        axis=-1,
    )
    camera_points = rays * source_depth[:, None]
    rotation = extrinsic[center, :3, :3]
    translation = extrinsic[center, :3, 3]
    world_points = (camera_points - translation) @ rotation

    height, width = depth.shape[1:3]
    grid_x = np.minimum(
        (source_xy[:, 0] * grid_size / max(width, 1)).astype(np.int64),
        grid_size - 1,
    )
    grid_y = np.minimum(
        (source_xy[:, 1] * grid_size / max(height, 1)).astype(np.int64),
        grid_size - 1,
    )
    grid_cells = grid_y * grid_size + grid_x
    return {
        "xy": source_xy,
        "world_points": world_points,
        "grid_cells": grid_cells,
        "num_grid_cells": int(np.unique(grid_cells).size),
    }


def _score_projected_overlap_pair(
    center,
    neighbor,
    source,
    depth,
    depth_conf,
    confidence_thresholds,
    extrinsic,
    intrinsics,
    reprojection_threshold,
):
    source_xy = source["xy"]
    world_points = source["world_points"]
    num_source = int(source_xy.shape[0])
    if num_source == 0:
        return {
            "source_samples": 0,
            "visible_samples": 0,
            "depth_valid_samples": 0,
            "consistent_samples": 0,
            "projected_visible_ratio": 0.0,
            "projected_depth_valid_ratio": 0.0,
            "projected_overlap": 0.0,
            "projected_grid_coverage": 0.0,
        }

    target_rotation = extrinsic[neighbor, :3, :3]
    target_translation = extrinsic[neighbor, :3, 3]
    target_camera = world_points @ target_rotation.T + target_translation
    target_z = target_camera[:, 2]
    target_intrinsic = intrinsics[neighbor]
    safe_z = np.where(np.abs(target_z) > 1e-8, target_z, 1.0)
    target_xy = np.stack(
        [
            target_intrinsic[0, 0] * target_camera[:, 0] / safe_z
            + target_intrinsic[0, 2],
            target_intrinsic[1, 1] * target_camera[:, 1] / safe_z
            + target_intrinsic[1, 2],
        ],
        axis=-1,
    )

    height, width = depth.shape[1:3]
    visible = (
        np.isfinite(target_xy).all(axis=1)
        & np.isfinite(target_z)
        & (target_z > 1e-6)
        & (target_xy[:, 0] >= 0.0)
        & (target_xy[:, 0] <= width - 1)
        & (target_xy[:, 1] >= 0.0)
        & (target_xy[:, 1] <= height - 1)
    )
    sampled_depth = _bilinear_sample_numpy(depth[neighbor], target_xy)
    depth_valid = visible & np.isfinite(sampled_depth) & (sampled_depth > 1e-6)
    if depth_conf is not None:
        sampled_conf = _bilinear_sample_numpy(depth_conf[neighbor], target_xy)
        depth_valid &= np.isfinite(sampled_conf) & (
            sampled_conf >= confidence_thresholds[neighbor]
        )

    target_rays = np.stack(
        [
            (target_xy[:, 0] - target_intrinsic[0, 2]) / target_intrinsic[0, 0],
            (target_xy[:, 1] - target_intrinsic[1, 2]) / target_intrinsic[1, 1],
            np.ones(num_source, dtype=np.float64),
        ],
        axis=-1,
    )
    target_camera_from_depth = target_rays * sampled_depth[:, None]
    target_world_from_depth = (
        target_camera_from_depth - target_translation
    ) @ target_rotation

    source_rotation = extrinsic[center, :3, :3]
    source_translation = extrinsic[center, :3, 3]
    source_camera_roundtrip = (
        target_world_from_depth @ source_rotation.T + source_translation
    )
    source_z_roundtrip = source_camera_roundtrip[:, 2]
    safe_source_z = np.where(np.abs(source_z_roundtrip) > 1e-8, source_z_roundtrip, 1.0)
    source_intrinsic = intrinsics[center]
    source_xy_roundtrip = np.stack(
        [
            source_intrinsic[0, 0] * source_camera_roundtrip[:, 0] / safe_source_z
            + source_intrinsic[0, 2],
            source_intrinsic[1, 1] * source_camera_roundtrip[:, 1] / safe_source_z
            + source_intrinsic[1, 2],
        ],
        axis=-1,
    )
    reprojection_error = np.linalg.norm(source_xy_roundtrip - source_xy, axis=1)
    consistent = (
        depth_valid
        & np.isfinite(source_xy_roundtrip).all(axis=1)
        & np.isfinite(source_z_roundtrip)
        & (source_z_roundtrip > 1e-6)
        & np.isfinite(reprojection_error)
        & (reprojection_error < reprojection_threshold)
    )

    num_visible = int(visible.sum())
    num_depth_valid = int(depth_valid.sum())
    num_consistent = int(consistent.sum())
    consistent_grid_cells = int(np.unique(source["grid_cells"][consistent]).size)
    return {
        "source_samples": num_source,
        "visible_samples": num_visible,
        "depth_valid_samples": num_depth_valid,
        "consistent_samples": num_consistent,
        "projected_visible_ratio": float(num_visible / num_source),
        "projected_depth_valid_ratio": float(num_depth_valid / num_source),
        "projected_overlap": float(num_consistent / num_source),
        "projected_grid_coverage": (
            float(consistent_grid_cells / source["num_grid_cells"])
            if source["num_grid_cells"] > 0
            else 0.0
        ),
    }


def build_projected_overlap_groups(
    pairs,
    extrinsic,
    intrinsics,
    depth,
    depth_conf,
    retrieval_sim_matrix,
    max_neighbors,
    rotation_threshold,
    dino_candidates=30,
    max_samples=2048,
    reprojection_threshold=4.0,
    confidence_quantile=0.2,
    selected_centers=None,
):
    from ffba.geometry import camera_centers_from_w2c, camera_viewing_axes_from_w2c
    from ffba.matching.vggsfm_groups import summarize_groups
    from ffba.reporting.statistics import summarize_distribution, summarize_numeric

    t_start = time.time()
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    depth = np.asarray(depth)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth_conf is not None:
        depth_conf = np.asarray(depth_conf)
        if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
            depth_conf = depth_conf[..., 0]

    num_images = extrinsic.shape[0]
    if extrinsic.shape != (num_images, 3, 4):
        raise ValueError(
            f"Expected extrinsic shape ({num_images}, 3, 4), got {extrinsic.shape}"
        )
    if intrinsics.shape != (num_images, 3, 3):
        raise ValueError(
            f"Expected intrinsics shape ({num_images}, 3, 3), got {intrinsics.shape}"
        )
    if depth.shape[0] != num_images:
        raise ValueError("Depth image count does not match extrinsics")
    if depth_conf is not None and depth_conf.shape != depth.shape:
        raise ValueError(
            f"Expected depth_conf shape {depth.shape}, got {depth_conf.shape}"
        )
    retrieval_sim_matrix = np.asarray(retrieval_sim_matrix, dtype=np.float64)
    if retrieval_sim_matrix.shape != (num_images, num_images):
        raise ValueError(
            "Expected retrieval similarity shape "
            f"({num_images}, {num_images}), got {retrieval_sim_matrix.shape}"
        )
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be >= 1")
    if dino_candidates <= 0:
        raise ValueError("dino_candidates must be >= 1")
    if reprojection_threshold <= 0:
        raise ValueError("reprojection_threshold must be > 0")

    adjacency = defaultdict(set)
    for i, j in np.asarray(pairs, dtype=np.int64).reshape(-1, 2).tolist():
        adjacency[int(i)].add(int(j))
        adjacency[int(j)].add(int(i))

    centers = camera_centers_from_w2c(extrinsic)
    viewing_axes = camera_viewing_axes_from_w2c(extrinsic)
    sample_xy = _regular_grid_samples(depth.shape[1], depth.shape[2], max_samples)
    confidence_thresholds = _depth_confidence_thresholds(
        depth,
        depth_conf,
        confidence_quantile,
    )

    if selected_centers is None:
        selected_centers = list(range(num_images))
    else:
        selected_centers = [int(center) for center in selected_centers]
        if len(set(selected_centers)) != len(selected_centers):
            raise ValueError("selected_centers must not contain duplicates")
        if any(center < 0 or center >= num_images for center in selected_centers):
            raise ValueError("selected_centers contains an out-of-range image index")

    groups = []
    candidate_details = {}
    candidate_counts = []
    pose_candidate_counts = []
    dino_candidate_counts = []
    selected_overlaps = []
    selected_grid_coverages = []
    selected_visible_ratios = []
    selected_pose_only = 0
    selected_dino_only = 0
    selected_both = 0

    dino_k = min(int(dino_candidates), max(num_images - 1, 0))
    for center in selected_centers:
        source = _prepare_projected_overlap_source(
            center,
            sample_xy,
            depth,
            depth_conf,
            confidence_thresholds,
            extrinsic,
            intrinsics,
        )
        pose_candidates = set()
        for neighbor in adjacency.get(center, set()):
            dot = float(np.dot(viewing_axes[center], viewing_axes[neighbor]))
            angle = float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
            if angle < rotation_threshold:
                pose_candidates.add(int(neighbor))

        row = np.nan_to_num(
            retrieval_sim_matrix[center],
            nan=-np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        ).copy()
        row[center] = -np.inf
        dino_order = np.argsort(-row, kind="stable")[:dino_k]
        dino_candidates_set = {int(idx) for idx in dino_order if idx != center}
        candidates = sorted(pose_candidates | dino_candidates_set)

        candidate_counts.append(len(candidates))
        pose_candidate_counts.append(len(pose_candidates))
        dino_candidate_counts.append(len(dino_candidates_set))
        details = []
        for neighbor in candidates:
            pair_stats = _score_projected_overlap_pair(
                center,
                neighbor,
                source,
                depth,
                depth_conf,
                confidence_thresholds,
                extrinsic,
                intrinsics,
                reprojection_threshold,
            )
            in_pose = neighbor in pose_candidates
            in_dino = neighbor in dino_candidates_set
            dot = float(np.dot(viewing_axes[center], viewing_axes[neighbor]))
            rotation_angle = float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
            detail = {
                "image_index": int(neighbor),
                "candidate_sources": [
                    source_name
                    for source_name, present in (("pose", in_pose), ("dino", in_dino))
                    if present
                ],
                "dino_similarity": float(retrieval_sim_matrix[center, neighbor]),
                "rotation_angle_deg": rotation_angle,
                "rotation_valid": rotation_angle < rotation_threshold,
                "camera_center_distance": float(
                    np.linalg.norm(centers[center] - centers[neighbor])
                ),
                **pair_stats,
            }
            details.append(detail)

        details.sort(
            key=lambda item: (
                -item["projected_overlap"],
                -item["projected_grid_coverage"],
                -item["projected_visible_ratio"],
                -item["dino_similarity"],
                item["camera_center_distance"],
                item["image_index"],
            )
        )
        selected = details[: min(int(max_neighbors), len(details))]
        for rank, detail in enumerate(selected, start=1):
            detail["selected"] = True
            detail["selected_rank"] = rank
            selected_overlaps.append(detail["projected_overlap"])
            selected_grid_coverages.append(detail["projected_grid_coverage"])
            selected_visible_ratios.append(detail["projected_visible_ratio"])
            sources = set(detail["candidate_sources"])
            if sources == {"pose", "dino"}:
                selected_both += 1
            elif sources == {"pose"}:
                selected_pose_only += 1
            elif sources == {"dino"}:
                selected_dino_only += 1
        for detail in details[len(selected) :]:
            detail["selected"] = False
            detail["selected_rank"] = None
        candidate_details[center] = details
        if selected:
            groups.append([center, *[item["image_index"] for item in selected]])

    stats = {
        "strategy": "projected_overlap_hybrid",
        "candidate_pool": "rotation_valid_pose_pairs_union_dino_topk",
        "input_pairs": int(np.asarray(pairs).reshape(-1, 2).shape[0]),
        "max_neighbors": int(max_neighbors),
        "pose_rotation_threshold": float(rotation_threshold),
        "dino_candidates_per_center": int(dino_k),
        "max_source_samples": int(max_samples),
        "actual_regular_grid_samples": int(sample_xy.shape[0]),
        "reprojection_threshold_lowres_px": float(reprojection_threshold),
        "depth_confidence_quantile": float(confidence_quantile),
        "scheduled_centers": selected_centers,
        "num_scheduled_centers": int(len(selected_centers)),
        "num_unscheduled_centers": int(num_images - len(selected_centers)),
        "seconds": float(time.time() - t_start),
        "candidate_count": summarize_numeric(candidate_counts),
        "pose_candidate_count": summarize_numeric(pose_candidate_counts),
        "dino_candidate_count": summarize_numeric(dino_candidate_counts),
        "selected_projected_overlap": summarize_distribution(selected_overlaps),
        "selected_projected_grid_coverage": summarize_distribution(
            selected_grid_coverages
        ),
        "selected_projected_visible_ratio": summarize_distribution(
            selected_visible_ratios
        ),
        "selected_candidate_source": {
            "pose_only": int(selected_pose_only),
            "dino_only": int(selected_dino_only),
            "both": int(selected_both),
        },
        **summarize_groups(groups, num_images),
    }
    return groups, stats, candidate_details
