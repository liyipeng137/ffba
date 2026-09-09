"""matching / scheduling for the formal SIFT + prior + BAE pipeline."""

import numpy as np


def select_sift_first_centers(num_images, valid_edges, max_center_gap, order=None):
    from ffba.matching.sift import canonicalize_pair_array

    num_images = int(num_images)
    max_center_gap = int(max_center_gap)
    if num_images < 0:
        raise ValueError("num_images must be >= 0")
    if max_center_gap <= 0:
        raise ValueError("max_center_gap must be >= 1")
    valid_set = {
        tuple(pair)
        for pair in canonicalize_pair_array(
            valid_edges,
            num_images=num_images,
        ).tolist()
    }
    if num_images == 0:
        return {
            "selected_centers": [],
            "owner": [],
            "frames": [],
            "num_selected": 0,
            "num_skipped": 0,
        }

    order = list(range(num_images)) if order is None else [int(i) for i in order]
    if sorted(order) != list(range(num_images)):
        raise ValueError("Scheduling order must be a permutation of image indices")
    selected_centers = [order[0]]
    owner = [None] * num_images
    owner[order[0]] = order[0]
    frames = [
        {
            "image_index": order[0],
            "owner": order[0],
            "selected": True,
            "reason": "first_frame",
        }
    ]
    last_center = order[0]
    last_center_position = 0
    for position, image_idx in enumerate(order[1:], start=1):
        gap = position - last_center_position
        supported = tuple(sorted((last_center, image_idx))) in valid_set
        if gap >= max_center_gap:
            selected = True
            reason = "max_gap"
        elif not supported:
            selected = True
            reason = "insufficient_sift_support"
        else:
            selected = False
            reason = "skipped_supported"

        if selected:
            last_center = image_idx
            last_center_position = position
            selected_centers.append(image_idx)
            owner[image_idx] = image_idx
        else:
            owner[image_idx] = last_center
        frames.append(
            {
                "image_index": image_idx,
                "owner": int(owner[image_idx]),
                "selected": selected,
                "reason": reason,
                "supported_by_previous_center": supported,
            }
        )

    for image_idx, current_owner in enumerate(owner):
        if current_owner is None:
            raise AssertionError(f"Frame {image_idx} has no owner")
        if image_idx != current_owner:
            edge = tuple(sorted((image_idx, int(current_owner))))
            if edge not in valid_set:
                raise AssertionError(f"Frame {image_idx} has invalid owner edge {edge}")
    return {
        "selected_centers": selected_centers,
        "owner": [int(value) for value in owner],
        "frames": sorted(frames, key=lambda frame: frame["image_index"]),
        "num_selected": int(len(selected_centers)),
        "num_skipped": int(num_images - len(selected_centers)),
    }


def simulate_sift_schedule_thresholds(
    pair_records,
    num_images,
    max_center_gap,
    *,
    inlier_thresholds=(64, 96, 128),
    coverage_thresholds=(0.10, 0.15, 0.20),
):
    simulations = []
    for min_inliers in inlier_thresholds:
        for min_coverage in coverage_thresholds:
            valid_edges = [
                record["pair"]
                for record in pair_records
                if int(record["inlier_count"]) >= int(min_inliers)
                and float(record["source_grid_coverage"]) >= float(min_coverage)
                and float(record["target_grid_coverage"]) >= float(min_coverage)
            ]
            selection = select_sift_first_centers(
                num_images,
                valid_edges,
                max_center_gap,
            )
            simulations.append(
                {
                    "min_pair_inliers": int(min_inliers),
                    "min_grid_coverage": float(min_coverage),
                    "valid_schedule_pair_count": int(len(valid_edges)),
                    "selected_center_count": int(selection["num_selected"]),
                    "skipped_center_count": int(selection["num_skipped"]),
                    "selected_center_ratio": (
                        float(selection["num_selected"] / num_images)
                        if num_images
                        else 0.0
                    ),
                }
            )
    return simulations


def build_scheduling_order(similarity, input_order):
    """A deterministic DINO traversal; never changes image IDs or creates pairs."""
    similarity = np.asarray(similarity)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("DINO similarity must be square")
    if not np.isfinite(similarity).all():
        raise ValueError("DINO similarity must be finite")
    n = len(similarity)
    if input_order == "ordered":
        return list(range(n))
    if input_order != "unordered":
        raise ValueError("input_order must be ordered or unordered")
    if n == 0:
        return []
    weights = similarity.copy()
    np.fill_diagonal(weights, 0)
    current = int(np.argmax(weights.sum(axis=1)))
    order = [current]
    unseen = np.ones(n, dtype=bool)
    unseen[current] = False
    while unseen.any():
        candidates = np.flatnonzero(unseen)
        current = int(candidates[np.argmax(weights[current, candidates])])
        order.append(current)
        unseen[current] = False
    return order


def resolve_group_strategy(requested, has_depth):
    if requested not in {"projected_overlap", "sift_pose_dino"}:
        raise ValueError("Unsupported VGGSfM group strategy")
    fallback = requested == "projected_overlap" and not has_depth
    return {
        "requested_strategy": requested,
        "effective_strategy": "sift_pose_dino" if fallback else requested,
        "fallback_reason": "initial_geometry_has_no_depth" if fallback else None,
    }
