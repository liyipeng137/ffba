"""matching / vggsfm_groups for the formal SIFT + prior + BAE pipeline."""

from collections import defaultdict
import numpy as np


def build_sift_pose_dino_candidate_details(
    selected_centers,
    sift_pair_records,
    pose_pairs,
    temporal_pairs,
    retrieval_sim_matrix,
    extrinsic,
    *,
    rotation_threshold,
    dino_candidates=30,
):
    from ffba.geometry import camera_centers_from_w2c, camera_viewing_axes_from_w2c
    from ffba.matching.sift import canonicalize_pair_array
    from ffba.reporting.statistics import summarize_numeric

    selected_centers = [int(center) for center in selected_centers]
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    num_images = int(extrinsic.shape[0])
    retrieval_sim_matrix = np.asarray(retrieval_sim_matrix, dtype=np.float64)
    if extrinsic.shape != (num_images, 3, 4):
        raise ValueError(
            f"Expected extrinsic shape ({num_images}, 3, 4), got {extrinsic.shape}"
        )
    if retrieval_sim_matrix.shape != (num_images, num_images):
        raise ValueError(
            "Expected retrieval similarity shape "
            f"({num_images}, {num_images}), got {retrieval_sim_matrix.shape}"
        )
    dino_k = min(max(int(dino_candidates), 0), max(num_images - 1, 0))

    pose_set = {
        tuple(pair)
        for pair in canonicalize_pair_array(pose_pairs, num_images=num_images).tolist()
    }
    temporal_set = {
        tuple(pair)
        for pair in canonicalize_pair_array(
            temporal_pairs,
            num_images=num_images,
        ).tolist()
    }
    pair_records = {}
    for record in sift_pair_records:
        pair = tuple(
            canonicalize_pair_array(
                [record["pair"]],
                num_images=num_images,
            )[0].tolist()
        )
        pair_records[pair] = record
    pair_neighbors = defaultdict(set)
    for i, j in pose_set | temporal_set | set(pair_records):
        pair_neighbors[int(i)].add(int(j))
        pair_neighbors[int(j)].add(int(i))

    centers = camera_centers_from_w2c(extrinsic)
    viewing_axes = camera_viewing_axes_from_w2c(extrinsic)
    details_by_center = {}
    candidate_counts = []
    valid_sift_counts = []
    for center in selected_centers:
        row = np.nan_to_num(
            retrieval_sim_matrix[center],
            nan=-np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        ).copy()
        row[center] = -np.inf
        dino_order = np.argsort(-row, kind="stable")[:dino_k]
        dino_set = {int(idx) for idx in dino_order if int(idx) != center}

        candidates = set(dino_set) | pair_neighbors.get(center, set())

        details = []
        for neighbor in sorted(candidates):
            pair = tuple(sorted((center, neighbor)))
            record = pair_records.get(pair)
            inlier_count = int(record["inlier_count"]) if record is not None else 0
            source_coverage = (
                float(record["source_grid_coverage"]) if record is not None else 0.0
            )
            target_coverage = (
                float(record["target_grid_coverage"]) if record is not None else 0.0
            )
            min_coverage = min(source_coverage, target_coverage)
            valid_sift = bool(
                record is not None and record.get("valid_schedule_edge", False)
            )
            dot = float(np.dot(viewing_axes[center], viewing_axes[neighbor]))
            rotation_angle = float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
            sources = []
            if pair in pose_set and rotation_angle < float(rotation_threshold):
                sources.append("pose")
            if pair in temporal_set:
                sources.append("temporal")
            if neighbor in dino_set:
                sources.append("dino")
            details.append(
                {
                    "image_index": int(neighbor),
                    "candidate_sources": sources,
                    "valid_schedule_edge": valid_sift,
                    "inlier_count": inlier_count,
                    "source_grid_coverage": source_coverage,
                    "target_grid_coverage": target_coverage,
                    "min_grid_coverage": min_coverage,
                    "dino_similarity": float(row[neighbor]),
                    "rotation_angle_deg": rotation_angle,
                    "rotation_valid": rotation_angle < float(rotation_threshold),
                    "camera_center_distance": float(
                        np.linalg.norm(centers[center] - centers[neighbor])
                    ),
                }
            )

        details.sort(
            key=lambda item: (
                -int(item["valid_schedule_edge"]),
                -item["min_grid_coverage"],
                -item["inlier_count"],
                -item["dino_similarity"],
                item["rotation_angle_deg"],
                item["camera_center_distance"],
                item["image_index"],
            )
        )
        details_by_center[center] = details
        candidate_counts.append(len(details))
        valid_sift_counts.append(
            sum(int(detail["valid_schedule_edge"]) for detail in details)
        )

    stats = {
        "strategy": "sift_pose_dino",
        "candidate_pool": "sift_pairs_union_rotation_valid_pose_temporal_dino_topk",
        "scheduled_centers": selected_centers,
        "num_scheduled_centers": int(len(selected_centers)),
        "num_unscheduled_centers": int(num_images - len(selected_centers)),
        "pose_pair_count": int(len(pose_set)),
        "temporal_pair_count": int(len(temporal_set)),
        "sift_pair_record_count": int(len(pair_records)),
        "dino_candidates_per_center": int(dino_k),
        "candidate_count": summarize_numeric(candidate_counts),
        "valid_sift_candidate_count": summarize_numeric(valid_sift_counts),
        "ranking": [
            "valid_schedule_edge_desc",
            "min_grid_coverage_desc",
            "inlier_count_desc",
            "dino_similarity_desc",
            "rotation_angle_asc",
            "camera_center_distance_asc",
        ],
    }
    return details_by_center, stats


def build_three_layer_vggsfm_groups(
    selected_centers,
    owner,
    valid_sift_edges,
    projected_candidate_details,
    num_images,
    max_neighbors,
    *,
    fill_source_name="projected_overlap_fill",
    strategy_name="sift_first_three_layer_projected_overlap",
):
    from ffba.matching.sift import canonicalize_pair_array

    selected_centers = [int(center) for center in selected_centers]
    owner = [int(value) for value in owner]
    num_images = int(num_images)
    max_neighbors = int(max_neighbors)
    if len(owner) != num_images:
        raise ValueError(f"Expected {num_images} owner entries, got {len(owner)}")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be >= 1")
    if len(set(selected_centers)) != len(selected_centers):
        raise ValueError("selected_centers must not contain duplicates")
    selected_set = set(selected_centers)
    if any(center < 0 or center >= num_images for center in selected_centers):
        raise ValueError("selected_centers contains an out-of-range image index")
    if any(value not in selected_set for value in owner):
        raise ValueError("Every owner must be a selected center")

    valid_set = {
        tuple(pair)
        for pair in canonicalize_pair_array(
            valid_sift_edges,
            num_images=num_images,
        ).tolist()
    }
    owned_by_center = defaultdict(list)
    for image_idx, center in enumerate(owner):
        if image_idx != center:
            owned_by_center[center].append(image_idx)

    groups = []
    group_records = []
    layer_counts = defaultdict(int)
    overflow_groups = 0
    for center_pos, center in enumerate(selected_centers):
        owned = sorted(set(owned_by_center.get(center, [])))
        bridges = []
        adjacent = []
        if center_pos > 0:
            adjacent.append(selected_centers[center_pos - 1])
        if center_pos + 1 < len(selected_centers):
            adjacent.append(selected_centers[center_pos + 1])
        for neighbor in adjacent:
            if tuple(sorted((center, neighbor))) in valid_set and neighbor not in owned:
                bridges.append(neighbor)

        members = []
        member_sources = {}
        for neighbor in owned:
            if neighbor not in member_sources:
                members.append(neighbor)
                member_sources[neighbor] = "owned_frame"
                layer_counts["owned_frame"] += 1
        for neighbor in bridges:
            if neighbor not in member_sources:
                members.append(neighbor)
                member_sources[neighbor] = "adjacent_center_bridge"
                layer_counts["adjacent_center_bridge"] += 1

        forced_count = len(members)
        remaining = max(0, max_neighbors - forced_count)
        projected_fill = []
        details = projected_candidate_details.get(center, [])
        for detail in details:
            if remaining <= 0:
                break
            neighbor = int(detail["image_index"])
            if neighbor == center or neighbor in member_sources:
                continue
            members.append(neighbor)
            projected_fill.append(neighbor)
            member_sources[neighbor] = fill_source_name
            layer_counts[fill_source_name] += 1
            remaining -= 1

        overflow = forced_count > max_neighbors
        overflow_groups += int(overflow)
        if members:
            groups.append([center, *members])
        group_records.append(
            {
                "center": center,
                "owned_frames": owned,
                "adjacent_bridges": bridges,
                fill_source_name: projected_fill,
                "final_members": members,
                "member_sources": {
                    str(neighbor): member_sources[neighbor] for neighbor in members
                },
                "forced_neighbor_count": forced_count,
                "group_size": int(1 + len(members)),
                "overflow": overflow,
            }
        )

    stats = {
        "strategy": strategy_name,
        "max_neighbors": max_neighbors,
        "scheduled_centers": selected_centers,
        "num_scheduled_centers": int(len(selected_centers)),
        "num_unscheduled_centers": int(num_images - len(selected_centers)),
        "layer_member_counts": {
            "owned_frame": int(layer_counts["owned_frame"]),
            "adjacent_center_bridge": int(layer_counts["adjacent_center_bridge"]),
            fill_source_name: int(layer_counts[fill_source_name]),
        },
        "overflow_groups": int(overflow_groups),
        "group_records": group_records,
        **summarize_groups(groups, num_images),
    }
    return groups, stats


def summarize_groups(groups, num_images):
    from ffba.reporting.statistics import summarize_numeric

    group_sizes = [len(group) for group in groups]
    neighbor_counts = [max(len(group) - 1, 0) for group in groups]
    centers = [int(group[0]) for group in groups]
    missing_centers = sorted(set(range(num_images)) - set(centers))
    return {
        "num_groups": int(len(groups)),
        "group_size": summarize_numeric(group_sizes),
        "neighbors": summarize_numeric(neighbor_counts),
        "missing_centers": missing_centers,
    }


def build_pose_groups(
    pairs,
    num_images,
    neighbors_per_center,
    centers=None,
    viewing_axes=None,
    rotation_threshold=None,
):
    adjacency = defaultdict(list)
    for i, j in pairs.tolist():
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))

    rotation_cos_threshold = None
    if viewing_axes is not None and rotation_threshold is not None:
        viewing_axes = np.asarray(viewing_axes, dtype=np.float64)
        if viewing_axes.shape != (num_images, 3):
            raise ValueError(
                "Expected viewing_axes shape "
                f"({num_images}, 3), got {viewing_axes.shape}"
            )
        if not 0.0 <= rotation_threshold <= 180.0:
            raise ValueError("rotation_threshold must be in [0, 180] degrees")
        rotation_cos_threshold = float(np.cos(np.deg2rad(rotation_threshold)))

    def is_unfiltered(center, neighbor):
        dot = float(np.dot(viewing_axes[center], viewing_axes[neighbor]))
        return bool(np.clip(dot, -1.0, 1.0) <= rotation_cos_threshold)

    groups = []
    for center in range(num_images):
        if rotation_cos_threshold is not None and centers is not None:
            neighbors = sorted(
                set(adjacency.get(center, [])),
                key=lambda x: (
                    is_unfiltered(center, x),
                    float(np.linalg.norm(centers[center] - centers[x])),
                    abs(x - center),
                    x,
                ),
            )
        elif rotation_cos_threshold is not None:
            neighbors = sorted(
                set(adjacency.get(center, [])),
                key=lambda x: (
                    is_unfiltered(center, x),
                    abs(x - center),
                    x,
                ),
            )
        elif centers is None:
            neighbors = sorted(
                set(adjacency.get(center, [])),
                key=lambda x: (abs(x - center), x),
            )
        else:
            neighbors = sorted(
                set(adjacency.get(center, [])),
                key=lambda x: (
                    float(np.linalg.norm(centers[center] - centers[x])),
                    abs(x - center),
                    x,
                ),
            )
        neighbors = neighbors[:neighbors_per_center]
        if neighbors:
            groups.append([center, *neighbors])
    return groups


def build_vggsfm_groups(
    args,
    pairs,
    num_images,
    image_names,
    image_size_hw,
    centers=None,
    viewing_axes=None,
):
    from ffba.reporting.statistics import summarize_numeric

    rotation_threshold = getattr(args, "pair_pose_rotation_threshold", None)
    groups = build_pose_groups(
        pairs,
        num_images,
        args.neighbors_per_center,
        centers=centers,
        viewing_axes=viewing_axes,
        rotation_threshold=rotation_threshold,
    )
    stats = {
        "strategy": "pose",
        "neighbor_order": (
            "rotation_valid_then_camera_distance"
            if viewing_axes is not None and rotation_threshold is not None
            else "camera_distance"
        ),
        "pose_rotation_threshold": (
            float(rotation_threshold) if rotation_threshold is not None else None
        ),
        "input_pairs": int(np.asarray(pairs).reshape(-1, 2).shape[0]),
        "max_neighbors": int(args.neighbors_per_center),
        **summarize_groups(groups, num_images),
    }
    if viewing_axes is not None and rotation_threshold is not None:
        valid_counts = []
        unfiltered_counts = []
        for group in groups:
            center = int(group[0])
            neighbors = np.asarray(group[1:], dtype=np.int64)
            if neighbors.size == 0:
                valid_counts.append(0)
                unfiltered_counts.append(0)
                continue
            dots = np.einsum("j,nj->n", viewing_axes[center], viewing_axes[neighbors])
            angles = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))
            num_valid = int(np.sum(angles < rotation_threshold))
            valid_counts.append(num_valid)
            unfiltered_counts.append(int(neighbors.size - num_valid))
        stats["selected_rotation_valid_neighbors"] = summarize_numeric(valid_counts)
        stats["selected_unfiltered_neighbors"] = summarize_numeric(unfiltered_counts)
    return groups, stats
