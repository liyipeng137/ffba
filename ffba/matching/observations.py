"""matching / observations for the formal SIFT + prior + BAE pipeline."""

from collections import defaultdict
import numpy as np
import torch
from scipy.spatial import cKDTree


def count_track_observations(tracks, num_images):
    counts = np.zeros(num_images, dtype=np.int64)
    for track in tracks:
        seen = set()
        for image_idx, _xy in track:
            if image_idx in seen:
                continue
            seen.add(image_idx)
            counts[int(image_idx)] += 1
    return counts


def remap_pairs(pairs, old_to_new):
    remapped = set()
    for i, j in pairs.tolist():
        if int(i) not in old_to_new or int(j) not in old_to_new:
            continue
        ni = old_to_new[int(i)]
        nj = old_to_new[int(j)]
        if ni == nj:
            continue
        remapped.add(tuple(sorted((ni, nj))))
    return np.asarray(sorted(remapped), dtype=np.int64)


def remap_prior_tracks(tracks, old_to_new):
    remapped_tracks = []
    for track in tracks:
        remapped = []
        seen = set()
        for image_idx, xy in track:
            image_idx = int(image_idx)
            if image_idx not in old_to_new:
                continue
            new_idx = old_to_new[image_idx]
            if new_idx in seen:
                continue
            seen.add(new_idx)
            remapped.append((new_idx, xy))
        if len(remapped) >= 2:
            remapped_tracks.append(remapped)
    return remapped_tracks


def filter_low_coverage_frames(
    image_names,
    images,
    extrinsic,
    features,
    pairs,
    prior_tracks,
    s_counts,
    p_counts,
    min_frame_observations,
    enabled=True,
):
    total_counts = s_counts + p_counts
    if not enabled:
        keep_indices = np.arange(len(image_names), dtype=np.int64)
    else:
        keep_indices = np.where(total_counts >= min_frame_observations)[0]

    if len(keep_indices) == len(image_names):
        return (
            image_names,
            images,
            extrinsic,
            features,
            pairs,
            prior_tracks,
            {
                "enabled": enabled,
                "min_frame_observations": min_frame_observations,
                "kept_indices": keep_indices.tolist(),
                "dropped_indices": [],
                "dropped_names": [],
                "s_observations": s_counts.tolist(),
                "p_observations": p_counts.tolist(),
                "total_observations": total_counts.tolist(),
            },
        )

    if len(keep_indices) == 0:
        raise ValueError(
            "All frames are below the observation threshold; cannot refine."
        )

    old_to_new = {int(old): new for new, old in enumerate(keep_indices.tolist())}
    dropped_indices = [idx for idx in range(len(image_names)) if idx not in old_to_new]
    dropped_names = [image_names[idx] for idx in dropped_indices]

    filtered_image_names = [image_names[idx] for idx in keep_indices]
    filtered_images = images[torch.as_tensor(keep_indices, device=images.device)]
    filtered_extrinsic = extrinsic[keep_indices]
    filtered_features = [features[idx] for idx in keep_indices]
    filtered_pairs = remap_pairs(pairs, old_to_new)
    filtered_prior_tracks = remap_prior_tracks(prior_tracks, old_to_new)

    return (
        filtered_image_names,
        filtered_images,
        filtered_extrinsic,
        filtered_features,
        filtered_pairs,
        filtered_prior_tracks,
        {
            "enabled": enabled,
            "min_frame_observations": min_frame_observations,
            "kept_indices": keep_indices.tolist(),
            "dropped_indices": dropped_indices,
            "dropped_names": dropped_names,
            "s_observations": s_counts.tolist(),
            "p_observations": p_counts.tolist(),
            "total_observations": total_counts.tolist(),
        },
    )


def merge_keypoints_per_image(raw_keypoints, merge_threshold):
    merged_keypoints = []
    raw_to_merged = []
    stats = {
        "merge_threshold": float(merge_threshold),
        "raw_keypoints": [],
        "merged_keypoints": [],
    }

    for points in raw_keypoints:
        if len(points) == 0:
            merged_keypoints.append(np.empty((0, 2), dtype=np.float32))
            raw_to_merged.append(np.empty((0,), dtype=np.int64))
            stats["raw_keypoints"].append(0)
            stats["merged_keypoints"].append(0)
            continue

        points_np = np.stack(points).astype(np.float32)
        stats["raw_keypoints"].append(int(points_np.shape[0]))

        if merge_threshold <= 0 or points_np.shape[0] == 1:
            merged_keypoints.append(points_np)
            raw_to_merged.append(np.arange(points_np.shape[0], dtype=np.int64))
            stats["merged_keypoints"].append(int(points_np.shape[0]))
            continue

        uf = UnionFind(points_np.shape[0])
        tree = cKDTree(points_np)
        for i, j in tree.query_pairs(r=merge_threshold):
            uf.union(int(i), int(j))

        components = defaultdict(list)
        for idx in range(points_np.shape[0]):
            components[uf.find(idx)].append(idx)

        remap = np.empty(points_np.shape[0], dtype=np.int64)
        merged = []
        for new_idx, indices in enumerate(components.values()):
            remap[indices] = new_idx
            merged.append(points_np[indices].mean(axis=0))

        merged_np = np.stack(merged).astype(np.float32)
        merged_keypoints.append(merged_np)
        raw_to_merged.append(remap)
        stats["merged_keypoints"].append(int(merged_np.shape[0]))

    stats["raw_total"] = int(sum(stats["raw_keypoints"]))
    stats["merged_total"] = int(sum(stats["merged_keypoints"]))
    stats["merged_reduction"] = int(stats["raw_total"] - stats["merged_total"])
    return merged_keypoints, raw_to_merged, stats


def snap_prior_tracks_to_features(
    tracks,
    features,
    snap_threshold=1.0,
    keep_unsnapped=True,
):
    keypoint_trees = []
    for feats in features:
        keypoints = np.asarray(feats["keypoints"], dtype=np.float32)
        keypoint_trees.append(cKDTree(keypoints) if keypoints.shape[0] > 0 else None)

    snapped_tracks = []
    stats = {
        "enabled": True,
        "snap_threshold": float(snap_threshold),
        "keep_unsnapped": bool(keep_unsnapped),
        "input_tracks": int(len(tracks)),
        "input_observations": 0,
        "output_tracks": 0,
        "output_observations": 0,
        "snapped_observations": 0,
        "unsnapped_kept_observations": 0,
        "dropped_observations": 0,
        "center_observations": 0,
        "center_snapped_observations": 0,
        "center_unsnapped_kept_observations": 0,
        "center_dropped_observations": 0,
        "neighbor_observations": 0,
        "neighbor_snapped_observations": 0,
        "neighbor_unsnapped_kept_observations": 0,
        "neighbor_dropped_observations": 0,
        "snap_distance_mean": 0.0,
        "snap_distance_max": 0.0,
    }
    snap_distances = []

    # Keep observations in track order, but group their nearest-neighbor queries
    # by image.  Reconstructing from these original slots is important for the
    # star match topology, where the first observation is the track center.
    prepared_tracks = []
    query_locations = [[] for _ in features]
    for track in tracks:
        prepared_obs = []
        seen_images = set()
        for obs_idx, (image_idx, xy) in enumerate(track):
            image_idx = int(image_idx)
            if image_idx in seen_images:
                continue
            seen_images.add(image_idx)
            stats["input_observations"] += 1
            prefix = "center" if obs_idx == 0 else "neighbor"
            stats[f"{prefix}_observations"] += 1

            xy = np.asarray(xy, dtype=np.float32)
            tree = keypoint_trees[image_idx]
            prepared_obs.append([image_idx, xy, prefix, None])
            if tree is not None:
                query_locations[image_idx].append(
                    (len(prepared_tracks), len(prepared_obs) - 1)
                )
        prepared_tracks.append(prepared_obs)

    for image_idx, locations in enumerate(query_locations):
        if not locations:
            continue
        query_points = np.stack(
            [prepared_tracks[track_idx][obs_idx][1] for track_idx, obs_idx in locations]
        )
        distances, keypoint_indices = keypoint_trees[image_idx].query(
            query_points,
            k=1,
            workers=1,
        )
        for location, distance, keypoint_idx in zip(
            locations,
            distances,
            keypoint_indices,
        ):
            track_idx, obs_idx = location
            prepared_tracks[track_idx][obs_idx][3] = (
                float(distance),
                int(keypoint_idx),
            )

    for prepared_obs in prepared_tracks:
        snapped_obs = []
        for image_idx, xy, prefix, query_result in prepared_obs:
            if query_result is None:
                if keep_unsnapped:
                    snapped_obs.append((image_idx, xy))
                    stats["unsnapped_kept_observations"] += 1
                    stats[f"{prefix}_unsnapped_kept_observations"] += 1
                else:
                    stats["dropped_observations"] += 1
                    stats[f"{prefix}_dropped_observations"] += 1
                continue

            distance, keypoint_idx = query_result
            if float(distance) <= snap_threshold:
                snapped_xy = features[image_idx]["keypoints"][int(keypoint_idx)].astype(
                    np.float32
                )
                snapped_obs.append((image_idx, snapped_xy))
                stats["snapped_observations"] += 1
                stats[f"{prefix}_snapped_observations"] += 1
                snap_distances.append(float(distance))
            elif keep_unsnapped:
                snapped_obs.append((image_idx, xy))
                stats["unsnapped_kept_observations"] += 1
                stats[f"{prefix}_unsnapped_kept_observations"] += 1
            else:
                stats["dropped_observations"] += 1
                stats[f"{prefix}_dropped_observations"] += 1

        if len(snapped_obs) >= 2:
            snapped_tracks.append(snapped_obs)
            stats["output_observations"] += len(snapped_obs)

    stats["output_tracks"] = int(len(snapped_tracks))
    if snap_distances:
        stats["snap_distance_mean"] = float(np.mean(snap_distances))
        stats["snap_distance_max"] = float(np.max(snap_distances))
    return snapped_tracks, stats


def tracks_to_keypoints_and_matches(
    tracks,
    num_images,
    merge_threshold=1e-3,
    match_topology="all_pairs",
):
    if match_topology not in {"all_pairs", "star"}:
        raise ValueError(
            f"match_topology must be 'all_pairs' or 'star', got {match_topology!r}"
        )

    raw_keypoints = [[] for _ in range(num_images)]
    raw_tracks = []

    for track in tracks:
        raw_obs = []
        seen = set()
        for image_idx, xy in track:
            if image_idx in seen:
                continue
            seen.add(image_idx)
            raw_idx = len(raw_keypoints[image_idx])
            raw_keypoints[image_idx].append(np.asarray(xy, dtype=np.float32))
            raw_obs.append((image_idx, raw_idx))
        if len(raw_obs) >= 2:
            raw_tracks.append(raw_obs)

    keypoints_np, raw_to_merged, merge_stats = merge_keypoints_per_image(
        raw_keypoints,
        merge_threshold,
    )

    pair_matches = defaultdict(set)
    kept_tracks = 0
    track_lengths = []
    for raw_obs in raw_tracks:
        obs_indices = []
        seen_images = set()
        for image_idx, raw_idx in raw_obs:
            if image_idx in seen_images:
                continue
            seen_images.add(image_idx)
            obs_indices.append((image_idx, int(raw_to_merged[image_idx][raw_idx])))
        obs_indices = list(dict.fromkeys(obs_indices))
        if len(obs_indices) < 2:
            continue
        kept_tracks += 1
        track_lengths.append(len(obs_indices))
        if match_topology == "all_pairs":
            index_pairs = (
                (a, b)
                for a in range(len(obs_indices))
                for b in range(a + 1, len(obs_indices))
            )
        else:
            # Match GlueMap TrackEstablishment's star-style prior: each
            # tracker group emits correspondences between the center view
            # (first observation) and each visible neighbor.
            index_pairs = ((0, b) for b in range(1, len(obs_indices)))
        for a, b in index_pairs:
            i, pi = obs_indices[a]
            j, pj = obs_indices[b]
            if i == j:
                continue
            if i > j:
                i, j = j, i
                pi, pj = pj, pi
            pair_matches[(i, j)].add((pi, pj))

    matches_np = {
        key: np.asarray(sorted(value), dtype=np.uint32)
        for key, value in pair_matches.items()
    }

    merge_stats["input_tracks"] = int(len(tracks))
    merge_stats["kept_tracks"] = int(kept_tracks)
    merge_stats["match_topology"] = match_topology
    merge_stats["track_length_mean"] = (
        float(np.mean(track_lengths)) if track_lengths else 0.0
    )
    merge_stats["track_length_median"] = (
        float(np.median(track_lengths)) if track_lengths else 0.0
    )
    merge_stats["track_length_max"] = int(max(track_lengths)) if track_lengths else 0
    return keypoints_np, matches_np, merge_stats


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a
