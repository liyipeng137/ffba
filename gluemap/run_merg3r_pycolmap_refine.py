import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree


def _lazy_import_pycolmap():
    try:
        import pycolmap  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycolmap is required. Run this script in the Gluemap environment."
        ) from exc
    return pycolmap


def _ensure_gluemap_imports():
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import thirdparty.path_to_thirdparty  # noqa: F401, PLC0415


def parse_args():
    parser = argparse.ArgumentParser(
        "Run pycolmap refinement from Merg3r coarse pose and exported "
        + "S/P tracks."
    )
    parser.add_argument("--artifact_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--path_tracker",
        type=str,
        default="/root/.cache/torch/hub/checkpoints/vggsfm_v2_tracker.pt",
    )
    parser.add_argument(
        "--track_mode", type=str, default="SP", choices=["S", "P", "SP"]
    )
    parser.add_argument("--neighbors_per_center", type=int, default=8)
    parser.add_argument("--vggsfm_query_points", type=int, default=1024)
    parser.add_argument(
        "--vggsfm_query_source",
        type=str,
        default="aliked",
        choices=["superpoint", "aliked"],
    )
    parser.add_argument(
        "--aliked_detection_threshold", type=float, default=0.005
    )
    parser.add_argument("--vggsfm_vis_threshold", type=float, default=0.5)
    parser.add_argument("--vggsfm_score_threshold", type=float, default=0.0)
    parser.add_argument("--vggsfm_fine_tracking", action="store_true")
    parser.add_argument(
        "--prior_snap_to_superpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--prior_snap_threshold", type=float, default=1.0)
    parser.add_argument(
        "--prior_keep_unsnapped",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prior_keypoint_merge_threshold",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--drop_low_coverage_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min_frame_observations", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--camera_model", type=str, default=None)
    parser.add_argument("--ba_max_num_iterations", type=int, default=100)
    parser.add_argument("--tri_min_angle", type=float, default=1.0)
    parser.add_argument("--tri_create_max_angle_error", type=float, default=2.0)
    parser.add_argument(
        "--enable_select_tracks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--select_track_min_support", type=int, default=512)
    parser.add_argument(
        "--enable_reprojection_filter",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--filter_reproj_error_type",
        type=str,
        default="angular",
        choices=["angular", "pixel", "normalized"],
    )
    parser.add_argument(
        "--filter_reproj_error_threshold", type=float, default=0.5
    )
    parser.add_argument(
        "--debug_print",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def debug(args, message):
    if args.debug_print:
        print(f"[MERG3R-REFINE] {message}", flush=True)


def summarize_counts(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": 0, "median": 0, "max": 0, "mean": 0.0, "zero": 0}
    return {
        "min": int(values.min()),
        "median": float(np.median(values)),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "zero": int(np.sum(values == 0)),
    }


def format_count_summary(label, values):
    summary = summarize_counts(values)
    return (
        f"{label}: min={summary['min']}, median={summary['median']:.1f}, "
        f"mean={summary['mean']:.1f}, max={summary['max']}, "
        f"zero={summary['zero']}"
    )


def read_json(path):
    with open(path) as f:
        return json.load(f)


def load_images(images_dir, image_names, device):
    tensors = []
    for name in image_names:
        image = Image.open(images_dir / name).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors, dim=0).to(device)


def camera_centers_from_w2c(extrinsic):
    rotations = extrinsic[:, :3, :3]
    translations = extrinsic[:, :3, 3]
    return np.einsum(
        "nij,nj->ni",
        -np.transpose(rotations, (0, 2, 1)),
        translations,
    )


def normalize_extrinsic(extrinsic):
    extrinsic = np.asarray(extrinsic)
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(
            "Expected extrinsic shape (N,3,4) or (N,4,4), got "
            + f"{extrinsic.shape}"
        )
    return extrinsic[:, :3, :4]


def global_pose_dicts_from_w2c(extrinsic):
    rotations = {}
    centers = {}
    for idx in range(extrinsic.shape[0]):
        rotations[idx] = np.asarray(extrinsic[idx, :3, :3], dtype=np.float64)
        centers[idx] = np.asarray(
            -rotations[idx].T @ extrinsic[idx, :3, 3], dtype=np.float64
        )
    return rotations, centers


def shared_intrinsics(intrinsic):
    k = np.asarray(intrinsic, dtype=np.float64).mean(axis=0)
    return [torch.from_numpy(k).to(torch.float64).unsqueeze(0)]


def load_lightglue_features(features_dir, num_images):
    features = []
    for idx in range(num_images):
        data = np.load(features_dir / f"frame_{idx:06d}.npz")
        scores = (
            data["keypoint_scores"]
            if "keypoint_scores" in data
            else np.ones(len(data["keypoints"]), dtype=np.float32)
        )
        features.append(
            {
                "keypoints": np.asarray(data["keypoints"], dtype=np.float32),
                "descriptors": (
                    np.asarray(data["descriptors"], dtype=np.float32)
                    if "descriptors" in data
                    else None
                ),
                "scores": np.asarray(scores, dtype=np.float32),
            }
        )
    return features


def load_lightglue_matches(path):
    data = np.load(path)
    matches = {}
    for key in data.files:
        i, j = (int(x) for x in key.split("_"))
        matches[(i, j)] = np.asarray(data[key], dtype=np.uint32)
    return matches


def count_lightglue_observations(matches, num_images):
    counts = np.zeros(num_images, dtype=np.int64)
    for (i, j), match_array in matches.items():
        num_matches = int(match_array.shape[0])
        counts[int(i)] += num_matches
        counts[int(j)] += num_matches
    return counts


def build_vggsfm_groups(pairs, num_images, neighbors_per_center, centers=None):
    adjacency = defaultdict(list)
    for i, j in pairs.tolist():
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))

    groups = []
    for center in range(num_images):
        if centers is None:
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


def sample_query_points(keypoints, max_points):
    if keypoints.shape[0] <= max_points:
        return keypoints
    indices = np.linspace(0, keypoints.shape[0] - 1, max_points, dtype=np.int64)
    return keypoints[indices]


@torch.no_grad()
def build_vggsfm_query_points(args, images, features):
    if args.vggsfm_query_source == "superpoint":
        query_points = [
            np.asarray(feats["keypoints"], dtype=np.float32)
            for feats in features
        ]
        return query_points, {
            "query_source": "superpoint",
            "query_counts": [int(points.shape[0]) for points in query_points],
            "query_extraction_time": 0.0,
        }

    _ensure_gluemap_imports()
    from lightglue import ALIKED  # noqa: PLC0415

    t0 = time.time()
    extractor = (
        ALIKED(
            max_num_keypoints=args.vggsfm_query_points,
            detection_threshold=args.aliked_detection_threshold,
        )
        .eval()
        .to(args.device)
    )
    query_points = []
    for idx in range(images.shape[0]):
        feats = extractor.extract(images[idx : idx + 1])
        keypoints = (
            feats["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        )
        query_points.append(keypoints)

    return query_points, {
        "query_source": "aliked",
        "aliked_detection_threshold": args.aliked_detection_threshold,
        "query_counts": [int(points.shape[0]) for points in query_points],
        "query_extraction_time": time.time() - t0,
    }


@torch.no_grad()
def run_vggsfm_prior_tracks(args, images, features, pairs, metadata, extrinsic):
    if "P" not in args.track_mode:
        return [], {"num_groups": 0, "num_tracks": 0, "num_observations": 0}
    if not args.path_tracker:
        raise ValueError(
            "--path_tracker is required when --track_mode includes P"
        )

    _ensure_gluemap_imports()
    from vggsfm.vggsfm_tracker import TrackerPredictor  # noqa: PLC0415

    tracker = TrackerPredictor().eval().to(args.device)
    tracker.load_state_dict(
        torch.load(args.path_tracker, map_location="cpu", weights_only=False)
    )

    centers = camera_centers_from_w2c(extrinsic)
    groups = build_vggsfm_groups(
        pairs,
        images.shape[0],
        args.neighbors_per_center,
        centers=centers,
    )
    tracks = []
    observations = 0
    query_points_per_image, query_stats = build_vggsfm_query_points(
        args, images, features
    )

    for group in groups:
        center = group[0]
        query_np = sample_query_points(
            query_points_per_image[center], args.vggsfm_query_points
        )
        if query_np.shape[0] == 0:
            continue
        group_tensor = images[group].unsqueeze(0)
        query = (
            torch.from_numpy(query_np)
            .to(args.device, dtype=torch.float32)
            .unsqueeze(0)
        )
        pred_track, _, pred_vis, pred_score = tracker(
            group_tensor,
            query,
            fine_tracking=args.vggsfm_fine_tracking,
        )
        pred_track = pred_track[0].detach().cpu().numpy()
        pred_vis = pred_vis[0].detach().cpu().numpy()
        pred_score = pred_score[0].detach().cpu().numpy()

        for point_idx in range(query_np.shape[0]):
            obs = [(center, query_np[point_idx].astype(np.float32))]
            for local_idx, image_idx in enumerate(group[1:], start=1):
                if pred_vis[local_idx, point_idx] < args.vggsfm_vis_threshold:
                    continue
                if (
                    pred_score[local_idx, point_idx]
                    < args.vggsfm_score_threshold
                ):
                    continue
                xy = pred_track[local_idx, point_idx].astype(np.float32)
                h, w = metadata["image_size_hw"]
                if not (0 <= xy[0] < w and 0 <= xy[1] < h):
                    continue
                obs.append((int(image_idx), xy))
            if len(obs) >= 2:
                observations += len(obs)
                tracks.append(obs)

    return tracks, {
        "num_groups": len(groups),
        "num_tracks": len(tracks),
        "num_observations": observations,
        "neighbors_per_center": args.neighbors_per_center,
        "query_points": args.vggsfm_query_points,
        **query_stats,
    }


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


def remap_lightglue_matches(matches, old_to_new):
    remapped = {}
    for (i, j), match_array in matches.items():
        if int(i) not in old_to_new or int(j) not in old_to_new:
            continue
        ni = old_to_new[int(i)]
        nj = old_to_new[int(j)]
        if ni < nj:
            remapped[(ni, nj)] = match_array
        else:
            remapped[(nj, ni)] = match_array[:, [1, 0]]
    return remapped


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
    lightglue_matches,
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
            lightglue_matches,
            prior_tracks,
            {
                "enabled": enabled,
                "min_frame_observations": min_frame_observations,
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

    old_to_new = {
        int(old): new for new, old in enumerate(keep_indices.tolist())
    }
    dropped_indices = [
        idx for idx in range(len(image_names)) if idx not in old_to_new
    ]
    dropped_names = [image_names[idx] for idx in dropped_indices]

    filtered_image_names = [image_names[idx] for idx in keep_indices]
    filtered_images = images[
        torch.as_tensor(keep_indices, device=images.device)
    ]
    filtered_extrinsic = extrinsic[keep_indices]
    filtered_features = [features[idx] for idx in keep_indices]
    filtered_pairs = remap_pairs(pairs, old_to_new)
    filtered_lightglue_matches = remap_lightglue_matches(
        lightglue_matches, old_to_new
    )
    filtered_prior_tracks = remap_prior_tracks(prior_tracks, old_to_new)

    return (
        filtered_image_names,
        filtered_images,
        filtered_extrinsic,
        filtered_features,
        filtered_pairs,
        filtered_lightglue_matches,
        filtered_prior_tracks,
        {
            "enabled": enabled,
            "min_frame_observations": min_frame_observations,
            "dropped_indices": dropped_indices,
            "dropped_names": dropped_names,
            "s_observations": s_counts.tolist(),
            "p_observations": p_counts.tolist(),
            "total_observations": total_counts.tolist(),
        },
    )


def _write_cameras_and_images(
    database, pycolmap, image_names, image_size_hw, intrinsic, camera_model
):
    from gluemap.utils.colmap import (
        camera_from_intrinsics_matrix,
    )  # noqa: I001, PLC0415

    height, width = image_size_hw
    camera = camera_from_intrinsics_matrix(
        intrinsic,
        camera_model,
        width,
        height,
        1,
    )
    database.write_camera(camera)
    for idx, name in enumerate(image_names):
        image = pycolmap.Image()
        image.image_id = idx + 1
        image.camera_id = 1
        image.name = name
        database.write_image(image, use_image_id=True)


def write_lightglue_database(
    db_path,
    image_names,
    image_size_hw,
    intrinsic,
    camera_model,
    features,
    matches,
):
    pycolmap = _lazy_import_pycolmap()
    if os.path.exists(db_path):
        os.remove(db_path)
    database = pycolmap.Database.open(db_path)
    _write_cameras_and_images(
        database, pycolmap, image_names, image_size_hw, intrinsic, camera_model
    )
    for idx, feats in enumerate(features):
        database.write_keypoints(idx + 1, feats["keypoints"])
    for (i, j), match_array in matches.items():
        if match_array.shape[0] < 3:
            continue
        two_view_geometry = pycolmap.TwoViewGeometry()
        two_view_geometry.inlier_matches = match_array
        two_view_geometry.config = 2
        database.write_matches(i + 1, j + 1, match_array)
        database.write_two_view_geometry(i + 1, j + 1, two_view_geometry)
    database.close()
    return {
        "num_keypoints": [
            int(feats["keypoints"].shape[0]) for feats in features
        ],
        "num_pairs": int(len(matches)),
        "num_pairs_written": int(
            sum(
                1
                for match_array in matches.values()
                if match_array.shape[0] >= 3
            )
        ),
        "num_matches": int(
            sum(match_array.shape[0] for match_array in matches.values())
        ),
    }


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


def snap_prior_tracks_to_superpoint(
    tracks,
    features,
    snap_threshold=1.0,
    keep_unsnapped=True,
):
    keypoint_trees = []
    for feats in features:
        keypoints = np.asarray(feats["keypoints"], dtype=np.float32)
        keypoint_trees.append(
            cKDTree(keypoints) if keypoints.shape[0] > 0 else None
        )

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

    for track in tracks:
        snapped_obs = []
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
            if tree is None:
                if keep_unsnapped:
                    snapped_obs.append((image_idx, xy))
                    stats["unsnapped_kept_observations"] += 1
                    stats[f"{prefix}_unsnapped_kept_observations"] += 1
                else:
                    stats["dropped_observations"] += 1
                    stats[f"{prefix}_dropped_observations"] += 1
                continue

            distance, keypoint_idx = tree.query(xy, k=1)
            if float(distance) <= snap_threshold:
                snapped_xy = features[image_idx]["keypoints"][
                    int(keypoint_idx)
                ].astype(np.float32)
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
):
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
            obs_indices.append(
                (image_idx, int(raw_to_merged[image_idx][raw_idx]))
            )
        obs_indices = list(dict.fromkeys(obs_indices))
        if len(obs_indices) < 2:
            continue
        kept_tracks += 1
        track_lengths.append(len(obs_indices))
        for a in range(len(obs_indices)):
            for b in range(a + 1, len(obs_indices)):
                i, pi = obs_indices[a]
                j, pj = obs_indices[b]
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
    merge_stats["track_length_mean"] = (
        float(np.mean(track_lengths)) if track_lengths else 0.0
    )
    merge_stats["track_length_median"] = (
        float(np.median(track_lengths)) if track_lengths else 0.0
    )
    merge_stats["track_length_max"] = (
        int(max(track_lengths)) if track_lengths else 0
    )
    return keypoints_np, matches_np, merge_stats


def write_tracks_database(
    db_path,
    image_names,
    image_size_hw,
    intrinsic,
    camera_model,
    tracks,
    features=None,
    snap_to_features=True,
    snap_threshold=1.0,
    keep_unsnapped=True,
    merge_threshold=1e-3,
):
    pycolmap = _lazy_import_pycolmap()
    if os.path.exists(db_path):
        os.remove(db_path)
    if snap_to_features:
        if features is None:
            raise ValueError(
                "features are required when snap_to_features=True"
            )
        tracks_for_database, snap_stats = snap_prior_tracks_to_superpoint(
            tracks,
            features,
            snap_threshold=snap_threshold,
            keep_unsnapped=keep_unsnapped,
        )
    else:
        tracks_for_database = tracks
        snap_stats = {
            "enabled": False,
            "input_tracks": int(len(tracks)),
            "output_tracks": int(len(tracks)),
        }
    keypoints, matches, merge_stats = tracks_to_keypoints_and_matches(
        tracks_for_database,
        len(image_names),
        merge_threshold=merge_threshold,
    )
    database = pycolmap.Database.open(db_path)
    _write_cameras_and_images(
        database, pycolmap, image_names, image_size_hw, intrinsic, camera_model
    )
    for idx, keypoints_i in enumerate(keypoints):
        database.write_keypoints(idx + 1, keypoints_i)
    for (i, j), match_array in matches.items():
        if match_array.shape[0] < 3:
            continue
        two_view_geometry = pycolmap.TwoViewGeometry()
        two_view_geometry.inlier_matches = match_array
        two_view_geometry.config = 2
        database.write_matches(i + 1, j + 1, match_array)
        database.write_two_view_geometry(i + 1, j + 1, two_view_geometry)
    database.close()
    return {
        "num_tracks": len(tracks_for_database),
        "num_input_tracks": len(tracks),
        "num_keypoints": [int(k.shape[0]) for k in keypoints],
        "num_pairs": len(matches),
        "snap": snap_stats,
        "keypoint_merge": merge_stats,
    }


def write_coarse_reconstruction(
    output_dir, image_names, image_size_hw, extrinsic, intrinsic, camera_model
):
    from gluemap.utils.colmap import write_to_colmap_format  # noqa: PLC0415

    rotations, centers = global_pose_dicts_from_w2c(extrinsic)
    intrinsics = [torch.from_numpy(intrinsic).to(torch.float64).unsqueeze(0)]
    intrinsics_mapping = {idx: 0 for idx in range(extrinsic.shape[0])}
    write_to_colmap_format(
        str(output_dir),
        [tuple(image_size_hw) for _ in range(extrinsic.shape[0])],
        rotations,
        centers,
        intrinsics,
        intrinsics_mapping,
        images_list=image_names,
        camera_type=camera_model,
    )


@contextlib.contextmanager
def suppress_native_stdio(enabled=True):
    if not enabled:
        yield
        return
    saved_fds = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        os.close(devnull)
        os.close(saved_fds[0])
        os.close(saved_fds[1])


def triangulate(pycolmap, coarse_dir, database_path, output_dir, args):
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(str(coarse_dir))
    options = pycolmap.IncrementalPipelineOptions()
    options.triangulation.min_angle = args.tri_min_angle
    options.triangulation.ignore_two_view_tracks = False
    options.triangulation.create_max_angle_error = (
        args.tri_create_max_angle_error
    )
    options.ba_global_max_refinements = 0
    if output_dir.exists():
        shutil.rmtree(output_dir)
    with suppress_native_stdio():
        reconstruction = pycolmap.triangulate_points(
            reconstruction,
            str(database_path),
            ".",
            str(output_dir),
            clear_points=True,
            refine_intrinsics=False,
            options=options,
        )
    return reconstruction


def run_bundle_adjustment(pycolmap, reconstruction, max_num_iterations):
    if hasattr(pycolmap, "bundle_adjustment"):
        options = pycolmap.BundleAdjustmentOptions()
        if hasattr(options, "solver_options"):
            options.solver_options.max_num_iterations = max_num_iterations
        elif hasattr(options, "ceres"):
            options.ceres.solver_options.max_num_iterations = max_num_iterations
        return pycolmap.bundle_adjustment(reconstruction, options)

    from gluemap.estimators.augmented_bundle_adjustment import (
        bundle_adjustment,
    )  # noqa: PLC0415

    reconstruction, _, summary = bundle_adjustment(
        reconstruction,
        None,
        {},
        max_num_iterations=max_num_iterations,
    )
    return summary


def classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count):
    counts = {"total": 0, "s": 0, "non_s": 0, "mixed": 0}
    for point3d in reconstruction.points3D.values():
        flags = []
        for elem in point3d.track.elements:
            flags.append(
                int(elem.point2D_idx) < s_keypoint_count.get(elem.image_id, 0)
            )
        if not flags:
            continue
        counts["total"] += 1
        if all(flags):
            counts["s"] += 1
        elif any(flags):
            counts["mixed"] += 1
        else:
            counts["non_s"] += 1
    return counts


def run_select_tracks(reconstruction, features, min_num_support_abs):
    from gluemap.controllers.global_refinement import (  # noqa: PLC0415
        select_tracks_from_merged,
    )

    s_keypoint_count = {
        image_id: int(features[image_id - 1]["keypoints"].shape[0])
        for image_id in reconstruction.images
    }
    before = classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count)
    pair_count = select_tracks_from_merged(
        reconstruction=reconstruction,
        sift_count=s_keypoint_count,
        min_num_support_abs=min_num_support_abs,
    )
    after = classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count)
    return {
        "enabled": True,
        "min_num_support_abs": int(min_num_support_abs),
        "before": before,
        "after": after,
        "removed_points3D": int(before["total"] - after["total"]),
        "pair_count_entries": int(len(pair_count)),
    }


def run_reprojection_filter(reconstruction, error_type, error_threshold):
    from gluemap.math.reprojection_error import (  # noqa: PLC0415
        ReprojectionErrorType,
        filter_reconstruction_by_reprojection_error,
    )

    error_type_map = {
        "angular": ReprojectionErrorType.ANGULAR,
        "pixel": ReprojectionErrorType.PIXEL,
        "normalized": ReprojectionErrorType.NORMALIZED,
    }
    before_points = len(reconstruction.points3D)
    observations_removed, tracks_removed = (
        filter_reconstruction_by_reprojection_error(
            reconstruction,
            error_type_map[error_type],
            error_threshold,
            log_prefix="real: ",
        )
    )
    return {
        "enabled": True,
        "error_type": error_type,
        "error_threshold": float(error_threshold),
        "observations_removed": int(observations_removed),
        "tracks_removed": int(tracks_removed),
        "points3D_before": int(before_points),
        "points3D_after": int(len(reconstruction.points3D)),
    }


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --device cpu is used.")

    _ensure_gluemap_imports()
    pycolmap = _lazy_import_pycolmap()

    artifact_dir = Path(args.artifact_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else artifact_dir / "pycolmap_refine"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    metadata = read_json(artifact_dir / "metadata.json")
    poses = np.load(artifact_dir / "coarse_poses.npz")
    extrinsic = normalize_extrinsic(poses["extrinsic"]).astype(np.float64)
    intrinsic_all = np.asarray(poses["intrinsic"], dtype=np.float64)
    intrinsic = intrinsic_all.mean(axis=0)
    pairs = np.load(artifact_dir / "pairs.npy")
    image_names = metadata["artifact_image_names"]
    image_size_hw = tuple(metadata["image_size_hw"])
    camera_model = args.camera_model or metadata.get("camera_model", "PINHOLE")

    debug(
        args,
        "Loaded artifacts: "
        f"images={len(image_names)}, pairs={pairs.shape[0]}, "
        f"image_size_hw={image_size_hw}, camera_model={camera_model}, "
        f"track_mode={args.track_mode}",
    )
    debug(
        args,
        "Intrinsics shared mean: "
        f"fx={intrinsic[0, 0]:.2f}, fy={intrinsic[1, 1]:.2f}, "
        f"cx={intrinsic[0, 2]:.2f}, cy={intrinsic[1, 2]:.2f}",
    )

    features = load_lightglue_features(
        artifact_dir / "features_lightglue", len(image_names)
    )
    lightglue_matches = load_lightglue_matches(
        artifact_dir / "matches_lightglue.npz"
    )
    images = load_images(artifact_dir / "images", image_names, args.device)

    stats = {"track_mode": args.track_mode, "timing": {}}

    prior_tracks = []
    if "P" in args.track_mode:
        debug(
            args,
            "Running VGGSfM prior tracking: "
            f"neighbors_per_center={args.neighbors_per_center}, "
            f"query_points={args.vggsfm_query_points}, "
            f"query_source={args.vggsfm_query_source}, "
            f"fine_tracking={args.vggsfm_fine_tracking}",
        )
        t0 = time.time()
        prior_tracks, prior_stats = run_vggsfm_prior_tracks(
            args, images, features, pairs, metadata, extrinsic
        )
        stats["timing"]["vggsfm_prior_tracks"] = time.time() - t0
        stats["vggsfm"] = prior_stats
        debug(
            args,
            "VGGSfM prior done: "
            f"groups={prior_stats['num_groups']}, "
            f"tracks={prior_stats['num_tracks']}, "
            f"observations={prior_stats['num_observations']}, "
            f"query_source={prior_stats['query_source']}, "
            f"query_extract_time="
            f"{prior_stats['query_extraction_time']:.2f}s, "
            f"time={stats['timing']['vggsfm_prior_tracks']:.2f}s",
        )

    s_counts = (
        count_lightglue_observations(lightglue_matches, len(image_names))
        if "S" in args.track_mode
        else np.zeros(len(image_names), dtype=np.int64)
    )
    p_counts = (
        count_track_observations(prior_tracks, len(image_names))
        if "P" in args.track_mode
        else np.zeros(len(image_names), dtype=np.int64)
    )
    debug(args, format_count_summary("S observations/frame", s_counts))
    debug(args, format_count_summary("P observations/frame", p_counts))
    debug(
        args,
        format_count_summary("S+P observations/frame", s_counts + p_counts),
    )

    (
        image_names,
        images,
        extrinsic,
        features,
        pairs,
        lightglue_matches,
        prior_tracks,
        coverage_stats,
    ) = filter_low_coverage_frames(
        image_names,
        images,
        extrinsic,
        features,
        pairs,
        lightglue_matches,
        prior_tracks,
        s_counts,
        p_counts,
        args.min_frame_observations,
        enabled=args.drop_low_coverage_frames,
    )
    stats["frame_filtering"] = coverage_stats
    stats["num_images_after_filter"] = len(image_names)
    debug(
        args,
        "Frame filtering: "
        f"enabled={coverage_stats['enabled']}, "
        f"min_obs={coverage_stats['min_frame_observations']}, "
        f"dropped={len(coverage_stats['dropped_indices'])}, "
        f"remaining={len(image_names)}",
    )
    if coverage_stats["dropped_indices"]:
        debug(
            args,
            "Dropped frames: "
            + ", ".join(
                f"{idx}:{name}"
                for idx, name in zip(
                    coverage_stats["dropped_indices"],
                    coverage_stats["dropped_names"],
                    strict=False,
                )
            ),
        )

    if "S" in args.track_mode:
        debug(args, "Writing LightGlue COLMAP database")
        t0 = time.time()
        lightglue_db_stats = write_lightglue_database(
            str(output_dir / "database_lightglue.db"),
            image_names,
            image_size_hw,
            intrinsic,
            camera_model,
            features,
            lightglue_matches,
        )
        stats["timing"]["write_lightglue_db"] = time.time() - t0
        stats["lightglue"] = {
            "num_pairs": len(lightglue_matches),
            "num_matches": int(
                sum(m.shape[0] for m in lightglue_matches.values())
            ),
            "database": lightglue_db_stats,
        }
        debug(
            args,
            "LightGlue DB written: "
            f"pairs={lightglue_db_stats['num_pairs']}, "
            f"pairs_written={lightglue_db_stats['num_pairs_written']}, "
            f"matches={lightglue_db_stats['num_matches']}, "
            f"time={stats['timing']['write_lightglue_db']:.2f}s",
        )

    if "P" in args.track_mode:
        debug(args, "Writing VGGSfM prior COLMAP database")
        t0 = time.time()
        stats["prior_database"] = write_tracks_database(
            str(output_dir / "database_vggsfm_prior.db"),
            image_names,
            image_size_hw,
            intrinsic,
            camera_model,
            prior_tracks,
            features=features,
            snap_to_features=args.prior_snap_to_superpoint,
            snap_threshold=args.prior_snap_threshold,
            keep_unsnapped=args.prior_keep_unsnapped,
            merge_threshold=args.prior_keypoint_merge_threshold,
        )
        stats["timing"]["write_prior_db"] = time.time() - t0
        prior_db_stats = stats["prior_database"]
        snap_stats = prior_db_stats["snap"]
        if snap_stats["enabled"]:
            center_snap_rate = (
                snap_stats["center_snapped_observations"]
                / snap_stats["center_observations"]
                if snap_stats["center_observations"]
                else 0.0
            )
            neighbor_snap_rate = (
                snap_stats["neighbor_snapped_observations"]
                / snap_stats["neighbor_observations"]
                if snap_stats["neighbor_observations"]
                else 0.0
            )
            debug(
                args,
                "Prior snap to SuperPoint: "
                f"threshold={snap_stats['snap_threshold']}, "
                f"snapped={snap_stats['snapped_observations']}, "
                f"unsnapped_kept="
                f"{snap_stats['unsnapped_kept_observations']}, "
                f"dropped={snap_stats['dropped_observations']}, "
                f"mean_dist={snap_stats['snap_distance_mean']:.3f}, "
                f"max_dist={snap_stats['snap_distance_max']:.3f}, "
                f"center_snap={center_snap_rate:.1%}, "
                f"neighbor_snap={neighbor_snap_rate:.1%}",
            )
        debug(
            args,
            "Prior DB written: "
            f"input_tracks={prior_db_stats['num_input_tracks']}, "
            f"tracks={prior_db_stats['num_tracks']}, "
            f"pairs={prior_db_stats['num_pairs']}, "
            f"raw_kp={prior_db_stats['keypoint_merge']['raw_total']}, "
            f"merged_kp={prior_db_stats['keypoint_merge']['merged_total']}, "
            f"reduced={prior_db_stats['keypoint_merge']['merged_reduction']}, "
            f"time={stats['timing']['write_prior_db']:.2f}s",
        )

    from gluemap.utils.colmap import merge_colmap_databases  # noqa: PLC0415

    t0 = time.time()
    if args.track_mode == "S":
        debug(args, "Using LightGlue database as merged database")
        shutil.copy2(
            output_dir / "database_lightglue.db",
            output_dir / "database_merged.db",
        )
    elif args.track_mode == "P":
        debug(args, "Using VGGSfM prior database as merged database")
        shutil.copy2(
            output_dir / "database_vggsfm_prior.db",
            output_dir / "database_merged.db",
        )
    else:
        debug(args, "Merging LightGlue and VGGSfM prior databases")
        merge_colmap_databases(
            str(output_dir / "database_lightglue.db"),
            str(output_dir / "database_vggsfm_prior.db"),
            str(output_dir / "database_merged.db"),
            primary_features_first=True,
        )
    stats["timing"]["merge_databases"] = time.time() - t0
    debug(
        args,
        f"Merged database ready in {stats['timing']['merge_databases']:.2f}s",
    )

    coarse_dir = output_dir / "coarse"
    debug(args, f"Writing coarse reconstruction: {coarse_dir}")
    t0 = time.time()
    write_coarse_reconstruction(
        coarse_dir,
        image_names,
        image_size_hw,
        extrinsic,
        intrinsic,
        camera_model,
    )
    stats["timing"]["write_coarse"] = time.time() - t0
    debug(
        args,
        "Coarse reconstruction written in "
        + f"{stats['timing']['write_coarse']:.2f}s",
    )

    debug(
        args,
        "Triangulating points: "
        f"min_angle={args.tri_min_angle}, "
        f"create_max_angle_error={args.tri_create_max_angle_error}",
    )
    t0 = time.time()
    reconstruction = triangulate(
        pycolmap,
        coarse_dir,
        output_dir / "database_merged.db",
        output_dir / "triangulated",
        args,
    )
    stats["timing"]["triangulation"] = time.time() - t0
    stats["triangulation"] = {
        "num_images": len(reconstruction.images),
        "num_points3D": len(reconstruction.points3D),
    }
    debug(
        args,
        "Triangulation done: "
        f"images={stats['triangulation']['num_images']}, "
        f"points3D={stats['triangulation']['num_points3D']}, "
        f"time={stats['timing']['triangulation']:.2f}s",
    )

    if (
        args.enable_select_tracks
        and "S" in args.track_mode
        and "P" in args.track_mode
    ):
        debug(
            args,
            "Running SelectTrack: "
            f"min_support={args.select_track_min_support}",
        )
        t0 = time.time()
        stats["select_tracks"] = run_select_tracks(
            reconstruction,
            features,
            args.select_track_min_support,
        )
        stats["timing"]["select_tracks"] = time.time() - t0
        select_stats = stats["select_tracks"]
        before = select_stats["before"]
        after = select_stats["after"]
        debug(
            args,
            "SelectTrack done: "
            f"points={before['total']} -> {after['total']}, "
            f"removed={select_stats['removed_points3D']}, "
            f"S={after['s']}, mixed={after['mixed']}, "
            f"nonS={after['non_s']}, "
            f"time={stats['timing']['select_tracks']:.2f}s",
        )
    else:
        stats["select_tracks"] = {
            "enabled": False,
            "reason": "disabled or track_mode does not include both S and P",
        }

    if args.enable_reprojection_filter:
        debug(
            args,
            "Running reprojection filter: "
            f"type={args.filter_reproj_error_type}, "
            f"threshold={args.filter_reproj_error_threshold}",
        )
        t0 = time.time()
        stats["reprojection_filter"] = run_reprojection_filter(
            reconstruction,
            args.filter_reproj_error_type,
            args.filter_reproj_error_threshold,
        )
        stats["timing"]["reprojection_filter"] = time.time() - t0
        filter_stats = stats["reprojection_filter"]
        debug(
            args,
            "Reprojection filter done: "
            f"points={filter_stats['points3D_before']} -> "
            f"{filter_stats['points3D_after']}, "
            f"obs_removed={filter_stats['observations_removed']}, "
            f"tracks_removed={filter_stats['tracks_removed']}, "
            f"time={stats['timing']['reprojection_filter']:.2f}s",
        )
    else:
        stats["reprojection_filter"] = {"enabled": False}

    debug(
        args,
        f"Running bundle adjustment: max_iters={args.ba_max_num_iterations}",
    )
    t0 = time.time()
    summary = run_bundle_adjustment(
        pycolmap, reconstruction, args.ba_max_num_iterations
    )
    stats["timing"]["bundle_adjustment"] = time.time() - t0
    stats["bundle_adjustment"] = {"summary": str(summary)}
    debug(
        args,
        "Bundle adjustment done in "
        + f"{stats['timing']['bundle_adjustment']:.2f}s",
    )
    debug(args, f"BA summary: {summary}")

    refined_dir = output_dir / "refined_pycolmap"
    refined_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(refined_dir))
    stats["timing"]["total"] = time.time() - t_start
    stats["output"] = {
        "coarse_dir": str(coarse_dir),
        "database_merged": str(output_dir / "database_merged.db"),
        "refined_dir": str(refined_dir),
    }
    with open(output_dir / "refine_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[REFINE] Wrote refined reconstruction to {refined_dir}")
    print(
        "[REFINE] "
        + f"triangulated_points={stats['triangulation']['num_points3D']}"
    )


if __name__ == "__main__":
    main()
