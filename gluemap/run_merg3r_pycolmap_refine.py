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
    parser.add_argument("--path_tracker", type=str, default=None)
    parser.add_argument(
        "--track_mode", type=str, default="SP", choices=["S", "P", "SP"]
    )
    parser.add_argument("--neighbors_per_center", type=int, default=8)
    parser.add_argument("--vggsfm_query_points", type=int, default=1024)
    parser.add_argument("--vggsfm_vis_threshold", type=float, default=0.5)
    parser.add_argument("--vggsfm_score_threshold", type=float, default=0.0)
    parser.add_argument("--vggsfm_fine_tracking", action="store_true")
    parser.add_argument(
        "--drop_low_coverage_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min_frame_observations", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--camera_model", type=str, default=None)
    parser.add_argument("--ba_max_num_iterations", type=int, default=100)
    parser.add_argument("--tri_min_angle", type=float, default=1.0)
    parser.add_argument("--tri_create_max_angle_error", type=float, default=2.0)
    return parser.parse_args()


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

    for group in groups:
        center = group[0]
        query_np = sample_query_points(
            features[center]["keypoints"], args.vggsfm_query_points
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


def tracks_to_keypoints_and_matches(tracks, num_images):
    keypoints = [[] for _ in range(num_images)]
    pair_matches = defaultdict(list)
    for track in tracks:
        obs_indices = []
        seen = set()
        for image_idx, xy in track:
            if image_idx in seen:
                continue
            seen.add(image_idx)
            point_idx = len(keypoints[image_idx])
            keypoints[image_idx].append(np.asarray(xy, dtype=np.float32))
            obs_indices.append((image_idx, point_idx))
        if len(obs_indices) < 2:
            continue
        for a in range(len(obs_indices)):
            for b in range(a + 1, len(obs_indices)):
                i, pi = obs_indices[a]
                j, pj = obs_indices[b]
                if i > j:
                    i, j = j, i
                    pi, pj = pj, pi
                pair_matches[(i, j)].append((pi, pj))
    keypoints_np = [
        np.stack(k).astype(np.float32)
        if k
        else np.empty((0, 2), dtype=np.float32)
        for k in keypoints
    ]
    matches_np = {
        k: np.asarray(v, dtype=np.uint32) for k, v in pair_matches.items()
    }
    return keypoints_np, matches_np


def write_tracks_database(
    db_path, image_names, image_size_hw, intrinsic, camera_model, tracks
):
    pycolmap = _lazy_import_pycolmap()
    if os.path.exists(db_path):
        os.remove(db_path)
    keypoints, matches = tracks_to_keypoints_and_matches(
        tracks,
        len(image_names),
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
        "num_tracks": len(tracks),
        "num_keypoints": [int(k.shape[0]) for k in keypoints],
        "num_pairs": len(matches),
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
        t0 = time.time()
        prior_tracks, prior_stats = run_vggsfm_prior_tracks(
            args, images, features, pairs, metadata, extrinsic
        )
        stats["timing"]["vggsfm_prior_tracks"] = time.time() - t0
        stats["vggsfm"] = prior_stats

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

    if "S" in args.track_mode:
        t0 = time.time()
        write_lightglue_database(
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
        }

    if "P" in args.track_mode:
        t0 = time.time()
        stats["prior_database"] = write_tracks_database(
            str(output_dir / "database_vggsfm_prior.db"),
            image_names,
            image_size_hw,
            intrinsic,
            camera_model,
            prior_tracks,
        )
        stats["timing"]["write_prior_db"] = time.time() - t0

    from gluemap.utils.colmap import merge_colmap_databases  # noqa: PLC0415

    t0 = time.time()
    if args.track_mode == "S":
        shutil.copy2(
            output_dir / "database_lightglue.db",
            output_dir / "database_merged.db",
        )
    elif args.track_mode == "P":
        shutil.copy2(
            output_dir / "database_vggsfm_prior.db",
            output_dir / "database_merged.db",
        )
    else:
        merge_colmap_databases(
            str(output_dir / "database_lightglue.db"),
            str(output_dir / "database_vggsfm_prior.db"),
            str(output_dir / "database_merged.db"),
            primary_features_first=True,
        )
    stats["timing"]["merge_databases"] = time.time() - t0

    coarse_dir = output_dir / "coarse"
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

    t0 = time.time()
    summary = run_bundle_adjustment(
        pycolmap, reconstruction, args.ba_max_num_iterations
    )
    stats["timing"]["bundle_adjustment"] = time.time() - t0
    stats["bundle_adjustment"] = {"summary": str(summary)}

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
