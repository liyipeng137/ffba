import contextlib
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

# Extracted SPV/SIFT/ALIKED utility functions for the integrated pipeline.

BAE_MIN_OBSERVATIONS_PER_IMAGE = 64


def _lazy_import_pycolmap():
    try:
        import pycolmap  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycolmap is required. Run this script in the Gluemap environment."
        ) from exc
    return pycolmap


def _ensure_gluemap_imports():
    repo_root = Path(__file__).resolve().parents[1] / "third_party" / "gluemap"
    if not (repo_root / "gluemap").is_dir():
        raise FileNotFoundError(f"GlueMap directory not found: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import thirdparty.path_to_thirdparty  # noqa: F401, PLC0415


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


def camera_centers_from_w2c(extrinsic):
    rotations = extrinsic[:, :3, :3]
    translations = extrinsic[:, :3, 3]
    return np.einsum(
        "nij,nj->ni",
        -np.transpose(rotations, (0, 2, 1)),
        translations,
    )


def camera_viewing_axes_from_w2c(extrinsic):
    rotations_c2w = np.transpose(np.asarray(extrinsic)[:, :3, :3], (0, 2, 1))
    return rotations_c2w[:, :, -1]


def to_homogeneous_w2c(extrinsic):
    extrinsic = np.asarray(extrinsic)
    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic
    if extrinsic.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected w2c extrinsic shape (...,3,4), got {extrinsic.shape}"
        )
    bottom_shape = extrinsic.shape[:-2] + (1, 4)
    bottom = np.zeros(bottom_shape, dtype=extrinsic.dtype)
    bottom[..., 0, 3] = 1.0
    return np.concatenate([extrinsic, bottom], axis=-2)


def center_local_extrinsics(extrinsic, group):
    w2c = to_homogeneous_w2c(extrinsic[group])
    center_inv = np.linalg.inv(w2c[0])
    local = np.einsum("nij,jk->nik", w2c, center_inv)
    return local[:, :3, :4]


def global_pose_dicts_from_w2c(extrinsic):
    rotations = {}
    centers = {}
    for idx in range(extrinsic.shape[0]):
        rotations[idx] = np.asarray(extrinsic[idx, :3, :3], dtype=np.float64)
        centers[idx] = np.asarray(
            -rotations[idx].T @ extrinsic[idx, :3, 3], dtype=np.float64
        )
    return rotations, centers


def average_intrinsics_with_gluemap(initial_intrinsics, camera_model):
    from gluemap.estimators.intrinsics_averaging import (  # noqa: PLC0415
        intrinsics_averaging,
    )

    initial_intrinsics = np.asarray(initial_intrinsics, dtype=np.float64)
    num_images = initial_intrinsics.shape[0]
    intrinsics_mapping = {idx: 0 for idx in range(num_images)}
    communities = [list(range(num_images))]
    intrinsics_all = [
        torch.from_numpy(initial_intrinsics).to(torch.float64).unsqueeze(0)
    ]
    global_intrinsics = intrinsics_averaging(
        intrinsics_all,
        communities,
        intrinsics_mapping,
        camera_model=camera_model,
    )
    if not global_intrinsics or global_intrinsics[0] is None:
        raise ValueError("intrinsics_averaging produced no shared intrinsics")
    averaged_intrinsics = np.repeat(
        global_intrinsics[0].detach().cpu().numpy(),
        num_images,
        axis=0,
    )
    return averaged_intrinsics, global_intrinsics, intrinsics_mapping


def summarize_intrinsics(initial_intrinsics, averaged_intrinsics, camera_model):
    initial_intrinsics = np.asarray(initial_intrinsics, dtype=np.float64)
    averaged_intrinsics = np.asarray(averaged_intrinsics, dtype=np.float64)
    initial_fx = initial_intrinsics[:, 0, 0]
    initial_fy = initial_intrinsics[:, 1, 1]
    initial_cx = initial_intrinsics[:, 0, 2]
    initial_cy = initial_intrinsics[:, 1, 2]
    averaged_k = averaged_intrinsics[0]

    delta = averaged_intrinsics - initial_intrinsics
    return {
        "method": "gluemap.intrinsics_averaging",
        "camera_model": camera_model,
        "num_images": int(initial_intrinsics.shape[0]),
        "intrinsics_mapping": "shared",
        "initial": {
            "fx_min": float(initial_fx.min()),
            "fx_median": float(np.median(initial_fx)),
            "fx_max": float(initial_fx.max()),
            "fy_min": float(initial_fy.min()),
            "fy_median": float(np.median(initial_fy)),
            "fy_max": float(initial_fy.max()),
            "cx_median": float(np.median(initial_cx)),
            "cy_median": float(np.median(initial_cy)),
        },
        "averaged": {
            "fx": float(averaged_k[0, 0]),
            "fy": float(averaged_k[1, 1]),
            "cx": float(averaged_k[0, 2]),
            "cy": float(averaged_k[1, 2]),
        },
        "delta_abs": {
            "fx_mean": float(np.mean(np.abs(delta[:, 0, 0]))),
            "fx_max": float(np.max(np.abs(delta[:, 0, 0]))),
            "fy_mean": float(np.mean(np.abs(delta[:, 1, 1]))),
            "fy_max": float(np.max(np.abs(delta[:, 1, 1]))),
            "cx_mean": float(np.mean(np.abs(delta[:, 0, 2]))),
            "cx_max": float(np.max(np.abs(delta[:, 0, 2]))),
            "cy_mean": float(np.mean(np.abs(delta[:, 1, 2]))),
            "cy_max": float(np.max(np.abs(delta[:, 1, 2]))),
        },
    }


def save_intrinsics_artifacts(
    output_dir,
    initial_intrinsics,
    averaged_intrinsics,
    intrinsics_mapping,
    image_names,
):
    mapping_array = np.asarray(
        [intrinsics_mapping[idx] for idx in range(len(intrinsics_mapping))],
        dtype=np.int64,
    )
    np.savez_compressed(
        output_dir / "intrinsics_refine_inputs.npz",
        initial_intrinsics=initial_intrinsics.astype(np.float64, copy=False),
        averaged_intrinsics=averaged_intrinsics.astype(np.float64, copy=False),
        shared_global_intrinsic=averaged_intrinsics[0].astype(np.float64, copy=False),
        intrinsics_mapping=mapping_array,
        image_names=np.asarray(image_names),
    )


def summarize_numeric(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def summarize_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def clone_virtual_tracks(predictions_dict):
    return {
        idx: (
            predictions_dict["tracks_virtual"][idx].clone(),
            predictions_dict["valid_virtual"][idx].clone(),
        )
        for idx in range(len(predictions_dict["indexes"]))
    }


def summarize_track_displacement(before, predictions_dict):
    deltas = []
    for idx, (tracks_before, valid_before) in before.items():
        tracks_after = predictions_dict["tracks_virtual"][idx]
        valid_after = predictions_dict["valid_virtual"][idx]
        if tracks_before.shape != tracks_after.shape:
            continue
        valid = (valid_before > 0) & (valid_after > 0)
        if not valid.any():
            continue
        delta = torch.linalg.norm(tracks_after - tracks_before, dim=-1)
        deltas.append(delta[valid].detach().cpu().numpy())
    if not deltas:
        return summarize_numeric([])
    return summarize_numeric(np.concatenate(deltas))


def summarize_pose_scores(predictions_dict):
    scores = []
    for idx in range(len(predictions_dict["indexes"])):
        current = predictions_dict["pose_scores"][idx][0]
        if current.numel() > 1:
            scores.append(current[1:].detach().cpu().numpy())
    if not scores:
        return summarize_numeric([])
    return summarize_numeric(np.concatenate(scores))


def summarize_virtual_valid(predictions_dict, num_images):
    point_counts = []
    view_counts = []
    frame_valid = np.zeros(num_images, dtype=np.int64)
    total_valid = 0
    center_valid = 0
    neighbor_valid = 0
    negative = 0
    for idx, group in enumerate(predictions_dict["indexes"]):
        valid = predictions_dict["valid_virtual"][idx] > 0
        point_counts.append(int(valid.shape[-1]))
        view_counts.append(len(group))
        total_valid += int(valid.sum().item())
        center_valid += int(valid[:, 0].sum().item())
        if valid.shape[1] > 1:
            neighbor_valid += int(valid[:, 1:].sum().item())
        for local_idx, image_idx in enumerate(group):
            frame_valid[int(image_idx)] += int(valid[:, local_idx].sum().item())
        if "isnegative_virtual" in predictions_dict:
            negative += int(
                (predictions_dict["isnegative_virtual"][idx] > 0).sum().item()
            )

    return {
        "num_groups": int(len(predictions_dict["indexes"])),
        "views_per_group": summarize_numeric(view_counts),
        "points_per_group": summarize_numeric(point_counts),
        "valid_observations": int(total_valid),
        "center_valid_observations": int(center_valid),
        "neighbor_valid_observations": int(neighbor_valid),
        "negative_observations": int(negative),
        "frame_valid_observations": summarize_counts(frame_valid),
        "frames_without_virtual_observations": int(np.sum(frame_valid == 0)),
    }


def summarize_virtual_pair_coverage(predictions_dict):
    counts = []
    pair_counts = {}
    for idx, group in enumerate(predictions_dict["indexes"]):
        if len(group) <= 1:
            continue
        valid = predictions_dict["valid_virtual"][idx] > 0
        center_valid = valid[0, 0]
        center = int(group[0])
        for local_idx, image_idx in enumerate(group[1:], start=1):
            count = int((center_valid & valid[0, local_idx]).sum().item())
            counts.append(count)
            pair = tuple(sorted((center, int(image_idx))))
            key = f"{pair[0]}_{pair[1]}"
            pair_counts[key] = pair_counts.get(key, 0) + count
    return {
        "pairs": int(len(pair_counts)),
        "coverage": summarize_numeric(counts),
        "zero_pairs": (
            int(np.sum(np.asarray(counts, dtype=np.int64) == 0)) if counts else 0
        ),
        "pair_counts": pair_counts,
    }


@torch.no_grad()
def build_virtual_track_diagnostics(
    args,
    output_dir,
    depth,
    depth_conf,
    extrinsic,
    depth_intrinsics,
    global_intrinsics,
    intrinsics_mapping,
    pairs,
    image_names,
    image_size_hw,
    depth_image_size_hw=None,
):
    from gluemap.estimators.covisibility_extraction import (  # noqa: PLC0415
        CovisibilityExtraction,
    )
    from gluemap.estimators.virtual_tracks import (  # noqa: PLC0415
        VirtualTrackPreparation,
    )

    num_images = len(image_names)
    if depth_image_size_hw is None:
        depth_image_size_hw = image_size_hw
    if tuple(depth.shape[1:3]) != tuple(depth_image_size_hw):
        raise ValueError(
            f"Depth resolution {tuple(depth.shape[1:3])} does not match "
            f"depth_image_size_hw={tuple(depth_image_size_hw)}"
        )

    centers = camera_centers_from_w2c(extrinsic)
    viewing_axes = camera_viewing_axes_from_w2c(extrinsic)
    groups, group_stats = build_vggsfm_groups(
        args,
        pairs,
        num_images,
        image_names,
        image_size_hw,
        centers=centers,
        viewing_axes=viewing_axes,
    )
    center_set = {group[0] for group in groups}
    skipped_centers = [idx for idx in range(num_images) if idx not in center_set]

    predictions_dict = {
        "indexes": [],
        "extrinsics": [],
        "intrinsics": [],
        "pose_scores": [],
        "tracks_virtual": [],
        "points3d_virtual": [],
        "valid_virtual": [],
        "isnegative_virtual": [],
    }
    extractor = CovisibilityExtraction(include_track=False)
    device = torch.device(args.device)
    virtual_verify_mode = getattr(args, "virtual_verify_mode", "n2")
    if virtual_verify_mode not in {"n2", "center"}:
        raise ValueError(
            "virtual_verify_mode must be 'n2' or 'center', "
            f"got {virtual_verify_mode!r}"
        )
    t0 = time.time()
    for group in groups:
        group_np = np.asarray(group, dtype=np.int64)
        local_extrinsic = center_local_extrinsics(extrinsic, group_np)
        predictions = {
            "depth": torch.from_numpy(depth[group_np]).to(
                device=device, dtype=torch.float32
            )[None],
            "extrinsics": torch.from_numpy(local_extrinsic).to(
                device=device, dtype=torch.float32
            )[None],
            "intrinsics": torch.from_numpy(depth_intrinsics[group_np]).to(
                device=device, dtype=torch.float32
            )[None],
        }
        if depth_conf is not None:
            predictions["depth_conf"] = torch.from_numpy(depth_conf[group_np]).to(
                device=device, dtype=torch.float32
            )[None]

        # Call the three generation stages directly so diagnostics use
        # project_tracks' true return order: valid_mask, then is_negative.
        depth_transformed = extractor._convert_from_depth_to_world_points(
            predictions["depth"],
            predictions["extrinsics"],
            predictions["intrinsics"],
        )
        if virtual_verify_mode == "n2":
            pose_scores, reprojection_valid_mask = extractor._verify_by_reprojection_n2(
                depth_transformed,
                predictions["extrinsics"],
                predictions["intrinsics"],
            )
        else:
            pose_scores, reprojection_valid_mask = extractor._verify_by_reprojection(
                depth_transformed,
                predictions["extrinsics"],
                predictions["intrinsics"],
            )
        (
            tracks_virtual,
            points3d_virtual,
            valid_virtual,
            isnegative_virtual,
        ) = extractor._calculate_virtual_tracks(
            predictions["depth"],
            predictions["extrinsics"],
            predictions["intrinsics"],
            reprojection_valid_mask,
        )

        predictions_dict["indexes"].append(group)
        predictions_dict["extrinsics"].append(predictions["extrinsics"].detach().cpu())
        predictions_dict["intrinsics"].append(predictions["intrinsics"].detach().cpu())
        predictions_dict["pose_scores"].append(pose_scores.detach().cpu())
        predictions_dict["tracks_virtual"].append(tracks_virtual.detach().cpu())
        predictions_dict["points3d_virtual"].append(points3d_virtual.detach().cpu())
        predictions_dict["valid_virtual"].append(valid_virtual.detach().cpu())
        predictions_dict["isnegative_virtual"].append(isnegative_virtual.detach().cpu())

    generation_seconds = time.time() - t0
    torch.cuda.empty_cache()

    stats = {
        "enabled": True,
        "num_images": int(num_images),
        "num_groups": int(len(groups)),
        "neighbors_per_center": int(args.neighbors_per_center),
        "group_strategy": args.group_strategy,
        "verify_mode": virtual_verify_mode,
        "group_stats": group_stats,
        "skipped_centers": skipped_centers,
        "depth": {
            "shape": list(depth.shape),
            "image_size_hw": list(depth_image_size_hw),
            "refine_image_size_hw": list(image_size_hw),
            "has_depth_conf": depth_conf is not None,
            "positive_pixels": int(np.sum(depth[..., 0] > 0)),
        },
        "generation": {
            "seconds": generation_seconds,
            "pose_scores": summarize_pose_scores(predictions_dict),
            "virtual": summarize_virtual_valid(predictions_dict, num_images),
            "pair_coverage": summarize_virtual_pair_coverage(predictions_dict),
        },
    }

    rotations, global_centers = global_pose_dicts_from_w2c(extrinsic)
    preparation = VirtualTrackPreparation()
    indexes_range = range(len(predictions_dict["indexes"]))
    for idx in indexes_range:
        predictions_dict["intrinsics"][idx] = (
            torch.stack(
                [
                    global_intrinsics[intrinsics_mapping[image_idx]]
                    for image_idx in predictions_dict["indexes"][idx]
                ],
                dim=1,
            )
            .detach()
            .cpu()
        )

    before_update = clone_virtual_tracks(predictions_dict)
    t0 = time.time()
    preparation._update_virtual_tracks(
        predictions_dict,
        global_intrinsics,
        intrinsics_mapping,
        indexes_range,
    )
    stats["update_virtual_tracks"] = {
        "seconds": time.time() - t0,
        "displacement_px": summarize_track_displacement(
            before_update, predictions_dict
        ),
        "virtual": summarize_virtual_valid(predictions_dict, num_images),
        "pair_coverage": summarize_virtual_pair_coverage(predictions_dict),
    }

    t0 = time.time()
    before_subsample_points = [
        int(predictions_dict["points3d_virtual"][idx].shape[-2])
        for idx in indexes_range
    ]
    preparation._subsample_virtual_tracks(predictions_dict, indexes_range)
    after_subsample_points = [
        int(predictions_dict["points3d_virtual"][idx].shape[-2])
        for idx in indexes_range
    ]
    stats["subsample_virtual_tracks"] = {
        "seconds": time.time() - t0,
        "points_before": summarize_numeric(before_subsample_points),
        "points_after": summarize_numeric(after_subsample_points),
        "virtual": summarize_virtual_valid(predictions_dict, num_images),
        "pair_coverage": summarize_virtual_pair_coverage(predictions_dict),
    }

    before_global = clone_virtual_tracks(predictions_dict)
    t0 = time.time()
    preparation._update_virtual_tracks_global(
        predictions_dict,
        global_intrinsics,
        intrinsics_mapping,
        rotations,
        global_centers,
        indexes_range,
    )
    stats["update_virtual_tracks_global"] = {
        "seconds": time.time() - t0,
        "displacement_px": summarize_track_displacement(
            before_global, predictions_dict
        ),
        "virtual": summarize_virtual_valid(predictions_dict, num_images),
        "pair_coverage": summarize_virtual_pair_coverage(predictions_dict),
    }

    stats_path = output_dir / "virtual_track_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    if args.save_virtual_tracks_debug:
        torch.save(
            {
                "predictions_dict": predictions_dict,
                "image_names": image_names,
                "intrinsics_mapping": intrinsics_mapping,
            },
            output_dir / "virtual_tracks_debug.pt",
        )
    return predictions_dict, stats


def load_database_keypoint_features(database_path, image_names):
    pycolmap = _lazy_import_pycolmap()
    database = pycolmap.Database.open(str(database_path))
    features = []
    try:
        for name in image_names:
            image = database.read_image_with_name(str(Path(name)))
            keypoints = database.read_keypoints(image.image_id)
            if keypoints is None or len(keypoints) == 0:
                keypoints_xy = np.empty((0, 2), dtype=np.float32)
            else:
                keypoints_xy = np.asarray(keypoints[:, :2], dtype=np.float32)
            features.append(
                {
                    "keypoints": keypoints_xy,
                    "descriptors": None,
                    "scores": np.ones(keypoints_xy.shape[0], dtype=np.float32),
                }
            )
    finally:
        database.close()
    return features


def load_database_keypoints_per_image(database_path, image_names):
    """Load 0-indexed image keypoints while preserving database indices."""
    pycolmap = _lazy_import_pycolmap()
    database = pycolmap.Database.open(str(database_path))
    keypoints_per_image = {}
    try:
        for image_idx, name in enumerate(image_names):
            image = database.read_image_with_name(str(Path(name)))
            expected_image_id = image_idx + 1
            if image.image_id != expected_image_id:
                raise ValueError(
                    "Merged database image IDs do not match reconstruction "
                    f"indices: {name!r} has ID {image.image_id}, expected "
                    f"{expected_image_id}"
                )
            keypoints = database.read_keypoints(image.image_id)
            if keypoints is None or len(keypoints) == 0:
                keypoints_per_image[image_idx] = np.empty((0, 2), dtype=np.float64)
            else:
                keypoints_per_image[image_idx] = np.ascontiguousarray(
                    keypoints[:, :2], dtype=np.float64
                )
    finally:
        database.close()
    return keypoints_per_image


def summarize_database_matches(database_path, image_names):
    pycolmap = _lazy_import_pycolmap()
    database = pycolmap.Database.open(str(database_path))
    counts = np.zeros(len(image_names), dtype=np.int64)
    try:
        image_id_to_name = {
            image.image_id: image.name for image in database.read_all_images()
        }
        name_to_idx = {str(Path(name)): idx for idx, name in enumerate(image_names)}
        pair_ids, matches_list = database.read_all_matches()
        num_pairs = 0
        num_matches = 0
        for pair_id, matches in zip(pair_ids, matches_list, strict=False):
            if matches is None or len(matches) == 0:
                continue
            image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
            name1 = image_id_to_name.get(image_id1)
            name2 = image_id_to_name.get(image_id2)
            if name1 not in name_to_idx or name2 not in name_to_idx:
                continue
            match_count = int(len(matches))
            counts[name_to_idx[name1]] += match_count
            counts[name_to_idx[name2]] += match_count
            num_pairs += 1
            num_matches += match_count
    finally:
        database.close()
    return {
        "num_pairs": int(num_pairs),
        "num_matches": int(num_matches),
        "observations_per_image": counts.tolist(),
    }


def prepare_sift_database_for_refine(
    args,
    output_dir,
    artifact_dir,
    image_names,
    pairs,
    camera_model,
    intrinsics_mapping,
):
    from gluemap.utils.colmap import prepare_sift_database  # noqa: PLC0415

    prepare_sift_database(
        str(output_dir),
        str(artifact_dir / "images"),
        image_names,
        intrinsics_mapping,
        pairs,
        device=args.device,
        camera_model=camera_model,
        skip_matching=False,
        remove_existing=True,
    )
    database_path = output_dir / "database_sift.db"
    features = load_database_keypoint_features(database_path, image_names)
    match_stats = summarize_database_matches(database_path, image_names)
    keypoint_counts = [int(feat["keypoints"].shape[0]) for feat in features]
    return features, {
        "database": str(database_path),
        "num_keypoints": keypoint_counts,
        "num_keypoints_total": int(sum(keypoint_counts)),
        **match_stats,
    }


def _copy_colmap_camera(pycolmap, camera, camera_id):
    return pycolmap.Camera(
        camera_id=camera_id,
        model=camera.model_name,
        params=camera.params,
        width=camera.width,
        height=camera.height,
    )


def _read_database_descriptors(database, image_id):
    if not hasattr(database, "read_descriptors"):
        return None
    return database.read_descriptors(image_id)


def _write_database_descriptors(database, image_id, descriptors):
    if descriptors is None:
        return
    if not hasattr(database, "write_descriptors"):
        return
    with contextlib.suppress(TypeError, ValueError, RuntimeError):
        database.write_descriptors(image_id, descriptors)


def filter_sift_database_for_refine(
    source_db_path,
    target_db_path,
    source_image_names,
    kept_indices,
    pairs,
    intrinsics_mapping=None,
    pairs_txt_path=None,
):
    """Rewrite a prefilter SIFT DB for the frame-filtered image subset."""
    pycolmap = _lazy_import_pycolmap()
    source_db_path = Path(source_db_path)
    target_db_path = Path(target_db_path)
    target_db_path.parent.mkdir(parents=True, exist_ok=True)

    source_image_names = [str(Path(name)) for name in source_image_names]
    kept_indices = [int(idx) for idx in kept_indices]
    filtered_image_names = [source_image_names[idx] for idx in kept_indices]
    if intrinsics_mapping is None:
        intrinsics_mapping = {idx: 0 for idx in range(len(filtered_image_names))}
    else:
        intrinsics_mapping = {
            int(idx): int(camera_idx) for idx, camera_idx in intrinsics_mapping.items()
        }

    in_place = source_db_path.resolve() == target_db_path.resolve()
    write_db_path = target_db_path
    if in_place:
        write_db_path = target_db_path.with_name(
            f"{target_db_path.stem}.filtered.tmp{target_db_path.suffix}"
        )
    if write_db_path.exists():
        os.remove(write_db_path)

    src_db = pycolmap.Database.open(str(source_db_path))
    dst_db = pycolmap.Database.open(str(write_db_path))
    try:
        source_images = {
            str(Path(image.name)): image for image in src_db.read_all_images()
        }
        source_cameras = {
            camera.camera_id: camera for camera in src_db.read_all_cameras()
        }

        old_index_to_new_index = {
            old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)
        }
        source_id_to_old_index = {}
        for old_idx, name in enumerate(source_image_names):
            image = source_images.get(name)
            if image is not None:
                source_id_to_old_index[image.image_id] = old_idx

        missing_names = [
            name for name in filtered_image_names if name not in source_images
        ]
        if missing_names:
            raise ValueError(
                "Filtered image names are missing from source SIFT database: "
                + ", ".join(missing_names[:10])
            )

        camera_bucket_to_source_camera_id = {}
        for new_idx, old_idx in enumerate(kept_indices):
            name = source_image_names[old_idx]
            camera_bucket = intrinsics_mapping[new_idx]
            camera_bucket_to_source_camera_id.setdefault(
                camera_bucket,
                source_images[name].camera_id,
            )

        for camera_bucket, source_camera_id in sorted(
            camera_bucket_to_source_camera_id.items()
        ):
            source_camera = source_cameras[source_camera_id]
            dst_db.write_camera(
                _copy_colmap_camera(
                    pycolmap,
                    source_camera,
                    camera_bucket + 1,
                )
            )

        for new_idx, old_idx in enumerate(kept_indices):
            name = source_image_names[old_idx]
            source_image = source_images[name]
            new_image = pycolmap.Image()
            new_image.image_id = new_idx + 1
            new_image.camera_id = intrinsics_mapping[new_idx] + 1
            new_image.name = name
            dst_db.write_image(new_image, use_image_id=True)

            keypoints = src_db.read_keypoints(source_image.image_id)
            if keypoints is not None and len(keypoints) > 0:
                dst_db.write_keypoints(new_image.image_id, keypoints)

            descriptors = _read_database_descriptors(
                src_db,
                source_image.image_id,
            )
            _write_database_descriptors(
                dst_db,
                new_image.image_id,
                descriptors,
            )

        allowed_pairs = {tuple(sorted((int(i), int(j)))) for i, j in pairs.tolist()}
        pair_ids, matches_list = src_db.read_all_matches()
        for pair_id, matches in zip(pair_ids, matches_list, strict=False):
            if matches is None or len(matches) == 0:
                continue
            source_id1, source_id2 = pycolmap.pair_id_to_image_pair(pair_id)
            old_idx1 = source_id_to_old_index.get(source_id1)
            old_idx2 = source_id_to_old_index.get(source_id2)
            if old_idx1 not in old_index_to_new_index:
                continue
            if old_idx2 not in old_index_to_new_index:
                continue

            new_idx1 = old_index_to_new_index[old_idx1]
            new_idx2 = old_index_to_new_index[old_idx2]
            new_pair = tuple(sorted((new_idx1, new_idx2)))
            if new_pair not in allowed_pairs:
                continue

            match_array = np.asarray(matches).copy()
            if new_idx1 <= new_idx2:
                new_image_id1 = new_idx1 + 1
                new_image_id2 = new_idx2 + 1
            else:
                new_image_id1 = new_idx2 + 1
                new_image_id2 = new_idx1 + 1
                match_array = match_array[:, [1, 0]]
            dst_db.write_matches(new_image_id1, new_image_id2, match_array)

            with contextlib.suppress(Exception):
                geometry = src_db.read_two_view_geometry(source_id1, source_id2)
                if (
                    geometry is None
                    or geometry.inlier_matches is None
                    or len(geometry.inlier_matches) == 0
                ):
                    continue
                inlier_matches = np.asarray(geometry.inlier_matches).copy()
                if new_idx1 > new_idx2:
                    inlier_matches = inlier_matches[:, [1, 0]]
                new_geometry = pycolmap.TwoViewGeometry()
                new_geometry.inlier_matches = inlier_matches
                new_geometry.config = geometry.config
                dst_db.write_two_view_geometry(
                    new_image_id1,
                    new_image_id2,
                    new_geometry,
                )
    finally:
        src_db.close()
        dst_db.close()

    if in_place:
        os.replace(write_db_path, target_db_path)

    if pairs_txt_path is None:
        pairs_txt_path = target_db_path.parent / "pairs.txt"
    pairs_txt_path = Path(pairs_txt_path)
    with open(pairs_txt_path, "w") as f:
        for i, j in sorted(allowed_pairs):
            f.write(f"{filtered_image_names[i]} {filtered_image_names[j]}\n")

    features = load_database_keypoint_features(target_db_path, filtered_image_names)
    match_stats = summarize_database_matches(target_db_path, filtered_image_names)
    keypoint_counts = [int(feat["keypoints"].shape[0]) for feat in features]
    return features, {
        "database": str(target_db_path),
        "source_database": str(source_db_path),
        "pairs_txt": str(pairs_txt_path),
        "kept_indices": kept_indices,
        "num_images": len(filtered_image_names),
        "num_keypoints": keypoint_counts,
        "num_keypoints_total": int(sum(keypoint_counts)),
        **match_stats,
    }


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
):
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
    for center in range(num_images):
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


def summarize_groups(groups, num_images):
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


def sample_query_points(keypoints, max_points):
    if keypoints.shape[0] <= max_points:
        return keypoints
    indices = np.linspace(0, keypoints.shape[0] - 1, max_points, dtype=np.int64)
    return keypoints[indices]


def apply_image_change(points, image_change):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 0] = points[..., 0] * image_change[0] + image_change[2]
    points[..., 1] = points[..., 1] * image_change[1] + image_change[3]
    return points


def invert_image_change(points, image_change):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 0] = (points[..., 0] - image_change[2]) / image_change[0]
    points[..., 1] = (points[..., 1] - image_change[3]) / image_change[1]
    return points


def prepare_vggsfm_tracker_images(args, images):
    if args.vggsfm_tracker_input == "native":
        return (
            images,
            None,
            {
                "tracker_input": "native",
                "tracker_image_size_hw": [
                    int(images.shape[-2]),
                    int(images.shape[-1]),
                ],
            },
        )

    _ensure_gluemap_imports()
    from gluemap.utils.load_fn import (  # noqa: PLC0415
        load_and_preprocess_images_1024,
    )

    images_cpu = [images[idx].detach().cpu() for idx in range(images.shape[0])]
    images_1024, image_changes_1024 = load_and_preprocess_images_1024(images_cpu)
    image_changes_1024 = np.asarray(image_changes_1024, dtype=np.float32)
    return (
        images_1024,
        image_changes_1024,
        {
            "tracker_input": "1024",
            "tracker_image_size_hw": [
                int(images_1024.shape[-2]),
                int(images_1024.shape[-1]),
            ],
        },
    )


@torch.no_grad()
def build_vggsfm_query_points(
    args, tracker_images, features=None, tracker_image_changes=None
):
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
    for idx in range(tracker_images.shape[0]):
        image = tracker_images[idx : idx + 1].to(args.device)
        feats = extractor.extract(image)
        keypoints = feats["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        query_points.append(keypoints)

    return query_points, {
        "query_source": "aliked",
        "aliked_detection_threshold": args.aliked_detection_threshold,
        "query_counts": [int(points.shape[0]) for points in query_points],
        "query_extraction_time": time.time() - t0,
    }


@torch.no_grad()
def precompute_vggsfm_tracker_fmaps(tracker, args, tracker_images, chunk_size=32):
    t0 = time.time()
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")

    num_images = int(tracker_images.shape[0])
    tracker_device = torch.device(args.device)
    use_cuda_cache = tracker_device.type == "cuda"
    cache_budget_fraction = 0.5
    tracker_fmaps = None
    storage_dtype = torch.float32
    free_cuda_bytes_before_cache = None
    estimated_cache_bytes = None
    fallback_reason = None

    # Keep the CUDA allocator cache warm across VGGSfM preprocessing chunks.
    # if use_cuda_cache:
    #     torch.cuda.empty_cache()

    for start in range(0, num_images, chunk_size):
        end = min(start + chunk_size, num_images)
        images_chunk = tracker_images[start:end].to(tracker_device, non_blocking=True)
        fmaps_chunk = tracker.process_images_to_fmaps(images_chunk)

        if tracker_fmaps is None:
            full_shape = (num_images, *fmaps_chunk.shape[1:])
            if use_cuda_cache:
                storage_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                element_size = torch.empty((), dtype=storage_dtype).element_size()
                estimated_cache_bytes = int(np.prod(full_shape)) * element_size
                free_cuda_bytes_before_cache, _ = torch.cuda.mem_get_info(
                    tracker_device
                )
                cache_budget_bytes = int(
                    free_cuda_bytes_before_cache * cache_budget_fraction
                )
                if estimated_cache_bytes > cache_budget_bytes:
                    fallback_reason = (
                        "estimated compressed fmap cache exceeds 50% of "
                        "currently free CUDA memory"
                    )
                else:
                    try:
                        tracker_fmaps = torch.empty(
                            full_shape,
                            dtype=storage_dtype,
                            device=tracker_device,
                        )
                    except torch.OutOfMemoryError:
                        fallback_reason = "CUDA allocation failed"
                        # torch.cuda.empty_cache()

            if tracker_fmaps is None:
                # Preserve the previous FP32 CPU-cache behavior when the
                # compressed resident cache would leave too little workspace
                # for the tracker itself.
                storage_dtype = torch.float32
                tracker_fmaps = torch.empty(
                    full_shape,
                    dtype=storage_dtype,
                    device="cpu",
                )

        tracker_fmaps[start:end].copy_(fmaps_chunk.detach(), non_blocking=True)
        del images_chunk, fmaps_chunk
        # Keep the CUDA allocator cache warm for the next chunk.
        # if use_cuda_cache:
        #     torch.cuda.empty_cache()

    cache_bytes = tracker_fmaps.numel() * tracker_fmaps.element_size()
    resident_on_tracker_device = tracker_fmaps.device.type == tracker_device.type and (
        tracker_device.index is None
        or tracker_fmaps.device.index == tracker_device.index
    )
    return tracker_fmaps, {
        "enabled": True,
        "chunk_size": int(chunk_size),
        "seconds": time.time() - t0,
        "shape": [int(v) for v in tracker_fmaps.shape],
        "storage_device": str(tracker_fmaps.device),
        "storage_dtype": str(tracker_fmaps.dtype).removeprefix("torch."),
        "resident_on_tracker_device": resident_on_tracker_device,
        "cache_bytes": int(cache_bytes),
        "estimated_compressed_cache_bytes": estimated_cache_bytes,
        "free_cuda_bytes_before_cache": free_cuda_bytes_before_cache,
        "cuda_cache_budget_fraction": (
            cache_budget_fraction if use_cuda_cache else None
        ),
        "fallback_reason": fallback_reason,
    }


@torch.no_grad()
def run_vggsfm_prior_tracks(
    args,
    images,
    features,
    pairs,
    metadata,
    extrinsic,
    image_names,
    groups=None,
    group_stats=None,
):
    if not args.path_tracker:
        raise ValueError("--path_tracker is required")
    group_batch_size = int(getattr(args, "vggsfm_group_batch_size", 1))
    if group_batch_size <= 0:
        raise ValueError("vggsfm_group_batch_size must be >= 1")

    _ensure_gluemap_imports()
    from vggsfm.vggsfm_tracker import TrackerPredictor  # noqa: PLC0415

    tracker = TrackerPredictor().eval().to(args.device)
    tracker.load_state_dict(
        torch.load(args.path_tracker, map_location="cpu", weights_only=False)
    )

    centers = camera_centers_from_w2c(extrinsic)
    viewing_axes = camera_viewing_axes_from_w2c(extrinsic)
    if (groups is None) != (group_stats is None):
        raise ValueError("groups and group_stats must be provided together")
    if groups is None:
        groups, group_stats = build_vggsfm_groups(
            args,
            pairs,
            images.shape[0],
            image_names,
            metadata["image_size_hw"],
            centers=centers,
            viewing_axes=viewing_axes,
        )
    observations = 0
    tracker_images, tracker_image_changes, tracker_stats = (
        prepare_vggsfm_tracker_images(args, images)
    )
    query_points_per_image, query_stats = build_vggsfm_query_points(
        args,
        tracker_images,
        features,
        tracker_image_changes=tracker_image_changes,
    )
    tracker_fmaps, fmaps_stats = precompute_vggsfm_tracker_fmaps(
        tracker,
        args,
        tracker_images,
    )
    tracker_parameter = next(tracker.parameters())
    tracker_device = tracker_parameter.device
    tracker_dtype = tracker_parameter.dtype

    neighbor_rank_stats = [
        {
            "rank": rank,
            "pair_count": 0,
            "rotation_valid_pairs": 0,
            "unfiltered_pairs": 0,
            "unclassified_pairs": 0,
            "pairs_executed": 0,
            "pairs_with_accepted_observations": 0,
            "attempted_queries": 0,
            "visibility_pass": 0,
            "score_pass": 0,
            "in_bounds_pass": 0,
            "accepted_observations": 0,
        }
        for rank in range(1, int(args.neighbors_per_center) + 1)
    ]
    rotation_threshold = getattr(args, "pair_pose_rotation_threshold", None)
    total_neighbor_slots = 0
    for group in groups:
        center = int(group[0])
        total_neighbor_slots += max(len(group) - 1, 0)
        for rank, image_idx in enumerate(group[1:], start=1):
            rank_stats = neighbor_rank_stats[rank - 1]
            rank_stats["pair_count"] += 1
            if rotation_threshold is None:
                rank_stats["unclassified_pairs"] += 1
                continue
            dot = float(np.dot(viewing_axes[center], viewing_axes[int(image_idx)]))
            angle = float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
            if angle < rotation_threshold:
                rank_stats["rotation_valid_pairs"] += 1
            else:
                rank_stats["unfiltered_pairs"] += 1

    total_queries = 0
    valid_center_queries = 0
    attempted_query_views = 0
    tracks_by_group = {}
    track_lengths_by_group = {}
    t_group = time.time()
    tracking_buckets = defaultdict(list)
    zero_query_centers = []
    for group_order, group in enumerate(groups):
        group = [int(image_idx) for image_idx in group]
        center = int(group[0])
        query_np = sample_query_points(
            query_points_per_image[center], args.vggsfm_query_points
        )
        if query_np.shape[0] == 0:
            zero_query_centers.append(center)
            continue
        num_queries = int(query_np.shape[0])
        total_queries += num_queries
        attempted_query_views += num_queries * max(len(group) - 1, 0)
        tracking_buckets[(len(group), num_queries)].append(
            (group_order, group, query_np)
        )

    bucket_stats = []
    num_forward_calls = 0
    effective_batch_size_histogram = defaultdict(int)
    for bucket_index, ((group_size, query_count), jobs) in enumerate(
        sorted(tracking_buckets.items()),
        start=1,
    ):
        group_count = len(jobs)
        batch_count = (group_count + group_batch_size - 1) // group_batch_size
        tail_batch_size = group_count % group_batch_size or min(
            group_batch_size, group_count
        )
        bucket_stat = {
            "bucket_index": int(bucket_index),
            "group_size": int(group_size),
            "neighbors_per_group": int(group_size - 1),
            "query_points": int(query_count),
            "group_count": int(group_count),
            "batch_count": int(batch_count),
            "tail_batch_size": int(tail_batch_size),
        }
        bucket_stats.append(bucket_stat)
        debug(
            args,
            "VGGSfM batch bucket "
            f"{bucket_index}/{len(tracking_buckets)}: "
            f"group_size={group_size}, query_points={query_count}, "
            f"groups={group_count}, batches={batch_count}, "
            f"tail_batch_size={tail_batch_size}",
        )

    debug(
        args,
        "VGGSfM batch bucketing done: "
        f"buckets={len(bucket_stats)}, batch_size={group_batch_size}, "
        f"runnable_groups={sum(len(jobs) for jobs in tracking_buckets.values())}, "
        f"zero_query_groups={len(zero_query_centers)}, "
        f"query_points_per_bucket="
        f"{[item['query_points'] for item in bucket_stats]}",
    )

    h, w = metadata["image_size_hw"]
    for (group_size, query_count), jobs in sorted(tracking_buckets.items()):
        for batch_start in range(0, len(jobs), group_batch_size):
            batch_jobs = jobs[batch_start : batch_start + group_batch_size]
            actual_batch_size = len(batch_jobs)
            num_forward_calls += 1
            effective_batch_size_histogram[actual_batch_size] += 1

            group_indices_np = np.asarray(
                [group for _group_order, group, _query_np in batch_jobs],
                dtype=np.int64,
            )
            group_tensor = None
            if args.vggsfm_fine_tracking:
                tracker_image_indices = torch.as_tensor(
                    group_indices_np,
                    dtype=torch.long,
                    device=tracker_images.device,
                )
                group_tensor = tracker_images[tracker_image_indices].to(tracker_device)
            tracker_fmap_indices = torch.as_tensor(
                group_indices_np,
                dtype=torch.long,
                device=tracker_fmaps.device,
            )
            group_fmaps = tracker_fmaps[tracker_fmap_indices].to(
                device=tracker_device,
                dtype=tracker_dtype,
                non_blocking=True,
            )
            query = torch.from_numpy(
                np.stack([query_np for _group_order, _group, query_np in batch_jobs])
            ).to(tracker_device, dtype=torch.float32)
            pred_track_batch, _, pred_vis_batch, pred_score_batch = tracker(
                group_tensor,
                query,
                fmaps=group_fmaps,
                fine_tracking=args.vggsfm_fine_tracking,
            )
            del group_fmaps, group_tensor, query
            pred_track_batch = pred_track_batch.detach().cpu().numpy()
            pred_vis_batch = pred_vis_batch.detach().cpu().numpy()
            pred_score_batch = pred_score_batch.detach().cpu().numpy()

            for batch_index, (group_order, group, query_np) in enumerate(batch_jobs):
                center = int(group[0])
                num_queries = int(query_np.shape[0])
                pred_track = pred_track_batch[batch_index]
                pred_vis = pred_vis_batch[batch_index]
                pred_score = pred_score_batch[batch_index]

                center_points = query_np.astype(np.float32, copy=True)
                if tracker_image_changes is not None:
                    center_points = invert_image_change(
                        center_points, tracker_image_changes[center]
                    )
                center_valid = (
                    (center_points[:, 0] >= 0)
                    & (center_points[:, 0] < w)
                    & (center_points[:, 1] >= 0)
                    & (center_points[:, 1] < h)
                )
                valid_center_queries += int(center_valid.sum())

                accepted_neighbor_masks = {}
                mapped_neighbor_tracks = {}
                for local_idx, image_idx in enumerate(group[1:], start=1):
                    rank_stats = neighbor_rank_stats[local_idx - 1]
                    rank_stats["pairs_executed"] += 1
                    rank_stats["attempted_queries"] += num_queries

                    visibility_values = np.asarray(pred_vis[local_idx]).reshape(-1)
                    score_values = np.asarray(pred_score[local_idx]).reshape(-1)
                    if (
                        visibility_values.size != num_queries
                        or score_values.size != num_queries
                    ):
                        raise ValueError(
                            "VGGSfM visibility/score output does not match query "
                            f"count: queries={num_queries}, "
                            f"visibility={visibility_values.shape}, "
                            f"score={score_values.shape}"
                        )
                    visibility_pass = visibility_values >= args.vggsfm_vis_threshold
                    score_pass = visibility_pass & (
                        score_values >= args.vggsfm_score_threshold
                    )
                    neighbor_points = pred_track[local_idx].astype(
                        np.float32, copy=True
                    )
                    if tracker_image_changes is not None:
                        neighbor_points = invert_image_change(
                            neighbor_points,
                            tracker_image_changes[int(image_idx)],
                        )
                    in_bounds_pass = (
                        score_pass
                        & (neighbor_points[:, 0] >= 0)
                        & (neighbor_points[:, 0] < w)
                        & (neighbor_points[:, 1] >= 0)
                        & (neighbor_points[:, 1] < h)
                    )
                    accepted = in_bounds_pass & center_valid

                    rank_stats["visibility_pass"] += int(visibility_pass.sum())
                    rank_stats["score_pass"] += int(score_pass.sum())
                    rank_stats["in_bounds_pass"] += int(in_bounds_pass.sum())
                    rank_stats["accepted_observations"] += int(accepted.sum())
                    if accepted.any():
                        rank_stats["pairs_with_accepted_observations"] += 1
                    accepted_neighbor_masks[local_idx] = accepted
                    mapped_neighbor_tracks[local_idx] = neighbor_points

                group_tracks = []
                group_track_lengths = []
                for point_idx in range(num_queries):
                    if not center_valid[point_idx]:
                        continue
                    obs = [(center, center_points[point_idx])]
                    for local_idx, image_idx in enumerate(group[1:], start=1):
                        if not accepted_neighbor_masks[local_idx][point_idx]:
                            continue
                        obs.append(
                            (
                                int(image_idx),
                                mapped_neighbor_tracks[local_idx][point_idx],
                            )
                        )
                    if len(obs) >= 2:
                        observations += len(obs)
                        group_tracks.append(obs)
                        group_track_lengths.append(len(obs))
                tracks_by_group[group_order] = group_tracks
                track_lengths_by_group[group_order] = group_track_lengths
    tracks = []
    track_lengths = []
    for group_order in range(len(groups)):
        tracks.extend(tracks_by_group.get(group_order, []))
        track_lengths.extend(track_lengths_by_group.get(group_order, []))
    group_tracking_time = time.time() - t_group
    debug(
        args,
        "VGGSfM batch tracking done: "
        f"forward_calls={num_forward_calls}, "
        f"effective_batch_size_histogram="
        f"{dict(sorted(effective_batch_size_histogram.items()))}",
    )

    for rank_stats in neighbor_rank_stats:
        attempted = rank_stats["attempted_queries"]
        for count_key, rate_key in (
            ("visibility_pass", "visibility_pass_rate"),
            ("score_pass", "score_pass_rate"),
            ("in_bounds_pass", "in_bounds_pass_rate"),
            ("accepted_observations", "accepted_observation_rate"),
        ):
            rank_stats[rate_key] = (
                float(rank_stats[count_key] / attempted) if attempted > 0 else 0.0
            )

    accepted_neighbor_observations = int(
        sum(rank["accepted_observations"] for rank in neighbor_rank_stats)
    )
    queries_forming_tracks = len(tracks)
    return tracks, {
        "num_groups": len(groups),
        "num_tracks": len(tracks),
        "num_observations": observations,
        "neighbors_per_center": args.neighbors_per_center,
        "group_strategy": args.group_strategy,
        "batching": {
            "configured_batch_size": int(group_batch_size),
            "num_buckets": int(len(bucket_stats)),
            "num_forward_calls": int(num_forward_calls),
            "zero_query_group_count": int(len(zero_query_centers)),
            "zero_query_centers": [int(center) for center in zero_query_centers],
            "effective_batch_size_histogram": {
                str(batch_size): int(count)
                for batch_size, count in sorted(effective_batch_size_histogram.items())
            },
            "buckets": bucket_stats,
        },
        "group_stats": group_stats,
        "query_points": args.vggsfm_query_points,
        "workload": {
            "total_neighbor_slots": int(total_neighbor_slots),
            "actual_query_points": int(total_queries),
            "attempted_query_views": int(attempted_query_views),
        },
        "neighbor_rank_stats": neighbor_rank_stats,
        "query_track_stats": {
            "total_queries": int(total_queries),
            "valid_center_queries": int(valid_center_queries),
            "invalid_center_queries": int(total_queries - valid_center_queries),
            "queries_forming_tracks": int(queries_forming_tracks),
            "queries_without_accepted_neighbor": int(
                valid_center_queries - queries_forming_tracks
            ),
            "forming_track_rate": (
                float(queries_forming_tracks / total_queries)
                if total_queries > 0
                else 0.0
            ),
            "forming_track_rate_valid_centers": (
                float(queries_forming_tracks / valid_center_queries)
                if valid_center_queries > 0
                else 0.0
            ),
            "track_length": summarize_distribution(track_lengths),
            "accepted_neighbor_observations": accepted_neighbor_observations,
            "observation_count_consistent": bool(
                accepted_neighbor_observations == observations - len(tracks)
            ),
        },
        "precompute_fmaps": fmaps_stats,
        "group_tracking_time": group_tracking_time,
        **tracker_stats,
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
    keep_indices=None,
):
    total_counts = s_counts + p_counts
    if keep_indices is not None:
        keep_indices = np.asarray(keep_indices, dtype=np.int64)
    elif not enabled:
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


def _write_cameras_and_images(
    database,
    pycolmap,
    image_names,
    image_size_hw,
    intrinsics,
    camera_model,
    intrinsics_mapping=None,
):
    from gluemap.utils.colmap import (
        camera_from_intrinsics_matrix,
    )  # noqa: I001, PLC0415

    height, width = image_size_hw
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None]
    if intrinsics_mapping is None:
        intrinsics_mapping = {idx: 0 for idx in range(len(image_names))}
    for camera_idx, intrinsic in enumerate(intrinsics):
        camera = camera_from_intrinsics_matrix(
            intrinsic,
            camera_model,
            width,
            height,
            camera_idx + 1,
        )
        database.write_camera(camera, use_camera_id=True)
    for idx, name in enumerate(image_names):
        image = pycolmap.Image()
        image.image_id = idx + 1
        image.camera_id = int(intrinsics_mapping[idx]) + 1
        image.name = name
        database.write_image(image, use_image_id=True)


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
            "match_topology must be 'all_pairs' or 'star', " f"got {match_topology!r}"
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
    snap_target="features",
    match_topology="all_pairs",
    intrinsics_mapping=None,
):
    pycolmap = _lazy_import_pycolmap()
    if os.path.exists(db_path):
        os.remove(db_path)
    if snap_to_features:
        if features is None:
            raise ValueError("features are required when snap_to_features=True")
        tracks_for_database, snap_stats = snap_prior_tracks_to_features(
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
    snap_stats["target"] = snap_target
    keypoints, matches, merge_stats = tracks_to_keypoints_and_matches(
        tracks_for_database,
        len(image_names),
        merge_threshold=merge_threshold,
        match_topology=match_topology,
    )
    database = pycolmap.Database.open(db_path)
    _write_cameras_and_images(
        database,
        pycolmap,
        image_names,
        image_size_hw,
        intrinsic,
        camera_model,
        intrinsics_mapping=intrinsics_mapping,
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


def classify_point3d_track_source(point3d, s_keypoint_count):
    has_s_observation = False
    has_p_observation = False
    for elem in point3d.track.elements:
        if int(elem.point2D_idx) < s_keypoint_count.get(elem.image_id, 0):
            has_s_observation = True
        else:
            has_p_observation = True

    if has_s_observation and has_p_observation:
        return "mixed"
    if has_s_observation:
        return "s_only"
    if has_p_observation:
        return "p_only"
    return "empty"


def build_s_keypoint_count(reconstruction, features):
    return {
        image_id: int(features[image_id - 1]["keypoints"].shape[0])
        for image_id in reconstruction.images
    }


def classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count):
    counts = {"total": 0, "s": 0, "non_s": 0, "mixed": 0}
    for point3d in reconstruction.points3D.values():
        source = classify_point3d_track_source(point3d, s_keypoint_count)
        if source == "empty":
            continue
        counts["total"] += 1
        if source == "s_only":
            counts["s"] += 1
        elif source == "mixed":
            counts["mixed"] += 1
        else:
            counts["non_s"] += 1
    return counts


def summarize_angular_errors_by_track_source(
    reconstruction,
    features,
    error_threshold,
    return_errors_per_track=False,
):
    if reconstruction is None:
        return {"enabled": False, "reason": "missing reconstruction"}

    from gluemap.math.reprojection_error import (  # noqa: PLC0415
        ReprojectionErrorType,
        compute_all_errors_from_reconstruction,
    )

    s_keypoint_count = build_s_keypoint_count(reconstruction, features)
    errors_per_track = compute_all_errors_from_reconstruction(
        reconstruction,
        ReprojectionErrorType.ANGULAR,
        negative_depth_observations={},
    )
    bucket_order = ("s_only", "p_only", "mixed")
    buckets = {
        source: {
            "track_count": 0,
            "track_observation_count": 0,
            "error_observation_count": 0,
            "finite_error_count": 0,
            "nonfinite_error_count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "lt_threshold_count": 0,
            "lt_threshold_ratio": 0.0,
        }
        for source in bucket_order
    }
    finite_errors_by_bucket = {source: [] for source in bucket_order}

    for point3D_id, point3d in reconstruction.points3D.items():
        source = classify_point3d_track_source(point3d, s_keypoint_count)
        if source == "empty":
            continue

        bucket = buckets[source]
        bucket["track_count"] += 1
        bucket["track_observation_count"] += len(list(point3d.track.elements))

        track_errors = errors_per_track.get(point3D_id, [])
        bucket["error_observation_count"] += len(track_errors)
        for _, _, error in track_errors:
            if np.isfinite(error):
                finite_errors_by_bucket[source].append(float(error))
            else:
                bucket["nonfinite_error_count"] += 1

    for source, finite_errors in finite_errors_by_bucket.items():
        bucket = buckets[source]
        values = np.asarray(finite_errors, dtype=np.float64)
        bucket["finite_error_count"] = int(values.size)
        if values.size == 0:
            continue
        bucket["min"] = float(values.min())
        bucket["median"] = float(np.median(values))
        bucket["mean"] = float(values.mean())
        bucket["p90"] = float(np.percentile(values, 90))
        bucket["max"] = float(values.max())
        bucket["lt_threshold_count"] = int(np.sum(values < error_threshold))
        bucket["lt_threshold_ratio"] = float(bucket["lt_threshold_count"] / values.size)

    result = {
        "enabled": True,
        "error_type": "angular",
        "error_threshold": float(error_threshold),
        "buckets": buckets,
    }
    if return_errors_per_track:
        return result, errors_per_track
    return result


def log_angular_errors_by_track_source(args, iteration, stats):
    if not stats.get("enabled"):
        debug(
            args,
            "Angular errors by track source skipped: "
            f"{stats.get('reason', 'unknown reason')}",
        )
        return

    threshold = stats["error_threshold"]
    labels = {
        "s_only": "S-only",
        "p_only": "P-only",
        "mixed": "mixed",
    }
    for source in ("s_only", "p_only", "mixed"):
        bucket = stats["buckets"][source]
        if bucket["finite_error_count"] > 0:
            error_summary = (
                f"mean={bucket['mean']:.4f}, "
                f"median={bucket['median']:.4f}, "
                f"p90={bucket['p90']:.4f}, "
                f"max={bucket['max']:.4f}, "
                f"<{threshold:g}deg="
                f"{bucket['lt_threshold_ratio'] * 100:.1f}%"
            )
        else:
            error_summary = (
                "mean=n/a, median=n/a, p90=n/a, max=n/a, " f"<{threshold:g}deg=n/a"
            )

        debug(
            args,
            "Angular errors by track source "
            f"iter={iteration}, bucket={labels[source]}: "
            f"tracks={bucket['track_count']}, "
            f"track_obs={bucket['track_observation_count']}, "
            f"error_obs={bucket['error_observation_count']}, "
            f"finite={bucket['finite_error_count']}, "
            f"nonfinite={bucket['nonfinite_error_count']}, "
            f"{error_summary}",
        )


def run_select_tracks(
    reconstruction,
    features,
    min_num_support_abs,
    return_pair_count=False,
):
    from gluemap.controllers.global_refinement import (  # noqa: PLC0415
        select_tracks_from_merged,
    )

    s_keypoint_count = build_s_keypoint_count(reconstruction, features)
    before = classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count)
    pair_count = select_tracks_from_merged(
        reconstruction=reconstruction,
        sift_count=s_keypoint_count,
        min_num_support_abs=min_num_support_abs,
    )
    after = classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count)
    stats = {
        "enabled": True,
        "min_num_support_abs": int(min_num_support_abs),
        "before": before,
        "after": after,
        "removed_points3D": int(before["total"] - after["total"]),
        "pair_count_entries": int(len(pair_count)),
    }
    if return_pair_count:
        return stats, pair_count
    return stats


def count_reconstruction_observations(reconstruction):
    if reconstruction is None:
        return 0
    return int(
        sum(
            len(list(point3d.track.elements))
            for point3d in reconstruction.points3D.values()
        )
    )


def summarize_reconstruction(reconstruction):
    if reconstruction is None:
        return {"points3D": 0, "observations": 0}
    return {
        "points3D": int(len(reconstruction.points3D)),
        "observations": count_reconstruction_observations(reconstruction),
    }


def summarize_ba_solver_result(summary):
    """Convert a backend-specific BA summary into JSON-compatible stats."""
    if summary is None or isinstance(summary, dict):
        return summary

    result = {"type": type(summary).__name__}
    scalar_attributes = (
        "initial_cost",
        "final_cost",
        "fixed_cost",
        "num_successful_steps",
        "num_unsuccessful_steps",
        "num_inner_iteration_steps",
        "preprocessor_time_in_seconds",
        "minimizer_time_in_seconds",
        "postprocessor_time_in_seconds",
        "total_time_in_seconds",
        "message",
    )
    for name in scalar_attributes:
        if not hasattr(summary, name):
            continue
        value = getattr(summary, name)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value

    if hasattr(summary, "termination_type"):
        result["termination_type"] = str(summary.termination_type)

    for output_name, method_name in (
        ("brief_report", "BriefReport"),
        ("full_report", "FullReport"),
    ):
        method = getattr(summary, method_name, None)
        if callable(method):
            result[output_name] = str(method())

    return result


def run_reprojection_filter_with_stats(
    reconstruction,
    error_type,
    error_threshold,
    negative_depth_observations=None,
    log_prefix="",
):
    if reconstruction is None:
        return {"enabled": False, "reason": "missing reconstruction"}

    from gluemap.math.reprojection_error import (  # noqa: PLC0415
        ReprojectionErrorType,
        filter_reconstruction_by_reprojection_error,
    )

    error_type_map = {
        "angular": ReprojectionErrorType.ANGULAR,
        "pixel": ReprojectionErrorType.PIXEL,
        "normalized": ReprojectionErrorType.NORMALIZED,
    }
    before = summarize_reconstruction(reconstruction)
    observations_removed, tracks_removed = filter_reconstruction_by_reprojection_error(
        reconstruction,
        error_type_map[error_type],
        error_threshold,
        negative_depth_observations=negative_depth_observations,
        log_prefix=log_prefix,
    )
    after = summarize_reconstruction(reconstruction)
    return {
        "enabled": True,
        "error_type": error_type,
        "error_threshold": float(error_threshold),
        "before": before,
        "after": after,
        "observations_removed": int(observations_removed),
        "tracks_removed": int(tracks_removed),
    }


def _camera_centers_from_reconstruction(reconstruction):
    centers = {}
    for image_id, image in reconstruction.images.items():
        pose = image.cam_from_world()
        rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(pose.translation, dtype=np.float64)
        centers[int(image_id)] = -(rotation.T @ translation)
    return centers


def _track_max_triangulation_angle_degrees(point_xyz, image_ids, camera_centers):
    point_xyz = np.asarray(point_xyz, dtype=np.float64)
    rays = []
    for image_id in image_ids:
        center = camera_centers.get(int(image_id))
        if center is None:
            continue
        ray = point_xyz - center
        norm = float(np.linalg.norm(ray))
        if not np.isfinite(norm) or norm <= 1e-12:
            continue
        rays.append(ray / norm)
    if len(rays) < 2:
        return 0.0

    rays = np.asarray(rays, dtype=np.float64)
    cosine = rays @ rays.T
    upper = cosine[np.triu_indices(len(rays), k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(upper.min(), -1.0, 1.0))))


def _current_track_angular_error_p90(elements, track_errors):
    error_by_observation = {
        (int(image_id), int(point2d_idx)): float(error)
        for image_id, point2d_idx, error in track_errors
    }
    errors = []
    for elem in elements:
        key = (int(elem.image_id), int(elem.point2D_idx))
        error = error_by_observation.get(key, float("inf"))
        errors.append(error)
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("inf")
    return float(np.percentile(values, 90))


def _summarize_track_records_by_source(track_records):
    result = {
        source: {"tracks": 0, "observations": 0}
        for source in ("s_only", "p_only", "mixed", "empty")
    }
    for record in track_records:
        bucket = result[record["source"]]
        bucket["tracks"] += 1
        bucket["observations"] += int(record["track_length"])
    return result


def _plan_bae_track_deletions(
    track_records,
    image_observations,
    max_observations,
    min_observations_per_image=BAE_MIN_OBSERVATIONS_PER_IMAGE,
):
    """Plan deterministic whole-track deletion without mutating reconstruction."""
    current_image_observations = {
        int(image_id): int(count) for image_id, count in image_observations.items()
    }
    image_floors = {
        image_id: min(count, int(min_observations_per_image))
        for image_id, count in current_image_observations.items()
    }
    current_observations = int(
        sum(int(record["track_length"]) for record in track_records)
    )
    deleted_records = []

    ordered_records = sorted(
        track_records,
        key=lambda record: (
            int(record["track_length"]),
            -float(record["angular_error_p90"]),
            float(record["max_triangulation_angle_deg"]),
            int(record["point3D_id"]),
        ),
    )
    for record in ordered_records:
        if current_observations <= max_observations:
            break

        removals_by_image = defaultdict(int)
        for image_id in record["image_ids"]:
            removals_by_image[int(image_id)] += 1
        if any(
            current_image_observations[image_id] - removal_count
            < image_floors[image_id]
            for image_id, removal_count in removals_by_image.items()
        ):
            continue

        deleted_records.append(record)
        current_observations -= int(record["track_length"])
        for image_id, removal_count in removals_by_image.items():
            current_image_observations[image_id] -= removal_count

    return {
        "reached_budget": current_observations <= max_observations,
        "remaining_observations": int(current_observations),
        "deleted_records": deleted_records,
        "remaining_image_observations": current_image_observations,
        "image_floors": image_floors,
    }


def prune_reconstruction_for_bae_observation_budget(
    reconstruction,
    features,
    max_observations,
    angular_errors_per_track=None,
    min_observations_per_image=BAE_MIN_OBSERVATIONS_PER_IMAGE,
):
    """Prune whole real tracks in place before BAE to meet an observation cap."""
    max_observations = int(max_observations)
    before = summarize_reconstruction(reconstruction)
    if max_observations <= 0:
        return {
            "enabled": False,
            "applied": False,
            "reason": "bae_max_observations_disabled",
            "max_observations": max_observations,
            "before": before,
            "after": before,
        }
    if before["observations"] <= max_observations:
        return {
            "enabled": True,
            "applied": False,
            "reason": "within_budget",
            "max_observations": max_observations,
            "min_observations_per_image": int(min_observations_per_image),
            "before": before,
            "after": before,
        }

    if angular_errors_per_track is None:
        from gluemap.math.reprojection_error import (  # noqa: PLC0415
            ReprojectionErrorType,
            compute_all_errors_from_reconstruction,
        )

        angular_errors_per_track = compute_all_errors_from_reconstruction(
            reconstruction,
            ReprojectionErrorType.ANGULAR,
            negative_depth_observations={},
        )

    camera_centers = _camera_centers_from_reconstruction(reconstruction)
    s_keypoint_count = build_s_keypoint_count(reconstruction, features)
    image_observations = {int(image_id): 0 for image_id in reconstruction.images}
    track_records = []
    for point3D_id, point3d in reconstruction.points3D.items():
        elements = list(point3d.track.elements)
        image_ids = tuple(int(elem.image_id) for elem in elements)
        for image_id in image_ids:
            image_observations[image_id] += 1
        track_records.append(
            {
                "point3D_id": int(point3D_id),
                "image_ids": image_ids,
                "track_length": int(len(elements)),
                "angular_error_p90": _current_track_angular_error_p90(
                    elements,
                    angular_errors_per_track.pop(point3D_id, []),
                ),
                "max_triangulation_angle_deg": (
                    _track_max_triangulation_angle_degrees(
                        point3d.xyz,
                        image_ids,
                        camera_centers,
                    )
                ),
                "source": classify_point3d_track_source(
                    point3d,
                    s_keypoint_count,
                ),
            }
        )

    plan = _plan_bae_track_deletions(
        track_records,
        image_observations,
        max_observations,
        min_observations_per_image=min_observations_per_image,
    )
    if not plan["reached_budget"]:
        message = (
            "BAE observation budget cannot be reached without violating the "
            f"per-image floor: requested={max_observations}, "
            f"minimum_reachable={plan['remaining_observations']}, "
            f"min_observations_per_image={min_observations_per_image}"
        )
        print(f"[MERG3R-REFINE] {message}", flush=True)
        raise RuntimeError(message)

    deleted_records = plan["deleted_records"]
    for record in deleted_records:
        reconstruction.delete_point3D(record["point3D_id"])

    after = summarize_reconstruction(reconstruction)
    if after["observations"] != plan["remaining_observations"]:
        raise RuntimeError(
            "BAE observation pruning produced an inconsistent count: "
            f"planned={plan['remaining_observations']}, "
            f"actual={after['observations']}"
        )

    finite_deleted_errors = [
        record["angular_error_p90"]
        for record in deleted_records
        if np.isfinite(record["angular_error_p90"])
    ]
    deleted_by_source = _summarize_track_records_by_source(deleted_records)
    before_by_source = _summarize_track_records_by_source(track_records)
    after_by_source = {
        source: {
            "tracks": before_by_source[source]["tracks"]
            - deleted_by_source[source]["tracks"],
            "observations": before_by_source[source]["observations"]
            - deleted_by_source[source]["observations"],
        }
        for source in before_by_source
    }
    return {
        "enabled": True,
        "applied": True,
        "max_observations": max_observations,
        "min_observations_per_image": int(min_observations_per_image),
        "before": before,
        "after": after,
        "removed_tracks": int(len(deleted_records)),
        "removed_observations": int(before["observations"] - after["observations"]),
        "before_by_source": before_by_source,
        "deleted_by_source": deleted_by_source,
        "after_by_source": after_by_source,
        "per_image_observations_before": summarize_distribution(
            list(image_observations.values())
        ),
        "per_image_observations_after": summarize_distribution(
            list(plan["remaining_image_observations"].values())
        ),
        "deleted_track_lengths": summarize_distribution(
            [record["track_length"] for record in deleted_records]
        ),
        "deleted_angular_error_p90": summarize_distribution(finite_deleted_errors),
        "deleted_nonfinite_angular_error_tracks": int(
            len(deleted_records) - len(finite_deleted_errors)
        ),
        "deleted_max_triangulation_angle_deg": summarize_distribution(
            [record["max_triangulation_angle_deg"] for record in deleted_records]
        ),
    }


def triangulate_from_seed_reconstruction(
    pycolmap,
    seed_reconstruction,
    database_path,
    output_dir,
    args,
):
    options = pycolmap.IncrementalPipelineOptions()
    options.triangulation.min_angle = args.tri_min_angle
    options.triangulation.merge_max_reproj_error = 15.0
    options.triangulation.complete_max_reproj_error = 15.0
    options.triangulation.ignore_two_view_tracks = False
    options.triangulation.create_max_angle_error = args.tri_create_max_angle_error
    options.ba_global_max_refinements = 0
    if output_dir.exists():
        shutil.rmtree(output_dir)
    with suppress_native_stdio():
        reconstruction = pycolmap.triangulate_points(
            deepcopy(seed_reconstruction),
            str(database_path),
            ".",
            str(output_dir),
            clear_points=True,
            refine_intrinsics=False,
            options=options,
        )
    return reconstruction


def _bae_huber_delta_for_iteration(args, outer_iter):
    final_delta = getattr(args, "final_bae_huber_delta", None)
    is_final_iteration = outer_iter == int(args.num_refinement_iterations) - 1
    if is_final_iteration and final_delta is not None:
        return float(final_delta)
    return float(getattr(args, "bae_huber_delta", 1.0))


def run_merg3r_augmented_refinement_loop(
    args,
    pycolmap,
    output_dir,
    image_names,
    image_size_hw,
    camera_model,
    extrinsic,
    global_intrinsics,
    intrinsics_mapping,
    virtual_predictions_dict,
    features,
    database_path,
):
    from gluemap.controllers.augmented_bundle_adjustment import (  # noqa: PLC0415
        IterativeBAOptions,
        build_negative_depth_observations,
        build_reconstruction_for_ba,
        build_seed_reconstruction_for_ba,
        initialize_world_points,
        iterative_bundle_adjustment,
    )
    from gluemap.controllers.global_refinement import (  # noqa: PLC0415
        select_virtual_tracks_from_merged,
    )
    from gluemap.estimators.track_establishment import (  # noqa: PLC0415
        TrackEstablishmentOptions,
        establish_tracks_from_predictions_dict,
    )
    from gluemap.utils.colmap import (  # noqa: PLC0415
        camera_from_intrinsics_matrix,
    )

    ba_backend = getattr(args, "ba_backend", "ceres")
    if ba_backend not in {"ceres", "bae"}:
        raise ValueError(
            f"Unknown BA backend {ba_backend!r}, expected 'ceres' or 'bae'"
        )
    use_virtual_tracks = ba_backend == "ceres"
    final_bae_huber_delta = getattr(args, "final_bae_huber_delta", None)
    if final_bae_huber_delta is not None and final_bae_huber_delta <= 0:
        raise ValueError("final_bae_huber_delta must be positive")
    if use_virtual_tracks and virtual_predictions_dict is None:
        raise ValueError(
            "track_mode=SPV requires virtual predictions, but they were not built"
        )

    num_images = len(image_names)
    image_shapes = [tuple(image_size_hw) for _ in range(num_images)]
    rotations, centers = global_pose_dicts_from_w2c(extrinsic)
    stats = {
        "enabled": True,
        "refinement_mode": "SPV" if use_virtual_tracks else "SP",
        "ba_backend": ba_backend,
        "num_refinement_iterations": int(args.num_refinement_iterations),
        "filter_reproj_error_threshold": float(
            args.filter_reproj_error_threshold
        ),
        "bae_huber_delta": float(getattr(args, "bae_huber_delta", 1.0)),
        "final_bae_huber_delta": (
            float(final_bae_huber_delta)
            if final_bae_huber_delta is not None
            else None
        ),
        "setup": {},
        "iterations": [],
    }

    if use_virtual_tracks:
        t0 = time.time()
        track_options = TrackEstablishmentOptions(track_min_num_views_per_track=2)
        (
            points3D,
            keypoints_per_image,
            pts2d_idx_inv,
            pts2d_idx_virtual_inv,
            images_points2d_virtual_isnegative,
        ) = establish_tracks_from_predictions_dict(
            predictions_dict=virtual_predictions_dict,
            num_images=num_images,
            options=track_options,
            add_tracks=False,
            add_virtual_points=True,
            device=args.device,
        )
        torch.cuda.empty_cache()
        stats["setup"]["establish_virtual_tracks_seconds"] = time.time() - t0
        stats["setup"]["established_virtual_tracks"] = int(len(points3D))

        height, width = image_size_hw
        cameras = [
            (
                camera_from_intrinsics_matrix(
                    intr[0],
                    camera_model,
                    width=width,
                    height=height,
                    camera_id=camera_id + 1,
                )
                if intr is not None
                else None
            )
            for camera_id, intr in enumerate(global_intrinsics)
        ]
        negative_depth_observations = build_negative_depth_observations(
            pts2d_idx_inv, images_points2d_virtual_isnegative
        )
        virtual_init_threshold = args.virtual_init_angular_error_threshold
        if virtual_init_threshold is None:
            virtual_init_threshold = (
                args.filter_reproj_error_threshold
                if args.filter_reproj_error_type == "angular"
                else 0.5
            )
        stats["setup"]["virtual_init_angular_error_threshold"] = float(
            virtual_init_threshold
        )

        t0 = time.time()
        points3D = initialize_world_points(
            virtual_predictions_dict,
            rotations,
            centers,
            points3D,
            pts2d_idx_inv,
            pts2d_idx_virtual_inv,
            keypoints_per_image=keypoints_per_image,
            cameras=cameras,
            intrinsics_mapping=intrinsics_mapping,
            angular_error_threshold_deg=virtual_init_threshold,
            negative_depth_observations=negative_depth_observations,
        )
        stats["setup"]["initialize_virtual_points_seconds"] = time.time() - t0
        stats["setup"]["initialized_virtual_tracks"] = int(len(points3D))

        t0 = time.time()
        virtual_reconstruction = build_reconstruction_for_ba(
            rotations,
            centers,
            global_intrinsics,
            intrinsics_mapping,
            points3D,
            keypoints_per_image,
            image_sizes=image_shapes,
            images_list=image_names,
            camera_model=camera_model,
        )
        stats["setup"]["build_virtual_reconstruction_seconds"] = time.time() - t0
        stats["setup"]["virtual_reconstruction"] = summarize_reconstruction(
            virtual_reconstruction
        )
        seed_reconstruction = virtual_reconstruction
    else:
        t0 = time.time()
        keypoints_per_image = load_database_keypoints_per_image(
            database_path, image_names
        )
        seed_reconstruction = build_seed_reconstruction_for_ba(
            rotations,
            centers,
            global_intrinsics,
            intrinsics_mapping,
            keypoints_per_image,
            image_sizes=image_shapes,
            images_list=image_names,
            camera_model=camera_model,
        )
        virtual_reconstruction = None
        negative_depth_observations = {}
        stats["setup"].update(
            {
                "virtual_tracks_enabled": False,
                "virtual_tracks_skip_reason": "BAE uses real tracks only",
                "seed_reconstruction_seconds": time.time() - t0,
                "seed_keypoints": int(
                    sum(len(points) for points in keypoints_per_image.values())
                ),
                "seed_reconstruction": summarize_reconstruction(seed_reconstruction),
            }
        )

    negative_depth_observations_1indexed = {
        image_id + 1: point2d_indices
        for image_id, point2d_indices in negative_depth_observations.items()
    }

    ba_options = IterativeBAOptions(
        max_ba_iterations=args.ba_max_num_iterations,
        max_filter_iterations=args.augmented_ba_max_filter_iterations,
        normalized_reproj_threshold=(args.augmented_ba_normalized_reproj_threshold),
        min_track_length=2,
        fix_rotations_first_pass=False,
        ba_backend=ba_backend,
        bae_device=getattr(args, "device", "cuda"),
        bae_max_iterations=getattr(args, "bae_max_num_iterations", None),
        bae_optimize_intrinsics=getattr(args, "bae_optimize_intrinsics", False),
        bae_fix_gauge=getattr(args, "bae_fix_gauge", "two_cams"),
        bae_robust_loss=getattr(args, "bae_robust_loss", "none"),
        bae_huber_delta=getattr(args, "bae_huber_delta", 1.0),
        # ceres keeps its virtual-driven re-BA loop; BAE filters through every
        # scaling once but never re-runs BA (the re-BA is near-useless once
        # huber has down-weighted the outliers).
        allow_re_ba_after_filter=(ba_backend == "ceres"),
    )

    reconstruction = None
    for outer_iter in range(args.num_refinement_iterations):
        is_final_round = outer_iter == args.num_refinement_iterations - 1
        bae_huber_delta = _bae_huber_delta_for_iteration(args, outer_iter)
        ba_options.bae_huber_delta = bae_huber_delta
        iter_stats = {
            "iteration": int(outer_iter + 1),
            "bae_huber_delta": bae_huber_delta,
        }
        t_iter = time.time()

        t0 = time.time()
        reconstruction = triangulate_from_seed_reconstruction(
            pycolmap,
            (virtual_reconstruction if use_virtual_tracks else seed_reconstruction),
            database_path,
            output_dir / f"triangulated_aug_iter_{outer_iter + 1}",
            args,
        )
        iter_stats["triangulation"] = {
            "seconds": time.time() - t0,
            **summarize_reconstruction(reconstruction),
        }

        pair_count = {}
        if args.enable_select_tracks:
            t0 = time.time()
            select_stats, pair_count = run_select_tracks(
                reconstruction,
                features,
                args.select_track_min_support,
                return_pair_count=True,
            )
            select_stats["seconds"] = time.time() - t0
            iter_stats["select_tracks"] = select_stats
        else:
            iter_stats["select_tracks"] = {
                "enabled": False,
                "reason": "disabled",
            }

        if use_virtual_tracks:
            t0 = time.time()
            before_virtual_select = summarize_reconstruction(virtual_reconstruction)
            pair_count = select_virtual_tracks_from_merged(
                virtual_reconstruction=virtual_reconstruction,
                pair_count=pair_count,
                min_num_support_abs=args.select_track_min_support,
            )
            after_virtual_select = summarize_reconstruction(virtual_reconstruction)
            iter_stats["select_virtual_tracks"] = {
                "enabled": True,
                "seconds": time.time() - t0,
                "before": before_virtual_select,
                "after": after_virtual_select,
                "removed_points3D": int(
                    before_virtual_select["points3D"] - after_virtual_select["points3D"]
                ),
                "pair_count_entries": int(len(pair_count)),
            }
        else:
            iter_stats["select_virtual_tracks"] = {
                "enabled": False,
                "reason": "BAE uses real tracks only",
            }

        angular_errors_per_track = None
        cache_angular_errors_for_bae_pruning = (
            ba_backend == "bae"
            and int(getattr(args, "bae_max_observations", 0)) > 0
            and count_reconstruction_observations(reconstruction)
            > int(getattr(args, "bae_max_observations", 0))
        )
        if args.filter_reproj_error_type == "angular":
            t0 = time.time()
            angular_result = summarize_angular_errors_by_track_source(
                reconstruction,
                features,
                args.filter_reproj_error_threshold,
                return_errors_per_track=cache_angular_errors_for_bae_pruning,
            )
            if cache_angular_errors_for_bae_pruning:
                angular_bucket_stats, angular_errors_per_track = angular_result
            else:
                angular_bucket_stats = angular_result
            angular_bucket_stats["seconds"] = time.time() - t0
            iter_stats["angular_errors_by_track_source"] = angular_bucket_stats
            log_angular_errors_by_track_source(
                args,
                outer_iter + 1,
                angular_bucket_stats,
            )
        else:
            iter_stats["angular_errors_by_track_source"] = {
                "enabled": False,
                "reason": ("only computed when filter_reproj_error_type=angular"),
            }

        if args.enable_reprojection_filter:
            t0 = time.time()
            real_filter_stats = run_reprojection_filter_with_stats(
                reconstruction,
                args.filter_reproj_error_type,
                args.filter_reproj_error_threshold,
                log_prefix="real: ",
            )
            if use_virtual_tracks:
                virtual_filter_stats = run_reprojection_filter_with_stats(
                    virtual_reconstruction,
                    args.filter_reproj_error_type,
                    args.filter_reproj_error_threshold,
                    negative_depth_observations=(negative_depth_observations_1indexed),
                    log_prefix="virtual: ",
                )
            else:
                virtual_filter_stats = {
                    "enabled": False,
                    "reason": "BAE uses real tracks only",
                }
            iter_stats["reprojection_filter"] = {
                "enabled": True,
                "seconds": time.time() - t0,
                "real": real_filter_stats,
                "virtual": virtual_filter_stats,
            }
        else:
            iter_stats["reprojection_filter"] = {"enabled": False}

        if ba_backend == "bae":
            t0 = time.time()
            bae_observation_pruning = prune_reconstruction_for_bae_observation_budget(
                reconstruction,
                features,
                getattr(args, "bae_max_observations", 0),
                angular_errors_per_track=angular_errors_per_track,
            )
            bae_observation_pruning["seconds"] = time.time() - t0
            iter_stats["bae_observation_pruning"] = bae_observation_pruning
            if bae_observation_pruning["applied"]:
                debug(
                    args,
                    "BAE observation pruning: "
                    f"observations={bae_observation_pruning['before']['observations']} "
                    f"-> {bae_observation_pruning['after']['observations']}, "
                    f"tracks_removed={bae_observation_pruning['removed_tracks']}, "
                    f"limit={bae_observation_pruning['max_observations']}, "
                    f"time={bae_observation_pruning['seconds']:.2f}s",
                )
        else:
            iter_stats["bae_observation_pruning"] = {
                "enabled": False,
                "applied": False,
                "reason": "Ceres backend does not use the BAE observation budget",
            }
        angular_errors_per_track = None

        # The post-BA filter only reaches the output in the final round; in
        # intermediate BAE rounds the next triangulation re-creates points with
        # clear_points=True, so its pruning is wasted. ceres rounds always
        # filter (their virtual-driven loop and per-round cleanup rely on it).
        ba_options.run_post_ba_filter = use_virtual_tracks or is_final_round

        t0 = time.time()
        before_ba = {
            "real": summarize_reconstruction(reconstruction),
            "virtual": summarize_reconstruction(virtual_reconstruction),
        }
        reconstruction, virtual_reconstruction = iterative_bundle_adjustment(
            reconstruction,
            virtual_reconstruction,
            negative_depth_observations_1indexed,
            options=ba_options,
        )
        if not use_virtual_tracks:
            # The optimized real poses/intrinsics seed the next triangulation.
            # Its points3D are discarded by clear_points=True.
            seed_reconstruction = reconstruction
        iter_stats["bundle_adjustment"] = {
            "seconds": time.time() - t0,
            "post_ba_filter_ran": ba_options.run_post_ba_filter,
            "before": before_ba,
            "after": {
                "real": summarize_reconstruction(reconstruction),
                "virtual": summarize_reconstruction(virtual_reconstruction),
            },
            "backend": ba_options.ba_backend,
            "summary": summarize_ba_solver_result(ba_options.last_ba_summary),
        }
        iter_stats["seconds"] = time.time() - t_iter
        stats["iterations"].append(iter_stats)

    if reconstruction is None:
        final_real_by_source = {
            "total": 0,
            "s_only": 0,
            "p_only": 0,
            "mixed": 0,
        }
        final_angular_errors = {
            "enabled": False,
            "reason": "missing reconstruction",
        }
    else:
        s_keypoint_count = build_s_keypoint_count(reconstruction, features)
        source_counts = classify_tracks_by_s_keypoints(
            reconstruction,
            s_keypoint_count,
        )
        final_real_by_source = {
            "total": int(source_counts["total"]),
            "s_only": int(source_counts["s"]),
            "p_only": int(source_counts["non_s"]),
            "mixed": int(source_counts["mixed"]),
        }
        if args.filter_reproj_error_type == "angular":
            t0 = time.time()
            final_angular_errors = summarize_angular_errors_by_track_source(
                reconstruction,
                features,
                args.filter_reproj_error_threshold,
            )
            final_angular_errors["seconds"] = time.time() - t0
        else:
            final_angular_errors = {
                "enabled": False,
                "reason": "only computed when filter_reproj_error_type=angular",
            }
    log_angular_errors_by_track_source(args, "final", final_angular_errors)

    stats["final"] = {
        "real": summarize_reconstruction(reconstruction),
        "real_by_source": final_real_by_source,
        "angular_errors_by_track_source": final_angular_errors,
        "virtual": summarize_reconstruction(virtual_reconstruction),
    }
    return reconstruction, virtual_reconstruction, stats
