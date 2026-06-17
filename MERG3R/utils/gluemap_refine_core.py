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


def _lazy_import_pycolmap():
    try:
        import pycolmap  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pycolmap is required. Run this script in the Gluemap environment."
        ) from exc
    return pycolmap


def _ensure_gluemap_imports():
    repo_root = Path(__file__).resolve().parents[1] / "gluemap"
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
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
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
    groups, group_stats = build_vggsfm_groups(
        args,
        pairs,
        num_images,
        image_names,
        image_size_hw,
        centers=centers,
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


def build_pose_groups(pairs, num_images, neighbors_per_center, centers=None):
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


def build_vggsfm_groups(
    args,
    pairs,
    num_images,
    image_names,
    image_size_hw,
    centers=None,
):
    groups = build_pose_groups(
        pairs,
        num_images,
        args.neighbors_per_center,
        centers=centers,
    )
    stats = {
        "strategy": "pose",
        "input_pairs": int(np.asarray(pairs).reshape(-1, 2).shape[0]),
        "max_neighbors": int(args.neighbors_per_center),
        **summarize_groups(groups, num_images),
    }
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

    fmaps_chunks = []
    num_images = int(tracker_images.shape[0])
    for start in range(0, num_images, chunk_size):
        end = min(start + chunk_size, num_images)
        images_chunk = tracker_images[start:end].to(args.device, non_blocking=True)
        fmaps_chunk = tracker.process_images_to_fmaps(images_chunk)
        fmaps_chunks.append(fmaps_chunk.detach().cpu())
        del images_chunk, fmaps_chunk
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    tracker_fmaps = torch.cat(fmaps_chunks, dim=0)
    return tracker_fmaps, {
        "enabled": True,
        "chunk_size": int(chunk_size),
        "seconds": time.time() - t0,
        "shape": [int(v) for v in tracker_fmaps.shape],
    }


@torch.no_grad()
def run_vggsfm_prior_tracks(
    args, images, features, pairs, metadata, extrinsic, image_names
):
    if not args.path_tracker:
        raise ValueError("--path_tracker is required")

    _ensure_gluemap_imports()
    from vggsfm.vggsfm_tracker import TrackerPredictor  # noqa: PLC0415

    tracker = TrackerPredictor().eval().to(args.device)
    tracker.load_state_dict(
        torch.load(args.path_tracker, map_location="cpu", weights_only=False)
    )

    centers = camera_centers_from_w2c(extrinsic)
    groups, group_stats = build_vggsfm_groups(
        args,
        pairs,
        images.shape[0],
        image_names,
        metadata["image_size_hw"],
        centers=centers,
    )
    tracks = []
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

    t_group = time.time()
    for group in groups:
        center = group[0]
        query_np = sample_query_points(
            query_points_per_image[center], args.vggsfm_query_points
        )
        if query_np.shape[0] == 0:
            continue
        group_tensor = tracker_images[group].unsqueeze(0)
        if args.vggsfm_fine_tracking:
            group_tensor = group_tensor.to(args.device)
        group_fmaps = tracker_fmaps[group].unsqueeze(0).to(args.device)
        query = (
            torch.from_numpy(query_np).to(args.device, dtype=torch.float32).unsqueeze(0)
        )
        pred_track, _, pred_vis, pred_score = tracker(
            group_tensor,
            query,
            fmaps=group_fmaps,
            fine_tracking=args.vggsfm_fine_tracking,
        )
        del group_fmaps
        pred_track = pred_track[0].detach().cpu().numpy()
        pred_vis = pred_vis[0].detach().cpu().numpy()
        pred_score = pred_score[0].detach().cpu().numpy()

        for point_idx in range(query_np.shape[0]):
            h, w = metadata["image_size_hw"]
            center_xy = query_np[point_idx].astype(np.float32)
            if tracker_image_changes is not None:
                center_xy = invert_image_change(
                    center_xy, tracker_image_changes[center]
                )
            if not (0 <= center_xy[0] < w and 0 <= center_xy[1] < h):
                continue
            obs = [(center, center_xy.astype(np.float32))]
            for local_idx, image_idx in enumerate(group[1:], start=1):
                if pred_vis[local_idx, point_idx] < args.vggsfm_vis_threshold:
                    continue
                if pred_score[local_idx, point_idx] < args.vggsfm_score_threshold:
                    continue
                xy = pred_track[local_idx, point_idx].astype(np.float32)
                if tracker_image_changes is not None:
                    xy = invert_image_change(xy, tracker_image_changes[int(image_idx)])
                if not (0 <= xy[0] < w and 0 <= xy[1] < h):
                    continue
                obs.append((int(image_idx), xy))
            if len(obs) >= 2:
                observations += len(obs)
                tracks.append(obs)
    group_tracking_time = time.time() - t_group

    return tracks, {
        "num_groups": len(groups),
        "num_tracks": len(tracks),
        "num_observations": observations,
        "neighbors_per_center": args.neighbors_per_center,
        "group_strategy": args.group_strategy,
        "group_stats": group_stats,
        "query_points": args.vggsfm_query_points,
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
        bucket["max"] = float(values.max())
        bucket["lt_threshold_count"] = int(np.sum(values < error_threshold))
        bucket["lt_threshold_ratio"] = float(bucket["lt_threshold_count"] / values.size)

    return {
        "enabled": True,
        "error_type": "angular",
        "error_threshold": float(error_threshold),
        "buckets": buckets,
    }


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
                f"max={bucket['max']:.4f}, "
                f"<{threshold:g}deg="
                f"{bucket['lt_threshold_ratio'] * 100:.1f}%"
            )
        else:
            error_summary = "mean=n/a, median=n/a, max=n/a, " f"<{threshold:g}deg=n/a"

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

    if virtual_predictions_dict is None:
        raise ValueError(
            "track_mode=SPV requires virtual predictions, but they were not " "built"
        )

    num_images = len(image_names)
    image_shapes = [tuple(image_size_hw) for _ in range(num_images)]
    rotations, centers = global_pose_dicts_from_w2c(extrinsic)
    stats = {
        "enabled": True,
        "num_refinement_iterations": int(args.num_refinement_iterations),
        "setup": {},
        "iterations": [],
    }

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
        ba_backend=getattr(args, "ba_backend", "ceres"),
        bae_device=getattr(args, "device", "cuda"),
        bae_max_iterations=getattr(args, "bae_max_num_iterations", None),
        bae_optimize_intrinsics=getattr(
            args, "bae_optimize_intrinsics", False
        ),
        bae_real_only=getattr(args, "bae_real_only", False),
        bae_fix_gauge=getattr(args, "bae_fix_gauge", "two_cams"),
    )

    reconstruction = None
    for outer_iter in range(args.num_refinement_iterations):
        iter_stats = {"iteration": int(outer_iter + 1)}
        t_iter = time.time()

        t0 = time.time()
        reconstruction = triangulate_from_seed_reconstruction(
            pycolmap,
            virtual_reconstruction,
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

        t0 = time.time()
        before_virtual_select = summarize_reconstruction(virtual_reconstruction)
        if virtual_reconstruction is not None:
            pair_count = select_virtual_tracks_from_merged(
                virtual_reconstruction=virtual_reconstruction,
                pair_count=pair_count,
                min_num_support_abs=args.select_track_min_support,
            )
        after_virtual_select = summarize_reconstruction(virtual_reconstruction)
        iter_stats["select_virtual_tracks"] = {
            "enabled": virtual_reconstruction is not None,
            "seconds": time.time() - t0,
            "before": before_virtual_select,
            "after": after_virtual_select,
            "removed_points3D": int(
                before_virtual_select["points3D"] - after_virtual_select["points3D"]
            ),
            "pair_count_entries": int(len(pair_count)),
        }

        if args.filter_reproj_error_type == "angular":
            t0 = time.time()
            angular_bucket_stats = summarize_angular_errors_by_track_source(
                reconstruction,
                features,
                args.filter_reproj_error_threshold,
            )
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
            virtual_filter_stats = run_reprojection_filter_with_stats(
                virtual_reconstruction,
                args.filter_reproj_error_type,
                args.filter_reproj_error_threshold,
                negative_depth_observations=negative_depth_observations_1indexed,
                log_prefix="virtual: ",
            )
            iter_stats["reprojection_filter"] = {
                "enabled": True,
                "seconds": time.time() - t0,
                "real": real_filter_stats,
                "virtual": virtual_filter_stats,
            }
        else:
            iter_stats["reprojection_filter"] = {"enabled": False}

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
        iter_stats["bundle_adjustment"] = {
            "seconds": time.time() - t0,
            "before": before_ba,
            "after": {
                "real": summarize_reconstruction(reconstruction),
                "virtual": summarize_reconstruction(virtual_reconstruction),
            },
            "backend": ba_options.ba_backend,
            "summary": ba_options.last_ba_summary,
        }
        iter_stats["seconds"] = time.time() - t_iter
        stats["iterations"].append(iter_stats)

    stats["final"] = {
        "real": summarize_reconstruction(reconstruction),
        "virtual": summarize_reconstruction(virtual_reconstruction),
    }
    return reconstruction, virtual_reconstruction, stats
