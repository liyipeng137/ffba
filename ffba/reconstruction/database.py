"""reconstruction / database for the formal SIFT + prior + BAE pipeline."""

import contextlib
import os
from pathlib import Path
import numpy as np
import torch


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


def load_database_keypoint_features(database_path, image_names):
    from ffba.runtime import _lazy_import_pycolmap

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
    from ffba.runtime import _lazy_import_pycolmap

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
    from ffba.runtime import _lazy_import_pycolmap

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
    from ffba.runtime import _lazy_import_pycolmap

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
    from ffba.matching.observations import (
        snap_prior_tracks_to_features,
        tracks_to_keypoints_and_matches,
    )
    from ffba.runtime import _lazy_import_pycolmap

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
    from ffba.geometry import global_pose_dicts_from_w2c
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
