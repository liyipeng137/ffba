import contextlib
import logging
import os
import shutil
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pycolmap
import pygluemap
import torch

from gluemap.controllers.augmented_bundle_adjustment import (
    IterativeBAOptions,
    build_negative_depth_observations,
    build_reconstruction_for_ba,
    initialize_world_points,
    iterative_bundle_adjustment,
)
from gluemap.estimators.track_establishment import (
    TrackEstablishmentOptions,
    establish_tracks_from_predictions_dict,
)
from gluemap.math.reprojection_error import (
    ReprojectionErrorType,
    filter_reconstruction_by_reprojection_error,
)
from gluemap.utils.colmap import (
    camera_from_intrinsics_matrix,
    merge_colmap_databases,
    prepare_glomap_prior,
)

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_native_stdio():
    """Redirect fd 1 and fd 2 to /dev/null for the duration of the block.

    Needed because pycolmap.triangulate_points writes via C++ std::cout
    and glog directly to file descriptors, so contextlib.redirect_stdout
    is not enough.
    """
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


def _extract_track_csr(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract track data from a reconstruction as CSR numpy arrays."""
    point3d_ids = []
    track_img_ids = []
    track_pt2d_idxs = []
    track_lengths = []
    for p3d_id, p3d in reconstruction.points3D.items():
        elems = list(p3d.track.elements)
        point3d_ids.append(p3d_id)
        track_lengths.append(len(elems))
        for e in elems:
            track_img_ids.append(e.image_id)
            track_pt2d_idxs.append(e.point2D_idx)
    return (
        np.array(point3d_ids, dtype=np.int64),
        np.array(track_img_ids, dtype=np.int64),
        np.array(track_pt2d_idxs, dtype=np.int64),
        np.array(track_lengths, dtype=np.int32),
    )


def _apply_deletions(
    reconstruction: pycolmap.Reconstruction,
    ids_to_delete: np.ndarray,
) -> None:
    """Delete point3D entries returned by a C++ track selector."""
    for p3d_id in ids_to_delete:
        reconstruction.delete_point3D(int(p3d_id))


def _summarize_values(values: list[int] | list[float]) -> dict[str, float]:
    """Small numeric summary used by refinement debug logging."""
    if len(values) == 0:
        return {
            "count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _format_summary(summary: dict[str, float]) -> str:
    return (
        f"min={summary['min']:.1f}, median={summary['median']:.1f}, "
        f"mean={summary['mean']:.1f}, p90={summary['p90']:.1f}, "
        f"max={summary['max']:.1f}"
    )


def _summarize_point_tracks(
    points3D: dict[int, pycolmap.Point3D],
    num_images: int | None = None,
    image_ids: list[int] | None = None,
) -> dict:
    track_lengths = []
    image_observations = defaultdict(int)

    for point3D in points3D.values():
        elements = list(point3D.track.elements)
        track_lengths.append(len(elements))
        for elem in elements:
            image_observations[int(elem.image_id)] += 1

    if image_ids is not None:
        per_image_counts = [
            int(image_observations.get(image_id, 0)) for image_id in image_ids
        ]
        zero_images = sum(1 for count in per_image_counts if count == 0)
    elif num_images is None:
        per_image_counts = list(image_observations.values())
        zero_images = 0
    else:
        per_image_counts = [
            int(image_observations.get(image_id, 0))
            for image_id in range(num_images)
        ]
        zero_images = sum(1 for count in per_image_counts if count == 0)

    return {
        "points": int(len(points3D)),
        "observations": int(sum(track_lengths)),
        "track_length": _summarize_values(track_lengths),
        "image_observations": _summarize_values(per_image_counts),
        "images_with_observations": int(len(image_observations)),
        "zero_observation_images": int(zero_images),
    }


def _log_point_tracks_debug(
    label: str,
    points3D: dict[int, pycolmap.Point3D],
    num_images: int | None = None,
) -> None:
    stats = _summarize_point_tracks(points3D, num_images)
    logger.info(
        "[RefineDebug] %s: points=%d, observations=%d, "
        "track_len=(%s), obs/image=(%s), images_with_obs=%d, "
        "zero_obs_images=%d",
        label,
        stats["points"],
        stats["observations"],
        _format_summary(stats["track_length"]),
        _format_summary(stats["image_observations"]),
        stats["images_with_observations"],
        stats["zero_observation_images"],
    )


def _log_reconstruction_debug(
    label: str,
    reconstruction: pycolmap.Reconstruction | None,
) -> None:
    if reconstruction is None:
        logger.info("[RefineDebug] %s: reconstruction=None", label)
        return

    stats = _summarize_point_tracks(
        reconstruction.points3D,
        image_ids=[int(image_id) for image_id in reconstruction.images],
    )
    logger.info(
        "[RefineDebug] %s: images=%d, cameras=%d, points=%d, "
        "observations=%d, track_len=(%s), obs/image=(%s), "
        "images_with_obs=%d, zero_obs_images=%d",
        label,
        len(reconstruction.images),
        len(reconstruction.cameras),
        stats["points"],
        stats["observations"],
        _format_summary(stats["track_length"]),
        _format_summary(stats["image_observations"]),
        stats["images_with_observations"],
        stats["zero_observation_images"],
    )


def _log_keypoint_debug(
    label: str,
    keypoints_per_image: dict[int, np.ndarray],
    num_images: int,
) -> None:
    counts = [
        len(keypoints_per_image[image_id])
        if image_id in keypoints_per_image
        else 0
        for image_id in range(num_images)
    ]
    logger.info(
        "[RefineDebug] %s: total_keypoints=%d, images_with_keypoints=%d, "
        "zero_keypoint_images=%d, keypoints/image=(%s)",
        label,
        int(sum(counts)),
        int(sum(1 for count in counts if count > 0)),
        int(sum(1 for count in counts if count == 0)),
        _format_summary(_summarize_values(counts)),
    )


def _log_negative_depth_debug(
    label: str,
    negative_depth_observations: dict[int, set] | dict[int, list],
) -> None:
    counts = [len(v) for v in negative_depth_observations.values()]
    logger.info(
        "[RefineDebug] %s: images=%d, observations=%d, obs/image=(%s)",
        label,
        len(negative_depth_observations),
        int(sum(counts)),
        _format_summary(_summarize_values(counts)),
    )


def _log_database_debug(label: str, db_path: str) -> None:
    if not os.path.exists(db_path):
        logger.info("[RefineDebug] %s: database missing at %s", label, db_path)
        return

    try:
        database = pycolmap.Database.open(db_path)
        images = database.read_all_images()
        keypoint_counts = []
        for image in images:
            keypoints = database.read_keypoints(image.image_id)
            keypoint_counts.append(0 if keypoints is None else len(keypoints))

        pair_ids, matches_list = database.read_all_matches()
        match_counts = [len(matches) for matches in matches_list]
        logger.info(
            "[RefineDebug] %s: images=%d, total_keypoints=%d, "
            "zero_keypoint_images=%d, keypoints/image=(%s), "
            "match_pairs=%d, total_matches=%d, matches/pair=(%s)",
            label,
            len(images),
            int(sum(keypoint_counts)),
            int(sum(1 for count in keypoint_counts if count == 0)),
            _format_summary(_summarize_values(keypoint_counts)),
            len(pair_ids),
            int(sum(match_counts)),
            _format_summary(_summarize_values(match_counts)),
        )
    except Exception as exc:
        logger.warning(
            "[RefineDebug] %s: failed to inspect database %s: %s",
            label,
            db_path,
            exc,
        )


def _log_predictions_debug(
    label: str,
    predictions_dict: dict,
) -> None:
    num_groups = len(predictions_dict.get("indexes", []))
    group_sizes = [
        len(indexes) for indexes in predictions_dict.get("indexes", [])
    ]
    logger.info(
        "[RefineDebug] %s: groups=%d, group_size=(%s)",
        label,
        num_groups,
        _format_summary(_summarize_values(group_sizes)),
    )

    if "tracks" in predictions_dict and "scores" in predictions_dict:
        real_points = []
        real_observations = []
        for idx in range(num_groups):
            tracks = predictions_dict["tracks"][idx]
            scores = predictions_dict["scores"][idx]
            real_points.append(int(tracks.shape[-2]))
            real_observations.append(int((scores > 0).sum().item()))
        logger.info(
            "[RefineDebug] %s real predictions: points/group=(%s), "
            "valid_obs/group=(%s), total_valid_obs=%d",
            label,
            _format_summary(_summarize_values(real_points)),
            _format_summary(_summarize_values(real_observations)),
            int(sum(real_observations)),
        )

    if (
        "tracks_virtual" in predictions_dict
        and "valid_virtual" in predictions_dict
    ):
        virtual_points = []
        virtual_observations = []
        for idx in range(num_groups):
            tracks_virtual = predictions_dict["tracks_virtual"][idx]
            valid_virtual = predictions_dict["valid_virtual"][idx]
            virtual_points.append(int(tracks_virtual.shape[-2]))
            virtual_observations.append(int(valid_virtual.sum().item()))
        logger.info(
            "[RefineDebug] %s virtual predictions: points/group=(%s), "
            "valid_obs/group=(%s), total_valid_obs=%d",
            label,
            _format_summary(_summarize_values(virtual_points)),
            _format_summary(_summarize_values(virtual_observations)),
            int(sum(virtual_observations)),
        )

    if "pose_inconsistent" in predictions_dict:
        pose_inconsistent = predictions_dict["pose_inconsistent"]
        masks = (
            pose_inconsistent.values()
            if hasattr(pose_inconsistent, "values")
            else pose_inconsistent
        )
        inconsistent_counts = [
            int(mask.sum().item()) for mask in masks
        ]
        logger.info(
            "[RefineDebug] %s pose_inconsistent: total=%d, per_group=(%s)",
            label,
            int(sum(inconsistent_counts)),
            _format_summary(_summarize_values(inconsistent_counts)),
        )


def select_tracks_from_merged(
    reconstruction: pycolmap.Reconstruction,
    sift_count: dict[int, int],
    min_num_support_abs: int = 512,
) -> dict[int, int]:
    """
    Selectively prune non-SIFT tracks from a merged reconstruction.
    Returns the image-pair coverage map {canonical_pair_key: count}.
    The canonical key encodes (img_low, img_high) as (img_low << 32) | img_high.
    """
    point3d_ids, track_img_ids, track_pt2d_idxs, track_lengths = (
        _extract_track_csr(reconstruction)
    )
    sc = {int(k): int(v) for k, v in sift_count.items()}
    points_before = len(reconstruction.points3D)
    observations_before = int(track_lengths.sum()) if len(track_lengths) else 0

    ids_to_delete, pair_count = pygluemap.compute_tracks_to_delete(
        point3d_ids,
        track_img_ids,
        track_pt2d_idxs,
        track_lengths,
        sc,
        min_num_support_abs,
    )
    _apply_deletions(reconstruction, ids_to_delete)
    observations_after = sum(
        len(list(point.track.elements))
        for point in reconstruction.points3D.values()
    )
    logger.info(
        "[RefineDebug] SelectTrack: points=%d -> %d, removed=%d, "
        "observations=%d -> %d, pair_count_entries=%d, "
        "pair_count=(%s)",
        points_before,
        len(reconstruction.points3D),
        len(ids_to_delete),
        observations_before,
        observations_after,
        len(pair_count),
        _format_summary(_summarize_values(list(pair_count.values()))),
    )
    return pair_count


def select_virtual_tracks_from_merged(
    virtual_reconstruction: pycolmap.Reconstruction,
    pair_count: dict[int, int],
    min_num_support_abs: int = 512,
) -> dict[int, int]:
    """
    Selectively prune virtual tracks using existing pair coverage.
    Removes tracks whose image pairs are all already sufficiently covered.
    Returns the updated pair_count.
    """
    if len(virtual_reconstruction.points3D) == 0:
        return pair_count

    point3d_ids, track_img_ids, track_pt2d_idxs, track_lengths = (
        _extract_track_csr(virtual_reconstruction)
    )
    points_before = len(virtual_reconstruction.points3D)
    observations_before = int(track_lengths.sum()) if len(track_lengths) else 0

    ids_to_delete, updated_pair_count = (
        pygluemap.compute_virtual_tracks_to_delete(
            point3d_ids,
            track_img_ids,
            track_pt2d_idxs,
            track_lengths,
            pair_count,
            min_num_support_abs,
        )
    )
    _apply_deletions(virtual_reconstruction, ids_to_delete)
    observations_after = sum(
        len(list(point.track.elements))
        for point in virtual_reconstruction.points3D.values()
    )
    logger.info(
        "[RefineDebug] SelectVirtualTrack: points=%d -> %d, removed=%d, "
        "observations=%d -> %d, pair_count_entries=%d, "
        "pair_count=(%s)",
        points_before,
        len(virtual_reconstruction.points3D),
        len(ids_to_delete),
        observations_before,
        observations_after,
        len(updated_pair_count),
        _format_summary(_summarize_values(list(updated_pair_count.values()))),
    )
    return updated_pair_count


def triangulate_with_pycolmap(
    reconstruction: pycolmap.Reconstruction,
    database_path: str,
    triangulated_output_path: str,
    options: pycolmap.IncrementalPipelineOptions,
) -> pycolmap.Reconstruction:
    """Run pycolmap.triangulate_points silently on a deep copy.

    The reconstruction is already 1-indexed (matching the COLMAP database
    written by prepare_glomap_prior), so no reindexing is required. The input
    reconstruction is deep-copied first so the returned reconstruction does
    not overwrite the caller's. Native stdout/stderr is suppressed because
    triangulate_points is verbose.
    """
    reconstruction = deepcopy(reconstruction)
    with _suppress_native_stdio():
        reconstruction = pycolmap.triangulate_points(
            reconstruction,
            database_path,
            ".",  # skip color extraction
            triangulated_output_path,
            clear_points=True,
            refine_intrinsics=False,
            options=options,
        )
    logger.info(
        f"pycolmap.triangulate_points produced "
        f"{len(reconstruction.points3D)} 3D points"
    )
    return reconstruction


def run_refinement_pipeline(
    args,
    predictions_dict: dict,
    global_rotations,
    global_centers,
    global_intrinsics,
    dataset_pair,
    num_images: int,
    use_triangulation_first: bool = True,
    angular_error_threshold_deg: float = 0.5,
    num_refinement_iterations: int = 2,
    track_mode: str = "SPV",
) -> pycolmap.Reconstruction:
    """
    Run the refinement pipeline.

    Triangulation, track establishment, and bundle adjustment.

    Args:
        args: Argument namespace (needs curr_path, images_path)
        predictions_dict: Predictions from star inference
        global_rotations: Global rotation matrices for all images
        global_centers: Global camera center positions
        global_intrinsics: Camera intrinsic parameters for all images
        dataset_pair: Dataset pair object (needs camera_model,
            intrinsics_mapping, images_shape_ori, images_list)
        num_images: Number of images in the dataset
        use_triangulation_first: If True, triangulate SIFT + real tracks
            first, then add only virtual points. If False (default),
            triangulate SIFT only and establish both real tracks and virtual
            points.
        track_mode: Combination of S(IFT), P(rior), V(irtual) tracks to use.
            Valid modes: "SPV", "SP", "SV", "PV", "S", "P".

    Returns:
        pycolmap.Reconstruction: The bundle-adjusted reconstruction
    """
    t_refinement_start = time.perf_counter()
    refinement_timing = {}

    # Indexing convention: all data stored in COLMAP format (the database
    # written by prepare_glomap_prior, the pycolmap.Reconstruction returned
    # by build_reconstruction_for_ba, and anything keyed against either) is
    # 1-indexed for image_id and camera_id. The upstream Python-side
    # structures (global_rotations, keypoints_per_image, points3D track
    # elements, negative_depth_observations, etc.) remain 0-indexed and are
    # shifted at the COLMAP boundary.

    # Parse track mode flags
    use_sift = "S" in track_mode
    use_prior = "P" in track_mode
    use_virtual = "V" in track_mode
    logger.info(
        f"Track mode: {track_mode} (SIFT={use_sift}, "
        f"Prior={use_prior}, Virtual={use_virtual})"
    )
    _log_predictions_debug("refinement input predictions", predictions_dict)

    # Step 1: Triangulate 3D points
    logger.info("Triangulating points with pycolmap...")
    t0 = time.perf_counter()
    suffix = getattr(args, "output_suffix", "")
    coarse_dir = f"coarse{suffix}"
    coarse_reconstruction = pycolmap.Reconstruction()
    coarse_reconstruction.read(args.curr_path + "/" + coarse_dir)
    refinement_timing["load_coarse"] = time.perf_counter() - t0
    _log_reconstruction_debug(
        "loaded coarse reconstruction", coarse_reconstruction
    )

    # Step 1b: Determine parameters based on track mode
    if use_prior:
        database_name = "database_tracks.db"
        add_tracks = True
        log_message = "Creating tracks database (with prior tracks)..."
    else:
        database_name = "database_empty.db"
        add_tracks = False
        log_message = "Creating tracks database (empty, no prior tracks)..."

    # Step 1c: Create database with tracks (or empty)
    logger.info(log_message)
    t0 = time.perf_counter()
    prepare_glomap_prior(
        args.curr_path,
        dataset_pair.images_shape_ori,
        dataset_pair.images_list,
        global_intrinsics,
        predictions_dict,
        dataset_pair.intrinsics_mapping,
        camera_model=dataset_pair.camera_model,
        add_tracks=add_tracks,
        add_virtual_points=False,
        database_name=database_name,
    )
    refinement_timing["prepare_prior"] = time.perf_counter() - t0
    _log_database_debug(
        f"prepared prior database ({database_name})",
        args.curr_path + "/" + database_name,
    )

    # Step 1c.5: Read SIFT DB keypoint counts (= sift_count per image)
    t0 = time.perf_counter()
    if use_sift:
        sift_db = pycolmap.Database.open(args.curr_path + "/database_sift.db")
        sift_count_by_name = {}
        for img in sift_db.read_all_images():
            kp = sift_db.read_keypoints(img.image_id)
            sift_count_by_name[img.name] = (
                len(kp) if kp is not None and len(kp) > 0 else 0
            )
    else:
        sift_count_by_name = {}
    refinement_timing["read_sift"] = time.perf_counter() - t0
    if use_sift:
        sift_counts = list(sift_count_by_name.values())
        logger.info(
            "[RefineDebug] SIFT database keypoints: images=%d, "
            "total_keypoints=%d, zero_keypoint_images=%d, "
            "keypoints/image=(%s)",
            len(sift_counts),
            int(sum(sift_counts)),
            int(sum(1 for count in sift_counts if count == 0)),
            _format_summary(_summarize_values(sift_counts)),
        )
        _log_database_debug(
            "SIFT database", args.curr_path + "/database_sift.db"
        )

    # Step 1d: Merge SIFT database with the created database (or copy if
    # no SIFT)
    t0 = time.perf_counter()
    merged_db_path = args.curr_path + "/database_merged.db"
    if use_sift:
        logger.info("Merging SIFT and tracks databases...")
        merge_colmap_databases(
            db_path_primary=args.curr_path + "/" + database_name,
            db_path_secondary=args.curr_path + "/database_sift.db",
            output_path=merged_db_path,
            # SIFT features should be at the front for correct indexing
            primary_features_first=False,
        )
    else:
        logger.info("Copying tracks database (no SIFT merge)...")
        shutil.copy2(args.curr_path + "/" + database_name, merged_db_path)
    refinement_timing["merge_databases"] = time.perf_counter() - t0
    _log_database_debug("merged database", merged_db_path)

    # Step 2: Establish tracks from predictions
    t0 = time.perf_counter()
    track_options = TrackEstablishmentOptions(track_min_num_views_per_track=2)

    add_virtual_points_flag = use_virtual
    if use_triangulation_first and use_prior:
        # Real tracks already in DB for triangulation; only establish
        # virtual points
        add_tracks_flag = False
    elif use_prior:
        # Establish real tracks into reconstruction directly
        add_tracks_flag = True
    else:
        # No prior tracks requested
        add_tracks_flag = False

    (
        points3D,
        keypoints_per_image,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
        images_points2d_virtual_isnegative,
    ) = establish_tracks_from_predictions_dict(
        predictions_dict=predictions_dict,
        num_images=num_images,
        options=track_options,
        add_tracks=add_tracks_flag,
        add_virtual_points=add_virtual_points_flag,
        device="cuda",
    )
    torch.cuda.empty_cache()
    refinement_timing["establish_tracks"] = time.perf_counter() - t0
    _log_keypoint_debug(
        "TrackEstablishment keypoints", keypoints_per_image, num_images
    )
    _log_point_tracks_debug(
        "TrackEstablishment points before initialization", points3D, num_images
    )

    # Step 3: Initialize 3D world points
    t0 = time.perf_counter()
    cameras = [
        (
            camera_from_intrinsics_matrix(intr[0], dataset_pair.camera_model)
            if intr is not None
            else None
        )
        for intr in global_intrinsics
    ]
    negative_depth_observations = build_negative_depth_observations(
        pts2d_idx_inv, images_points2d_virtual_isnegative
    )
    _log_negative_depth_debug(
        "negative-depth virtual observations before reconstruction build",
        negative_depth_observations,
    )
    points3D = initialize_world_points(
        predictions_dict,
        global_rotations,
        global_centers,
        points3D,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
        keypoints_per_image=keypoints_per_image,
        cameras=cameras,
        intrinsics_mapping=dataset_pair.intrinsics_mapping,
        angular_error_threshold_deg=angular_error_threshold_deg,
        negative_depth_observations=negative_depth_observations,
    )
    refinement_timing["initialize_points"] = time.perf_counter() - t0
    _log_point_tracks_debug(
        "TrackEstablishment points after initialization", points3D, num_images
    )

    # Step 4: Configure bundle adjustment
    ba_options = IterativeBAOptions(
        max_ba_iterations=200,
        max_filter_iterations=3,
        normalized_reproj_threshold=1e-2,
        min_track_length=2,
        fix_rotations_first_pass=False,
    )

    # Step 5: Build reconstruction from current data
    t0 = time.perf_counter()
    virtual_reconstruction = build_reconstruction_for_ba(
        global_rotations,
        global_centers,
        global_intrinsics,
        dataset_pair.intrinsics_mapping,
        points3D,
        keypoints_per_image,
        image_sizes=dataset_pair.images_shape_ori,
        images_list=dataset_pair.images_list,
        camera_model=dataset_pair.camera_model,
    )
    refinement_timing["build_reconstruction"] = time.perf_counter() - t0
    _log_reconstruction_debug(
        "virtual reconstruction after build", virtual_reconstruction
    )

    # build_reconstruction_for_ba emits 1-indexed image_ids, so consumers that
    # join against the reconstruction (BA, reprojection-error filter) need a
    # 1-indexed view of negative_depth_observations.
    negative_depth_observations_1indexed = {
        image_id + 1: pt_set
        for image_id, pt_set in negative_depth_observations.items()
    }

    database_path = args.curr_path + "/database_merged.db"
    triangulated_output_path = args.curr_path + "/coarse_triangulated"

    iteration_timings = []
    for outer_iter in range(num_refinement_iterations):
        logger.info(f"{'=' * 60}")
        logger.info(
            f"Refinement iteration {outer_iter + 1}/{num_refinement_iterations}"
        )
        logger.info(f"{'=' * 60}")
        t_iter_start = time.perf_counter()
        _log_reconstruction_debug(
            f"iter {outer_iter + 1} virtual seed before triangulation",
            virtual_reconstruction,
        )

        # Step 1e: Triangulate on merged database
        t_tri_start = time.perf_counter()
        opt_triang = pycolmap.IncrementalPipelineOptions()
        opt_triang.triangulation.min_angle = 1.0
        opt_triang.triangulation.merge_max_reproj_error = 15.0
        opt_triang.triangulation.complete_max_reproj_error = 15.0
        opt_triang.triangulation.ignore_two_view_tracks = False
        opt_triang.triangulation.create_max_angle_error = (
            angular_error_threshold_deg
        )
        opt_triang.ba_global_max_refinements = 0

        reconstruction = triangulate_with_pycolmap(
            virtual_reconstruction,
            database_path,
            triangulated_output_path,
            options=opt_triang,
        )
        t_tri_end = time.perf_counter()
        _log_reconstruction_debug(
            f"iter {outer_iter + 1} real after triangulation",
            reconstruction,
        )

        # Step 7a: Selectively prune prior/virtual tracks (SelectTrack logic)
        sift_count = {}
        for recon_id, img in reconstruction.images.items():
            sift_count[recon_id] = sift_count_by_name.get(img.name, 0)

        pair_count = select_tracks_from_merged(
            reconstruction=reconstruction,
            sift_count=sift_count,
            min_num_support_abs=512,
        )
        _log_reconstruction_debug(
            f"iter {outer_iter + 1} real after SelectTrack",
            reconstruction,
        )

        # Step 7a.2: Prune virtual tracks using pair coverage from real
        # selection
        if virtual_reconstruction is not None:
            select_virtual_tracks_from_merged(
                virtual_reconstruction=virtual_reconstruction,
                pair_count=pair_count,
                min_num_support_abs=512,
            )
            _log_reconstruction_debug(
                f"iter {outer_iter + 1} virtual after SelectVirtualTrack",
                virtual_reconstruction,
            )

        # Step 7.5: Filter tracks before bundle adjustment
        t_filter_start = time.perf_counter()
        if angular_error_threshold_deg > 0:
            _log_reconstruction_debug(
                f"iter {outer_iter + 1} real before angular filter",
                reconstruction,
            )
            _log_reconstruction_debug(
                f"iter {outer_iter + 1} virtual before angular filter",
                virtual_reconstruction,
            )
            for recon, neg_depth, label in (
                (reconstruction, None, "real: "),
                (
                    virtual_reconstruction,
                    negative_depth_observations_1indexed,
                    "virtual: ",
                ),
            ):
                if recon is None:
                    continue
                filter_reconstruction_by_reprojection_error(
                    recon,
                    ReprojectionErrorType.ANGULAR,
                    angular_error_threshold_deg,
                    negative_depth_observations=neg_depth,
                    log_prefix=label,
                )
            _log_reconstruction_debug(
                f"iter {outer_iter + 1} real after angular filter",
                reconstruction,
            )
            _log_reconstruction_debug(
                f"iter {outer_iter + 1} virtual after angular filter",
                virtual_reconstruction,
            )

        t_filter_end = time.perf_counter()

        # Step 7c: Limit number of tracks before BA
        max_num_tracks = getattr(args, "max_num_tracks", None)
        if (
            max_num_tracks is not None
            and len(reconstruction.points3D) > max_num_tracks
        ):
            sorted_ids = sorted(
                reconstruction.points3D.keys(),
                key=lambda pid: len(
                    list(reconstruction.points3D[pid].track.elements)
                ),
                reverse=True,
            )
            ids_to_remove = sorted_ids[max_num_tracks:]
            for pid in ids_to_remove:
                reconstruction.delete_point3D(pid)
            logger.info(
                f"  Track limit: kept {max_num_tracks}, "
                f"removed {len(ids_to_remove)} tracks"
            )
            _log_reconstruction_debug(
                f"iter {outer_iter + 1} real after max_num_tracks",
                reconstruction,
            )

        # Step 8: Run iterative bundle adjustment
        t_ba_start = time.perf_counter()
        _log_reconstruction_debug(
            f"iter {outer_iter + 1} real before augmented BA",
            reconstruction,
        )
        _log_reconstruction_debug(
            f"iter {outer_iter + 1} virtual before augmented BA",
            virtual_reconstruction,
        )
        reconstruction, virtual_reconstruction = iterative_bundle_adjustment(
            reconstruction,
            virtual_reconstruction,
            negative_depth_observations_1indexed,
            options=ba_options,
        )
        t_ba_end = time.perf_counter()
        _log_reconstruction_debug(
            f"iter {outer_iter + 1} real after augmented BA",
            reconstruction,
        )
        _log_reconstruction_debug(
            f"iter {outer_iter + 1} virtual after augmented BA",
            virtual_reconstruction,
        )

        iter_timing = {
            "triangulation": t_tri_end - t_tri_start,
            "filter": t_filter_end - t_filter_start,
            "ba": t_ba_end - t_ba_start,
            "total": t_ba_end - t_iter_start,
        }
        iteration_timings.append(iter_timing)
        logger.info(
            f"[Profiling] Iteration {outer_iter + 1}: "
            f"triangulation={iter_timing['triangulation']:.2f}s, "
            f"filter={iter_timing['filter']:.2f}s, "
            f"ba={iter_timing['ba']:.2f}s, total={iter_timing['total']:.2f}s"
        )

    # Clean up triangulated reconstruction output
    if os.path.exists(triangulated_output_path):
        shutil.rmtree(triangulated_output_path)

    # Step 9: Write bundle adjusted results to COLMAP format
    t0 = time.perf_counter()
    suffix = getattr(args, "output_suffix", "")
    file_dir = f"gluemap_aba{suffix}"
    logger.info(
        "Writing bundle adjusted reconstruction: %s",
        args.curr_path + "/" + file_dir,
    )
    os.makedirs(args.curr_path + "/" + file_dir, exist_ok=True)
    _log_reconstruction_debug(
        "final real reconstruction before write", reconstruction
    )
    _log_reconstruction_debug(
        "final virtual reconstruction before write", virtual_reconstruction
    )
    reconstruction.write(args.curr_path + "/" + file_dir)
    refinement_timing["write_output"] = time.perf_counter() - t0

    refinement_timing["iterations"] = iteration_timings
    refinement_timing["total"] = time.perf_counter() - t_refinement_start

    logger.info("[Profiling] Refinement Summary:")
    logger.info(
        f"  Setup: load_coarse={refinement_timing['load_coarse']:.2f}s, "
        f"prepare_prior={refinement_timing['prepare_prior']:.2f}s, "
        f"merge_db={refinement_timing['merge_databases']:.2f}s, "
        f"establish_tracks={refinement_timing['establish_tracks']:.2f}s, "
        f"init_points={refinement_timing['initialize_points']:.2f}s, "
        f"build_recon={refinement_timing['build_reconstruction']:.2f}s"
    )
    logger.info(
        f"  Iterations: {sum(it['total'] for it in iteration_timings):.2f}s "
        f"({len(iteration_timings)} iters)"
    )

    logger.info(f"  Total refinement: {refinement_timing['total']:.2f}s")

    return file_dir, refinement_timing
