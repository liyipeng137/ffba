"""reconstruction / tracks for the formal SIFT + prior + BAE pipeline."""

import shutil
from copy import deepcopy


def run_select_tracks(
    reconstruction,
    features,
    min_num_support_abs,
    return_pair_count=False,
):
    from ffba.reporting.tracks import (
        build_s_keypoint_count,
        classify_tracks_by_s_keypoints,
    )
    from gluemap.controllers.global_refinement import (  # noqa: PLC0415
        select_tracks_from_merged,
    )

    s_keypoint_count = build_s_keypoint_count(reconstruction, features)
    before = classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count)
    # Zero protected feature counts put every source through the existing
    # seeded-shuffle/pair-support selector. Keep the real counts for auditing.
    pair_count = select_tracks_from_merged(
        reconstruction=reconstruction,
        sift_count={},
        min_num_support_abs=min_num_support_abs,
    )
    after = classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count)
    stats = {
        "enabled": True,
        "selection_scope": "all_sources",
        "min_num_support_abs": int(min_num_support_abs),
        "before": before,
        "after": after,
        "removed_points3D": int(before["total"] - after["total"]),
        "pair_count_entries": int(len(pair_count)),
    }
    if return_pair_count:
        return stats, pair_count
    return stats


def run_reprojection_filter_with_stats(
    reconstruction,
    error_type,
    error_threshold,
    negative_depth_observations=None,
    log_prefix="",
):
    from ffba.reporting.tracks import summarize_reconstruction

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
    from ffba.runtime import suppress_native_stdio

    options = pycolmap.IncrementalPipelineOptions()
    options.triangulation.min_angle = args.tri_min_angle
    options.triangulation.merge_max_reproj_error = 15.0
    options.triangulation.complete_max_reproj_error = 15.0
    options.triangulation.ignore_two_view_tracks = True
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
