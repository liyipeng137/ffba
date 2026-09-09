"""reconstruction / refinement for the formal SIFT + prior + BAE pipeline."""

import time


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
    from ffba.geometry import global_pose_dicts_from_w2c
    from ffba.reconstruction.budget import (
        prune_reconstruction_for_bae_observation_budget,
    )
    from ffba.reconstruction.database import load_database_keypoints_per_image
    from ffba.reconstruction.tracks import (
        run_reprojection_filter_with_stats,
        run_select_tracks,
        triangulate_from_seed_reconstruction,
    )
    from ffba.reporting.tracks import (
        build_s_keypoint_count,
        classify_tracks_by_s_keypoints,
        count_reconstruction_observations,
        log_angular_errors_by_track_source,
        summarize_angular_errors_by_track_source,
        summarize_ba_solver_result,
        summarize_reconstruction,
    )
    from ffba.runtime import debug
    from ffba.reconstruction.bae import (
        IterativeBAOptions,
        build_seed_reconstruction_for_ba,
        iterative_bundle_adjustment,
    )

    final_bae_huber_delta = getattr(args, "final_bae_huber_delta", None)
    if final_bae_huber_delta is not None and final_bae_huber_delta <= 0:
        raise ValueError("final_bae_huber_delta must be positive")
    num_images = len(image_names)
    image_shapes = [tuple(image_size_hw) for _ in range(num_images)]
    rotations, centers = global_pose_dicts_from_w2c(extrinsic)
    stats = {
        "enabled": True,
        "refinement_mode": "SP",
        "ba_backend": "bae",
        "num_refinement_iterations": int(args.num_refinement_iterations),
        "filter_reproj_error_threshold": float(args.filter_reproj_error_threshold),
        "bae_huber_delta": float(getattr(args, "bae_huber_delta", 1.0)),
        "final_bae_huber_delta": float(final_bae_huber_delta)
        if final_bae_huber_delta is not None
        else None,
        "setup": {},
        "iterations": [],
    }
    t0 = time.time()
    keypoints_per_image = load_database_keypoints_per_image(database_path, image_names)
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
                sum((len(points) for points in keypoints_per_image.values()))
            ),
            "seed_reconstruction": summarize_reconstruction(seed_reconstruction),
        }
    )
    negative_depth_observations_1indexed = {
        image_id + 1: point2d_indices
        for image_id, point2d_indices in negative_depth_observations.items()
    }
    ba_options = IterativeBAOptions(
        max_filter_iterations=args.augmented_ba_max_filter_iterations,
        normalized_reproj_threshold=args.augmented_ba_normalized_reproj_threshold,
        min_track_length=2,
        bae_device=getattr(args, "device", "cuda"),
        bae_max_iterations=getattr(args, "bae_max_num_iterations", None),
        bae_optimize_intrinsics=getattr(args, "bae_optimize_intrinsics", False),
        bae_fix_gauge=getattr(args, "bae_fix_gauge", "two_cams"),
        bae_robust_loss=getattr(args, "bae_robust_loss", "none"),
        bae_huber_delta=getattr(args, "bae_huber_delta", 1.0),
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
            seed_reconstruction,
            database_path,
            output_dir / f"triangulated_aug_iter_{outer_iter + 1}",
            args,
        )
        iter_stats["triangulation"] = {
            "ignore_two_view_tracks": True,
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
            iter_stats["select_tracks"] = {"enabled": False, "reason": "disabled"}
        iter_stats["select_virtual_tracks"] = {
            "enabled": False,
            "reason": "BAE uses real tracks only",
        }
        angular_errors_per_track = None
        cache_angular_errors_for_bae_pruning = int(
            getattr(args, "bae_max_observations", 0)
        ) > 0 and count_reconstruction_observations(reconstruction) > int(
            getattr(args, "bae_max_observations", 0)
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
                args, outer_iter + 1, angular_bucket_stats
            )
        else:
            iter_stats["angular_errors_by_track_source"] = {
                "enabled": False,
                "reason": "only computed when filter_reproj_error_type=angular",
            }
        if args.enable_reprojection_filter:
            t0 = time.time()
            real_filter_stats = run_reprojection_filter_with_stats(
                reconstruction,
                args.filter_reproj_error_type,
                args.filter_reproj_error_threshold,
                log_prefix="real: ",
            )
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
                f"BAE observation pruning: observations={bae_observation_pruning['before']['observations']} -> {bae_observation_pruning['after']['observations']}, tracks_removed={bae_observation_pruning['removed_tracks']}, limit={bae_observation_pruning['max_observations']}, time={bae_observation_pruning['seconds']:.2f}s",
            )
        angular_errors_per_track = None
        ba_options.run_post_ba_filter = is_final_round
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
        seed_reconstruction = reconstruction
        iter_stats["bundle_adjustment"] = {
            "seconds": time.time() - t0,
            "post_ba_filter_ran": ba_options.run_post_ba_filter,
            "before": before_ba,
            "after": {
                "real": summarize_reconstruction(reconstruction),
                "virtual": summarize_reconstruction(virtual_reconstruction),
            },
            "backend": "bae",
            "summary": summarize_ba_solver_result(ba_options.last_ba_summary),
        }
        iter_stats["seconds"] = time.time() - t_iter
        stats["iterations"].append(iter_stats)
    if reconstruction is None:
        final_real_by_source = {"total": 0, "s_only": 0, "p_only": 0, "mixed": 0}
        final_angular_errors = {"enabled": False, "reason": "missing reconstruction"}
    else:
        s_keypoint_count = build_s_keypoint_count(reconstruction, features)
        source_counts = classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count)
        final_real_by_source = {
            "total": int(source_counts["total"]),
            "s_only": int(source_counts["s"]),
            "p_only": int(source_counts["non_s"]),
            "mixed": int(source_counts["mixed"]),
        }
        if args.filter_reproj_error_type == "angular":
            t0 = time.time()
            final_angular_errors = summarize_angular_errors_by_track_source(
                reconstruction, features, args.filter_reproj_error_threshold
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
    return (reconstruction, virtual_reconstruction, stats)
