from ffba.refinement_config import GluemapSpvRefineConfig as GluemapSpvRefineConfig
from ffba.refinement_config import GluemapSpvRefineResult, _make_refine_args
import gc
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ffba.reporting.depth import export_prediction_depth_maps
from ffba import api as ref
from ffba.matching.scheduling import build_scheduling_order, resolve_group_strategy

CAMERA_MODEL = "SIMPLE_PINHOLE"
S_DATABASE_MODE = "sift"
QUERY_SOURCE = "aliked"
GROUP_STRATEGY = "projected_overlap"
TRACK_MODE = "SP"
TRACKER_INPUT = "1024"


def _debug(args, message):
    if args.debug_print:
        print(f"[PIPELINE-REFINE] {message}", flush=True)


def _cuda_memory_snapshot(device):
    allocated_bytes = int(torch.cuda.memory_allocated(device))
    reserved_bytes = int(torch.cuda.memory_reserved(device))
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": allocated_bytes,
        "reserved_bytes": reserved_bytes,
        "driver_free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _format_cuda_memory_snapshot(snapshot):
    gib = 1024**3
    return (
        f"allocated={snapshot['allocated_bytes'] / gib:.2f} GiB, "
        f"reserved={snapshot['reserved_bytes'] / gib:.2f} GiB, "
        f"driver_free={snapshot['driver_free_bytes'] / gib:.2f}/"
        f"{snapshot['total_bytes'] / gib:.2f} GiB"
    )


def _save_one_work_image(item):
    idx, image, images_dir = item
    name = f"frame_{idx:06d}.png"
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(images_dir / name)
    return name


def _save_work_images(images, output_dir, num_workers=16):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images_cpu = images.detach().cpu().float().clamp(0, 1)

    num_workers = int(num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1")
    if int(images_cpu.shape[0]) == 0:
        raise ValueError("Cannot save work images from an empty tensor")
    worker_count = min(num_workers, int(images_cpu.shape[0]))

    work_items = [(idx, image, images_dir) for idx, image in enumerate(images_cpu)]
    if worker_count == 1:
        image_names = [_save_one_work_image(item) for item in work_items]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            image_names = list(executor.map(_save_one_work_image, work_items))

    print(
        "[PIPELINE-REFINE] Saved work images: "
        f"images={len(image_names)}, workers={worker_count}, images_dir={images_dir}"
    )
    return images_dir, image_names


def _write_json(path, payload):
    with open(path, "w") as f:
        # default=str keeps the stats dump from crashing after expensive compute
        # if a value (e.g. a Path or numpy scalar) is not JSON-serializable.
        json.dump(payload, f, indent=2, default=str)


def run_gluemap_spv_refinement(coarse_state, output_dir, config):
    ref._ensure_gluemap_imports()
    pycolmap = ref._lazy_import_pycolmap()
    args = _make_refine_args(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    t0 = time.time()
    images_dir, image_names = _save_work_images(
        coarse_state.high_images, output_dir, num_workers=config.work_image_workers
    )
    save_work_images_seconds = time.time() - t0
    image_size_hw = tuple(coarse_state.high_image_size_hw)
    depth_image_size_hw = (
        tuple(coarse_state.low_image_size_hw)
        if coarse_state.raw_depth is not None
        else None
    )
    initial_intrinsics_high_all = np.asarray(
        coarse_state.intrinsic_high, dtype=np.float64
    )
    initial_intrinsics_low_all = np.asarray(
        coarse_state.intrinsic_low, dtype=np.float64
    )
    extrinsic = np.asarray(coarse_state.extrinsic, dtype=np.float64)
    legacy_pose_pairs = ref.canonicalize_pair_array(
        coarse_state.pairs, num_images=len(image_names)
    )
    is_loma = args.prior_provider == "loma"
    input_order = args.input_order
    if input_order == "unordered":
        args.sift_temporal_window = 0
    scheduling_order = build_scheduling_order(
        coarse_state.retrieval_sim_matrix, input_order
    )
    group_resolution = None
    if not is_loma:
        group_resolution = resolve_group_strategy(
            args.group_strategy, coarse_state.raw_depth is not None
        )
        args.group_strategy = group_resolution["effective_strategy"]
        _debug(args, f"VGGSfM groups: {group_resolution}")

    schedule_mode = "sift_guided" if is_loma else "sift_first_sparse"
    if args.sift_temporal_window < 0:
        raise ValueError("sift_temporal_window must be >= 0")
    if not is_loma and args.vggsfm_max_center_gap <= 0:
        raise ValueError("vggsfm_max_center_gap must be >= 1")
    sift_candidate_pairs, sift_pair_graph_stats = ref.build_sift_candidate_pairs(
        legacy_pose_pairs, len(image_names), args.sift_temporal_window
    )
    temporal_pairs = ref.build_temporal_pairs(
        len(image_names), args.sift_temporal_window
    )
    pairs = sift_candidate_pairs
    metadata = {"image_size_hw": image_size_hw}
    stats = {
        "initial_geometry_source": getattr(
            coarse_state, "initial_geometry_source", "feedforward"
        ),
        "has_initial_depth": coarse_state.raw_depth is not None,
        "track_mode": args.track_mode,
        "camera_model": CAMERA_MODEL,
        "s_database_mode": S_DATABASE_MODE,
        "prior_provider": args.prior_provider,
        "vggsfm_query_source": None if is_loma else QUERY_SOURCE,
        "vggsfm_tracker_input": None if is_loma else TRACKER_INPUT,
        "group_strategy": "pose_union_dino_union_temporal"
        if is_loma
        else args.group_strategy,
        "vggsfm_schedule_mode": None if is_loma else schedule_mode,
        "pair_graphs": {
            **sift_pair_graph_stats,
            "active_sift_pair_graph": "legacy_pose_pairs_union_temporal",
            "active_sift_pair_count": int(pairs.shape[0]),
        },
        "ba_backend": config.ba_backend,
        "bae_optimize_intrinsics": config.bae_optimize_intrinsics,
        "bae_fix_gauge": config.bae_fix_gauge,
        "bae_robust_loss": config.bae_robust_loss,
        "bae_huber_delta": config.bae_huber_delta,
        "final_bae_huber_delta": config.final_bae_huber_delta,
        "timing": {"save_work_images": save_work_images_seconds},
        "work_images": {
            "images_dir": str(images_dir),
            "image_names": image_names,
            "num_workers": int(config.work_image_workers),
            "source_low_image_names": list(coarse_state.low_image_names),
            "source_high_image_names": list(coarse_state.high_image_names),
            "low_image_size_hw": list(depth_image_size_hw)
            if depth_image_size_hw is not None
            else None,
            "high_image_size_hw": list(image_size_hw),
            "image_pyramid": coarse_state.image_pyramid,
        },
    }
    _debug(
        args,
        f"Loaded coarse state: images={len(image_names)}, pairs={pairs.shape[0]}, high_image_size_hw={image_size_hw}, low_image_size_hw={depth_image_size_hw}, camera_model={CAMERA_MODEL}",
    )
    prefilter_image_names = list(image_names)
    prefilter_intrinsics_mapping = {idx: 0 for idx in range(len(image_names))}
    features = None
    sift_prefilter_stats = None
    sift_schedule_stats = None
    center_selection = {
        "selected_centers": list(range(len(image_names))),
        "owner": list(range(len(image_names))),
        "frames": [
            {
                "image_index": idx,
                "owner": idx,
                "selected": True,
                "reason": "full_center_mode",
            }
            for idx in range(len(image_names))
        ],
        "num_selected": len(image_names),
        "num_skipped": 0,
    }
    _debug(
        args,
        f"Preparing SIFT database before {('LoMa' if is_loma else 'VGGSfM')}: mode={schedule_mode}, legacy_pairs={legacy_pose_pairs.shape[0]}, sift_pairs={pairs.shape[0]}, temporal_new={sift_pair_graph_stats['temporal_new_pair_count']}",
    )
    t0 = time.time()
    features, sift_prefilter_stats = ref.prepare_sift_database_for_refine(
        args,
        output_dir,
        output_dir,
        image_names,
        pairs,
        CAMERA_MODEL,
        prefilter_intrinsics_mapping,
    )
    stats["timing"]["prepare_sift_database_prefilter"] = time.time() - t0
    t0 = time.time()
    sift_schedule_stats = ref.analyze_sift_schedule_graph(
        output_dir / "database_sift.db",
        image_names,
        pairs,
        image_size_hw,
        legacy_pose_pairs=legacy_pose_pairs,
        temporal_pairs=temporal_pairs,
        grid_size=args.sift_schedule_grid_size,
        min_inliers_per_cell=args.sift_schedule_min_inliers_per_cell,
        min_pair_inliers=args.sift_schedule_min_pair_inliers,
        min_grid_coverage=args.sift_schedule_min_grid_coverage,
    )
    stats["timing"]["sift_schedule_analysis"] = time.time() - t0
    if not is_loma:
        center_selection = ref.select_sift_first_centers(
            len(image_names),
            sift_schedule_stats["valid_edges"],
            args.vggsfm_max_center_gap,
            order=scheduling_order,
        )
    stats["s_database"] = {
        "mode": S_DATABASE_MODE,
        "prefilter": sift_prefilter_stats,
        "prefilter_database_ready": True,
    }
    stats["sift_schedule"] = {
        key: value for key, value in sift_schedule_stats.items() if key != "pairs"
    }
    _debug(
        args,
        f"SIFT schedule ready: verified={sift_schedule_stats['verified_pair_count']}, valid={sift_schedule_stats['valid_schedule_pair_count']}"
        + (
            " (LoMa support annotation only)"
            if is_loma
            else f", centers={center_selection['num_selected']}/{len(image_names)}"
        ),
    )
    stats["input_order"] = input_order
    stats["scheduling_order"] = scheduling_order
    stats["scheduling_order_source"] = (
        "input" if input_order == "ordered" else "dino_greedy"
    )
    stats["group_resolution"] = group_resolution
    loma_result = None
    if is_loma:
        from ffba.matching.loma import (
            annotate_loma_pairs_with_sift,
            build_loma_candidate_pairs,
            run_loma_prior,
            select_loma_pairs,
        )

        t0 = time.time()
        loma_records, graph_stats = build_loma_candidate_pairs(
            legacy_pose_pairs,
            temporal_pairs,
            coarse_state.retrieval_sim_matrix,
            dino_topk=args.loma_dino_candidates,
        )
        loma_records = annotate_loma_pairs_with_sift(
            loma_records, pairs, sift_schedule_stats
        )
        loma_records, selection_stats = select_loma_pairs(
            loma_records,
            coarse_state.retrieval_sim_matrix,
            mode=args.loma_pair_selection,
            sufficient_neighbors=args.loma_sufficient_neighbors,
            insufficient_neighbors=args.loma_insufficient_neighbors,
            untried_neighbors=args.loma_untried_neighbors,
            temporal_window=args.sift_temporal_window,
        )
        stats["pair_graphs"]["loma"] = graph_stats
        stats["pair_graphs"]["loma_selection"] = selection_stats
        stats["timing"]["loma_pair_build"] = time.time() - t0
        _write_json(
            output_dir / "prior_loma_pairs.json",
            {
                "image_names": image_names,
                "graph": graph_stats,
                "selection": selection_stats,
                "sift_support_config": sift_schedule_stats["config"],
                "pairs": loma_records,
            },
        )
        selected_records = [record for record in loma_records if record["selected"]]
        _debug(
            args,
            f"Running LoMa-B prior: selected={len(selected_records)}/{len(loma_records)} pairs, selection={args.loma_pair_selection}, no track/observation cap",
        )
        loma_result = run_loma_prior(
            [images_dir / name for name in image_names],
            image_size_hw,
            initial_intrinsics_high_all,
            selected_records,
            device=args.device,
            match_batch_size=args.loma_match_batch_size,
            extract_batch_size=args.loma_extract_batch_size,
            preprocess_workers=args.loma_preprocess_workers,
            geometry_workers=args.loma_geometry_workers,
            feature_cache=args.loma_feature_cache,
        )
        stats["loma"] = loma_result.stats
        stats["loma"]["pair_selection"] = selection_stats
        stats["timing"]["loma_prior_matches"] = loma_result.stats["timing"]["total"]
        _write_json(output_dir / "prior_loma_stats.json", loma_result.stats)
        executed_records = {
            tuple(record["pair"]): record for record in loma_result.pair_records
        }
        _write_json(
            output_dir / "prior_loma_pairs.json",
            {
                "image_names": image_names,
                "graph": graph_stats,
                "selection": selection_stats,
                "sift_support_config": sift_schedule_stats["config"],
                "pairs": [
                    executed_records.get(tuple(record["pair"]), record)
                    for record in loma_records
                ],
            },
        )
        prior_tracks = []
        _debug(
            args,
            f"LoMa prior done: verified_pairs={len(loma_result.geometries)}, unique_observations={loma_result.stats['unique_matched_observations']}",
        )
    else:
        _debug(
            args,
            f"Running VGGSfM prior tracking: schedule_mode={schedule_mode}, group_strategy={args.group_strategy}, group_batch_size={args.vggsfm_group_batch_size}, neighbors_per_center={args.neighbors_per_center}, query_points={args.vggsfm_query_points}, query_source={QUERY_SOURCE}, tracker_input={TRACKER_INPUT}",
        )
        tracking_groups = None
        tracking_group_stats = None
        if args.group_strategy == "projected_overlap":
            if coarse_state.raw_depth is None:
                raise ValueError(
                    "vggsfm_group_strategy='projected_overlap' requires depth"
                )
            if coarse_state.retrieval_sim_matrix is None:
                raise ValueError(
                    "retrieval_sim_matrix is required when vggsfm_group_strategy='projected_overlap'"
                )
            t0 = time.time()
            projected_groups, projected_group_stats, projected_candidate_details = (
                ref.build_projected_overlap_groups(
                    pairs=legacy_pose_pairs,
                    extrinsic=extrinsic,
                    intrinsics=initial_intrinsics_low_all,
                    depth=coarse_state.raw_depth,
                    depth_conf=coarse_state.raw_depth_conf,
                    retrieval_sim_matrix=coarse_state.retrieval_sim_matrix,
                    max_neighbors=int(args.neighbors_per_center),
                    rotation_threshold=float(args.pair_pose_rotation_threshold),
                    dino_candidates=int(config.projected_overlap_dino_candidates),
                    max_samples=int(config.projected_overlap_samples),
                    reprojection_threshold=float(
                        config.projected_overlap_reproj_threshold
                    ),
                    confidence_quantile=float(config.projected_overlap_conf_quantile),
                    selected_centers=center_selection["selected_centers"]
                    if not is_loma
                    else None,
                )
            )
            if not is_loma:
                tracking_groups, tracking_group_stats = (
                    ref.build_three_layer_vggsfm_groups(
                        center_selection["selected_centers"],
                        center_selection["owner"],
                        sift_schedule_stats["valid_edges"],
                        projected_candidate_details,
                        len(image_names),
                        int(args.neighbors_per_center),
                    )
                )
                tracking_group_stats["projected_overlap_base"] = projected_group_stats
            else:
                tracking_groups = projected_groups
                tracking_group_stats = projected_group_stats
            stats["timing"]["vggsfm_group_build"] = time.time() - t0
            _debug(
                args,
                f"Built projected-overlap VGGSfM groups: groups={len(tracking_groups)}, group_size={tracking_group_stats['group_size']}",
            )
        elif args.group_strategy == "sift_pose_dino":
            if coarse_state.retrieval_sim_matrix is None:
                raise ValueError(
                    "retrieval_sim_matrix is required when vggsfm_group_strategy='sift_pose_dino'"
                )
            t0 = time.time()
            sift_pose_dino_details, sift_pose_dino_stats = (
                ref.build_sift_pose_dino_candidate_details(
                    center_selection["selected_centers"],
                    sift_schedule_stats["pairs"],
                    legacy_pose_pairs,
                    temporal_pairs,
                    coarse_state.retrieval_sim_matrix,
                    extrinsic,
                    rotation_threshold=float(args.pair_pose_rotation_threshold),
                    dino_candidates=int(config.projected_overlap_dino_candidates),
                )
            )
            tracking_groups, tracking_group_stats = ref.build_three_layer_vggsfm_groups(
                center_selection["selected_centers"],
                center_selection["owner"],
                sift_schedule_stats["valid_edges"],
                sift_pose_dino_details,
                len(image_names),
                int(args.neighbors_per_center),
                fill_source_name="sift_pose_dino_fill",
                strategy_name="sift_first_three_layer_sift_pose_dino",
            )
            tracking_group_stats["sift_pose_dino_base"] = sift_pose_dino_stats
            stats["timing"]["vggsfm_group_build"] = time.time() - t0
            _debug(
                args,
                f"Built depth-free SIFT/Pose/DINO VGGSfM groups: groups={len(tracking_groups)}, group_size={tracking_group_stats['group_size']}",
            )
        else:
            raise ValueError(
                f"Unsupported effective group strategy: {args.group_strategy}"
            )
        schedule_audit = {
            "input_order": input_order,
            "scheduling_order": scheduling_order,
            "group_resolution": group_resolution,
            "mode": schedule_mode,
            "config": {
                "sift_temporal_window": int(args.sift_temporal_window),
                "sift_schedule_grid_size": int(args.sift_schedule_grid_size),
                "sift_schedule_min_inliers_per_cell": int(
                    args.sift_schedule_min_inliers_per_cell
                ),
                "sift_schedule_min_pair_inliers": int(
                    args.sift_schedule_min_pair_inliers
                ),
                "sift_schedule_min_grid_coverage": float(
                    args.sift_schedule_min_grid_coverage
                ),
                "vggsfm_max_center_gap": int(args.vggsfm_max_center_gap),
                "neighbors_per_center": int(args.neighbors_per_center),
            },
            "pair_graphs": sift_pair_graph_stats,
            "sift_graph": sift_schedule_stats,
            "center_selection": center_selection,
            "groups": tracking_group_stats,
        }
        _write_json(output_dir / "vggsfm_schedule.json", schedule_audit)
        stats["vggsfm_schedule"] = {
            "path": str(output_dir / "vggsfm_schedule.json"),
            "selected_centers": center_selection["num_selected"],
            "skipped_centers": center_selection["num_skipped"],
        }
        t0 = time.time()
        prior_tracks, prior_stats = ref.run_vggsfm_prior_tracks(
            args,
            coarse_state.high_images,
            None,
            legacy_pose_pairs,
            metadata,
            extrinsic,
            image_names,
            groups=tracking_groups,
            group_stats=tracking_group_stats,
        )
        stats["timing"]["vggsfm_prior_tracks"] = time.time() - t0
        stats["vggsfm"] = prior_stats
        group_stats = prior_stats["group_stats"]
        valid_neighbors = group_stats.get("selected_rotation_valid_neighbors")
        unfiltered_neighbors = group_stats.get("selected_unfiltered_neighbors")
        neighbor_priority_summary = ""
        if valid_neighbors is not None and unfiltered_neighbors is not None:
            neighbor_priority_summary = f"neighbor_order={group_stats['neighbor_order']}, valid_neighbors_mean={valid_neighbors['mean']:.2f}, unfiltered_neighbors_mean={unfiltered_neighbors['mean']:.2f}, "
        workload = prior_stats["workload"]
        query_track_stats = prior_stats["query_track_stats"]
        track_length = query_track_stats["track_length"]
        _debug(
            args,
            f"VGGSfM prior done: groups={prior_stats['num_groups']}, tracks={prior_stats['num_tracks']}, observations={prior_stats['num_observations']}, {neighbor_priority_summary}attempted_query_views={workload['attempted_query_views']}, forming_track_rate={query_track_stats['forming_track_rate']:.3f}, track_length_median={track_length['median']:.2f}, track_length_p90={track_length['p90']:.2f}, fmap_precompute={prior_stats['precompute_fmaps']['seconds']:.2f}s, fmap_cache={prior_stats['precompute_fmaps']['storage_dtype']}@{prior_stats['precompute_fmaps']['storage_device']}, fmap_resident={prior_stats['precompute_fmaps']['resident_on_tracker_device']}, group_tracking={prior_stats['group_tracking_time']:.2f}s, time={stats['timing']['vggsfm_prior_tracks']:.2f}s",
        )
    s_counts = np.asarray(
        sift_prefilter_stats["observations_per_image"], dtype=np.int64
    )
    p_counts = (
        loma_result.observation_counts()
        if is_loma
        else ref.count_track_observations(prior_tracks, len(image_names))
    )
    _debug(args, ref.format_count_summary("S observations/frame", s_counts))
    _debug(args, ref.format_count_summary("P observations/frame", p_counts))
    _debug(
        args, ref.format_count_summary("S+P observations/frame", s_counts + p_counts)
    )
    image_names, images, extrinsic, features, pairs, prior_tracks, coverage_stats = (
        ref.filter_low_coverage_frames(
            image_names,
            coarse_state.high_images,
            extrinsic,
            features,
            pairs,
            prior_tracks,
            s_counts,
            p_counts,
            args.min_frame_observations,
            enabled=True,
        )
    )
    stats["frame_filtering"] = coverage_stats
    stats["num_images_after_filter"] = len(image_names)
    _debug(
        args,
        f"Frame filtering: min_obs={coverage_stats['min_frame_observations']}, dropped={len(coverage_stats['dropped_indices'])}, remaining={len(image_names)}",
    )
    if coverage_stats["dropped_indices"]:
        _debug(
            args,
            "Dropped frames: "
            + ", ".join(
                (
                    f"{idx}:{name}"
                    for idx, name in zip(
                        coverage_stats["dropped_indices"],
                        coverage_stats["dropped_names"],
                        strict=False,
                    )
                )
            ),
        )
    kept_indices = np.asarray(coverage_stats["kept_indices"], dtype=np.int64)
    if is_loma:
        loma_result = loma_result.subset(kept_indices)
        stats["loma"]["frame_filtering"] = {
            "kept_original_indices": kept_indices.tolist(),
            "remaining_verified_pairs": len(loma_result.geometries),
            "remaining_unique_observations": int(
                loma_result.observation_counts().sum()
            ),
        }
    initial_intrinsics_high = initial_intrinsics_high_all[kept_indices]
    initial_intrinsics_low_all[kept_indices]
    depth = (
        coarse_state.raw_depth[kept_indices]
        if coarse_state.raw_depth is not None
        else None
    )
    depth_conf = (
        coarse_state.raw_depth_conf[kept_indices]
        if coarse_state.raw_depth_conf is not None
        else None
    )
    if depth is not None:
        t0 = time.time()
        depth_predictions = {"depth": depth}
        depth_conf_threshold = None
        if depth_conf is not None:
            depth_predictions["depth_conf"] = depth_conf
            depth_conf_threshold = 2.0
        stats["depth_export"] = export_prediction_depth_maps(
            depth_predictions,
            image_names,
            output_dir / "pred_depth",
            conf_threshold=depth_conf_threshold,
        )
        stats["depth_export"]["enabled"] = True
        stats["timing"]["depth_export"] = time.time() - t0
    else:
        stats["depth_export"] = {
            "enabled": False,
            "reason": "initial geometry uses prior pose without depth",
        }
        stats["timing"]["depth_export"] = 0.0
    t0 = time.time()
    averaged_intrinsics, global_intrinsics, intrinsics_mapping = (
        ref.average_intrinsics_with_gluemap(initial_intrinsics_high, CAMERA_MODEL)
    )
    stats["timing"]["intrinsics_averaging"] = time.time() - t0
    intrinsic = averaged_intrinsics[0]
    stats["intrinsics"] = ref.summarize_intrinsics(
        initial_intrinsics_high, averaged_intrinsics, CAMERA_MODEL
    )
    ref.save_intrinsics_artifacts(
        output_dir,
        initial_intrinsics_high,
        averaged_intrinsics,
        intrinsics_mapping,
        image_names,
    )
    _debug(
        args,
        f"Intrinsics averaged: fx={intrinsic[0, 0]:.2f}, fy={intrinsic[1, 1]:.2f}, cx={intrinsic[0, 2]:.2f}, cy={intrinsic[1, 2]:.2f}, time={stats['timing']['intrinsics_averaging']:.2f}s",
    )
    virtual_predictions_dict = None
    stats["virtual_tracks"] = {
        "enabled": False,
        "reason": "BAE backend uses SP real tracks only",
    }
    stats["timing"]["virtual_tracks"] = 0.0
    _debug(args, "Skipping virtual tracks for BAE SP refinement")
    if coverage_stats["dropped_indices"]:
        _debug(args, "Filtering SIFT database after frame filtering")
        t0 = time.time()
        features, sift_final_stats = ref.filter_sift_database_for_refine(
            output_dir / "database_sift.db",
            output_dir / "database_sift.db",
            prefilter_image_names,
            coverage_stats["kept_indices"],
            pairs,
            intrinsics_mapping,
        )
        stats["timing"]["filter_sift_database_final"] = time.time() - t0
        stats["timing"]["prepare_sift_database_final"] = 0.0
    else:
        sift_final_stats = stats["s_database"]["prefilter"]
        stats["timing"]["filter_sift_database_final"] = 0.0
        stats["timing"]["prepare_sift_database_final"] = 0.0
    stats["s_database"]["final"] = sift_final_stats
    _debug(
        args,
        f"SIFT DB ready: keypoints={sift_final_stats['num_keypoints_total']}, pairs={sift_final_stats['num_pairs']}, matches={sift_final_stats['num_matches']}",
    )
    prior_database_path = output_dir / (
        "database_loma_prior.db" if is_loma else "database_vggsfm_prior.db"
    )
    if is_loma:
        from ffba.matching.loma import write_loma_database

        t0 = time.time()
        stats["prior_database"] = write_loma_database(
            prior_database_path, image_names, image_size_hw, intrinsic, loma_result
        )
        stats["timing"]["write_prior_db"] = time.time() - t0
    else:
        _debug(args, "Writing VGGSfM prior database")
        t0 = time.time()
        stats["prior_database"] = ref.write_tracks_database(
            str(output_dir / "database_vggsfm_prior.db"),
            image_names,
            image_size_hw,
            intrinsic,
            CAMERA_MODEL,
            prior_tracks,
            features=features,
            snap_to_features=True,
            snap_threshold=args.prior_snap_threshold,
            keep_unsnapped=True,
            merge_threshold=args.prior_keypoint_merge_threshold,
            snap_target="sift",
            match_topology=args.prior_match_topology,
        )
        stats["timing"]["write_prior_db"] = time.time() - t0
        prior_db_stats = stats["prior_database"]
        snap_stats = prior_db_stats["snap"]
        _debug(
            args,
            f"Prior DB written: tracks={prior_db_stats['num_tracks']}, topology={prior_db_stats['keypoint_merge']['match_topology']}, pairs={prior_db_stats['num_pairs']}, raw_kp={prior_db_stats['keypoint_merge']['raw_total']}, merged_kp={prior_db_stats['keypoint_merge']['merged_total']}, snapped={snap_stats['snapped_observations']}",
        )
    from gluemap.utils.colmap import merge_colmap_databases

    _debug(args, f"Merging {args.prior_provider} prior and SIFT databases")
    t0 = time.time()
    merge_colmap_databases(
        str(prior_database_path),
        str(output_dir / "database_sift.db"),
        str(output_dir / "database_merged.db"),
        primary_features_first=False,
    )
    stats["timing"]["merge_databases"] = time.time() - t0
    coarse_dir = output_dir / "coarse"
    _debug(args, f"Writing coarse reconstruction: {coarse_dir}")
    t0 = time.time()
    ref.write_coarse_reconstruction(
        coarse_dir, image_names, image_size_hw, extrinsic, intrinsic, CAMERA_MODEL
    )
    stats["timing"]["write_coarse"] = time.time() - t0
    if args.device.startswith("cuda") and torch.cuda.is_available():
        cuda_device = torch.device(args.device)
        torch.cuda.synchronize(cuda_device)
        cuda_memory_before = _cuda_memory_snapshot(cuda_device)
        print(
            f"[PIPELINE-REFINE] CUDA memory before augmented refinement cleanup: {_format_cuda_memory_snapshot(cuda_memory_before)}",
            flush=True,
        )
        gc_collected = int(gc.collect())
        torch.cuda.empty_cache()
        torch.cuda.synchronize(cuda_device)
        cuda_memory_after = _cuda_memory_snapshot(cuda_device)
        print(
            f"[PIPELINE-REFINE] CUDA memory after augmented refinement cleanup: {_format_cuda_memory_snapshot(cuda_memory_after)}, gc_collected={gc_collected}",
            flush=True,
        )
        stats["augmented_refinement_cuda_cleanup"] = {
            "enabled": True,
            "device": str(cuda_device),
            "gc_collected": gc_collected,
            "before": cuda_memory_before,
            "after": cuda_memory_after,
            "reserved_bytes_released": int(
                cuda_memory_before["reserved_bytes"]
                - cuda_memory_after["reserved_bytes"]
            ),
            "driver_free_bytes_gained": int(
                cuda_memory_after["driver_free_bytes"]
                - cuda_memory_before["driver_free_bytes"]
            ),
        }
    else:
        stats["augmented_refinement_cuda_cleanup"] = {
            "enabled": False,
            "reason": "CUDA device is not active",
        }
    _debug(
        args,
        f"Running augmented refinement: iterations={args.num_refinement_iterations}, ba_max_iters={args.bae_max_num_iterations}, filter_reproj_threshold={args.filter_reproj_error_threshold}, bae_huber_delta={args.bae_huber_delta}, final_bae_huber_delta={args.final_bae_huber_delta}",
    )
    t0 = time.time()
    reconstruction, virtual_reconstruction, augmented_stats = (
        ref.run_merg3r_augmented_refinement_loop(
            args,
            pycolmap,
            output_dir,
            image_names,
            image_size_hw,
            CAMERA_MODEL,
            extrinsic,
            global_intrinsics,
            intrinsics_mapping,
            virtual_predictions_dict,
            features,
            output_dir / "database_merged.db",
        )
    )
    stats["timing"]["augmented_refinement"] = time.time() - t0
    stats["augmented_refinement"] = augmented_stats
    if is_loma:
        from ffba.matching.loma import summarize_final_tracks

        t0 = time.time()
        stats["loma"]["final_tracks"] = summarize_final_tracks(
            reconstruction,
            ref.build_s_keypoint_count(reconstruction, features),
            loma_result,
        )
        stats["timing"]["loma_final_track_audit"] = time.time() - t0
        _write_json(output_dir / "prior_loma_stats.json", stats["loma"])
    refined_dir = output_dir / "refined_gluemap_aba"
    refined_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(refined_dir))
    virtual_dir = None
    if virtual_reconstruction is not None:
        virtual_dir = output_dir / "virtual_gluemap_aba"
        virtual_dir.mkdir(parents=True, exist_ok=True)
        virtual_reconstruction.write(str(virtual_dir))
    stats["timing"]["total"] = time.time() - t_start
    stats["output"] = {
        "coarse_dir": str(coarse_dir),
        "database_merged": str(output_dir / "database_merged.db"),
        "refined_dir": str(refined_dir),
        "virtual_refined_dir": str(virtual_dir) if virtual_dir is not None else None,
    }
    _write_json(output_dir / "refine_stats.json", stats)
    final_sources = augmented_stats["final"]["real_by_source"]
    _debug(
        args,
        f"Augmented refinement done: real_points={augmented_stats['final']['real']['points3D']}, s_only={final_sources['s_only']}, p_only={final_sources['p_only']}, mixed={final_sources['mixed']}, virtual_points={augmented_stats['final']['virtual']['points3D']}, time={stats['timing']['augmented_refinement']:.2f}s",
    )
    return GluemapSpvRefineResult(
        image_names=image_names,
        extrinsic=extrinsic,
        pairs=pairs,
        intrinsic=intrinsic,
        intrinsics_mapping=intrinsics_mapping,
        stats=stats,
        refined_dir=refined_dir,
        virtual_refined_dir=virtual_dir,
    )
