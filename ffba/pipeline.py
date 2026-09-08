from ffba.refinement_config import GluemapSpvRefineConfig
import json
import time
from pathlib import Path
import torch
from ffba.refinement import (
    CAMERA_MODEL as PIPELINE_CAMERA_MODEL,
    QUERY_SOURCE as PIPELINE_QUERY_SOURCE,
    S_DATABASE_MODE as PIPELINE_S_DATABASE_MODE,
    TRACK_MODE as PIPELINE_TRACK_MODE,
    TRACKER_INPUT as PIPELINE_TRACKER_INPUT,
    run_gluemap_spv_refinement,
)
from ffba.initialization.stage import (
    run_merg3r_coarse_stage,
    run_prior_pose_initial_stage,
    write_stage_a_summary,
)

PIPELINE_MODEL = "pi3x"


def run_pipeline(args):
    pipeline_t_start = time.time()
    if args.device.startswith("cuda") and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA is required unless --device cpu is used.")
    prior_pose_mode = args.prior_transforms_json is not None
    is_loma = args.prior_provider == "loma"
    if is_loma:
        from ffba.matching.loma_execution import validate_execution

        validate_execution(
            args.device,
            args.loma_match_batch_size,
            args.loma_extract_batch_size,
            args.loma_preprocess_workers,
            args.loma_geometry_workers,
            args.loma_feature_cache,
        )
        if args.bae_max_observations > 0:
            raise ValueError(
                "LoMa V1 has no observation cap; use --bae_max_observations 0"
            )
        if args.loma_dino_candidates <= 0:
            raise ValueError("loma_dino_candidates must be positive")
        if (
            min(
                args.loma_sufficient_neighbors,
                args.loma_insufficient_neighbors,
                args.loma_untried_neighbors,
            )
            < 0
        ):
            raise ValueError("LoMa neighbor counts must be nonnegative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from ffba.config import save_resolved_config

    save_resolved_config(args, output_dir)
    with open(output_dir / "pipeline_config.json", "w") as f:
        json.dump(
            {
                "pipeline_defaults": {
                    "model": None if prior_pose_mode else PIPELINE_MODEL,
                    "initial_geometry_source": "nerfstudio_prior"
                    if prior_pose_mode
                    else "feedforward",
                    "camera_model": PIPELINE_CAMERA_MODEL,
                    "s_database_mode": PIPELINE_S_DATABASE_MODE,
                    "prior_provider": args.prior_provider,
                    "loma_dino_candidates": args.loma_dino_candidates
                    if is_loma
                    else None,
                    "loma_pair_selection": args.loma_pair_selection
                    if is_loma
                    else None,
                    "vggsfm_query_source": None if is_loma else PIPELINE_QUERY_SOURCE,
                    "vggsfm_tracker_input": None if is_loma else PIPELINE_TRACKER_INPUT,
                    "group_strategy": "pose_union_dino_union_temporal"
                    if is_loma
                    else args.vggsfm_group_strategy,
                    "vggsfm_schedule_mode": None
                    if is_loma
                    else args.vggsfm_schedule_mode,
                    "track_mode": "SP" if is_loma else PIPELINE_TRACK_MODE,
                },
                "args": vars(args),
            },
            f,
            indent=2,
        )
    if prior_pose_mode:
        print(
            "[PIPELINE] Prior-pose mode enabled: using original-resolution images and skipping Pi3X/MERG3R/depth.",
            flush=True,
        )
        state, timing = run_prior_pose_initial_stage(args, output_dir)
    else:
        state, timing = run_merg3r_coarse_stage(args, output_dir)
    write_stage_a_summary(output_dir, args, state, timing)
    print(f"[PIPELINE] Stage A done: output_dir={output_dir}")
    print(
        f"[PIPELINE] images={state.extrinsic.shape[0]}, pairs={state.pairs.shape[0]}, low_image_size_hw={state.low_image_size_hw}, high_image_size_hw={state.high_image_size_hw}, camera_model={PIPELINE_CAMERA_MODEL}"
    )
    print(
        f"[PIPELINE] pair degree: min={state.pair_graph_stats['degree_min']}, median={state.pair_graph_stats['degree_median']:.1f}, mean={state.pair_graph_stats['degree_mean']:.1f}, p90={state.pair_graph_stats['degree_p90']:.1f}, max={state.pair_graph_stats['degree_max']}, zero={state.pair_graph_stats['zero_degree_images']}"
    )
    refine_config = GluemapSpvRefineConfig.from_namespace(args)
    refine_result = run_gluemap_spv_refinement(state, output_dir, refine_config)
    save_resolved_config(
        args, output_dir, group_resolution=refine_result.stats.get("group_resolution")
    )
    pipeline_summary = {
        "input_order": args.input_order,
        "mode": args.mode,
        "group_resolution": refine_result.stats.get("group_resolution"),
        "prior_provider": args.prior_provider,
        "loma_execution": refine_result.stats.get("loma", {}).get("execution")
        if is_loma
        else None,
        "loma_pair_selection": refine_result.stats.get("pair_graphs", {}).get(
            "loma_selection"
        )
        if is_loma
        else None,
        "vggsfm_schedule_mode": None if is_loma else args.vggsfm_schedule_mode,
        "num_input_images": int(state.extrinsic.shape[0]),
        "num_output_images": int(len(refine_result.image_names)),
        "num_dropped_images": int(
            len(refine_result.stats["frame_filtering"]["dropped_indices"])
        ),
        "selected_centers": None
        if is_loma
        else refine_result.stats.get("vggsfm_schedule", {}).get(
            "selected_centers", int(state.extrinsic.shape[0])
        ),
        "timing": {
            "stage_a_seconds": float(timing["seconds"]),
            "refinement_seconds": float(refine_result.stats["timing"]["total"]),
            "end_to_end_seconds": float(time.time() - pipeline_t_start),
        },
        "output": {
            "refined_dir": str(refine_result.refined_dir),
            "virtual_refined_dir": str(refine_result.virtual_refined_dir)
            if refine_result.virtual_refined_dir is not None
            else None,
        },
    }
    with open(output_dir / "pipeline_run_summary.json", "w") as f:
        json.dump(pipeline_summary, f, indent=2)
    print(
        f"[PIPELINE] Refinement done: refined_dir={refine_result.refined_dir}, virtual_dir={refine_result.virtual_refined_dir}, end_to_end={pipeline_summary['timing']['end_to_end_seconds']:.2f}s"
    )
