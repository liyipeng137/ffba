import gc
import json
import time
import numpy as np
import torch
from ffba.initialization.alignment import align_extrinsics
from ffba.initialization.partition import create_sequence
from ffba.initialization.retrieval import get_sim_matrix
from ffba.initialization.alignment import restore_predictions_order
from ffba.initialization.feedforward import load_model, run_inference_step_by_step
from ffba.initialization.images import (
    build_two_resolution_image_tensors,
    load_image_tensors_from_dir,
    scale_intrinsics_with_pyramid_records,
)
from ffba.initialization.prior_pose import load_nerfstudio_prior
from ffba.refinement import (
    CAMERA_MODEL as PIPELINE_CAMERA_MODEL,
    QUERY_SOURCE as PIPELINE_QUERY_SOURCE,
    S_DATABASE_MODE as PIPELINE_S_DATABASE_MODE,
    TRACK_MODE as PIPELINE_TRACK_MODE,
    TRACKER_INPUT as PIPELINE_TRACKER_INPUT,
)
from ffba.types import Merg3rCoarseState

PIPELINE_MODEL = "pi3x"


def _as_numpy_prediction(prediction):
    for key in list(prediction.keys()):
        value = prediction[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
            if value.shape[:1] == (1,):
                value = value.squeeze(0)
            prediction[key] = value


def _camera_centers_from_w2c(extrinsic):
    rotations = extrinsic[:, :3, :3]
    translations = extrinsic[:, :3, 3]
    return np.einsum("nij,nj->ni", -np.transpose(rotations, (0, 2, 1)), translations)


def _viewing_axis_angles_from_w2c(extrinsic):
    rotations_c2w = np.transpose(extrinsic[:, :3, :3], (0, 2, 1))
    axes = rotations_c2w[:, :, -1]
    dots = np.einsum("mi,ni->mn", axes, axes, optimize=True)
    return np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0)))


def _normalize_extrinsic(extrinsic):
    extrinsic = np.asarray(extrinsic)
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(
            f"Expected extrinsic shape (N,3,4) or (N,4,4), got {extrinsic.shape}"
        )
    return extrinsic[:, :3, :4]


def _normalize_depth_like(array, num_images, name):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        array = array[..., None]
    if array.ndim != 4 or array.shape[0] != num_images or array.shape[-1] != 1:
        raise ValueError(
            f"Expected {name} shape (N,H,W), (N,H,W,1), or (1,N,H,W,1), "
            f"got {array.shape}"
        )
    return array


def _add_pair(pairs, i, j, n):
    if i == j or i < 0 or j < 0 or i >= n or j >= n:
        return
    a, b = sorted((int(i), int(j)))
    pairs.add((a, b))


def build_pose_pairs(
    num_images,
    extrinsic,
    k_pose,
    pose_rotation_threshold,
):
    n = int(num_images)
    pairs = set()

    if k_pose > 0 and n > 1:
        centers = _camera_centers_from_w2c(extrinsic)
        dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        np.fill_diagonal(dists, np.inf)
        angle_diffs = _viewing_axis_angles_from_w2c(extrinsic)
        invalid = angle_diffs >= pose_rotation_threshold
        np.fill_diagonal(invalid, True)
        for i in range(n):
            ordered = [int(j) for j in np.argsort(dists[i]) if j != i]
            valid_ordered = [j for j in ordered if not invalid[i, j]]
            selected = valid_ordered[: min(k_pose, n - 1)]
            for j in selected:
                _add_pair(pairs, i, int(j), n)

    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def summarize_pair_graph(pairs, num_images):
    degrees = np.zeros(num_images, dtype=np.int64)
    for i, j in pairs.tolist():
        degrees[int(i)] += 1
        degrees[int(j)] += 1
    if num_images == 0:
        return {
            "num_pairs": int(pairs.shape[0]),
            "degree_min": 0,
            "degree_median": 0.0,
            "degree_mean": 0.0,
            "degree_p90": 0.0,
            "degree_max": 0,
            "zero_degree_images": 0,
        }
    return {
        "num_pairs": int(pairs.shape[0]),
        "degree_min": int(degrees.min()),
        "degree_median": float(np.median(degrees)),
        "degree_mean": float(degrees.mean()),
        "degree_p90": float(np.percentile(degrees, 90)),
        "degree_max": int(degrees.max()),
        "zero_degree_images": int(np.sum(degrees == 0)),
    }


def run_merg3r_coarse_stage(args, output_dir):
    t0 = time.time()
    image_pyramid_result = None
    if args.image_pyramid:
        stage2_scale_factor = (
            None if args.stage2_scale_factor == 0 else args.stage2_scale_factor
        )
        image_pyramid_result = build_two_resolution_image_tensors(
            args.dataset,
            output_dir / "image_pyramid",
            stage1_downscale_n=args.stage1_downscale_n,
            multiple=args.stage1_multiple,
            stage2_scale_factor=stage2_scale_factor,
            recursive=args.multi_dirs,
            num_workers=args.image_pyramid_workers,
            subsample=args.subsample,
            num_images=args.num_images,
        )
        low_images = image_pyramid_result.low_images
        high_images = image_pyramid_result.high_images
        low_image_names = list(image_pyramid_result.image_names)
        high_image_names = list(image_pyramid_result.image_names)
        low_image_size_hw = tuple(int(x) for x in low_images.shape[-2:])
        high_image_size_hw = tuple(int(x) for x in high_images.shape[-2:])
        image_pyramid_metadata = {
            "enabled": True,
            "materialization": "memory",
            "manifest_path": str(image_pyramid_result.manifest_path),
            "low_dir": None,
            "high_dir": None,
            "stage1_downscale_n": int(args.stage1_downscale_n),
            "stage1_multiple": int(args.stage1_multiple),
            "num_workers": int(args.image_pyramid_workers),
            "stage2_scale_factor": (
                int(args.stage1_downscale_n)
                if args.stage2_scale_factor == 0
                else int(args.stage2_scale_factor)
            ),
        }
    else:
        low_images, low_image_names = load_image_tensors_from_dir(
            args.dataset,
            device="cpu",
            subsample=args.subsample,
            num_images=args.num_images,
            recursive=args.multi_dirs,
        )
        low_image_size_hw = tuple(int(x) for x in low_images.shape[-2:])
        high_images = low_images.detach().cpu()
        high_image_names = list(low_image_names)
        high_image_size_hw = low_image_size_hw
        image_pyramid_metadata = None

    sequence = create_sequence(
        low_images,
        low_image_names,
        sequence_type=args.sequence_type,
        subset_size=args.subset_size,
        overlap=args.overlap,
        save_path=str(output_dir),
        alpha=args.alpha,
        splitting_type=args.splitting_type,
    )
    retrieval_sim_matrix = getattr(sequence, "retrieval_sim_matrix", None)
    if retrieval_sim_matrix is not None:
        retrieval_sim_matrix = retrieval_sim_matrix.numpy().astype(
            np.float32, copy=False
        )
    if retrieval_sim_matrix is None:
        print(
            "[PIPELINE] Computing DINO retrieval matrix for all images before "
            "feed-forward inference.",
            flush=True,
        )
        retrieval_sim_matrix = (
            get_sim_matrix(low_images, alpha=args.alpha, device=args.device)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    batches = sequence.image_split
    for idx, batch in enumerate(batches):
        batches[idx] = batch.to("cpu")

    # Sequence construction may run DINO retrieval on CUDA.  Its tensors are
    # no longer needed once the split and CPU retrieval matrix are retained,
    # but the caching allocator can otherwise keep several GiB reserved and
    # fragment the following Pi3X allocation on full-length scenes.
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    model, _ = load_model(PIPELINE_MODEL, device=args.device)
    sequence.predictions = run_inference_step_by_step(
        model,
        batches,
        low_image_size_hw,
        args.device,
        need_features=False,
        pi3x_intrinsics_method=args.pi3x_intrinsics_method,
    )
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    for prediction in sequence.predictions:
        _as_numpy_prediction(prediction)

    sequence.images = None
    sequence.image_split = []
    if image_pyramid_result is not None:
        image_pyramid_result.low_images = None
    del batches, low_images

    final_predictions, _, _ = align_extrinsics(
        sequence, method=args.alignment_type, ba=False
    )
    restore_predictions_order(final_predictions)

    extrinsic = _normalize_extrinsic(final_predictions["extrinsic"]).astype(np.float32)
    intrinsic = np.asarray(final_predictions["intrinsic"], dtype=np.float32)
    if image_pyramid_result is not None:
        intrinsic_high = scale_intrinsics_with_pyramid_records(
            intrinsic,
            image_pyramid_result.records,
        ).astype(np.float32)
    else:
        intrinsic_high = intrinsic.astype(np.float32, copy=True)
    image_ids = np.asarray(
        final_predictions.get("image_ids", np.arange(extrinsic.shape[0])),
        dtype=np.int64,
    )
    pairs = build_pose_pairs(
        extrinsic.shape[0],
        extrinsic,
        args.pair_k_pose,
        args.pair_pose_rotation_threshold,
    )
    pair_graph_stats = summarize_pair_graph(pairs, extrinsic.shape[0])
    raw_depth = _normalize_depth_like(
        final_predictions["depth"], extrinsic.shape[0], "depth"
    )
    raw_depth_conf = None
    if "depth_conf" in final_predictions:
        raw_depth_conf = _normalize_depth_like(
            final_predictions["depth_conf"],
            extrinsic.shape[0],
            "depth_conf",
        )

    state = Merg3rCoarseState(
        high_images=high_images,
        low_image_names=list(low_image_names),
        high_image_names=list(high_image_names),
        low_image_size_hw=low_image_size_hw,
        high_image_size_hw=high_image_size_hw,
        final_predictions=final_predictions,
        extrinsic=extrinsic,
        intrinsic_low=intrinsic,
        intrinsic_high=intrinsic_high,
        image_ids=image_ids,
        pairs=pairs,
        pair_graph_stats=pair_graph_stats,
        raw_depth=raw_depth,
        raw_depth_conf=raw_depth_conf,
        retrieval_sim_matrix=retrieval_sim_matrix,
        image_pyramid=image_pyramid_metadata,
        initial_geometry_source="feedforward",
        source_metadata=None,
    )
    return state, {"seconds": time.time() - t0}


def run_prior_pose_initial_stage(args, output_dir):
    t0 = time.time()
    if args.prior_dino_batch_size <= 0:
        raise ValueError("prior_dino_batch_size must be >= 1")
    prior = load_nerfstudio_prior(
        args.prior_transforms_json,
        args.dataset,
        subsample=args.subsample,
        num_images=args.num_images,
        num_workers=args.image_pyramid_workers,
        retrieval_long_side=args.prior_dino_long_side,
    )
    print(
        "[PIPELINE] Computing DINO retrieval matrix from temporary prior-pose "
        f"retrieval images: shape={tuple(prior.retrieval_images.shape)}",
        flush=True,
    )
    retrieval_sim_matrix = (
        get_sim_matrix(
            prior.retrieval_images,
            alpha=args.alpha,
            device=args.device,
            subset_size=args.prior_dino_batch_size,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    prior.retrieval_images = None
    prior.audit["dino_batch_size"] = int(args.prior_dino_batch_size)
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    pairs = build_pose_pairs(
        prior.extrinsic.shape[0],
        prior.extrinsic,
        args.pair_k_pose,
        args.pair_pose_rotation_threshold,
    )
    pair_graph_stats = summarize_pair_graph(pairs, prior.extrinsic.shape[0])
    final_predictions = {
        "extrinsic": prior.extrinsic,
        "intrinsic": prior.intrinsic,
        "image_ids": prior.image_ids,
    }
    image_metadata = {
        "enabled": False,
        "reason": "prior_pose_uses_original_resolution",
        "materialization": "original_resolution_memory",
    }
    state = Merg3rCoarseState(
        high_images=prior.images,
        low_image_names=list(prior.image_names),
        high_image_names=list(prior.image_names),
        low_image_size_hw=prior.image_size_hw,
        high_image_size_hw=prior.image_size_hw,
        final_predictions=final_predictions,
        extrinsic=prior.extrinsic,
        intrinsic_low=prior.intrinsic,
        intrinsic_high=prior.intrinsic,
        image_ids=prior.image_ids,
        pairs=pairs,
        pair_graph_stats=pair_graph_stats,
        raw_depth=None,
        raw_depth_conf=None,
        retrieval_sim_matrix=retrieval_sim_matrix,
        image_pyramid=image_metadata,
        initial_geometry_source="nerfstudio_prior",
        source_metadata=prior.audit,
    )
    with open(output_dir / "prior_pose_import.json", "w") as handle:
        json.dump(prior.audit, handle, indent=2)
    return state, {"seconds": time.time() - t0}


def write_stage_a_summary(output_dir, args, state, timing):
    has_depth = state.raw_depth is not None
    feedforward_source = state.initial_geometry_source == "feedforward"
    timing_summary = {"initial_geometry_seconds": timing["seconds"]}
    if feedforward_source:
        timing_summary["merg3r_coarse_seconds"] = timing["seconds"]
    else:
        timing_summary["prior_pose_import_seconds"] = timing["seconds"]
    summary = {
        "input_order": args.input_order,
        "image_order_source": "transforms_json_frames"
        if args.prior_transforms_json
        else "sorted_image_paths",
        "source": (
            "run_merg3r_gluemap_pipeline.py"
            if feedforward_source
            else "nerfstudio_prior"
        ),
        "stage": "merg3r_coarse" if feedforward_source else "prior_pose_initial",
        "status": "stage_a_completed",
        "dataset": args.dataset,
        "prior_transforms_json": args.prior_transforms_json,
        "pipeline_defaults": {
            "model": (PIPELINE_MODEL if feedforward_source else None),
            "camera_model": PIPELINE_CAMERA_MODEL,
            "s_database_mode": PIPELINE_S_DATABASE_MODE,
            "vggsfm_query_source": PIPELINE_QUERY_SOURCE,
            "vggsfm_tracker_input": PIPELINE_TRACKER_INPUT,
            "group_strategy": args.vggsfm_group_strategy,
            "track_mode": PIPELINE_TRACK_MODE,
        },
        "low_image_names": state.low_image_names,
        "high_image_names": state.high_image_names,
        "low_image_size_hw": list(state.low_image_size_hw),
        "high_image_size_hw": list(state.high_image_size_hw),
        "num_images": int(state.extrinsic.shape[0]),
        "extrinsic_shape": list(state.extrinsic.shape),
        "intrinsic_low_shape": list(state.intrinsic_low.shape),
        "intrinsic_high_shape": list(state.intrinsic_high.shape),
        "image_pyramid": state.image_pyramid,
        "raw_geometry": {
            "has_depth": has_depth,
            "depth_shape": list(state.raw_depth.shape) if has_depth else None,
            "depth_coordinate_system": "low" if has_depth else None,
            "has_depth_conf": state.raw_depth_conf is not None,
            "depth_conf_shape": (
                list(state.raw_depth_conf.shape)
                if state.raw_depth_conf is not None
                else None
            ),
        },
        "retrieval_similarity": {
            "available": state.retrieval_sim_matrix is not None,
            "shape": (
                list(state.retrieval_sim_matrix.shape)
                if state.retrieval_sim_matrix is not None
                else None
            ),
        },
        "pair_selection": {
            "strategy": "pose_rotation_filtered",
            "pair_k_pose": args.pair_k_pose,
            "pair_pose_rotation_threshold": args.pair_pose_rotation_threshold,
            "num_pairs": int(state.pairs.shape[0]),
            "pair_graph": state.pair_graph_stats,
        },
        "source_metadata": state.source_metadata,
        "timing": timing_summary,
        "config": vars(args),
    }
    with open(output_dir / "pipeline_stage_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
