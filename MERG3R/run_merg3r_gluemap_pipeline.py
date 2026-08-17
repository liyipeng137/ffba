import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from algos.alignment import align_extrinsics
from algos.sequence import create_sequence
from algos.utils import (
    get_sim_matrix,
    restore_predictions_order,
)
from utils.feedforward import load_model, run_inference_step_by_step
from utils.image_pyramid import (
    build_two_resolution_image_pyramid,
    load_image_tensors_from_dir,
    load_matching_high_images,
    scale_intrinsics_low_to_high,
)
from utils.gluemap_spv_refine import (
    CAMERA_MODEL as PIPELINE_CAMERA_MODEL,
    GROUP_STRATEGY as PIPELINE_GROUP_STRATEGY,
    QUERY_SOURCE as PIPELINE_QUERY_SOURCE,
    S_DATABASE_MODE as PIPELINE_S_DATABASE_MODE,
    TRACK_MODE as PIPELINE_TRACK_MODE,
    TRACKER_INPUT as PIPELINE_TRACKER_INPUT,
    GluemapSpvRefineConfig,
    export_vggsfm_groups,
    run_gluemap_spv_refinement,
)

PIPELINE_MODEL = "pi3x"


@dataclass
class Merg3rCoarseState:
    low_images: torch.Tensor
    high_images: torch.Tensor
    low_image_names: list[str]
    high_image_names: list[str]
    low_image_size_hw: tuple[int, int]
    high_image_size_hw: tuple[int, int]
    final_predictions: dict
    extrinsic: np.ndarray
    intrinsic_low: np.ndarray
    intrinsic_high: np.ndarray
    image_ids: np.ndarray
    pairs: np.ndarray
    pair_graph_stats: dict
    raw_depth: np.ndarray
    raw_depth_conf: np.ndarray | None
    retrieval_sim_matrix: np.ndarray | None
    image_pyramid: dict | None


def parse_args():
    parser = argparse.ArgumentParser(
        "Run the integrated Merg3r + GlueMap pipeline with fixed "
        "pi3x + SIMPLE_PINHOLE + SIFT + ALIKED + configurable groups + SPV."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--pi3x_intrinsics_method",
        type=str,
        default="moge",
        choices=["lstsq", "moge"],
    )
    parser.add_argument("--num_images", type=int, default=-1)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--multi_dirs", action="store_true")
    parser.add_argument(
        "--image_pyramid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Preprocess the input image directory into low/high resolution "
            "sets. Low-res images feed MERG3R; high-res images feed SIFT, "
            "VGGSfM prior, refinement, and final outputs."
        ),
    )
    parser.add_argument("--stage1_downscale_n", type=int, default=4)
    parser.add_argument("--stage1_multiple", type=int, default=14)
    parser.add_argument(
        "--image_pyramid_workers",
        type=int,
        default=16,
        help="Number of concurrent workers for image pyramid preprocessing.",
    )
    parser.add_argument(
        "--stage2_scale_factor",
        type=int,
        default=0,
        help=(
            "High-res scale relative to stage1 after crop. 0 means use "
            "stage1_downscale_n, making stage2 approximately original size."
        ),
    )
    parser.add_argument("--sequence_type", type=str, default="shortest_path")
    parser.add_argument("--subset_size", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument(
        "--splitting_type",
        type=str,
        default="interleave",
        choices=[
            "interleave",
            "zigzag",
            "threshold",
            "original",
            "original_threshold",
        ],
    )
    parser.add_argument("--alignment_type", type=str, default="weighted_iterative")
    parser.add_argument("--pair_k_similarity", type=int, default=0)
    parser.add_argument("--pair_k_pose", type=int, default=25)
    parser.add_argument("--pair_temporal_window", type=int, default=0)
    parser.add_argument("--pair_pose_rotation_threshold", type=float, default=30.0)
    parser.add_argument(
        "--pair_pose_fill_unfiltered",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When pose-neighbor candidates passing the rotation threshold are "
            "fewer than pair_k_pose, fill the remaining slots by camera-center "
            "distance. This keeps the graph dense enough for pose groups."
        ),
    )
    parser.add_argument(
        "--path_tracker",
        type=str,
        default="/root/.cache/torch/hub/checkpoints/vggsfm_v2_tracker.pt",
    )
    parser.add_argument("--neighbors_per_center", type=int, default=25)
    parser.add_argument(
        "--vggsfm_group_strategy",
        type=str,
        default=PIPELINE_GROUP_STRATEGY,
        choices=["pose", "projected_overlap"],
        help=(
            "Group strategy used by formal VGGSfM prior tracking. "
            "'projected_overlap' ranks the union of rotation-valid pose and "
            "DINO retrieval candidates using coarse projected overlap."
        ),
    )
    parser.add_argument(
        "--export_vggsfm_groups_only",
        action="store_true",
        help=(
            "Stop after Stage A and export the selected VGGSfM audit groups as "
            "per-center images, contact sheets, JSON, and a label CSV."
        ),
    )
    parser.add_argument(
        "--vggsfm_group_audit_strategy",
        type=str,
        default="pose",
        choices=["pose", "projected_overlap", "both"],
        help=(
            "Group strategy exported by --export_vggsfm_groups_only. "
            "'projected_overlap' uses pose-no-fill plus DINO retrieval "
            "candidates; 'both' also exports the pose baseline."
        ),
    )
    parser.add_argument(
        "--projected_overlap_dino_candidates",
        type=int,
        default=30,
        help="Per-center DINO candidates added to the projected-overlap pool.",
    )
    parser.add_argument(
        "--projected_overlap_samples",
        type=int,
        default=2048,
        help="Maximum regular-grid source depth samples per center.",
    )
    parser.add_argument(
        "--projected_overlap_reproj_threshold",
        type=float,
        default=4.0,
        help="Round-trip reprojection threshold in low-resolution pixels.",
    )
    parser.add_argument(
        "--projected_overlap_conf_quantile",
        type=float,
        default=0.2,
        help="Drop the lowest source/target depth-confidence quantile.",
    )
    parser.add_argument("--vggsfm_query_points", type=int, default=1024)
    parser.add_argument("--aliked_detection_threshold", type=float, default=0.005)
    parser.add_argument("--vggsfm_vis_threshold", type=float, default=0.5)
    parser.add_argument("--vggsfm_score_threshold", type=float, default=0.0)
    parser.add_argument("--vggsfm_fine_tracking", action="store_true")
    parser.add_argument("--prior_snap_threshold", type=float, default=1.0)
    parser.add_argument("--prior_keypoint_merge_threshold", type=float, default=1e-3)
    parser.add_argument(
        "--prior_match_topology",
        type=str,
        default="star",
        choices=["all_pairs", "star"],
        help=(
            "Topology used to convert VGGSfM prior tracks into COLMAP pair "
            "matches. 'star' writes only center-neighbor correspondences, "
            "matching GlueMap TrackEstablishment more closely."
        ),
    )
    parser.add_argument("--min_frame_observations", type=int, default=10)
    parser.add_argument(
        "--ba_backend",
        type=str,
        default="bae",
        choices=["ceres", "bae"],
        help=(
            "Bundle adjustment backend for GlueMap augmented BA. 'ceres' keeps "
            "the original solver; 'bae' uses MERG3R/third_party/bae as an "
            "independent PyTorch backend."
        ),
    )
    parser.add_argument("--ba_max_num_iterations", type=int, default=100)
    parser.add_argument(
        "--bae_max_num_iterations",
        type=int,
        default=20,
        help="BAE optimizer iterations when --ba_backend=bae.",
    )
    parser.add_argument(
        "--bae_optimize_intrinsics",
        default=True,
        action="store_true",
        help=(
            "Let the BAE backend optimize SIMPLE_PINHOLE f while fixing cx/cy. "
            "Ignored by the Ceres backend."
        ),
    )
    parser.add_argument(
        "--bae_fix_gauge",
        type=str,
        default="two_cams",
        choices=["none", "two_cams", "three_points", "two_cams_full"],
        help=(
            "Gauge fixing strategy for the BAE backend. 'two_cams' mirrors "
            "COLMAP/Ceres TWO_CAMS_FROM_WORLD semantically: fix one pose and "
            "one translation DOF on a second pose, with three-point fallback."
        ),
    )
    parser.add_argument(
        "--bae_robust_loss",
        type=str,
        default="huber",
        choices=["none", "huber"],
        help=(
            "Robust loss for the BAE backend's real-track residuals. 'huber' "
            "applies IRLS Huber weighting (delta in pixels), mirroring the "
            "Ceres real-track Huber loss. Ignored by the Ceres backend."
        ),
    )
    parser.add_argument(
        "--bae_huber_delta",
        type=float,
        default=1.0,
        help="Huber delta in pixels when --bae_robust_loss=huber.",
    )
    parser.add_argument("--num_refinement_iterations", type=int, default=3)
    parser.add_argument("--augmented_ba_max_filter_iterations", type=int, default=3)
    parser.add_argument(
        "--augmented_ba_normalized_reproj_threshold",
        type=float,
        default=1e-2,
    )
    parser.add_argument("--tri_min_angle", type=float, default=1.0)
    parser.add_argument("--tri_create_max_angle_error", type=float, default=0.5)
    parser.add_argument("--select_track_min_support", type=int, default=512)
    parser.add_argument(
        "--filter_reproj_error_type",
        type=str,
        default="angular",
        choices=["angular", "pixel", "normalized"],
    )
    parser.add_argument("--filter_reproj_error_threshold", type=float, default=0.5)
    parser.add_argument("--virtual_init_angular_error_threshold", type=float)
    parser.add_argument(
        "--virtual_verify_mode",
        type=str,
        default="center",
        choices=["n2", "center"],
        help=(
            "Virtual-track covisibility verification mode. 'n2' matches "
            "GlueMap's full N^2 verification; 'center' verifies only the "
            "center-to-neighbor sweep used to build virtual-track masks."
        ),
    )
    parser.add_argument(
        "--save_virtual_tracks_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--debug_print",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


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


def build_mixed_pairs(
    images,
    extrinsic,
    k_similarity,
    k_pose,
    temporal_window,
    pose_rotation_threshold,
    pose_fill_unfiltered,
    similarity_matrix=None,
):
    n = int(images.shape[0])
    pairs = set()

    if temporal_window > 0:
        for i in range(n):
            for step in range(1, temporal_window + 1):
                _add_pair(pairs, i, i + step, n)

    if k_similarity > 0 and n > 1:
        sim_matrix = similarity_matrix
        if sim_matrix is None:
            sim_matrix = get_sim_matrix(images).detach().cpu()
        else:
            sim_matrix = torch.as_tensor(sim_matrix).detach().cpu()
        for i in range(n):
            row = sim_matrix[i].clone()
            row[i] = -float("inf")
            k = min(k_similarity, n - 1)
            for j in torch.topk(row, k).indices.tolist():
                _add_pair(pairs, i, j, n)

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
            if pose_fill_unfiltered and len(selected) < min(k_pose, n - 1):
                selected_set = set(selected)
                for j in ordered:
                    if j in selected_set:
                        continue
                    selected.append(j)
                    selected_set.add(j)
                    if len(selected) >= min(k_pose, n - 1):
                        break
            for j in selected:
                _add_pair(pairs, i, int(j), n)

    return np.asarray(sorted(pairs), dtype=np.int64)


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
    dataset_for_coarse = args.dataset
    if args.image_pyramid:
        stage2_scale_factor = (
            None if args.stage2_scale_factor == 0 else args.stage2_scale_factor
        )
        image_pyramid_result = build_two_resolution_image_pyramid(
            args.dataset,
            output_dir / "image_pyramid",
            stage1_downscale_n=args.stage1_downscale_n,
            multiple=args.stage1_multiple,
            stage2_scale_factor=stage2_scale_factor,
            recursive=args.multi_dirs,
            num_workers=args.image_pyramid_workers,
        )
        dataset_for_coarse = str(image_pyramid_result.low_dir)

    low_images, low_image_names = load_image_tensors_from_dir(
        dataset_for_coarse,
        device=args.device,
        subsample=args.subsample,
        num_images=args.num_images,
        recursive=args.multi_dirs,
    )
    low_image_size_hw = tuple(int(x) for x in low_images.shape[-2:])
    if image_pyramid_result is not None:
        high_images, high_image_names = load_matching_high_images(
            image_pyramid_result.high_dir,
            low_image_names,
            image_pyramid_result.low_dir,
            device="cpu",
            num_workers=args.image_pyramid_workers,
        )
        high_image_size_hw = tuple(int(x) for x in high_images.shape[-2:])
        image_pyramid_metadata = {
            "enabled": True,
            "manifest_path": str(image_pyramid_result.manifest_path),
            "low_dir": str(image_pyramid_result.low_dir),
            "high_dir": str(image_pyramid_result.high_dir),
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
    needs_projected_overlap = (
        args.export_vggsfm_groups_only
        and args.vggsfm_group_audit_strategy in {"projected_overlap", "both"}
    ) or (
        not args.export_vggsfm_groups_only
        and args.vggsfm_group_strategy == "projected_overlap"
    )
    if needs_projected_overlap and retrieval_sim_matrix is None:
        print(
            "[PIPELINE] DINO retrieval matrix was not produced by the sequence "
            "strategy; computing it for projected-overlap VGGSfM groups.",
            flush=True,
        )
        retrieval_sim_matrix = (
            get_sim_matrix(low_images, alpha=args.alpha)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    batches = sequence.image_split
    for idx, batch in enumerate(batches):
        batches[idx] = batch.to("cpu")

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

    for idx, prediction in enumerate(sequence.predictions):
        prediction["images"] = batches[idx]
        _as_numpy_prediction(prediction)

    final_predictions, _, _ = align_extrinsics(
        sequence, method=args.alignment_type, ba=False
    )
    restore_predictions_order(final_predictions)

    extrinsic = _normalize_extrinsic(final_predictions["extrinsic"]).astype(np.float32)
    intrinsic = np.asarray(final_predictions["intrinsic"], dtype=np.float32)
    if image_pyramid_result is not None:
        intrinsic_high = scale_intrinsics_low_to_high(
            intrinsic,
            low_image_names,
            image_pyramid_result.low_dir,
            image_pyramid_result.records,
        ).astype(np.float32)
    else:
        intrinsic_high = intrinsic.astype(np.float32, copy=True)
    image_ids = np.asarray(
        final_predictions.get("image_ids", np.arange(extrinsic.shape[0])),
        dtype=np.int64,
    )
    pairs = build_mixed_pairs(
        low_images,
        extrinsic,
        args.pair_k_similarity,
        args.pair_k_pose,
        args.pair_temporal_window,
        args.pair_pose_rotation_threshold,
        args.pair_pose_fill_unfiltered,
        similarity_matrix=retrieval_sim_matrix,
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
        low_images=low_images,
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
    )
    return state, {"seconds": time.time() - t0}


def write_stage_a_summary(output_dir, args, state, timing):
    summary = {
        "source": "MERG3R/run_merg3r_gluemap_pipeline.py",
        "stage": "merg3r_coarse",
        "status": "stage_a_completed",
        "dataset": args.dataset,
        "pipeline_defaults": {
            "model": PIPELINE_MODEL,
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
            "depth_shape": list(state.raw_depth.shape),
            "depth_coordinate_system": "low",
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
            "pair_k_similarity": args.pair_k_similarity,
            "pair_k_pose": args.pair_k_pose,
            "pair_temporal_window": args.pair_temporal_window,
            "pair_pose_rotation_threshold": args.pair_pose_rotation_threshold,
            "pair_pose_fill_unfiltered": args.pair_pose_fill_unfiltered,
            "num_pairs": int(state.pairs.shape[0]),
            "pair_graph": state.pair_graph_stats,
        },
        "timing": {"merg3r_coarse_seconds": timing["seconds"]},
        "config": vars(args),
    }
    with open(output_dir / "pipeline_stage_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required unless --device cpu is used.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pipeline_config.json", "w") as f:
        json.dump(
            {
                "pipeline_defaults": {
                    "model": PIPELINE_MODEL,
                    "camera_model": PIPELINE_CAMERA_MODEL,
                    "s_database_mode": PIPELINE_S_DATABASE_MODE,
                    "vggsfm_query_source": PIPELINE_QUERY_SOURCE,
                    "vggsfm_tracker_input": PIPELINE_TRACKER_INPUT,
                    "group_strategy": args.vggsfm_group_strategy,
                    "track_mode": PIPELINE_TRACK_MODE,
                },
                "args": vars(args),
            },
            f,
            indent=2,
        )

    state, timing = run_merg3r_coarse_stage(args, output_dir)
    write_stage_a_summary(output_dir, args, state, timing)

    print(f"[PIPELINE] Stage A done: output_dir={output_dir}")
    print(
        "[PIPELINE] "
        f"images={state.extrinsic.shape[0]}, "
        f"pairs={state.pairs.shape[0]}, "
        f"low_image_size_hw={state.low_image_size_hw}, "
        f"high_image_size_hw={state.high_image_size_hw}, "
        f"camera_model={PIPELINE_CAMERA_MODEL}"
    )
    print(
        "[PIPELINE] pair degree: "
        f"min={state.pair_graph_stats['degree_min']}, "
        f"median={state.pair_graph_stats['degree_median']:.1f}, "
        f"mean={state.pair_graph_stats['degree_mean']:.1f}, "
        f"p90={state.pair_graph_stats['degree_p90']:.1f}, "
        f"max={state.pair_graph_stats['degree_max']}, "
        f"zero={state.pair_graph_stats['zero_degree_images']}"
    )

    if args.export_vggsfm_groups_only:
        if args.vggsfm_group_audit_strategy == "both":
            audit_strategies = ["pose", "projected_overlap"]
        else:
            audit_strategies = [args.vggsfm_group_audit_strategy]
        manifest_paths = []
        for audit_strategy in audit_strategies:
            manifest_paths.append(
                export_vggsfm_groups(
                    state,
                    output_dir,
                    neighbors_per_center=args.neighbors_per_center,
                    pair_pose_rotation_threshold=(args.pair_pose_rotation_threshold),
                    num_workers=args.image_pyramid_workers,
                    selection_strategy=audit_strategy,
                    retrieval_sim_matrix=state.retrieval_sim_matrix,
                    projected_overlap_dino_candidates=(
                        args.projected_overlap_dino_candidates
                    ),
                    projected_overlap_samples=args.projected_overlap_samples,
                    projected_overlap_reproj_threshold=(
                        args.projected_overlap_reproj_threshold
                    ),
                    projected_overlap_conf_quantile=(
                        args.projected_overlap_conf_quantile
                    ),
                )
            )
        print(
            "[PIPELINE] Group export done; refinement was skipped: "
            f"manifests={[str(path) for path in manifest_paths]}",
            flush=True,
        )
        return

    refine_config = GluemapSpvRefineConfig(
        path_tracker=args.path_tracker,
        device=args.device,
        neighbors_per_center=args.neighbors_per_center,
        pair_pose_rotation_threshold=args.pair_pose_rotation_threshold,
        vggsfm_group_strategy=args.vggsfm_group_strategy,
        projected_overlap_dino_candidates=(args.projected_overlap_dino_candidates),
        projected_overlap_samples=args.projected_overlap_samples,
        projected_overlap_reproj_threshold=(args.projected_overlap_reproj_threshold),
        projected_overlap_conf_quantile=args.projected_overlap_conf_quantile,
        vggsfm_query_points=args.vggsfm_query_points,
        aliked_detection_threshold=args.aliked_detection_threshold,
        vggsfm_vis_threshold=args.vggsfm_vis_threshold,
        vggsfm_score_threshold=args.vggsfm_score_threshold,
        vggsfm_fine_tracking=args.vggsfm_fine_tracking,
        prior_snap_threshold=args.prior_snap_threshold,
        prior_keypoint_merge_threshold=args.prior_keypoint_merge_threshold,
        prior_match_topology=args.prior_match_topology,
        min_frame_observations=args.min_frame_observations,
        ba_backend=args.ba_backend,
        ba_max_num_iterations=args.ba_max_num_iterations,
        bae_max_num_iterations=args.bae_max_num_iterations,
        bae_optimize_intrinsics=args.bae_optimize_intrinsics,
        bae_fix_gauge=args.bae_fix_gauge,
        bae_robust_loss=args.bae_robust_loss,
        bae_huber_delta=args.bae_huber_delta,
        num_refinement_iterations=args.num_refinement_iterations,
        augmented_ba_max_filter_iterations=args.augmented_ba_max_filter_iterations,
        augmented_ba_normalized_reproj_threshold=(
            args.augmented_ba_normalized_reproj_threshold
        ),
        tri_min_angle=args.tri_min_angle,
        tri_create_max_angle_error=args.tri_create_max_angle_error,
        select_track_min_support=args.select_track_min_support,
        filter_reproj_error_type=args.filter_reproj_error_type,
        filter_reproj_error_threshold=args.filter_reproj_error_threshold,
        virtual_init_angular_error_threshold=args.virtual_init_angular_error_threshold,
        virtual_verify_mode=args.virtual_verify_mode,
        save_virtual_tracks_debug=args.save_virtual_tracks_debug,
        debug_print=args.debug_print,
        work_image_workers=args.image_pyramid_workers,
    )
    refine_result = run_gluemap_spv_refinement(state, output_dir, refine_config)
    print(
        "[PIPELINE] Refinement done: "
        f"refined_dir={refine_result.refined_dir}, "
        f"virtual_dir={refine_result.virtual_refined_dir}"
    )


if __name__ == "__main__":
    main()
