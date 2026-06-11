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
    load_model,
    process_images,
    restore_predictions_order,
    run_inference_step_by_step,
)

PIPELINE_MODEL = "pi3x"
PIPELINE_CAMERA_MODEL = "SIMPLE_PINHOLE"
PIPELINE_S_DATABASE_MODE = "sift"
PIPELINE_QUERY_SOURCE = "aliked"
PIPELINE_GROUP_STRATEGY = "pose"
PIPELINE_TRACK_MODE = "SPV"


@dataclass
class Merg3rCoarseState:
    images: torch.Tensor
    image_names: list[str]
    image_size_hw: tuple[int, int]
    final_predictions: dict
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    image_ids: np.ndarray
    pairs: np.ndarray
    pair_graph_stats: dict
    raw_depth: np.ndarray
    raw_depth_conf: np.ndarray | None


def parse_args():
    parser = argparse.ArgumentParser(
        "Run the integrated Merg3r + GlueMap pipeline. "
        "Current implementation stops after the Merg3r coarse stage."
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
):
    n = int(images.shape[0])
    pairs = set()

    if temporal_window > 0:
        for i in range(n):
            for step in range(1, temporal_window + 1):
                _add_pair(pairs, i, i + step, n)

    if k_similarity > 0 and n > 1:
        sim_matrix = get_sim_matrix(images).detach().cpu()
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
    images, image_names = process_images(
        args.dataset,
        args.subsample,
        args.device,
        args.num_images,
        args.multi_dirs,
        PIPELINE_MODEL,
    )
    image_size_hw = tuple(int(x) for x in images.shape[-2:])

    sequence = create_sequence(
        images,
        image_names,
        sequence_type=args.sequence_type,
        subset_size=args.subset_size,
        overlap=args.overlap,
        save_path=str(output_dir),
        alpha=args.alpha,
        splitting_type=args.splitting_type,
    )
    batches = sequence.image_split
    for idx, batch in enumerate(batches):
        batches[idx] = batch.to("cpu")

    model, _ = load_model(PIPELINE_MODEL, device=args.device)
    sequence.predictions = run_inference_step_by_step(
        model,
        batches,
        image_size_hw,
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
    image_ids = np.asarray(
        final_predictions.get("image_ids", np.arange(extrinsic.shape[0])),
        dtype=np.int64,
    )
    pairs = build_mixed_pairs(
        images,
        extrinsic,
        args.pair_k_similarity,
        args.pair_k_pose,
        args.pair_temporal_window,
        args.pair_pose_rotation_threshold,
        args.pair_pose_fill_unfiltered,
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
        images=images,
        image_names=list(image_names),
        image_size_hw=image_size_hw,
        final_predictions=final_predictions,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_ids=image_ids,
        pairs=pairs,
        pair_graph_stats=pair_graph_stats,
        raw_depth=raw_depth,
        raw_depth_conf=raw_depth_conf,
    )
    return state, {"seconds": time.time() - t0}


def write_stage_a_summary(output_dir, args, state, timing):
    summary = {
        "source": "MERG3R/run_merg3r_gluemap_pipeline.py",
        "stage": "merg3r_coarse",
        "status": "stops_after_stage_a",
        "dataset": args.dataset,
        "pipeline_defaults": {
            "model": PIPELINE_MODEL,
            "camera_model": PIPELINE_CAMERA_MODEL,
            "s_database_mode": PIPELINE_S_DATABASE_MODE,
            "vggsfm_query_source": PIPELINE_QUERY_SOURCE,
            "group_strategy": PIPELINE_GROUP_STRATEGY,
            "track_mode": PIPELINE_TRACK_MODE,
        },
        "image_names": state.image_names,
        "image_size_hw": list(state.image_size_hw),
        "num_images": int(state.extrinsic.shape[0]),
        "extrinsic_shape": list(state.extrinsic.shape),
        "intrinsic_shape": list(state.intrinsic.shape),
        "raw_geometry": {
            "depth_shape": list(state.raw_depth.shape),
            "has_depth_conf": state.raw_depth_conf is not None,
            "depth_conf_shape": (
                list(state.raw_depth_conf.shape)
                if state.raw_depth_conf is not None
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
                    "group_strategy": PIPELINE_GROUP_STRATEGY,
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
        f"image_size_hw={state.image_size_hw}, "
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


if __name__ == "__main__":
    main()
