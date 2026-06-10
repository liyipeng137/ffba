import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lightglue import LightGlue, SuperPoint

from algos.alignment import align_extrinsics
from algos.sequence import create_sequence
from algos.utils import (
    export_prediction_depth_maps,
    get_sim_matrix,
    load_model,
    process_images,
    rbd,
    restore_predictions_order,
    run_inference_step_by_step,
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Export Merg3r coarse poses and LightGlue artifacts for Gluemap-style refinement."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        default="pi3x",
        choices=["vggt", "pi3", "pi3x", "vggt_omega"],
    )
    parser.add_argument(
        "--pi3x_intrinsics_method", type=str, default="moge", choices=["lstsq", "moge"]
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
        choices=["interleave", "zigzag", "threshold", "original", "original_threshold"],
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
            "distance. This keeps the exported graph dense enough for "
            "Gluemap-style star groups."
        ),
    )
    parser.add_argument("--max_num_keypoints", type=int, default=4096)
    parser.add_argument("--artifact_dir", type=str, default=None)
    parser.add_argument(
        "--export_depth_maps",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--depth_conf_threshold", type=float, default=2.0)
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


def save_images(images, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    image_names = []
    images_cpu = images.detach().cpu().float().clamp(0, 1)
    for idx, image in enumerate(images_cpu):
        name = f"frame_{idx:06d}.png"
        array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(array).save(out_dir / name)
        image_names.append(name)
    return image_names


def _feature_to_numpy(feats):
    result = {}
    for key in ("keypoints", "descriptors", "keypoint_scores", "scores"):
        if key not in feats:
            continue
        value = feats[key]
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().numpy()
    if "keypoint_scores" not in result and "scores" in result:
        result["keypoint_scores"] = result["scores"]
    return result


@torch.no_grad()
def export_lightglue_features_and_matches(
    images, pairs, out_dir, max_num_keypoints, device
):
    features_dir = out_dir / "features_lightglue"
    features_dir.mkdir(parents=True, exist_ok=True)

    extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    all_features = []
    for idx in range(images.shape[0]):
        feats = extractor.extract(images[idx].to(device))
        all_features.append(feats)
        np.savez_compressed(
            features_dir / f"frame_{idx:06d}.npz", **_feature_to_numpy(rbd(feats))
        )

    matches_payload = {}
    match_counts = {}
    for i, j in pairs.tolist():
        feats0 = all_features[int(i)]
        feats1 = all_features[int(j)]
        matches01 = matcher({"image0": feats0, "image1": feats1})
        matches = rbd(matches01)["matches"].detach().cpu().numpy().astype(np.int32)
        key = f"{int(i):06d}_{int(j):06d}"
        matches_payload[key] = matches
        match_counts[key] = int(matches.shape[0])

    np.savez_compressed(out_dir / "matches_lightglue.npz", **matches_payload)
    return match_counts


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Merg3r export script.")

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = (
        Path(args.artifact_dir)
        if args.artifact_dir
        else output_dir / "gluemap_refine_inputs"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    images, image_names = process_images(
        args.dataset,
        args.subsample,
        device,
        args.num_images,
        args.multi_dirs,
        args.model,
    )
    size_hw = tuple(int(x) for x in images.shape[-2:])

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

    model, _ = load_model(args.model, device=device)
    sequence.predictions = run_inference_step_by_step(
        model,
        batches,
        size_hw,
        device,
        need_features=False,
        pi3x_intrinsics_method=args.pi3x_intrinsics_method,
    )
    del model
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
    np.save(artifact_dir / "pairs.npy", pairs)
    np.savez_compressed(
        artifact_dir / "coarse_poses.npz",
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_ids=image_ids,
    )
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
    raw_geometry_payload = {"depth": raw_depth}
    if raw_depth_conf is not None:
        raw_geometry_payload["depth_conf"] = raw_depth_conf
    np.savez_compressed(artifact_dir / "raw_geometry.npz", **raw_geometry_payload)

    artifact_image_names = save_images(images, artifact_dir / "images")
    depth_stats = None
    if args.export_depth_maps:
        depth_stats = export_prediction_depth_maps(
            final_predictions,
            artifact_image_names,
            str(artifact_dir / "single_frame_depth"),
            conf_threshold=args.depth_conf_threshold,
        )
    match_counts = export_lightglue_features_and_matches(
        images,
        pairs,
        artifact_dir,
        args.max_num_keypoints,
        device,
    )

    metadata = {
        "source": "MERG3R/export_merg3r_refine_inputs.py",
        "dataset": args.dataset,
        "original_image_names": list(image_names),
        "artifact_image_names": artifact_image_names,
        "image_size_hw": list(size_hw),
        "num_images": int(extrinsic.shape[0]),
        "camera_model": "PINHOLE",
        "intrinsics_mode": "shared_mean",
        "pair_selection": {
            "pair_k_similarity": args.pair_k_similarity,
            "pair_k_pose": args.pair_k_pose,
            "pair_temporal_window": args.pair_temporal_window,
            "pair_pose_rotation_threshold": args.pair_pose_rotation_threshold,
            "pair_pose_fill_unfiltered": args.pair_pose_fill_unfiltered,
            "num_pairs": int(pairs.shape[0]),
            "pair_graph": pair_graph_stats,
        },
        "lightglue": {
            "features": "superpoint",
            "max_num_keypoints": args.max_num_keypoints,
            "match_counts": match_counts,
        },
        "raw_geometry": {
            "path": "raw_geometry.npz",
            "depth_shape": list(raw_depth.shape),
            "has_depth_conf": raw_depth_conf is not None,
        },
        "depth_export": depth_stats,
        "timing": {"total_export_seconds": time.time() - t_start},
        "config": vars(args),
    }
    with open(artifact_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[EXPORT] Wrote artifacts to {artifact_dir}")
    print(f"[EXPORT] images={extrinsic.shape[0]}, pairs={pairs.shape[0]}")
    print(
        "[EXPORT] pair degree: "
        f"min={pair_graph_stats['degree_min']}, "
        f"median={pair_graph_stats['degree_median']:.1f}, "
        f"mean={pair_graph_stats['degree_mean']:.1f}, "
        f"p90={pair_graph_stats['degree_p90']:.1f}, "
        f"max={pair_graph_stats['degree_max']}, "
        f"zero={pair_graph_stats['zero_degree_images']}"
    )


if __name__ == "__main__":
    main()
