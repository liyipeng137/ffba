"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

Usage:
    # Streaming inference (frame-by-frame with KV cache)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/

    # Streaming inference with keyframe KV caching
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/ --mode streaming --keyframe_interval 6

    # Windowed inference (for very long sequences, >500 frames)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10 --mode windowed --window_size 64

    # From video with custom FPS sampling
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10
"""

import argparse
import contextlib
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png",
                first_k=None, stride=1, image_size=518, patch_size=14, num_workers=8):
    """Load images from folder or video and preprocess into a tensor.

    Returns:
        (images, paths, resolved_image_folder): preprocessed tensor, file paths,
        and the folder containing the source images (for sky mask caching etc.).
    """
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        resolved_folder = out_dir
        print(f"Extracted {len(paths)} frames from video ({total_frames} total, interval={interval})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed images to {w}x{h} using canonical crop mode")
    return images, paths, resolved_folder


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint."""
    if getattr(args, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.kv_cache_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )

    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Checkpoint loaded.")

    return model.to(device).eval()


# =============================================================================
# Post-processing
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    # Convert w2c to c2w
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4) # inverse
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


# =============================================================================
# Offline export
# =============================================================================

def _to_numpy(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    return value


def _to_4x4(transforms):
    transforms = _to_numpy(transforms, dtype=np.float32)
    if transforms.ndim == 2:
        transforms = transforms[None]
    if transforms.shape[-2:] == (4, 4):
        return transforms
    if transforms.shape[-2:] != (3, 4):
        raise ValueError(f"Expected transforms with shape (..., 3, 4) or (..., 4, 4), got {transforms.shape}")

    transforms_4x4 = np.zeros((*transforms.shape[:-2], 4, 4), dtype=np.float32)
    transforms_4x4[..., :3, :4] = transforms
    transforms_4x4[..., 3, 3] = 1.0
    return transforms_4x4


def save_preprocessed_images(images, output_dir):
    images_np = _to_numpy(images, dtype=np.float32)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = []
    for i in range(images_np.shape[0]):
        file_name = f"{i:06d}.png"
        output_path = os.path.join(output_dir, file_name)
        image_rgb = images_np[i].transpose(1, 2, 0)
        image_u8 = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))
        image_paths.append(f"./images/{file_name}")
    return image_paths


def save_lingbot_transforms_json(output_path, c2w_opencv, intrinsics, image_paths, image_size):
    """Save NeRF-style transforms.json using OpenGL c2w matrices.

    The per-frame intrinsics are always written even when all Ks are equal,
    because LingBot-MAP predicts intrinsics for each frame and later pipeline
    stages need to inspect that variance explicitly.
    """
    c2w_opencv = _to_4x4(c2w_opencv)
    intrinsics = _to_numpy(intrinsics, dtype=np.float32)
    h, w = image_size

    c2w_opengl = np.array(c2w_opencv, copy=True)
    c2w_opengl[:, :3, 1:3] *= -1

    frames = []
    for i in range(c2w_opengl.shape[0]):
        k = intrinsics[i]
        frames.append(
            {
                "file_path": image_paths[i],
                "transform_matrix": c2w_opengl[i].tolist(),
                "w": int(w),
                "h": int(h),
                "fl_x": float(k[0, 0]),
                "fl_y": float(k[1, 1]),
                "cx": float(k[0, 2]),
                "cy": float(k[1, 2]),
            }
        )

    data = {
        "camera_model": "OpenGL",
        "frames": frames,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def write_dense_ply(output_path, world_points, images, conf=None, conf_percentile=1.5, downsample_factor=1):
    world_points = _to_numpy(world_points, dtype=np.float32)
    images_np = _to_numpy(images, dtype=np.float32)

    if downsample_factor > 1:
        world_points = world_points[:, ::downsample_factor, ::downsample_factor]
        images_np = images_np[:, :, ::downsample_factor, ::downsample_factor]
        if conf is not None:
            conf = _to_numpy(conf, dtype=np.float32)[:, ::downsample_factor, ::downsample_factor]
    elif conf is not None:
        conf = _to_numpy(conf, dtype=np.float32)

    points = world_points.reshape(-1, 3)
    colors = np.transpose(images_np, (0, 2, 3, 1)).reshape(-1, 3)
    colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    valid = np.isfinite(points).all(axis=1)
    conf_flat = None
    frame_indices = None
    if conf is not None:
        conf_flat = conf.reshape(-1)
        valid &= np.isfinite(conf_flat)
        if conf_percentile is not None and conf_percentile > 0:
            finite_conf = conf_flat[np.isfinite(conf_flat)]
            if finite_conf.size > 0:
                threshold = np.percentile(finite_conf, conf_percentile)
                valid &= conf_flat >= threshold
        valid &= conf_flat > 1e-5
        frame_indices = np.repeat(
            np.arange(world_points.shape[0], dtype=np.int32),
            world_points.shape[1] * world_points.shape[2],
        )

    points = points[valid]
    colors = colors[valid]
    if conf_flat is not None:
        conf_flat = conf_flat[valid].astype(np.float32, copy=False)
        frame_indices = frame_indices[valid]

    dtype = [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    if conf_flat is not None:
        dtype.extend([("confidence", "<f4"), ("frame_index", "<i4")])

    vertices = np.empty(points.shape[0], dtype=dtype)
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    if conf_flat is not None:
        vertices["confidence"] = conf_flat
        vertices["frame_index"] = frame_indices

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {vertices.shape[0]}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    if conf_flat is not None:
        header_lines.extend(["property float confidence", "property int frame_index"])
    header_lines.append("end_header")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(("\n".join(header_lines) + "\n").encode("ascii"))
        vertices.tofile(f)

    return int(vertices.shape[0])


def compute_self_projection_report(world_points, w2c, intrinsics, max_samples=20000):
    world_points = _to_numpy(world_points, dtype=np.float32)
    w2c = _to_4x4(w2c)
    intrinsics = _to_numpy(intrinsics, dtype=np.float32)

    n, h, w = world_points.shape[:3]
    total = n * h * w
    sample_count = min(int(max_samples), total)
    if sample_count <= 0:
        return {"num_sampled": 0, "num_valid": 0}

    rng = np.random.default_rng(0)
    flat = rng.integers(0, total, size=sample_count)
    frame_idx = flat // (h * w)
    rem = flat % (h * w)
    y = rem // w
    x = rem % w

    points = world_points[frame_idx, y, x]
    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid):
        return {"num_sampled": int(sample_count), "num_valid": 0}

    frame_idx = frame_idx[valid]
    x = x[valid].astype(np.float32, copy=False)
    y = y[valid].astype(np.float32, copy=False)
    points = points[valid]

    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam_points = np.einsum("nij,nj->ni", w2c[frame_idx], points_h)[:, :3]
    z = cam_points[:, 2]
    valid_z = np.isfinite(z) & (z > 1e-6)
    if not np.any(valid_z):
        return {"num_sampled": int(sample_count), "num_valid": 0}

    frame_idx = frame_idx[valid_z]
    x = x[valid_z]
    y = y[valid_z]
    cam_points = cam_points[valid_z]
    z = cam_points[:, 2]
    k = intrinsics[frame_idx]

    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[:, 0, 0] * (cam_points[:, 0] / z) + k[:, 0, 2]
        v = k[:, 1, 1] * (cam_points[:, 1] / z) + k[:, 1, 2]

    finite_uv = np.isfinite(u) & np.isfinite(v)
    if not np.any(finite_uv):
        return {"num_sampled": int(sample_count), "num_valid": 0}

    u = u[finite_uv]
    v = v[finite_uv]
    x = x[finite_uv]
    y = y[finite_uv]

    err = np.sqrt((u - x) ** 2 + (v - y) ** 2)
    err_half = np.sqrt((u - (x + 0.5)) ** 2 + (v - (y + 0.5)) ** 2)
    return {
        "num_sampled": int(sample_count),
        "num_valid": int(err.shape[0]),
        "mean_px": float(np.mean(err)),
        "median_px": float(np.median(err)),
        "p95_px": float(np.percentile(err, 95.0)),
        "max_px": float(np.max(err)),
        "median_px_with_half_pixel_shift": float(np.median(err_half)),
    }


def export_lingbot_outputs(args, predictions, images_cpu, source_paths):
    output_dir = args.export_dir
    if output_dir is None:
        candidate_paths = [args.save_transforms, args.save_dense]
        output_dir = next((os.path.dirname(p) for p in candidate_paths if p and os.path.dirname(p)), ".")

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")

    # c2w = _to_4x4(predictions["extrinsic"])
    w2c = _to_4x4(predictions["extrinsic"])
    c2w = np.linalg.inv(w2c).astype(np.float32)
    # w2c = np.linalg.inv(c2w).astype(np.float32)
    intrinsics = _to_numpy(predictions["intrinsic"], dtype=np.float32)
    world_points = _to_numpy(predictions["world_points"], dtype=np.float32)
    world_points_conf = predictions.get("world_points_conf")
    if world_points_conf is not None:
        world_points_conf = _to_numpy(world_points_conf, dtype=np.float32)

    image_paths = save_preprocessed_images(images_cpu, images_dir)
    h, w = images_cpu.shape[-2:]

    np.save(os.path.join(output_dir, "intrinsics.npy"), intrinsics)
    np.save(os.path.join(output_dir, "c2w.npy"), c2w.astype(np.float32, copy=False))
    np.save(os.path.join(output_dir, "w2c.npy"), w2c)
    np.save(os.path.join(output_dir, "world_points.npy"), world_points)
    if world_points_conf is not None:
        np.save(os.path.join(output_dir, "world_points_conf.npy"), world_points_conf)
    if "depth" in predictions:
        np.save(os.path.join(output_dir, "depth.npy"), _to_numpy(predictions["depth"], dtype=np.float32))

    transforms_path = args.save_transforms or os.path.join(output_dir, "transforms.json")
    save_lingbot_transforms_json(
        output_path=transforms_path,
        c2w_opencv=c2w,
        intrinsics=intrinsics,
        image_paths=image_paths,
        image_size=(h, w),
    )

    dense_path = args.save_dense or os.path.join(output_dir, "dense.ply")
    conf_percentile = args.export_conf_percentile
    if conf_percentile is None:
        conf_percentile = args.conf_threshold
    num_dense_points = write_dense_ply(
        output_path=dense_path,
        world_points=world_points,
        images=images_cpu,
        conf=world_points_conf,
        conf_percentile=conf_percentile,
        downsample_factor=args.export_dense_downsample_factor,
    )

    eye_errors = np.linalg.norm(w2c @ c2w - np.eye(4, dtype=np.float32), axis=(-2, -1))
    meta = {
        "num_frames": int(c2w.shape[0]),
        "image_size": {"height": int(h), "width": int(w)},
        "source_images": [str(p) for p in source_paths],
        "coordinate_convention": {
            "intrinsics.npy": "OpenCV per-frame intrinsics in pixel coordinates",
            "c2w.npy": "OpenCV camera-to-world, +x right, +y down, +z forward",
            "w2c.npy": "OpenCV world-to-camera, inverse of c2w.npy",
            "transforms.json": "OpenGL/Blender camera-to-world, generated by flipping OpenCV y/z axes",
            "transform.json": "Alias of transforms.json when --save_transforms is not set",
        },
        "shapes": {
            "intrinsics": list(intrinsics.shape),
            "c2w": list(c2w.shape),
            "w2c": list(w2c.shape),
            "world_points": list(world_points.shape),
            "world_points_conf": list(world_points_conf.shape) if world_points_conf is not None else None,
        },
        "dense_ply": {
            "path": os.path.relpath(dense_path, output_dir),
            "num_points": num_dense_points,
            "confidence_percentile": float(conf_percentile) if conf_percentile is not None else None,
            "downsample_factor": int(args.export_dense_downsample_factor),
        },
        "sanity": {
            "max_w2c_c2w_identity_frobenius": float(np.max(eye_errors)),
            "mean_w2c_c2w_identity_frobenius": float(np.mean(eye_errors)),
            "self_projection": compute_self_projection_report(world_points, w2c, intrinsics),
        },
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)

    print("\n" + "=" * 60)
    print(f"Exported LingBot outputs to: {output_dir}")
    print(f"  images: {images_dir}")
    print(f"  transforms: {transforms_path}")
    print(f"  dense: {dense_path} ({num_dense_points} points)")
    print(f"  max ||w2c @ c2w - I||_F: {meta['sanity']['max_w2c_c2w_identity_frobenius']:.3e}")
    print("=" * 60 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo")

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Inference mode
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"],
                        help="streaming: frame-by-frame with KV cache; windowed: overlapping windows for long sequences")

    # Streaming options
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=512)
    parser.add_argument("--num_scale_frames", type=int, default=6)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=1,
        help="Streaming only. Every N-th frame after scale frames is kept as a keyframe. 1 = every frame.",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=16)
    parser.add_argument("--kv_cache_scale_frames", type=int, default=6)
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (no flashinfer needed). Default: FlashInfer")

    parser.add_argument("--camera_num_iterations", type=int, default=6,
                        help="Camera head iterative-refinement steps. Default 4; set 1 for faster inference "
                             "(skips 3 refinement passes at a small accuracy cost).")

    # Windowed options
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=16, help="Overlap between windows")


    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument("--sky_mask_dir", type=str, default=None,
                        help="Directory for cached sky masks (default: <image_folder>_sky_masks/)")
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                        help="Save sky mask visualizations (original | mask | overlay) to this directory")
    parser.add_argument("--export_preprocessed", type=str, default=None,
                        help="Export stride-sampled, resized/cropped images to this folder")
    parser.add_argument("--export_dir", type=str, default=None,
                        help="Optional directory to export LingBot poses, intrinsics, dense points, images, and metadata")
    parser.add_argument("--save_transforms", type=str, default=None,
                        help="Optional path to save transforms.json. Defaults to <export_dir>/transforms.json when exporting")
    parser.add_argument("--save_dense", type=str, default=None,
                        help="Optional path to save dense point cloud PLY. Defaults to <export_dir>/dense.ply when exporting")
    parser.add_argument("--export_conf_percentile", type=float, default=None,
                        help="Confidence percentile filtered out for dense.ply. Defaults to --conf_threshold")
    parser.add_argument("--export_dense_downsample_factor", type=int, default=1,
                        help="Spatial downsample factor for dense.ply only. NPY dense maps are always full resolution")
    parser.add_argument("--skip_viewer", action="store_true",
                        help="Skip the interactive viewer after inference/export")

    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"
    if args.export_dense_downsample_factor < 1:
        raise ValueError("--export_dense_downsample_factor must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
    )

    # Export preprocessed images if requested
    if args.export_preprocessed:
        os.makedirs(args.export_preprocessed, exist_ok=True)
        print(f"Exporting {images.shape[0]} preprocessed images to {args.export_preprocessed}...")
        for i in range(images.shape[0]):
            img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.export_preprocessed, f"{i:06d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        print(f"Exported to {args.export_preprocessed}")

    model = load_model(args, device)
    print(f"Total load time: {time.time() - t0:.1f}s")

    images = images.to(device)
    num_frames = images.shape[0]
    print(f"Input: {num_frames} frames, shape {tuple(images.shape)}")
    print(f"Mode: {args.mode}")

    if args.mode != "streaming" and args.keyframe_interval != 1:
        print("Warning: --keyframe_interval only applies to --mode streaming. Ignoring it for windowed inference.")
        args.keyframe_interval = 1
    elif args.mode == "streaming" and args.keyframe_interval > 1:
        print(
            f"Keyframe streaming enabled: interval={args.keyframe_interval} "
            f"(after the first {args.num_scale_frames} scale frames)."
        )

    # ── Inference ────────────────────────────────────────────────────────────
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_context = torch.amp.autocast("cuda", dtype=dtype)
    else:
        dtype = torch.float32
        autocast_context = contextlib.nullcontext()
    print(f"Running {args.mode} inference (dtype={dtype})...")
    t0 = time.time()

    with torch.no_grad(), autocast_context:
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
            )
        else:  # windowed
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
            )

    print(f"Inference done in {time.time() - t0:.1f}s")

    # ── Post-process ─────────────────────────────────────────────────────────
    predictions, images_cpu = postprocess(predictions, images)

    # ── Offline export ───────────────────────────────────────────────────────
    if args.export_dir or args.save_transforms or args.save_dense:
        export_lingbot_outputs(args, predictions, images_cpu, paths)

    if args.skip_viewer:
        print("Skipping viewer (--skip_viewer).")
        return

    # ── Visualize ────────────────────────────────────────────────────────────
    try:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=prepare_for_visualization(predictions, images_cpu),
            port=args.port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            mask_sky=args.mask_sky,
            image_folder=resolved_image_folder,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
            use_point_map=True
        )
        print(f"3D viewer at http://localhost:{args.port}")
        viewer.run()
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        print(f"Predictions contain keys: {list(predictions.keys())}")


if __name__ == "__main__":
    main()
