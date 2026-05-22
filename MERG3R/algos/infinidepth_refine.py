import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def _infinidepth_root():
    return Path(__file__).resolve().parents[1] / "third_party" / "InfiniDepth"


DEFAULT_INFINIDEPTH_MODEL_PATH = str(
    _infinidepth_root() / "checkpoints" / "depth" / "infinidepth_depthsensor.ckpt"
)


def _ensure_infinidepth_import_path():
    root = _infinidepth_root()
    if not root.exists():
        raise FileNotFoundError(f"InfiniDepth directory not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _image_to_numpy_rgb(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    if image.dtype != np.uint8:
        image_max = float(np.max(image)) if image.size else 0.0
        if image_max <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _depth_path_for_image(depth_dir, image_name, frame_idx):
    stem = os.path.splitext(os.path.basename(str(image_name)))[0]
    path = Path(depth_dir) / f"{stem}.npy"
    if path.exists():
        return path, stem

    fallback_stem = f"frame_{frame_idx:04d}"
    fallback = Path(depth_dir) / f"{fallback_stem}.npy"
    if fallback.exists():
        return fallback, fallback_stem
    return path, stem


def _depth_to_color_opencv(depth_map, vmin=None, vmax=None, colormap=cv2.COLORMAP_TURBO):
    valid_mask = np.isfinite(depth_map) & (depth_map > 0)
    depth_clean = depth_map.copy()
    depth_clean[~valid_mask] = 0
    if vmin is None:
        vmin = depth_clean[valid_mask].min() if valid_mask.any() else 0
    if vmax is None:
        vmax = depth_clean[valid_mask].max() if valid_mask.any() else 1
    depth_normalized = np.clip(
        (depth_clean - vmin) / (vmax - vmin + 1e-8) * 255,
        0,
        255,
    ).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_normalized, colormap)
    depth_colored[~valid_mask] = [0, 0, 0]
    return depth_colored


def _save_refined_depth(depth_pred, stem, output_dir):
    out_npy = output_dir / "depth_npy"
    out_vis = output_dir / "depth_vis"
    out_png = output_dir / "depth_png"
    for directory in (out_npy, out_vis, out_png):
        directory.mkdir(parents=True, exist_ok=True)

    depth_pred = np.nan_to_num(depth_pred, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    np.save(out_npy / f"{stem}.npy", depth_pred)

    depth_colored = _depth_to_color_opencv(depth_pred)
    cv2.imwrite(str(out_vis / f"{stem}.png"), depth_colored)

    depth_u16 = np.clip(depth_pred * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    cv2.imwrite(str(out_png / f"{stem}.png"), depth_u16)


def _write_temp_image(image_rgb, image_name, frame_idx, image_dir):
    stem = os.path.splitext(os.path.basename(str(image_name)))[0] if image_name else f"frame_{frame_idx:04d}"
    image_path = image_dir / f"{frame_idx:06d}_{stem}.png"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(image_path), image_bgr):
        raise RuntimeError(f"Failed to write temporary InfiniDepth RGB input: {image_path}")
    return image_path


@torch.no_grad()
def run_infinidepth_refinement(
    images,
    image_names,
    depth_npy_dir,
    output_dir,
    intrinsic,
    model_path=DEFAULT_INFINIDEPTH_MODEL_PATH,
    device="cuda",
    depth_image_names=None,
    input_size=(768, 1024),
):
    """
    Refine sparse sensor depth with InfiniDepth_DepthSensor.

    Args:
        images: RGB images, shape (N, 3, H, W) or (N, H, W, 3).
        image_names: Names used for output refined depth stems.
        depth_image_names: Optional names used to match low-resolution input depth by stem.
        depth_npy_dir: Directory containing float32 sparse/sensor depth .npy files.
        output_dir: Output directory; writes depth_npy, depth_vis, and depth_png.
        intrinsic: Camera intrinsics in the RGB image coordinate system, shape (3, 3) or (N, 3, 3).
    """
    _ensure_infinidepth_import_path()
    from inference_depth import DepthInferenceArgs, load_depth_model, run_depth_inference

    depth_npy_dir = Path(depth_npy_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_image_dir = output_dir / "_inputs" / "image"
    temp_image_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(images, torch.Tensor):
        num_frames = int(images.shape[0])
    else:
        num_frames = len(images)
    if len(image_names) != num_frames:
        print(
            "[INFINIDEPTH] Warning: image_names length does not match images; "
            f"names={len(image_names)}, images={num_frames}"
        )
    if depth_image_names is None:
        depth_image_names = image_names
    if len(depth_image_names) != num_frames:
        print(
            "[INFINIDEPTH] Warning: depth_image_names length does not match images; "
            f"names={len(depth_image_names)}, images={num_frames}"
        )

    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic.ndim == 2:
        intrinsic = np.repeat(intrinsic[None], num_frames, axis=0)
    if intrinsic.shape != (num_frames, 3, 3):
        raise ValueError(f"Expected intrinsic shape (3, 3) or ({num_frames}, 3, 3), got {intrinsic.shape}")

    frame_args = DepthInferenceArgs(
        input_image_path="",
        input_depth_path=None,
        depth_output_dir=None,
        pcd_output_dir=None,
        save_pcd=False,
        model_type="InfiniDepth_DepthSensor",
        depth_model_path=str(model_path),
        input_size=tuple(input_size),
        output_resolution_mode="original",
        upsample_ratio=1,
    )
    model, inf_device = load_depth_model(frame_args)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[INFINIDEPTH] CUDA requested but unavailable; using InfiniDepth-selected device.")

    processed = 0
    skipped = 0
    for frame_idx in range(num_frames):
        image_name = image_names[frame_idx] if frame_idx < len(image_names) else f"frame_{frame_idx:04d}.png"
        depth_image_name = (
            depth_image_names[frame_idx]
            if frame_idx < len(depth_image_names)
            else image_name
        )
        depth_path, _ = _depth_path_for_image(depth_npy_dir, depth_image_name, frame_idx)
        stem = os.path.splitext(os.path.basename(str(image_name)))[0]
        if not depth_path.exists():
            print(f"[INFINIDEPTH] skip {stem}: missing input depth {depth_path}")
            skipped += 1
            continue

        image_rgb = _image_to_numpy_rgb(images[frame_idx])
        image_path = _write_temp_image(image_rgb, image_name, frame_idx, temp_image_dir)
        K = intrinsic[frame_idx]

        result = run_depth_inference(
            frame_args,
            model=model,
            device=inf_device,
            input_image_path=str(image_path),
            input_depth_path=str(depth_path),
            fx_org=float(K[0, 0]),
            fy_org=float(K[1, 1]),
            cx_org=float(K[0, 2]),
            cy_org=float(K[1, 2]),
        )
        depth_pred = result.pred_depthmap.squeeze().detach().cpu().numpy()
        if depth_pred.shape != image_rgb.shape[:2]:
            depth_pred = cv2.resize(
                depth_pred.astype(np.float32, copy=False),
                (image_rgb.shape[1], image_rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        _save_refined_depth(depth_pred, stem, output_dir)
        processed += 1
        print(f"[INFINIDEPTH] refined {stem}")

        del result

    print(f"[INFINIDEPTH] Done -> {output_dir} (processed={processed}, skipped={skipped})")
    return {
        "model": str(model_path),
        "output_dir": str(output_dir),
        "num_processed": int(processed),
        "num_skipped": int(skipped),
        "input_size": tuple(input_size),
        "output_resolution_mode": "original",
    }
