import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


DEFAULT_LINGBOT_MODEL = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"


def _ensure_lingbot_import_path():
    lingbot_root = Path(__file__).resolve().parents[1] / "lingbot-depth"
    if not lingbot_root.exists():
        raise FileNotFoundError(f"LingBot-Depth directory not found: {lingbot_root}")
    lingbot_root_str = str(lingbot_root)
    if lingbot_root_str not in sys.path:
        sys.path.insert(0, lingbot_root_str)


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


def _normalize_intrinsic(intrinsic, width, height):
    K = np.asarray(intrinsic, dtype=np.float32).copy()
    if K.shape != (3, 3):
        raise ValueError(f"Expected intrinsic shape (3, 3), got {K.shape}")
    K[0, 0] /= width
    K[0, 2] /= width
    K[1, 1] /= height
    K[1, 2] /= height
    return K


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


@torch.no_grad()
def run_lingbot_depth_refinement(
    images,
    image_names,
    depth_npy_dir,
    output_dir,
    intrinsic,
    model_name=DEFAULT_LINGBOT_MODEL,
    device="cuda",
    use_fp16=True,
    enable_depth_mask=False,
):
    """
    Refine projected dense depth maps with LingBot-Depth.

    Args:
        images: Tensor/array/list of images in RGB order, shape (N, 3, H, W) or (N, H, W, 3).
        image_names: Names used to match depth_npy files by stem.
        depth_npy_dir: Directory containing projected input depths as float32 .npy files.
        output_dir: Output directory; writes depth_npy, depth_vis, and depth_png.
        intrinsic: Final BA intrinsics, either shared (3, 3) or per-frame (N, 3, 3).
    """
    _ensure_lingbot_import_path()
    from mdm.model.v2 import MDMModel

    depth_npy_dir = Path(depth_npy_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(images, torch.Tensor):
        num_frames = int(images.shape[0])
    else:
        num_frames = len(images)
    if len(image_names) != num_frames:
        print(
            "[LINGBOT DEPTH] Warning: image_names length does not match images; "
            f"names={len(image_names)}, images={num_frames}"
        )

    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic.ndim == 2:
        intrinsic = np.repeat(intrinsic[None], num_frames, axis=0)
    if intrinsic.shape != (num_frames, 3, 3):
        raise ValueError(f"Expected intrinsic shape (3, 3) or ({num_frames}, 3, 3), got {intrinsic.shape}")

    device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    model = MDMModel.from_pretrained(model_name).to(device)
    model.eval()

    processed = 0
    skipped = 0
    for frame_idx in range(num_frames):
        image_name = image_names[frame_idx] if frame_idx < len(image_names) else f"frame_{frame_idx:04d}.png"
        depth_path, stem = _depth_path_for_image(depth_npy_dir, image_name, frame_idx)
        if not depth_path.exists():
            print(f"[LINGBOT DEPTH] skip {stem}: missing input depth {depth_path}")
            skipped += 1
            continue

        image_rgb = _image_to_numpy_rgb(images[frame_idx])
        height, width = image_rgb.shape[:2]
        depth_np = np.load(depth_path).astype(np.float32)
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        if depth_np.shape != (height, width):
            depth_np = cv2.resize(depth_np, (width, height), interpolation=cv2.INTER_NEAREST)

        image_t = torch.as_tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
        depth_t = torch.as_tensor(depth_np, dtype=torch.float32, device=device)[None]
        intrinsics_t = torch.as_tensor(
            _normalize_intrinsic(intrinsic[frame_idx], width, height),
            dtype=torch.float32,
            device=device,
        )[None]

        output = model.infer(
            image_t,
            depth_in=depth_t,
            enable_depth_mask=enable_depth_mask,
            use_fp16=use_fp16,
            intrinsics=intrinsics_t,
        )
        depth_pred = output["depth"].squeeze().detach().cpu().numpy()
        _save_refined_depth(depth_pred, stem, output_dir)
        processed += 1
        print(f"[LINGBOT DEPTH] refined {stem}")

    print(f"[LINGBOT DEPTH] Done -> {output_dir} (processed={processed}, skipped={skipped})")
    return {
        "model": model_name,
        "output_dir": str(output_dir),
        "num_processed": int(processed),
        "num_skipped": int(skipped),
    }


def run_lingbot_depth_refinement_from_dirs(
    images_dir,
    depth_npy_dir,
    output_dir,
    intrinsic,
    model_name=DEFAULT_LINGBOT_MODEL,
    device="cuda",
    use_fp16=True,
    enable_depth_mask=False,
):
    images_dir = Path(images_dir)
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        image_paths.extend(images_dir.glob(pattern))
    image_paths = sorted(image_paths)

    images = []
    image_names = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"[LINGBOT DEPTH] skip unreadable image {image_path}")
            continue
        images.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_names.append(image_path.name)

    print(f"[LINGBOT DEPTH] Found {len(images)} images in {images_dir}")
    return run_lingbot_depth_refinement(
        images,
        image_names,
        depth_npy_dir,
        output_dir,
        intrinsic,
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        enable_depth_mask=enable_depth_mask,
    )
