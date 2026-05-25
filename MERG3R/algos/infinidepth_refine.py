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


def _confidence_path_for_image(confidence_dir, image_name, frame_idx):
    if confidence_dir is None:
        return None
    confidence_dir = Path(confidence_dir)
    stem = os.path.splitext(os.path.basename(str(image_name)))[0]
    path = confidence_dir / f"{stem}.png"
    if path.exists():
        return path

    fallback = confidence_dir / f"frame_{frame_idx:04d}.png"
    if fallback.exists():
        return fallback
    return path


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


def _image_to_infinidepth_tensors(image_rgb, input_size, device):
    org_h, org_w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (int(input_size[1]), int(input_size[0])), interpolation=cv2.INTER_AREA)
    org_img = torch.as_tensor(image_rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    image = torch.as_tensor(resized / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)[None]
    return org_img, image, org_h, org_w


def _depth_to_disparity(depth):
    disp = depth.clone()
    valid = disp > 0
    disp[valid] = 1.0 / disp[valid]
    return disp


def _make_2d_uniform_query(height, width, device):
    ys = ((torch.arange(height, device=device, dtype=torch.float32) + 0.5) / max(float(height), 1.0)) * 2.0 - 1.0
    xs = ((torch.arange(width, device=device, dtype=torch.float32) + 0.5) / max(float(width), 1.0)) * 2.0 - 1.0
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_y, grid_x], dim=-1).reshape(1, -1, 2).contiguous()


def _load_sensor_depth_prompt(
    depth_path,
    input_size,
    device,
    confidence_path=None,
    min_prompt=0.01,
    max_prompt=100.0,
    num_samples=1500,
):
    depth = np.load(depth_path).astype(np.float32)
    depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Expected sensor depth shape (H, W), got {depth.shape} from {depth_path}")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = cv2.resize(depth, (int(input_size[1]), int(input_size[0])), interpolation=cv2.INTER_NEAREST)

    depth_mask = ((depth > float(min_prompt)) & (depth < float(max_prompt))).astype(np.float32)
    if confidence_path is not None:
        confidence_path = Path(confidence_path)
        if confidence_path.exists():
            conf_mask = cv2.imread(str(confidence_path), cv2.IMREAD_GRAYSCALE)
            if conf_mask is None:
                raise ValueError(f"Failed to read confidence mask: {confidence_path}")
            conf_mask = cv2.resize(
                conf_mask,
                (int(input_size[1]), int(input_size[0])),
                interpolation=cv2.INTER_NEAREST,
            )
            depth_mask *= (conf_mask > 0).astype(np.float32)
        else:
            print(f"[INFINIDEPTH] Warning: missing confidence mask {confidence_path}; using depth range mask only")

    valid_depth = depth * depth_mask
    if int((valid_depth > float(min_prompt)).sum()) > int(num_samples):
        sample_depth = valid_depth.reshape(-1).copy()
        nonzero_index = np.flatnonzero(sample_depth > float(min_prompt))
        keep_index = np.random.permutation(nonzero_index)[: int(num_samples)]
        sampled = np.zeros_like(sample_depth)
        sampled[keep_index] = sample_depth[keep_index]
        sample_depth = sampled.reshape(depth.shape)
    else:
        sample_depth = valid_depth

    depth_t = torch.as_tensor(depth, dtype=torch.float32, device=device)[None, None]
    sample_depth_t = torch.as_tensor(sample_depth, dtype=torch.float32, device=device)[None, None]
    depth_mask_t = torch.as_tensor(depth_mask, dtype=torch.float32, device=device)[None, None]
    return depth_t, sample_depth_t, depth_mask_t


def _load_infinidepth_model(model_path, device):
    _ensure_infinidepth_import_path()
    from InfiniDepth.utils.model_utils import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run InfiniDepth refinement.")
    model = build_model("InfiniDepth_DepthSensor", model_path=str(model_path))
    model = model.to(device)
    model.eval()
    print(f"[INFINIDEPTH] Loaded model: {model.__class__.__name__}")
    return model


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
    confidence_dir=None,
    input_size=(768, 1024),
):
    """
    Refine sparse sensor depth with InfiniDepth_DepthSensor.

    Args:
        images: RGB images, shape (N, 3, H, W) or (N, H, W, 3).
        image_names: Names used for output refined depth stems.
        depth_image_names: Optional names used to match low-resolution input depth by stem.
        depth_npy_dir: Directory containing float32 sparse/sensor depth .npy files.
        confidence_dir: Optional directory containing confidence mask PNGs matched by depth_image_names.
        output_dir: Output directory; writes depth_npy, depth_vis, and depth_png.
        intrinsic: Camera intrinsics in the RGB image coordinate system, shape (3, 3) or (N, 3, 3).
    """
    depth_npy_dir = Path(depth_npy_dir)
    confidence_dir = None if confidence_dir is None else Path(confidence_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    inf_device = torch.device(device)
    model = _load_infinidepth_model(model_path, inf_device)

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
        confidence_path = _confidence_path_for_image(confidence_dir, depth_image_name, frame_idx)
        stem = os.path.splitext(os.path.basename(str(image_name)))[0]
        if not depth_path.exists():
            print(f"[INFINIDEPTH] skip {stem}: missing input depth {depth_path}")
            skipped += 1
            continue

        image_rgb = _image_to_numpy_rgb(images[frame_idx])
        _, image_t, org_h, org_w = _image_to_infinidepth_tensors(image_rgb, input_size, inf_device)
        gt_depth, prompt_depth, gt_depth_mask = _load_sensor_depth_prompt(
            depth_path,
            input_size,
            inf_device,
            confidence_path=confidence_path,
        )

        query_coord = _make_2d_uniform_query(org_h, org_w, inf_device)
        gt_disp = _depth_to_disparity(gt_depth)
        prompt_disp = _depth_to_disparity(prompt_depth)
        pred_2d_uniform_depth, _ = model.inference(
            image=image_t,
            query_coord=query_coord,
            gt_depth=gt_disp,
            gt_depth_mask=gt_depth_mask,
            prompt_depth=prompt_disp,
            prompt_mask=prompt_disp > 0,
        )
        depth_pred = pred_2d_uniform_depth.permute(0, 2, 1).view(1, 1, org_h, org_w)
        depth_pred = depth_pred.squeeze().detach().cpu().numpy()
        _save_refined_depth(depth_pred, stem, output_dir)
        processed += 1
        print(f"[INFINIDEPTH] refined {stem}")

        del gt_depth, prompt_depth, gt_depth_mask, gt_disp, prompt_disp, pred_2d_uniform_depth

    print(f"[INFINIDEPTH] Done -> {output_dir} (processed={processed}, skipped={skipped})")
    return {
        "model": str(model_path),
        "output_dir": str(output_dir),
        "num_processed": int(processed),
        "num_skipped": int(skipped),
        "input_size": tuple(input_size),
        "output_resolution_mode": "original",
    }
