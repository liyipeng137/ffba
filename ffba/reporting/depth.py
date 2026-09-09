import os
import numpy as np
import torch
import cv2


def _depth_stem_for_image(image_names, idx):
    if idx < len(image_names):
        return os.path.splitext(os.path.basename(str(image_names[idx])))[0]
    return f"frame_{idx:04d}"


def _save_depth_frame_pngs(depth_frame, stem, output_dir):
    depth_u16_dir = os.path.join(output_dir, "depth_u16")
    depth_vis_dir = os.path.join(output_dir, "depth_vis")
    depth_npy_dir = os.path.join(output_dir, "depth_npy")
    os.makedirs(depth_u16_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)
    os.makedirs(depth_npy_dir, exist_ok=True)

    depth_frame = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    depth_u16 = np.clip(depth_frame * 1000.0, 0, np.iinfo(np.uint16).max).astype(
        np.uint16
    )
    base_name = stem + ".png"

    cv2.imwrite(os.path.join(depth_u16_dir, base_name), depth_u16)
    np.save(
        os.path.join(depth_npy_dir, stem + ".npy"),
        depth_frame.astype(np.float32, copy=False),
    )

    valid_mask = np.isfinite(depth_frame) & (depth_frame > 0)
    if np.any(valid_mask):
        d = depth_frame[valid_mask]
        d_min = np.percentile(d, 2.0)
        d_max = np.percentile(d, 98.0)
        if d_max <= d_min:
            d_max = d_min + 1e-6

        depth_norm = (depth_frame - d_min) / (d_max - d_min)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        depth_vis_u8 = (depth_norm * 255.0).astype(np.uint8)
        depth_vis_u8[~valid_mask] = 0
        depth_color = cv2.applyColorMap(depth_vis_u8, cv2.COLORMAP_TURBO)
    else:
        h, w = depth_frame.shape[:2]
        depth_color = np.zeros((h, w, 3), dtype=np.uint8)

    cv2.imwrite(os.path.join(depth_vis_dir, base_name), depth_color)


def save_depth_pngs(depth_np, image_names, output_dir):
    """
    Save depth maps in three formats:
      1) uint16 millimeter PNGs under depth_u16
      2) uint8 pseudo-color PNGs under depth_vis
      3) float32 NPY files under depth_npy
    """
    for idx in range(depth_np.shape[0]):
        _save_depth_frame_pngs(
            depth_np[idx], _depth_stem_for_image(image_names, idx), output_dir
        )


def export_prediction_depth_maps(
    predictions, image_names, output_dir, conf_threshold=None
):
    """
    Save per-frame prediction depth maps without reprojecting the merged dense cloud.
    This is intended for temporary analysis against projected dense depth outputs.
    """
    if "depth" not in predictions:
        raise ValueError("predictions must contain a 'depth' entry")

    depth_src = predictions["depth"]
    depth_shape = (
        tuple(depth_src.shape)
        if isinstance(depth_src, torch.Tensor)
        else np.asarray(depth_src).shape
    )
    if len(depth_shape) == 4 and depth_shape[-1] == 1:
        depth_shape_3d = depth_shape[:3]
    else:
        depth_shape_3d = depth_shape
    if len(depth_shape_3d) != 3:
        raise ValueError(
            f"Expected prediction depth shape (N, H, W) or (N, H, W, 1), got {depth_shape}"
        )
    num_frames = int(depth_shape_3d[0])

    masked_pixels = 0
    conf_threshold_value = None
    conf_src = None
    if conf_threshold is not None:
        if "depth_conf" not in predictions:
            raise ValueError(
                "predictions must contain 'depth_conf' when conf_threshold is provided"
            )
        conf_src = predictions["depth_conf"]
        conf_shape = (
            tuple(conf_src.shape)
            if isinstance(conf_src, torch.Tensor)
            else np.asarray(conf_src).shape
        )
        if len(conf_shape) == 4 and conf_shape[-1] == 1:
            conf_shape_3d = conf_shape[:3]
        else:
            conf_shape_3d = conf_shape
        if tuple(conf_shape_3d) != tuple(depth_shape_3d):
            raise ValueError(
                f"Expected depth_conf shape {depth_shape_3d}, got {conf_shape}"
            )
        if conf_threshold == 0.0:
            conf_threshold_value = 0.0
        else:
            conf_for_percentile = (
                conf_src.detach().cpu().numpy()
                if isinstance(conf_src, torch.Tensor)
                else np.asarray(conf_src)
            )
            if conf_for_percentile.ndim == 4 and conf_for_percentile.shape[-1] == 1:
                conf_for_percentile = conf_for_percentile[..., 0]
            conf_threshold_value = float(
                np.percentile(conf_for_percentile, conf_threshold)
            )
            del conf_for_percentile

    if conf_src is not None:
        confidence_dir = os.path.join(output_dir, "confidence")
        os.makedirs(confidence_dir, exist_ok=True)

    nonzero_pixels = 0
    for idx in range(num_frames):
        if isinstance(depth_src, torch.Tensor):
            depth_frame = depth_src[idx].detach().cpu().numpy()
        else:
            depth_frame = np.asarray(depth_src[idx])
        if depth_frame.ndim == 3 and depth_frame.shape[-1] == 1:
            depth_frame = depth_frame[..., 0]
        depth_frame = np.asarray(depth_frame, dtype=np.float32)

        if conf_src is not None:
            if isinstance(conf_src, torch.Tensor):
                conf_frame = conf_src[idx].detach().cpu().numpy()
            else:
                conf_frame = np.asarray(conf_src[idx])
            if conf_frame.ndim == 3 and conf_frame.shape[-1] == 1:
                conf_frame = conf_frame[..., 0]
            conf_frame = np.asarray(conf_frame, dtype=np.float32)
            valid_conf = (
                np.isfinite(conf_frame)
                & (conf_frame >= conf_threshold_value)
                & (conf_frame > 1e-5)
            )
            masked_pixels += int(valid_conf.size - np.count_nonzero(valid_conf))
            depth_frame = np.where(valid_conf, depth_frame, 0.0).astype(
                np.float32, copy=False
            )

            stem = _depth_stem_for_image(image_names, idx)
            confidence_mask = valid_conf.astype(np.uint8) * 255
            cv2.imwrite(os.path.join(confidence_dir, stem + ".png"), confidence_mask)

        nonzero_pixels += int(
            np.count_nonzero(np.isfinite(depth_frame) & (depth_frame > 0))
        )
        _save_depth_frame_pngs(
            depth_frame, _depth_stem_for_image(image_names, idx), output_dir
        )

    print(
        f"[DEPTH EXPORT] Saved single-frame prediction depth maps to {output_dir} "
        f"(frames={num_frames}, nonzero_pixels={nonzero_pixels}, "
        f"conf_percentile={conf_threshold}, conf_threshold_value={conf_threshold_value}, "
        f"masked_pixels={masked_pixels})"
    )
    return {
        "num_depth_frames": int(num_frames),
        "num_nonzero_depth_pixels": nonzero_pixels,
        "conf_threshold": conf_threshold,
        "conf_threshold_value": conf_threshold_value,
        "num_masked_by_conf": masked_pixels,
        "output_dir": str(output_dir),
    }
