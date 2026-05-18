import json
import os

import cv2
import numpy as np
import torch
import trimesh

from algos.utils import export_dense_local_point_map_ply


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _as_w2c_3x4(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    if extrinsic.shape == (4, 4):
        return extrinsic[:3, :4]
    if extrinsic.shape == (3, 4):
        return extrinsic
    raise ValueError(f"Expected extrinsic shape (3, 4) or (4, 4), got {extrinsic.shape}")


def _image_to_uint8(image, target_hw):
    image = _to_numpy(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    height, width = target_hw
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height))
    if image.dtype != np.uint8:
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
    return image


def _frame_valid_mask(valid_track_mask, frame_idx, count):
    if valid_track_mask is None:
        return np.ones(count, dtype=bool)

    mask = valid_track_mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    frame_mask = np.asarray(mask[frame_idx])

    if frame_mask.size < count:
        padded = np.zeros(count, dtype=bool)
        padded[:frame_mask.size] = frame_mask.astype(bool)
        return padded
    return frame_mask[:count].astype(bool)


def _mad(values):
    values = np.asarray(values)
    if values.size == 0:
        return None
    med = np.median(values)
    return float(np.median(np.abs(values - med)))


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "mad": None,
            "p90_abs": None,
            "p95_abs": None,
            "max_abs": None,
        }
    abs_values = np.abs(values)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "mad": _mad(values),
        "p90_abs": float(np.percentile(abs_values, 90)),
        "p95_abs": float(np.percentile(abs_values, 95)),
        "max_abs": float(np.max(abs_values)),
    }


def _quantile_bins(x, y, num_bins=5):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size == 0:
        return []

    edges = np.quantile(x, np.linspace(0.0, 1.0, num_bins + 1))
    bins = []
    for idx in range(num_bins):
        low = edges[idx]
        high = edges[idx + 1]
        if idx == num_bins - 1:
            mask = (x >= low) & (x <= high)
        else:
            mask = (x >= low) & (x < high)
        if not np.any(mask):
            continue
        bins.append({
            "low": float(low),
            "high": float(high),
            "count": int(np.count_nonzero(mask)),
            "res_z_abs_median": float(np.median(np.abs(y[mask]))),
            "res_z_abs_p90": float(np.percentile(np.abs(y[mask]), 90)),
        })
    return bins


def _depth_edge_magnitude(depth):
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(valid, depth, 0.0).astype(np.float32)
    grad_x = cv2.Sobel(safe_depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(safe_depth, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    edge[~valid] = np.nan
    return edge


def _write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_point_cloud(path, points, colors=None):
    points = np.asarray(points, dtype=np.float32)
    if points.size == 0:
        points = np.zeros((1, 3), dtype=np.float32)
        colors = np.asarray([[255, 0, 0]], dtype=np.uint8)
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape[-1] == 3:
            colors = np.concatenate(
                [colors.astype(np.uint8), np.full((colors.shape[0], 1), 255, dtype=np.uint8)],
                axis=1,
            )
    trimesh.PointCloud(points, colors=colors).export(path)


def _save_sparse_heatmap(output_path, height, width, uv, values, title_scale=1.0):
    heat = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    if len(uv) > 0:
        xy = np.asarray(uv, dtype=np.int64)
        vals = np.asarray(values, dtype=np.float32)
        valid = (
            np.isfinite(vals)
            & (xy[:, 0] >= 0)
            & (xy[:, 0] < width)
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < height)
        )
        xy = xy[valid]
        vals = np.abs(vals[valid])
        np.add.at(heat, (xy[:, 1], xy[:, 0]), vals)
        np.add.at(counts, (xy[:, 1], xy[:, 0]), 1.0)

    valid = counts > 0
    heat[valid] /= counts[valid]
    if np.any(valid):
        vmax = np.percentile(heat[valid], 95)
        vmax = max(float(vmax), 1e-6)
        norm = np.clip(heat / (vmax * title_scale), 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        color[~valid] = 0
    else:
        color = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imwrite(output_path, color)


def build_sparse_dense_anchor_diagnostics(predictions, images, track, points_id, valid_track_mask=None):
    local_points = _to_numpy(predictions["local_points"]).astype(np.float32)
    extrinsic = _to_numpy(predictions["extrinsic"]).astype(np.float32)
    sparse_points = _to_numpy(predictions["points"]).astype(np.float32)
    depth_conf = _to_numpy(predictions.get("depth_conf", np.ones(local_points.shape[:3], dtype=np.float32)))
    if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
        depth_conf = depth_conf[..., 0]

    num_frames, height, width, _ = local_points.shape
    observations = []
    frame_observations = [[] for _ in range(num_frames)]

    for frame_idx, (track_i, point_ids_i) in enumerate(zip(track, points_id)):
        track_i = np.asarray(track_i)
        point_ids_i = np.asarray(point_ids_i)
        if track_i.size == 0 or point_ids_i.size == 0:
            continue

        track_i = track_i.reshape(-1, 2)
        point_ids_i = point_ids_i.reshape(-1).astype(np.int64)
        count = min(len(track_i), len(point_ids_i))
        if count == 0:
            continue

        valid_mask = _frame_valid_mask(valid_track_mask, frame_idx, count)
        w2c = _as_w2c_3x4(extrinsic[frame_idx])
        image_rgb = _image_to_uint8(images[frame_idx], (height, width))
        depth_edge = _depth_edge_magnitude(local_points[frame_idx, :, :, 2])

        for obs_idx in np.flatnonzero(valid_mask):
            point_id = int(point_ids_i[obs_idx])
            if point_id < 0 or point_id >= len(sparse_points):
                continue

            u_float, v_float = track_i[obs_idx]
            u = int(np.rint(u_float))
            v = int(np.rint(v_float))
            if u < 0 or u >= width or v < 0 or v >= height:
                continue

            dense_cam = local_points[frame_idx, v, u]
            sparse_world = sparse_points[point_id]
            sparse_cam = w2c[:, :3] @ sparse_world + w2c[:, 3]
            if not (np.isfinite(dense_cam).all() and np.isfinite(sparse_cam).all()):
                continue

            z_dense = float(dense_cam[2])
            z_sparse = float(sparse_cam[2])
            if z_dense <= 1e-8 or z_sparse <= 1e-8:
                continue

            res_xyz = sparse_cam - dense_cam
            obs = {
                "frame_idx": frame_idx,
                "obs_idx": int(obs_idx),
                "point_id": point_id,
                "u": u,
                "v": v,
                "u_float": float(u_float),
                "v_float": float(v_float),
                "z_dense": z_dense,
                "z_sparse": z_sparse,
                "inv_z_dense": float(1.0 / z_dense),
                "inv_z_sparse": float(1.0 / z_sparse),
                "res_z": float(z_sparse - z_dense),
                "res_invz": float((1.0 / z_sparse) - (1.0 / z_dense)),
                "res_xyz_norm": float(np.linalg.norm(res_xyz)),
                "dense_conf": float(depth_conf[frame_idx, v, u]),
                "dense_depth_edge": float(depth_edge[v, u]),
                "sparse_world": sparse_world.astype(np.float32),
                "sparse_cam": sparse_cam.astype(np.float32),
                "dense_cam": dense_cam.astype(np.float32),
                "rgb": image_rgb[v, u].astype(np.uint8),
            }
            observations.append(obs)
            frame_observations[frame_idx].append(obs)

    return observations, frame_observations


def export_dense_debug_outputs(
    output_dir,
    predictions,
    images,
    image_names,
    track,
    points_id,
    valid_track_mask=None,
    dense_max_points=2_000_000,
):
    if track is None or points_id is None or "points" not in predictions or "local_points" not in predictions:
        print("[DENSE DEBUG] Missing sparse/dense inputs; skipping dense debug export.")
        return None

    debug_dir = os.path.join(output_dir, "dense_debug")
    heatmap_dir = os.path.join(debug_dir, "sparse_dense_residual_heatmap")
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    observations, frame_observations = build_sparse_dense_anchor_diagnostics(
        predictions,
        images,
        track,
        points_id,
        valid_track_mask=valid_track_mask,
    )

    local_points = _to_numpy(predictions["local_points"])
    height, width = local_points.shape[1:3]

    all_res_z = np.asarray([obs["res_z"] for obs in observations], dtype=np.float64)
    all_res_invz = np.asarray([obs["res_invz"] for obs in observations], dtype=np.float64)
    all_res_xyz_norm = np.asarray([obs["res_xyz_norm"] for obs in observations], dtype=np.float64)
    all_conf = np.asarray([obs["dense_conf"] for obs in observations], dtype=np.float64)
    all_z_dense = np.asarray([obs["z_dense"] for obs in observations], dtype=np.float64)
    all_depth_edge = np.asarray([obs["dense_depth_edge"] for obs in observations], dtype=np.float64)

    frame_stats = []
    for frame_idx, obs_list in enumerate(frame_observations):
        res_z = np.asarray([obs["res_z"] for obs in obs_list], dtype=np.float64)
        res_invz = np.asarray([obs["res_invz"] for obs in obs_list], dtype=np.float64)
        res_xyz_norm = np.asarray([obs["res_xyz_norm"] for obs in obs_list], dtype=np.float64)
        frame_stats.append({
            "frame_idx": frame_idx,
            "image_name": os.path.basename(str(image_names[frame_idx])) if frame_idx < len(image_names) else str(frame_idx),
            "num_valid_anchors": int(len(obs_list)),
            "res_z": _summary(res_z),
            "res_invz": _summary(res_invz),
            "res_xyz_norm": _summary(res_xyz_norm),
        })

        uv = [(obs["u"], obs["v"]) for obs in obs_list]
        _save_sparse_heatmap(
            os.path.join(heatmap_dir, f"{frame_idx:05d}_res_z_abs.png"),
            height,
            width,
            uv,
            res_z,
        )
        _save_sparse_heatmap(
            os.path.join(heatmap_dir, f"{frame_idx:05d}_res_invz_abs.png"),
            height,
            width,
            uv,
            res_invz,
        )

    stats = {
        "num_frames": int(len(frame_observations)),
        "num_observations": int(len(observations)),
        "num_frames_with_anchors": int(sum(len(x) > 0 for x in frame_observations)),
        "res_z": _summary(all_res_z),
        "res_invz": _summary(all_res_invz),
        "res_xyz_norm": _summary(all_res_xyz_norm),
        "res_z_by_dense_conf_quantile": _quantile_bins(all_conf, all_res_z),
        "res_z_by_dense_depth_quantile": _quantile_bins(all_z_dense, all_res_z),
        "res_z_by_dense_depth_edge_quantile": _quantile_bins(all_depth_edge, all_res_z),
        "frames": frame_stats,
    }
    _write_json(os.path.join(debug_dir, "sparse_dense_residual_stats.json"), stats)

    camera_stats = {
        "num_frames": stats["num_frames"],
        "num_observations": stats["num_observations"],
        "frames": [
            {
                "frame_idx": item["frame_idx"],
                "image_name": item["image_name"],
                "num_valid_anchors": item["num_valid_anchors"],
                "z_res_median": item["res_z"]["median"],
                "z_res_mad": item["res_z"]["mad"],
                "invz_res_median": item["res_invz"]["median"],
                "xyz_res_norm_median": item["res_xyz_norm"]["median"],
            }
            for item in frame_stats
        ],
    }
    _write_json(os.path.join(debug_dir, "sparse_anchor_points_cam_stats.json"), camera_stats)

    sparse_points = np.asarray([obs["sparse_world"] for obs in observations], dtype=np.float32)
    colors = np.asarray([obs["rgb"] for obs in observations], dtype=np.uint8)
    _write_point_cloud(os.path.join(debug_dir, "sparse_ba_points.ply"), sparse_points, colors)

    dense_anchor_world = []
    overlay_points = []
    overlay_colors = []
    extrinsic = _to_numpy(predictions["extrinsic"]).astype(np.float32)
    for obs in observations:
        frame_idx = obs["frame_idx"]
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :4] = _as_w2c_3x4(extrinsic[frame_idx])
        c2w = np.linalg.inv(w2c)
        dense_cam = obs["dense_cam"]
        dense_world = dense_cam @ c2w[:3, :3].T + c2w[:3, 3]
        dense_anchor_world.append(dense_world)
        overlay_points.append(obs["sparse_world"])
        overlay_colors.append(np.asarray([255, 40, 40], dtype=np.uint8))
        overlay_points.append(dense_world)
        overlay_colors.append(np.asarray([40, 160, 255], dtype=np.uint8))

    _write_point_cloud(
        os.path.join(debug_dir, "dense_points_at_sparse_pixels.ply"),
        np.asarray(dense_anchor_world, dtype=np.float32),
        colors,
    )
    _write_point_cloud(
        os.path.join(debug_dir, "sparse_anchor_overlay.ply"),
        np.asarray(overlay_points, dtype=np.float32),
        np.asarray(overlay_colors, dtype=np.uint8),
    )

    export_dense_local_point_map_ply(
        os.path.join(debug_dir, "dense_before_correction.ply"),
        predictions["local_points"],
        predictions["extrinsic"],
        images,
        predictions["depth_conf"],
        conf_threshold=50.0,
        stride=1,
        max_points=dense_max_points,
    )

    print(
        "[DENSE DEBUG] Exported dense diagnostics to "
        f"{debug_dir} with {stats['num_observations']} sparse-dense anchors."
    )
    return stats
