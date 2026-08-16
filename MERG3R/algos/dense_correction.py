import json
import os

import numpy as np

from algos.dense_debug import build_sparse_dense_anchor_diagnostics


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
    median = np.median(values)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(median),
        "mad": float(np.median(np.abs(values - median))),
        "p90_abs": float(np.percentile(abs_values, 90)),
        "p95_abs": float(np.percentile(abs_values, 95)),
        "max_abs": float(np.max(abs_values)),
    }


def _weighted_lstsq_affine(x, y, weights):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 1e-8, None)

    design = np.stack([x, np.ones_like(x)], axis=1)
    sqrt_w = np.sqrt(weights)[:, None]
    lhs = design * sqrt_w
    rhs = y * sqrt_w[:, 0]
    scale, bias = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return float(scale), float(bias)


def _robust_fit_inverse_depth_affine(
    obs_list,
    min_anchors,
    conf_quantile,
    edge_quantile,
    mad_k,
    alpha,
    scale_min,
    scale_max,
    bias_abs_max,
):
    if len(obs_list) < min_anchors:
        return None, {
            "status": "skipped_too_few_anchors",
            "num_anchors": int(len(obs_list)),
            "num_fit_anchors": 0,
        }

    invz_dense = np.asarray([obs["inv_z_dense"] for obs in obs_list], dtype=np.float64)
    invz_sparse = np.asarray([obs["inv_z_sparse"] for obs in obs_list], dtype=np.float64)
    conf = np.asarray([obs["dense_conf"] for obs in obs_list], dtype=np.float64)
    edge = np.asarray([obs["dense_depth_edge"] for obs in obs_list], dtype=np.float64)

    finite = (
        np.isfinite(invz_dense)
        & np.isfinite(invz_sparse)
        & np.isfinite(conf)
        & np.isfinite(edge)
        & (invz_dense > 0)
        & (invz_sparse > 0)
    )
    if np.count_nonzero(finite) < min_anchors:
        return None, {
            "status": "skipped_too_few_finite_anchors",
            "num_anchors": int(len(obs_list)),
            "num_fit_anchors": int(np.count_nonzero(finite)),
        }

    fit_mask = finite.copy()
    if conf_quantile > 0:
        conf_threshold = np.quantile(conf[finite], conf_quantile)
        fit_mask &= conf >= conf_threshold
    if edge_quantile < 1:
        edge_threshold = np.quantile(edge[finite], edge_quantile)
        fit_mask &= edge <= edge_threshold

    residual = invz_sparse - invz_dense
    finite_residual = residual[fit_mask & np.isfinite(residual)]
    if finite_residual.size >= min_anchors:
        median_residual = np.median(finite_residual)
        mad = np.median(np.abs(finite_residual - median_residual))
        robust_sigma = max(1.4826 * mad, 1e-6)
        fit_mask &= np.abs(residual - median_residual) <= mad_k * robust_sigma

    if np.count_nonzero(fit_mask) < min_anchors:
        return None, {
            "status": "skipped_too_few_robust_anchors",
            "num_anchors": int(len(obs_list)),
            "num_fit_anchors": int(np.count_nonzero(fit_mask)),
        }
    if np.std(invz_dense[fit_mask]) < 1e-6:
        return None, {
            "status": "skipped_degenerate_depth_range",
            "num_anchors": int(len(obs_list)),
            "num_fit_anchors": int(np.count_nonzero(fit_mask)),
        }

    fit_conf = conf[fit_mask]
    conf_scale = np.median(fit_conf[np.isfinite(fit_conf) & (fit_conf > 0)])
    if not np.isfinite(conf_scale) or conf_scale <= 0:
        conf_scale = 1.0
    weights = np.clip(fit_conf / conf_scale, 0.25, 4.0)

    fit_edge = edge[fit_mask]
    edge_scale = np.percentile(fit_edge[np.isfinite(fit_edge)], 75) if np.any(np.isfinite(fit_edge)) else 0.0
    if np.isfinite(edge_scale) and edge_scale > 1e-8:
        weights *= 1.0 / (1.0 + fit_edge / edge_scale)

    scale, bias = _weighted_lstsq_affine(invz_dense[fit_mask], invz_sparse[fit_mask], weights)
    raw_scale, raw_bias = scale, bias
    scale = float(np.clip(scale, scale_min, scale_max))
    bias = float(np.clip(bias, -bias_abs_max, bias_abs_max))

    fitted_before = invz_dense[fit_mask]
    target = invz_sparse[fit_mask]
    fitted_after = (1.0 - alpha) * fitted_before + alpha * (scale * fitted_before + bias)
    before_residual = target - fitted_before
    after_residual = target - fitted_after
    before_summary = _summary(before_residual)
    after_summary = _summary(after_residual)

    status = "ok"
    if raw_scale != scale or raw_bias != bias:
        status = "clamped"
        before_p95 = before_summary["p95_abs"]
        after_p95 = after_summary["p95_abs"]
        before_median_abs = abs(before_summary["median"])
        after_median_abs = abs(after_summary["median"])
        if (
            before_p95 is not None
            and after_p95 is not None
            and after_p95 > before_p95
            and after_median_abs > before_median_abs
        ):
            return None, {
                "status": "skipped_clamped_worse_after_fit",
                "num_anchors": int(len(obs_list)),
                "num_fit_anchors": int(np.count_nonzero(fit_mask)),
                "scale": scale,
                "bias": bias,
                "raw_scale": float(raw_scale),
                "raw_bias": float(raw_bias),
                "before_res_invz": before_summary,
                "after_res_invz": after_summary,
            }

    return (scale, bias), {
        "status": status,
        "num_anchors": int(len(obs_list)),
        "num_fit_anchors": int(np.count_nonzero(fit_mask)),
        "scale": scale,
        "bias": bias,
        "raw_scale": float(raw_scale),
        "raw_bias": float(raw_bias),
        "before_res_invz": before_summary,
        "after_res_invz": after_summary,
    }


def _apply_inverse_depth_affine_to_frame(local_points_frame, scale, bias, alpha, depth_ratio_min, depth_ratio_max):
    local = np.asarray(local_points_frame, dtype=np.float32)
    corrected = local.copy()
    depth = local[..., 2]
    valid = np.isfinite(depth) & (depth > 1e-8)
    if not np.any(valid):
        return corrected, 0

    inv_depth = np.zeros_like(depth, dtype=np.float32)
    inv_depth[valid] = 1.0 / depth[valid]
    inv_corr = scale * inv_depth + bias
    inv_final = (1.0 - alpha) * inv_depth + alpha * inv_corr
    valid_corr = valid & np.isfinite(inv_final) & (inv_final > 1e-8)

    new_depth = depth.copy()
    new_depth[valid_corr] = 1.0 / inv_final[valid_corr]
    ratio = np.ones_like(depth, dtype=np.float32)
    ratio[valid_corr] = new_depth[valid_corr] / depth[valid_corr]
    ratio = np.clip(ratio, depth_ratio_min, depth_ratio_max)
    new_depth[valid_corr] = depth[valid_corr] * ratio[valid_corr]

    ray_xy = np.zeros_like(local[..., :2], dtype=np.float32)
    ray_xy[valid] = local[..., :2][valid] / depth[valid][:, None]
    corrected_xy = corrected[..., :2]
    corrected_xy[valid_corr] = ray_xy[valid_corr] * new_depth[valid_corr][:, None]
    corrected[..., :2] = corrected_xy
    corrected[..., 2][valid_corr] = new_depth[valid_corr]
    return corrected, int(np.count_nonzero(valid_corr))


def apply_inverse_depth_affine_correction(
    predictions,
    images,
    image_names,
    track,
    points_id,
    valid_track_mask=None,
    output_dir=None,
    min_anchors_per_frame=50,
    alpha=0.75,
    conf_quantile=0.2,
    edge_quantile=0.8,
    residual_mad_k=3.5,
    scale_min=0.5,
    scale_max=2.0,
    bias_abs_max=0.25,
    depth_ratio_min=0.7,
    depth_ratio_max=1.3,
):
    if track is None or points_id is None or "points" not in predictions or "local_points" not in predictions:
        print("[DENSE CORRECTION] Missing sparse/dense inputs; skipping inverse-depth affine correction.")
        return None

    local_points = np.asarray(predictions["local_points"], dtype=np.float32)
    corrected_local_points = local_points.copy()

    observations, frame_observations = build_sparse_dense_anchor_diagnostics(
        predictions,
        images,
        track,
        points_id,
        valid_track_mask=valid_track_mask,
    )

    frame_stats = []
    corrected_frames = 0
    skipped_frames = 0
    clamped_frames = 0
    corrected_pixels = 0

    for frame_idx, obs_list in enumerate(frame_observations):
        fit, stats = _robust_fit_inverse_depth_affine(
            obs_list,
            min_anchors=min_anchors_per_frame,
            conf_quantile=conf_quantile,
            edge_quantile=edge_quantile,
            mad_k=residual_mad_k,
            alpha=alpha,
            scale_min=scale_min,
            scale_max=scale_max,
            bias_abs_max=bias_abs_max,
        )
        stats["frame_idx"] = int(frame_idx)
        stats["image_name"] = os.path.basename(str(image_names[frame_idx])) if frame_idx < len(image_names) else str(frame_idx)

        if fit is None:
            skipped_frames += 1
            frame_stats.append(stats)
            continue

        scale, bias = fit
        corrected_frame, num_pixels = _apply_inverse_depth_affine_to_frame(
            corrected_local_points[frame_idx],
            scale,
            bias,
            alpha=alpha,
            depth_ratio_min=depth_ratio_min,
            depth_ratio_max=depth_ratio_max,
        )
        corrected_local_points[frame_idx] = corrected_frame
        corrected_frames += 1
        corrected_pixels += num_pixels
        if stats["status"] == "clamped":
            clamped_frames += 1
        stats["num_corrected_pixels"] = num_pixels
        frame_stats.append(stats)

    predictions["local_points"] = corrected_local_points
    predictions["depth"] = corrected_local_points[..., 2:3].astype(np.float32)

    before_all = []
    after_all = []
    status_counts = {}
    for item in frame_stats:
        status = item.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        before = item.get("before_res_invz")
        after = item.get("after_res_invz")
        if before and after:
            before_all.append(before["median"])
            after_all.append(after["median"])

    stats = {
        "method": "invz_affine",
        "num_frames": int(len(frame_observations)),
        "num_observations": int(len(observations)),
        "corrected_frames": int(corrected_frames),
        "skipped_frames": int(skipped_frames),
        "clamped_frames": int(clamped_frames),
        "corrected_pixels": int(corrected_pixels),
        "status_counts": status_counts,
        "params": {
            "min_anchors_per_frame": int(min_anchors_per_frame),
            "alpha": float(alpha),
            "conf_quantile": float(conf_quantile),
            "edge_quantile": float(edge_quantile),
            "residual_mad_k": float(residual_mad_k),
            "scale_min": float(scale_min),
            "scale_max": float(scale_max),
            "bias_abs_max": float(bias_abs_max),
            "depth_ratio_min": float(depth_ratio_min),
            "depth_ratio_max": float(depth_ratio_max),
        },
        "fit_frame_res_invz_median_before": _summary(before_all),
        "fit_frame_res_invz_median_after": _summary(after_all),
        "frames": frame_stats,
    }

    if output_dir is not None:
        correction_dir = os.path.join(output_dir, "dense_correction")
        os.makedirs(correction_dir, exist_ok=True)
        with open(os.path.join(correction_dir, "invz_affine_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    print(
        "[DENSE CORRECTION] invz_affine "
        f"corrected_frames={corrected_frames}, "
        f"skipped_frames={skipped_frames}, "
        f"clamped_frames={clamped_frames}, "
        f"observations={len(observations)}"
    )
    return stats
