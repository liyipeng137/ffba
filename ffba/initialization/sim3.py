import numpy as np
from numba import njit


def compute_alignment_error(
    point_map1, conf1, point_map2, conf2, conf_threshold, s, R, t
):
    """
    Compute the average point alignment error (using only original inputs)

    Args:
    point_map1: target point map (b, h, w, 3)
    conf1: target confidence map (b, h, w)
    point_map2: source point map (b, h, w, 3)
    conf2: source confidence map (b, h, w)
    conf_threshold: confidence threshold
    s, R, t: transformation parameters
    """
    b1, h1, w1, _ = point_map1.shape
    b2, h2, w2, _ = point_map2.shape
    b = min(b1, b2)
    h = min(h1, h2)
    w = min(w1, w2)

    target_points = []
    source_points = []

    for i in range(b):
        mask1 = conf1[i, :h, :w] > conf_threshold
        mask2 = conf2[i, :h, :w] > conf_threshold
        valid_mask = mask1 & mask2

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        t_pts = point_map1[i, :h, :w][idx]
        s_pts = point_map2[i, :h, :w][idx]

        target_points.append(t_pts)
        source_points.append(s_pts)

    if len(target_points) == 0:
        print("Warning: No matching point pairs found for error calculation")
        return np.nan

    all_target = np.concatenate(target_points, axis=0)
    all_source = np.concatenate(source_points, axis=0)

    transformed = (s * (R @ all_source.T)).T + t

    errors = np.linalg.norm(transformed - all_target, axis=1)

    mean_error = np.mean(errors)
    std_error = np.std(errors)
    median_error = np.median(errors)
    max_error = np.max(errors)

    print(
        f"Alignment error statistics [using {len(errors)} points]: "
        f"mean={mean_error:.4f}, std={std_error:.4f}, "
        f"median={median_error:.4f}, max={max_error:.4f}"
    )

    return mean_error


def weighted_estimate_sim3(source_points, target_points, weights):
    """
    source_points:  (Nx3)
    target_points:  (Nx3)
    :weights:  (N,) [0,1]
    """
    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        raise ValueError("Total weight too small for meaningful estimation")

    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
    scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
    s = scale_tgt / scale_src

    weighted_src = (s * src_centered) * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]

    H = weighted_src.T @ weighted_tgt

    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


def huber_loss(r, delta):
    abs_r = np.abs(r)
    return np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))


def robust_weighted_estimate_sim3(
    src, tgt, init_weights, delta=0.1, max_iters=20, tol=1e-9
):
    """
    src:  (Nx3)
    tgt:  (Nx3)
    init_weights:  (N,)
    """
    s, R, t = weighted_estimate_sim3(src, tgt, init_weights)
    prev_error = float("inf")

    for iter in range(max_iters):
        transformed = s * (src @ R.T) + t
        residuals = np.linalg.norm(tgt - transformed, axis=1)  # (N,)
        print(f"Residuals: {np.mean(residuals)}")

        abs_res = np.abs(residuals)
        huber_weights = np.ones_like(residuals)
        large_res_mask = abs_res > delta
        huber_weights[large_res_mask] = delta / abs_res[large_res_mask]

        combined_weights = init_weights * huber_weights

        combined_weights /= np.sum(combined_weights) + 1e-12

        s_new, R_new, t_new = weighted_estimate_sim3(src, tgt, combined_weights)

        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1) / 2)))
        current_error = np.sum(huber_loss(residuals, delta) * init_weights)

        if (param_change < tol and rot_angle < np.radians(0.1)) or (
            abs(prev_error - current_error) < tol * prev_error
        ):
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error

    return s, R, t


@njit(cache=True)
def _weighted_estimate_sim3_numba(source_points, target_points, weights):
    # Ensure float32
    source_points = source_points.astype(np.float32)
    target_points = target_points.astype(np.float32)
    weights = weights.astype(np.float32)

    total_weight = np.sum(weights)
    if total_weight < 1e-6:
        return (
            -1.0,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros((3, 3), dtype=np.float32),
        )

    normalized_weights = weights / total_weight

    mu_src = np.sum(normalized_weights[:, None] * source_points, axis=0)
    mu_tgt = np.sum(normalized_weights[:, None] * target_points, axis=0)

    src_centered = source_points - mu_src
    tgt_centered = target_points - mu_tgt

    scale_src = np.sqrt(np.sum(normalized_weights * np.sum(src_centered**2, axis=1)))
    scale_tgt = np.sqrt(np.sum(normalized_weights * np.sum(tgt_centered**2, axis=1)))
    s = scale_tgt / scale_src

    weighted_src = (s * src_centered) * np.sqrt(normalized_weights)[:, None]
    weighted_tgt = tgt_centered * np.sqrt(normalized_weights)[:, None]

    H = weighted_src.T @ weighted_tgt

    return s, mu_src, mu_tgt, H


def weighted_estimate_sim3_numba(source_points, target_points, weights):
    s, mu_src, mu_tgt, H = _weighted_estimate_sim3_numba(
        source_points, target_points, weights
    )

    if s < 0:
        raise ValueError("Total weight too small for meaningful estimation")

    # Ensure float32
    H = H.astype(np.float32)
    U, _, Vt = np.linalg.svd(H.astype(np.float32))  # float32 SVD

    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_tgt - s * R @ mu_src
    return s, R, t


@njit(cache=True)
def huber_loss_numba(r, delta):
    r = r.astype(np.float32)
    delta = np.float32(delta)
    abs_r = np.abs(r)
    result = np.where(abs_r <= delta, 0.5 * r**2, delta * (abs_r - 0.5 * delta))
    return result.astype(np.float32)


@njit(cache=True)
def compute_residuals_numba(tgt, transformed):
    residuals = np.empty(tgt.shape[0], dtype=np.float32)
    for i in range(tgt.shape[0]):
        diff = tgt[i] - transformed[i]
        residuals[i] = np.sqrt(np.sum(diff**2))
    return residuals


@njit(cache=True)
def compute_huber_weights_numba(residuals, delta):
    weights = np.ones(residuals.shape, dtype=np.float32)
    for i in range(residuals.shape[0]):
        r = residuals[i]
        if r > delta:
            weights[i] = delta / r
    return weights


@njit(cache=True)
def apply_transformation_numba(src, s, R, t):
    transformed = np.empty_like(src)
    for i in range(src.shape[0]):
        p = src[i]
        transformed[i] = s * (R @ p) + t
    return transformed


def robust_weighted_estimate_sim3_numba(
    src, tgt, init_weights, delta=0.1, max_iters=20, tol=1e-9
):
    src = src.astype(np.float32)
    tgt = tgt.astype(np.float32)
    init_weights = init_weights.astype(np.float32)

    s, R, t = weighted_estimate_sim3_numba(src, tgt, init_weights)

    prev_error = float("inf")

    for iter in range(max_iters):
        transformed = apply_transformation_numba(src, s, R, t)
        residuals = compute_residuals_numba(tgt, transformed)

        print(f"Residuals: {np.mean(residuals)}")

        huber_weights = compute_huber_weights_numba(residuals, delta)
        combined_weights = init_weights * huber_weights
        combined_weights /= np.sum(combined_weights) + 1e-12

        s_new, R_new, t_new = weighted_estimate_sim3_numba(src, tgt, combined_weights)

        param_change = np.abs(s_new - s) + np.linalg.norm(t_new - t)
        rot_angle = np.arccos(min(1.0, max(-1.0, (np.trace(R_new @ R.T) - 1) / 2)))

        current_error = np.sum(huber_loss_numba(residuals, delta) * init_weights)

        if (param_change < tol and rot_angle < np.radians(0.1)) or (
            abs(prev_error - current_error) < tol * prev_error
        ):
            break

        s, R, t = s_new, R_new, t_new
        prev_error = current_error

    return s, R, t


def weighted_align_point_maps(
    point_map1,
    conf1,
    point_map2,
    conf2,
    conf_threshold,
    delta=0.1,
    max_iters=5,
    tol=1e-9,
    align_method="numba",
):
    """point_map2 -> point_map1"""
    b1, _, _, _ = point_map1.shape
    b2, _, _, _ = point_map2.shape
    b = min(b1, b2)

    aligned_points1 = []
    aligned_points2 = []
    confidence_weights = []

    for i in range(b):
        mask1 = conf1[i] > conf_threshold
        mask2 = conf2[i] > conf_threshold
        valid_mask = mask1 & mask2

        idx = np.where(valid_mask)
        if len(idx[0]) == 0:
            continue

        pts1 = point_map1[i][idx]
        pts2 = point_map2[i][idx]

        combined_conf = np.sqrt(conf1[i][idx] * conf2[i][idx])

        aligned_points1.append(pts1)
        aligned_points2.append(pts2)
        confidence_weights.append(combined_conf)

    if len(aligned_points1) == 0:
        raise ValueError("No matching point pairs were found!")

    all_pts1 = np.concatenate(aligned_points1, axis=0)
    all_pts2 = np.concatenate(aligned_points2, axis=0)
    all_weights = np.concatenate(confidence_weights, axis=0)

    print(f"The number of corresponding points matched: {all_pts1.shape[0]}")

    if align_method == "numba":
        s, R, t = robust_weighted_estimate_sim3_numba(
            all_pts2, all_pts1, all_weights, delta=delta, max_iters=max_iters, tol=tol
        )
    else:  # numpy
        s, R, t = robust_weighted_estimate_sim3(
            all_pts2, all_pts1, all_weights, delta=delta, max_iters=max_iters, tol=tol
        )

    mean_error = compute_alignment_error(
        point_map1, conf1, point_map2, conf2, conf_threshold, s, R, t
    )
    print(f"Mean error: {mean_error}")

    return s, R, t
