import numpy as np

from algos.geometry import project_3d_points_to_image_numpy


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "n=0"
    return (
        f"n={values.size}, min={values.min():.3f}, p10={np.percentile(values, 10):.3f}, "
        f"median={np.median(values):.3f}, mean={values.mean():.3f}, "
        f"p90={np.percentile(values, 90):.3f}, max={values.max():.3f}"
    )


def _normalize_extrinsics(extrinsic):
    extrinsic = _to_numpy(extrinsic)
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected extrinsic shape (N, 3, 4) or (N, 4, 4), got {extrinsic.shape}")
    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic[:, :3, :4]
    return extrinsic


def _build_point_frames(points_id, num_points):
    point_frames = [[] for _ in range(num_points)]
    for frame_idx, point_ids_i in enumerate(points_id):
        point_ids_i = np.asarray(point_ids_i).reshape(-1)
        for point_id in point_ids_i:
            point_id = int(point_id)
            if 0 <= point_id < num_points:
                point_frames[point_id].append(frame_idx)
    return point_frames


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.parent[root_b] = root_a


def _camera_graph_stats(point_frames, num_frames):
    uf = _UnionFind(num_frames)
    degrees = np.zeros(num_frames, dtype=np.int64)
    edges = set()
    for frames in point_frames:
        unique_frames = sorted(set(frames))
        if len(unique_frames) < 2:
            continue
        base = unique_frames[0]
        for frame in unique_frames[1:]:
            uf.union(base, frame)
        for idx, frame_a in enumerate(unique_frames):
            for frame_b in unique_frames[idx + 1 :]:
                edge = (frame_a, frame_b)
                if edge not in edges:
                    edges.add(edge)
                    degrees[frame_a] += 1
                    degrees[frame_b] += 1

    roots = [uf.find(i) for i in range(num_frames) if degrees[i] > 0]
    if roots:
        component_sizes = np.asarray(
            [roots.count(root) for root in sorted(set(roots))],
            dtype=np.int64,
        )
        num_components = component_sizes.size
        largest_component = int(component_sizes.max())
    else:
        num_components = 0
        largest_component = 0

    return {
        "num_edges": len(edges),
        "num_components": num_components,
        "largest_component": largest_component,
        "degrees": degrees,
    }


def _reprojection_errors(track, points_id, points_3d, extrinsic, intrinsic):
    errors = []
    positive_depth = []
    extrinsic = _normalize_extrinsics(extrinsic)
    intrinsic = _to_numpy(intrinsic)
    points_3d = _to_numpy(points_3d)

    for frame_idx, (track_i, point_ids_i) in enumerate(zip(track, points_id)):
        track_i = np.asarray(track_i, dtype=np.float64).reshape(-1, 2)
        point_ids_i = np.asarray(point_ids_i).reshape(-1)
        count = min(len(track_i), len(point_ids_i))
        if count == 0:
            continue

        valid_ids = point_ids_i[:count].astype(np.int64)
        valid_mask = (valid_ids >= 0) & (valid_ids < len(points_3d))
        if not np.any(valid_mask):
            continue

        obs_xy = track_i[:count][valid_mask]
        obs_points = points_3d[valid_ids[valid_mask]]
        proj_xy, depth_mask = project_3d_points_to_image_numpy(
            obs_points,
            extrinsic[frame_idx, :3, :3],
            extrinsic[frame_idx, :3, 3:],
            intrinsic[frame_idx],
        )
        residual = np.linalg.norm(proj_xy - obs_xy, axis=-1)
        residual = residual[np.isfinite(residual)]
        if residual.size:
            errors.append(residual)
        positive_depth.append(depth_mask.astype(np.float32))

    if errors:
        errors = np.concatenate(errors)
    else:
        errors = np.empty((0,), dtype=np.float64)
    if positive_depth:
        positive_depth = np.concatenate(positive_depth)
    else:
        positive_depth = np.empty((0,), dtype=np.float32)
    return errors, positive_depth


def filter_tracks_by_reprojection(
    track,
    points_id,
    points_3d,
    points_conf,
    extrinsic,
    intrinsic,
    max_reproj_error=16.0,
    min_track_length=2,
    label="Tracking",
):
    extrinsic = _normalize_extrinsics(extrinsic)
    intrinsic = _to_numpy(intrinsic)
    points_3d = _to_numpy(points_3d)
    points_conf = None if points_conf is None else _to_numpy(points_conf)

    num_frames = len(track)
    num_points = len(points_3d)
    kept_observations_by_point = [[] for _ in range(num_points)]
    total_observations = 0
    kept_observations = 0

    for frame_idx, (track_i, point_ids_i) in enumerate(zip(track, points_id)):
        track_i = np.asarray(track_i, dtype=np.float64).reshape(-1, 2)
        point_ids_i = np.asarray(point_ids_i).reshape(-1)
        count = min(len(track_i), len(point_ids_i))
        total_observations += count
        if count == 0:
            continue

        point_ids_i = point_ids_i[:count].astype(np.int64)
        for obs_idx, point_id in enumerate(point_ids_i):
            if point_id < 0 or point_id >= num_points:
                continue
            point = points_3d[point_id : point_id + 1]
            proj_xy, depth_mask = project_3d_points_to_image_numpy(
                point,
                extrinsic[frame_idx, :3, :3],
                extrinsic[frame_idx, :3, 3:],
                intrinsic[frame_idx],
            )
            residual = float(np.linalg.norm(proj_xy[0] - track_i[obs_idx]))
            if np.isfinite(residual) and depth_mask[0] and residual <= max_reproj_error:
                kept_observations_by_point[point_id].append((frame_idx, track_i[obs_idx].astype(np.float32)))
                kept_observations += 1

    final_track = [[] for _ in range(num_frames)]
    final_points_id = [[] for _ in range(num_frames)]
    final_points = []
    final_points_conf = []
    kept_points = 0

    for old_point_id, observations in enumerate(kept_observations_by_point):
        unique_frames = {frame_idx for frame_idx, _ in observations}
        if len(unique_frames) < min_track_length:
            continue

        new_point_id = len(final_points)
        for frame_idx, xy in observations:
            final_track[frame_idx].append(xy)
            final_points_id[frame_idx].append(new_point_id)

        final_points.append(points_3d[old_point_id].astype(np.float32))
        if points_conf is not None and old_point_id < len(points_conf):
            final_points_conf.append(points_conf[old_point_id])
        kept_points += 1

    final_track = [np.stack(track_i).astype(np.float32) if track_i else np.array([]) for track_i in final_track]
    final_points_id = [np.stack(ids_i).astype(np.int64) if ids_i else np.array([]) for ids_i in final_points_id]
    if final_points:
        final_points = np.stack(final_points).astype(np.float32)
    else:
        final_points = np.empty((0, 3), dtype=np.float32)
    if final_points_conf:
        final_points_conf = np.asarray(final_points_conf, dtype=np.float32)
    elif points_conf is None:
        final_points_conf = None
    else:
        final_points_conf = np.empty((0,), dtype=np.float32)

    print(
        f"[TRACKING_FILTER][{label}] max_reproj_error={max_reproj_error}, "
        f"min_track_length={min_track_length}, "
        f"observations={total_observations}->{kept_observations}, "
        f"points={num_points}->{kept_points}"
    )
    return final_track, final_points_id, final_points, final_points_conf


def print_tracking_metrics(
    label,
    track,
    points_id,
    points_3d,
    extrinsic,
    intrinsic,
    points_conf=None,
    extra_stats=None,
):
    num_frames = len(track)
    num_points = len(points_3d)
    obs_counts = np.asarray([np.asarray(track_i).reshape(-1, 2).shape[0] for track_i in track], dtype=np.int64)
    num_observations = int(obs_counts.sum())
    frames_with_obs = int(np.count_nonzero(obs_counts))
    point_frames = _build_point_frames(points_id, num_points)
    track_lengths = np.asarray([len(set(frames)) for frames in point_frames], dtype=np.int64)
    graph_stats = _camera_graph_stats(point_frames, num_frames)
    reproj_errors, positive_depth = _reprojection_errors(track, points_id, points_3d, extrinsic, intrinsic)

    print(f"[TRACKING_METRICS][{label}] frames={num_frames}, frames_with_obs={frames_with_obs}, dropped_frames={num_frames - frames_with_obs}")
    print(f"[TRACKING_METRICS][{label}] points={num_points}, observations={num_observations}")
    print(f"[TRACKING_METRICS][{label}] observations_per_frame: {_stats(obs_counts)}")
    print(f"[TRACKING_METRICS][{label}] track_length: {_stats(track_lengths)}")
    if track_lengths.size:
        print(
            f"[TRACKING_METRICS][{label}] track_length_ratios: "
            f"len2={np.mean(track_lengths == 2):.3f}, "
            f"len>=3={np.mean(track_lengths >= 3):.3f}, "
            f"len>=5={np.mean(track_lengths >= 5):.3f}, "
            f"len>=10={np.mean(track_lengths >= 10):.3f}"
        )
    print(
        f"[TRACKING_METRICS][{label}] camera_graph: edges={graph_stats['num_edges']}, "
        f"components={graph_stats['num_components']}, largest_component={graph_stats['largest_component']}, "
        f"degree_stats={_stats(graph_stats['degrees'])}"
    )
    print(f"[TRACKING_METRICS][{label}] initial_reproj_px: {_stats(reproj_errors)}")
    if reproj_errors.size:
        print(
            f"[TRACKING_METRICS][{label}] reproj_outlier_ratios: "
            f">2px={np.mean(reproj_errors > 2.0):.3f}, "
            f">4px={np.mean(reproj_errors > 4.0):.3f}, "
            f">8px={np.mean(reproj_errors > 8.0):.3f}, "
            f">16px={np.mean(reproj_errors > 16.0):.3f}"
        )
    if positive_depth.size:
        print(f"[TRACKING_METRICS][{label}] positive_depth_ratio={np.mean(positive_depth > 0):.3f}")
    if points_conf is not None:
        print(f"[TRACKING_METRICS][{label}] points_conf: {_stats(points_conf)}")
    if extra_stats:
        for key, value in extra_stats.items():
            print(f"[TRACKING_METRICS][{label}] {key}: {value}")
