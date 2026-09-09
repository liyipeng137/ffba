"""reporting / tracks for the formal SIFT + prior + BAE pipeline."""

import numpy as np


def classify_point3d_track_source(point3d, s_keypoint_count):
    has_s_observation = False
    has_p_observation = False
    for elem in point3d.track.elements:
        if int(elem.point2D_idx) < s_keypoint_count.get(elem.image_id, 0):
            has_s_observation = True
        else:
            has_p_observation = True

    if has_s_observation and has_p_observation:
        return "mixed"
    if has_s_observation:
        return "s_only"
    if has_p_observation:
        return "p_only"
    return "empty"


def build_s_keypoint_count(reconstruction, features):
    return {
        image_id: int(features[image_id - 1]["keypoints"].shape[0])
        for image_id in reconstruction.images
    }


def classify_tracks_by_s_keypoints(reconstruction, s_keypoint_count):
    counts = {"total": 0, "s": 0, "non_s": 0, "mixed": 0}
    for point3d in reconstruction.points3D.values():
        source = classify_point3d_track_source(point3d, s_keypoint_count)
        if source == "empty":
            continue
        counts["total"] += 1
        if source == "s_only":
            counts["s"] += 1
        elif source == "mixed":
            counts["mixed"] += 1
        else:
            counts["non_s"] += 1
    return counts


def summarize_angular_errors_by_track_source(
    reconstruction,
    features,
    error_threshold,
    return_errors_per_track=False,
):
    if reconstruction is None:
        return {"enabled": False, "reason": "missing reconstruction"}

    from gluemap.math.reprojection_error import (  # noqa: PLC0415
        ReprojectionErrorType,
        compute_all_errors_from_reconstruction,
    )

    s_keypoint_count = build_s_keypoint_count(reconstruction, features)
    errors_per_track = compute_all_errors_from_reconstruction(
        reconstruction,
        ReprojectionErrorType.ANGULAR,
        negative_depth_observations={},
    )
    bucket_order = ("s_only", "p_only", "mixed")
    buckets = {
        source: {
            "track_count": 0,
            "track_observation_count": 0,
            "error_observation_count": 0,
            "finite_error_count": 0,
            "nonfinite_error_count": 0,
            "min": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "lt_threshold_count": 0,
            "lt_threshold_ratio": 0.0,
        }
        for source in bucket_order
    }
    finite_errors_by_bucket = {source: [] for source in bucket_order}

    for point3D_id, point3d in reconstruction.points3D.items():
        source = classify_point3d_track_source(point3d, s_keypoint_count)
        if source == "empty":
            continue

        bucket = buckets[source]
        bucket["track_count"] += 1
        bucket["track_observation_count"] += len(list(point3d.track.elements))

        track_errors = errors_per_track.get(point3D_id, [])
        bucket["error_observation_count"] += len(track_errors)
        for _, _, error in track_errors:
            if np.isfinite(error):
                finite_errors_by_bucket[source].append(float(error))
            else:
                bucket["nonfinite_error_count"] += 1

    for source, finite_errors in finite_errors_by_bucket.items():
        bucket = buckets[source]
        values = np.asarray(finite_errors, dtype=np.float64)
        bucket["finite_error_count"] = int(values.size)
        if values.size == 0:
            continue
        bucket["min"] = float(values.min())
        bucket["median"] = float(np.median(values))
        bucket["mean"] = float(values.mean())
        bucket["p90"] = float(np.percentile(values, 90))
        bucket["max"] = float(values.max())
        bucket["lt_threshold_count"] = int(np.sum(values < error_threshold))
        bucket["lt_threshold_ratio"] = float(bucket["lt_threshold_count"] / values.size)

    result = {
        "enabled": True,
        "error_type": "angular",
        "error_threshold": float(error_threshold),
        "buckets": buckets,
    }
    if return_errors_per_track:
        return result, errors_per_track
    return result


def log_angular_errors_by_track_source(args, iteration, stats):
    from ffba.runtime import debug

    if not stats.get("enabled"):
        debug(
            args,
            "Angular errors by track source skipped: "
            f"{stats.get('reason', 'unknown reason')}",
        )
        return

    threshold = stats["error_threshold"]
    labels = {
        "s_only": "S-only",
        "p_only": "P-only",
        "mixed": "mixed",
    }
    for source in ("s_only", "p_only", "mixed"):
        bucket = stats["buckets"][source]
        if bucket["finite_error_count"] > 0:
            error_summary = (
                f"mean={bucket['mean']:.4f}, "
                f"median={bucket['median']:.4f}, "
                f"p90={bucket['p90']:.4f}, "
                f"max={bucket['max']:.4f}, "
                f"<{threshold:g}deg="
                f"{bucket['lt_threshold_ratio'] * 100:.1f}%"
            )
        else:
            error_summary = (
                f"mean=n/a, median=n/a, p90=n/a, max=n/a, <{threshold:g}deg=n/a"
            )

        debug(
            args,
            "Angular errors by track source "
            f"iter={iteration}, bucket={labels[source]}: "
            f"tracks={bucket['track_count']}, "
            f"track_obs={bucket['track_observation_count']}, "
            f"error_obs={bucket['error_observation_count']}, "
            f"finite={bucket['finite_error_count']}, "
            f"nonfinite={bucket['nonfinite_error_count']}, "
            f"{error_summary}",
        )


def count_reconstruction_observations(reconstruction):
    if reconstruction is None:
        return 0
    return int(
        sum(
            len(list(point3d.track.elements))
            for point3d in reconstruction.points3D.values()
        )
    )


def summarize_reconstruction(reconstruction):
    if reconstruction is None:
        return {"points3D": 0, "observations": 0}
    return {
        "points3D": int(len(reconstruction.points3D)),
        "observations": count_reconstruction_observations(reconstruction),
    }


def summarize_ba_solver_result(summary):
    """Convert a backend-specific BA summary into JSON-compatible stats."""
    if summary is None or isinstance(summary, dict):
        return summary

    result = {"type": type(summary).__name__}
    scalar_attributes = (
        "initial_cost",
        "final_cost",
        "fixed_cost",
        "num_successful_steps",
        "num_unsuccessful_steps",
        "num_inner_iteration_steps",
        "preprocessor_time_in_seconds",
        "minimizer_time_in_seconds",
        "postprocessor_time_in_seconds",
        "total_time_in_seconds",
        "message",
    )
    for name in scalar_attributes:
        if not hasattr(summary, name):
            continue
        value = getattr(summary, name)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value

    if hasattr(summary, "termination_type"):
        result["termination_type"] = str(summary.termination_type)

    for output_name, method_name in (
        ("brief_report", "BriefReport"),
        ("full_report", "FullReport"),
    ):
        method = getattr(summary, method_name, None)
        if callable(method):
            result[output_name] = str(method())

    return result
