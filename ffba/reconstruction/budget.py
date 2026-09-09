"""reconstruction / budget for the formal SIFT + prior + BAE pipeline."""

from collections import defaultdict
import numpy as np

BAE_MIN_OBSERVATIONS_PER_IMAGE = 64


def _camera_centers_from_reconstruction(reconstruction):
    centers = {}
    for image_id, image in reconstruction.images.items():
        pose = image.cam_from_world()
        rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(pose.translation, dtype=np.float64)
        centers[int(image_id)] = -(rotation.T @ translation)
    return centers


def _track_max_triangulation_angle_degrees(point_xyz, image_ids, camera_centers):
    point_xyz = np.asarray(point_xyz, dtype=np.float64)
    rays = []
    for image_id in image_ids:
        center = camera_centers.get(int(image_id))
        if center is None:
            continue
        ray = point_xyz - center
        norm = float(np.linalg.norm(ray))
        if not np.isfinite(norm) or norm <= 1e-12:
            continue
        rays.append(ray / norm)
    if len(rays) < 2:
        return 0.0

    rays = np.asarray(rays, dtype=np.float64)
    cosine = rays @ rays.T
    upper = cosine[np.triu_indices(len(rays), k=1)]
    if upper.size == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(upper.min(), -1.0, 1.0))))


def _current_track_angular_error_p90(elements, track_errors):
    error_by_observation = {
        (int(image_id), int(point2d_idx)): float(error)
        for image_id, point2d_idx, error in track_errors
    }
    errors = []
    for elem in elements:
        key = (int(elem.image_id), int(elem.point2D_idx))
        error = error_by_observation.get(key, float("inf"))
        errors.append(error)
    values = np.asarray(errors, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        return float("inf")
    return float(np.percentile(values, 90))


def _summarize_track_records_by_source(track_records):
    result = {
        source: {"tracks": 0, "observations": 0}
        for source in ("s_only", "p_only", "mixed", "empty")
    }
    for record in track_records:
        bucket = result[record["source"]]
        bucket["tracks"] += 1
        bucket["observations"] += int(record["track_length"])
    return result


def _plan_bae_track_deletions(
    track_records,
    image_observations,
    max_observations,
    min_observations_per_image=BAE_MIN_OBSERVATIONS_PER_IMAGE,
):
    """Plan deterministic whole-track deletion without mutating reconstruction."""
    current_image_observations = {
        int(image_id): int(count) for image_id, count in image_observations.items()
    }
    image_floors = {
        image_id: min(count, int(min_observations_per_image))
        for image_id, count in current_image_observations.items()
    }
    current_observations = int(
        sum(int(record["track_length"]) for record in track_records)
    )
    deleted_records = []

    ordered_records = sorted(
        track_records,
        key=lambda record: (
            int(record["track_length"]),
            -float(record["angular_error_p90"]),
            float(record["max_triangulation_angle_deg"]),
            int(record["point3D_id"]),
        ),
    )
    for record in ordered_records:
        if current_observations <= max_observations:
            break

        removals_by_image = defaultdict(int)
        for image_id in record["image_ids"]:
            removals_by_image[int(image_id)] += 1
        if any(
            current_image_observations[image_id] - removal_count
            < image_floors[image_id]
            for image_id, removal_count in removals_by_image.items()
        ):
            continue

        deleted_records.append(record)
        current_observations -= int(record["track_length"])
        for image_id, removal_count in removals_by_image.items():
            current_image_observations[image_id] -= removal_count

    return {
        "reached_budget": current_observations <= max_observations,
        "remaining_observations": int(current_observations),
        "deleted_records": deleted_records,
        "remaining_image_observations": current_image_observations,
        "image_floors": image_floors,
    }


def prune_reconstruction_for_bae_observation_budget(
    reconstruction,
    features,
    max_observations,
    angular_errors_per_track=None,
    min_observations_per_image=BAE_MIN_OBSERVATIONS_PER_IMAGE,
):
    """Prune whole real tracks in place before BAE to meet an observation cap."""
    from ffba.reporting.statistics import summarize_distribution
    from ffba.reporting.tracks import (
        build_s_keypoint_count,
        classify_point3d_track_source,
        summarize_reconstruction,
    )

    max_observations = int(max_observations)
    before = summarize_reconstruction(reconstruction)
    if max_observations <= 0:
        return {
            "enabled": False,
            "applied": False,
            "reason": "bae_max_observations_disabled",
            "max_observations": max_observations,
            "before": before,
            "after": before,
        }
    if before["observations"] <= max_observations:
        return {
            "enabled": True,
            "applied": False,
            "reason": "within_budget",
            "max_observations": max_observations,
            "min_observations_per_image": int(min_observations_per_image),
            "before": before,
            "after": before,
        }

    if angular_errors_per_track is None:
        from gluemap.math.reprojection_error import (  # noqa: PLC0415
            ReprojectionErrorType,
            compute_all_errors_from_reconstruction,
        )

        angular_errors_per_track = compute_all_errors_from_reconstruction(
            reconstruction,
            ReprojectionErrorType.ANGULAR,
            negative_depth_observations={},
        )

    camera_centers = _camera_centers_from_reconstruction(reconstruction)
    s_keypoint_count = build_s_keypoint_count(reconstruction, features)
    image_observations = {int(image_id): 0 for image_id in reconstruction.images}
    track_records = []
    for point3D_id, point3d in reconstruction.points3D.items():
        elements = list(point3d.track.elements)
        image_ids = tuple(int(elem.image_id) for elem in elements)
        for image_id in image_ids:
            image_observations[image_id] += 1
        track_records.append(
            {
                "point3D_id": int(point3D_id),
                "image_ids": image_ids,
                "track_length": int(len(elements)),
                "angular_error_p90": _current_track_angular_error_p90(
                    elements,
                    angular_errors_per_track.pop(point3D_id, []),
                ),
                "max_triangulation_angle_deg": (
                    _track_max_triangulation_angle_degrees(
                        point3d.xyz,
                        image_ids,
                        camera_centers,
                    )
                ),
                "source": classify_point3d_track_source(
                    point3d,
                    s_keypoint_count,
                ),
            }
        )

    plan = _plan_bae_track_deletions(
        track_records,
        image_observations,
        max_observations,
        min_observations_per_image=min_observations_per_image,
    )
    if not plan["reached_budget"]:
        message = (
            "BAE observation budget cannot be reached without violating the "
            f"per-image floor: requested={max_observations}, "
            f"minimum_reachable={plan['remaining_observations']}, "
            f"min_observations_per_image={min_observations_per_image}"
        )
        print(f"[MERG3R-REFINE] {message}", flush=True)
        raise RuntimeError(message)

    deleted_records = plan["deleted_records"]
    for record in deleted_records:
        reconstruction.delete_point3D(record["point3D_id"])

    after = summarize_reconstruction(reconstruction)
    if after["observations"] != plan["remaining_observations"]:
        raise RuntimeError(
            "BAE observation pruning produced an inconsistent count: "
            f"planned={plan['remaining_observations']}, "
            f"actual={after['observations']}"
        )

    finite_deleted_errors = [
        record["angular_error_p90"]
        for record in deleted_records
        if np.isfinite(record["angular_error_p90"])
    ]
    deleted_by_source = _summarize_track_records_by_source(deleted_records)
    before_by_source = _summarize_track_records_by_source(track_records)
    after_by_source = {
        source: {
            "tracks": before_by_source[source]["tracks"]
            - deleted_by_source[source]["tracks"],
            "observations": before_by_source[source]["observations"]
            - deleted_by_source[source]["observations"],
        }
        for source in before_by_source
    }
    return {
        "enabled": True,
        "applied": True,
        "max_observations": max_observations,
        "min_observations_per_image": int(min_observations_per_image),
        "before": before,
        "after": after,
        "removed_tracks": int(len(deleted_records)),
        "removed_observations": int(before["observations"] - after["observations"]),
        "before_by_source": before_by_source,
        "deleted_by_source": deleted_by_source,
        "after_by_source": after_by_source,
        "per_image_observations_before": summarize_distribution(
            list(image_observations.values())
        ),
        "per_image_observations_after": summarize_distribution(
            list(plan["remaining_image_observations"].values())
        ),
        "deleted_track_lengths": summarize_distribution(
            [record["track_length"] for record in deleted_records]
        ),
        "deleted_angular_error_p90": summarize_distribution(finite_deleted_errors),
        "deleted_nonfinite_angular_error_tracks": int(
            len(deleted_records) - len(finite_deleted_errors)
        ),
        "deleted_max_triangulation_angle_deg": summarize_distribution(
            [record["max_triangulation_angle_deg"] for record in deleted_records]
        ),
    }
