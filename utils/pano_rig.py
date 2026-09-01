"""Fixed five-face cubemap rig helpers for the pano pipeline branch.

The rig uses the Front/Left/Right/Up/Down face orientation produced by
``pytorch360convert.e2c``. Images are ordered frame-major, with the stable
sensor order center, left, right, up, down for every source timestamp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SENSOR_NAMES = ("center", "left", "right", "up", "down")
SENSOR_CUBEMAP_FACES = ("Front", "Left", "Right", "Up", "Down")
SENSOR_YAWS_DEGREES = (0.0, -90.0, 90.0, 0.0, 0.0)
CENTER_SENSOR_INDEX = 0
CUBEMAP_FOV_DEGREES = 90.0

# OpenCV camera coordinates are +x right, +y down, +z forward. These complete
# sensor_from_rig rotations reproduce pytorch360convert.e2c's face pixel
# orientations, including the otherwise ambiguous roll of the Up/Down faces.
SENSOR_FROM_RIG_ROTATIONS = (
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # Front
    ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)),  # Left
    ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),  # Right
    ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),  # Up
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),  # Down
)


@dataclass(frozen=True)
class PanoRigMetadata:
    frame_names: list[str]
    frame_source_indices: list[int]
    image_frame_indices: list[int]
    image_sensor_indices: list[int]
    image_names: list[str]
    sensor_names: tuple[str, ...] = SENSOR_NAMES
    sensor_cubemap_faces: tuple[str, ...] = SENSOR_CUBEMAP_FACES
    sensor_yaws_degrees: tuple[float, ...] = SENSOR_YAWS_DEGREES
    sensor_from_rig_rotations: tuple[tuple[tuple[float, ...], ...], ...] = (
        SENSOR_FROM_RIG_ROTATIONS
    )
    center_sensor_index: int = CENTER_SENSOR_INDEX
    hfov_degrees: float = CUBEMAP_FOV_DEGREES
    vfov_degrees: float = CUBEMAP_FOV_DEGREES

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    @property
    def num_images(self) -> int:
        return len(self.image_names)

    @property
    def center_image_indices(self) -> list[int]:
        return [
            image_idx
            for image_idx, sensor_idx in enumerate(self.image_sensor_indices)
            if sensor_idx == self.center_sensor_index
        ]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["center_image_indices"] = self.center_image_indices
        payload["num_frames"] = self.num_frames
        payload["num_images"] = self.num_images
        payload["image_order"] = "frame_major_center_left_right_up_down"
        payload["sensor_order"] = list(self.sensor_names)
        payload["cubemap_face_order"] = list(self.sensor_cubemap_faces)
        payload["rig_reference_sensor"] = self.sensor_names[self.center_sensor_index]
        payload["rotation_semantics"] = "sensor_from_rig"
        payload["face_geometry"] = "square_90_degree_hfov_vfov"
        payload["translation_model"] = "co_located_zero_baseline"
        return payload


def make_pano_rig_metadata(frame_names, hfov_degrees=CUBEMAP_FOV_DEGREES):
    frame_names = [str(name) for name in frame_names]
    if not frame_names:
        raise ValueError("A pano rig requires at least one frame")
    if len(set(frame_names)) != len(frame_names):
        raise ValueError("Pano frame basenames must be unique")
    if not 0.0 < float(hfov_degrees) < 180.0:
        raise ValueError("hfov_degrees must be in (0, 180)")

    image_frame_indices = []
    image_sensor_indices = []
    image_names = []
    for frame_idx, frame_name in enumerate(frame_names):
        stem = Path(frame_name).stem
        for sensor_idx, sensor_name in enumerate(SENSOR_NAMES):
            image_frame_indices.append(frame_idx)
            image_sensor_indices.append(sensor_idx)
            image_names.append(f"frame_{frame_idx:06d}_{sensor_name}_{stem}.png")
    return PanoRigMetadata(
        frame_names=frame_names,
        frame_source_indices=list(range(len(frame_names))),
        image_frame_indices=image_frame_indices,
        image_sensor_indices=image_sensor_indices,
        image_names=image_names,
        hfov_degrees=float(hfov_degrees),
        vfov_degrees=float(hfov_degrees),
    )


def sensor_from_rig_rotation(sensor):
    """Return one e2c-compatible ``sensor_from_rig`` rotation matrix."""

    if isinstance(sensor, str):
        try:
            sensor_idx = SENSOR_NAMES.index(sensor)
        except ValueError as exc:
            raise ValueError(f"Unknown pano rig sensor: {sensor}") from exc
    else:
        sensor_idx = int(sensor)
    if not 0 <= sensor_idx < len(SENSOR_FROM_RIG_ROTATIONS):
        raise ValueError(f"Pano rig sensor index out of range: {sensor_idx}")
    return np.asarray(SENSOR_FROM_RIG_ROTATIONS[sensor_idx], dtype=np.float64)


def yaw_sensor_from_rig(yaw_degrees):
    """Return camera-coordinate ``sensor_from_rig`` for a yawed sensor.

    Camera coordinates use +x right, +y down and +z forward. Positive yaw
    points the virtual camera toward rig-right.
    """

    angle = np.deg2rad(float(yaw_degrees))
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array(
        [[cosine, 0.0, -sine], [0.0, 1.0, 0.0], [sine, 0.0, cosine]],
        dtype=np.float64,
    )


def expand_center_extrinsics(center_extrinsic, metadata):
    center_extrinsic = np.asarray(center_extrinsic, dtype=np.float64)
    if center_extrinsic.ndim != 3 or center_extrinsic.shape[1:] not in {
        (3, 4),
        (4, 4),
    }:
        raise ValueError(
            "Expected center extrinsics with shape (N,3,4) or (N,4,4), "
            f"got {center_extrinsic.shape}"
        )
    if center_extrinsic.shape[0] != metadata.num_frames:
        raise ValueError(
            "Center pose count does not match pano frames: "
            f"poses={center_extrinsic.shape[0]}, frames={metadata.num_frames}"
        )

    center_w2c = np.zeros((metadata.num_frames, 4, 4), dtype=np.float64)
    center_w2c[:, 3, 3] = 1.0
    center_w2c[:, :3, :4] = center_extrinsic[:, :3, :4]
    expanded = []
    for frame_idx in range(metadata.num_frames):
        for rotation in metadata.sensor_from_rig_rotations:
            sensor_from_rig = np.eye(4, dtype=np.float64)
            sensor_from_rig[:3, :3] = np.asarray(rotation, dtype=np.float64)
            expanded.append((sensor_from_rig @ center_w2c[frame_idx])[:3, :4])
    return np.asarray(expanded, dtype=np.float64)


def intrinsics_from_pinhole_crop(record, hfov_degrees):
    """Compute K from source HFOV after resize and centered crop.

    The source is assumed to have square pixels, so its focal lengths in pixel
    units are equal. Independent resize scales are then applied to fx and fy;
    this also remains correct if preprocessing introduces a small anisotropy.
    """

    source_width, source_height = record.source_size_wh
    if source_width <= 0 or source_height <= 0:
        raise ValueError(
            "Pano source dimensions must be positive, got "
            f"{source_width}x{source_height} for {record.source_path}"
        )
    # pytorch360convert.e2c samples each face with linspace(-0.5, 0.5, W).
    # Pixel 0 and pixel W-1 lie exactly on the two HFOV boundary rays, so the
    # matching pinhole differs by half a pixel from the usual edge convention.
    focal_source = (source_width - 1.0) / (
        2.0 * np.tan(np.deg2rad(float(hfov_degrees)) / 2.0)
    )
    source_cx = (source_width - 1.0) * 0.5
    source_cy = (source_height - 1.0) * 0.5

    def scaled(base_size_wh, crop_box):
        base_width, base_height = base_size_wh
        left, top, _right, _bottom = crop_box
        scale_x = base_width / source_width
        scale_y = base_height / source_height
        # PIL resize uses the standard pixel-center transform. Apply it before
        # the centered crop so K matches the actual resampled image.
        principal_x = (source_cx + 0.5) * scale_x - 0.5 - left
        principal_y = (source_cy + 0.5) * scale_y - 0.5 - top
        return np.array(
            [
                [focal_source * scale_x, 0.0, principal_x],
                [0.0, focal_source * scale_y, principal_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    return (
        scaled(record.low_base_size_wh, record.low_crop_box),
        scaled(record.high_base_size_wh, record.high_crop_box),
    )


def intrinsics_for_image_size(image_size_hw, hfov_degrees):
    height, width = (int(value) for value in image_size_hw)
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Pano processed dimensions must be positive, got {width}x{height}"
        )
    focal = (width - 1.0) / (
        2.0 * np.tan(np.deg2rad(float(hfov_degrees)) / 2.0)
    )
    return np.array(
        [
            [focal, 0.0, (width - 1.0) * 0.5],
            [0.0, focal, (height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_rig_projected_overlap_selection(
    extrinsic,
    metadata,
    frame_candidate_details,
    max_pair_neighbors,
    max_group_neighbors,
    max_axis_angle_degrees,
    group_rotation_threshold_degrees,
):
    """Expand depth-ranked frame candidates into five-face pairs and groups.

    ``frame_candidate_details`` is produced by projected-overlap scoring on
    the Stage-A center views. That scoring uses real depth in both frames.
    Here the score is transferred to the synchronized rig frame and combined
    with each virtual sensor's known viewing axis. No depth is fabricated for
    Left/Right/Up/Down.
    """

    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape != (metadata.num_images, 3, 4):
        raise ValueError(
            f"Expected expanded extrinsic shape ({metadata.num_images},3,4), "
            f"got {extrinsic.shape}"
        )
    max_pair_neighbors = int(max_pair_neighbors)
    max_group_neighbors = int(max_group_neighbors)
    if max_pair_neighbors <= 0 or max_group_neighbors <= 0:
        raise ValueError("Pair and group neighbor limits must be >= 1")
    if not 0.0 < float(max_axis_angle_degrees) <= 180.0:
        raise ValueError("max_axis_angle_degrees must be in (0, 180]")
    if not 0.0 <= float(group_rotation_threshold_degrees) <= 180.0:
        raise ValueError("group_rotation_threshold_degrees must be in [0, 180]")

    rotations = extrinsic[:, :3, :3]
    translations = extrinsic[:, :3, 3]
    centers = np.einsum(
        "nij,nj->ni", -rotations.transpose(0, 2, 1), translations
    )
    axes = rotations.transpose(0, 2, 1)[:, :, 2]
    frame_indices = np.asarray(metadata.image_frame_indices, dtype=np.int64)
    sensors_per_frame = len(metadata.sensor_names)
    pair_set = set()
    groups = []
    valid_neighbor_counts = []
    unfiltered_neighbor_counts = []

    def finite_score(detail, name, default=0.0):
        value = float(detail.get(name, default))
        return value if np.isfinite(value) else default

    for source_idx in range(metadata.num_images):
        source_frame = int(frame_indices[source_idx])
        candidates = []
        for frame_detail in frame_candidate_details.get(source_frame, []):
            target_frame = int(frame_detail["image_index"])
            if target_frame == source_frame:
                continue
            for target_sensor in range(sensors_per_frame):
                target_idx = target_frame * sensors_per_frame + target_sensor
                dot = float(np.dot(axes[source_idx], axes[target_idx]))
                angle = float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
                if angle >= float(max_axis_angle_degrees):
                    continue
                distance = float(
                    np.linalg.norm(centers[source_idx] - centers[target_idx])
                )
                candidates.append(
                    {
                        "image_index": target_idx,
                        "axis_angle": angle,
                        "distance": distance,
                        "projected_overlap": finite_score(
                            frame_detail, "projected_overlap"
                        ),
                        "projected_grid_coverage": finite_score(
                            frame_detail, "projected_grid_coverage"
                        ),
                        "projected_visible_ratio": finite_score(
                            frame_detail, "projected_visible_ratio"
                        ),
                        "dino_similarity": finite_score(
                            frame_detail, "dino_similarity", -np.inf
                        ),
                    }
                )

        pair_order = sorted(
            candidates,
            key=lambda item: (
                -item["projected_overlap"],
                -item["projected_grid_coverage"],
                -item["projected_visible_ratio"],
                -item["dino_similarity"],
                item["axis_angle"],
                item["distance"],
                item["image_index"],
            ),
        )
        for item in pair_order[:max_pair_neighbors]:
            pair_set.add(tuple(sorted((source_idx, int(item["image_index"])))))

        group_order = sorted(
            candidates,
            key=lambda item: (
                item["axis_angle"] >= float(group_rotation_threshold_degrees),
                -item["projected_overlap"],
                -item["projected_grid_coverage"],
                -item["projected_visible_ratio"],
                -item["dino_similarity"],
                item["axis_angle"],
                item["distance"],
                item["image_index"],
            ),
        )[:max_group_neighbors]
        if group_order:
            groups.append(
                [source_idx, *[int(item["image_index"]) for item in group_order]]
            )
            valid = sum(
                item["axis_angle"] < float(group_rotation_threshold_degrees)
                for item in group_order
            )
            valid_neighbor_counts.append(int(valid))
            unfiltered_neighbor_counts.append(int(len(group_order) - valid))

    pairs = np.asarray(sorted(pair_set), dtype=np.int64).reshape(-1, 2)

    def numeric_summary(values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}
        return {
            "min": int(values.min()),
            "median": float(np.median(values)),
            "mean": float(values.mean()),
            "max": int(values.max()),
        }

    group_sizes = [len(group) for group in groups]
    group_neighbors = [len(group) - 1 for group in groups]
    stats = {
        "strategy": "rig_center_depth_projected_overlap",
        "candidate_pool": "center_pose_union_global_center_dino",
        "depth_semantics": (
            "true round_trip projected overlap on center-center frame pairs; "
            "known rig viewing-axis expansion for non-center sensors"
        ),
        "neighbor_order": (
            "rotation_valid_then_projected_overlap_grid_visible_dino_geometry"
        ),
        "max_pair_neighbors": max_pair_neighbors,
        "max_neighbors": max_group_neighbors,
        "max_axis_angle_degrees": float(max_axis_angle_degrees),
        "pose_rotation_threshold": float(group_rotation_threshold_degrees),
        "num_groups": len(groups),
        "group_size": numeric_summary(group_sizes),
        "neighbors": numeric_summary(group_neighbors),
        "missing_centers": sorted(
            set(range(metadata.num_images)) - {int(group[0]) for group in groups}
        ),
        "selected_rotation_valid_neighbors": numeric_summary(
            valid_neighbor_counts
        ),
        "selected_unfiltered_neighbors": numeric_summary(
            unfiltered_neighbor_counts
        ),
    }
    return pairs, groups, stats


def build_rig_pose_pairs(
    extrinsic,
    metadata,
    max_neighbors,
    max_axis_angle_degrees,
):
    """Build cross-frame pairs with round-robin sensor coverage.

    Same-frame virtual cameras are deliberately excluded because they share an
    optical center and therefore provide no triangulation baseline. Candidate
    target sensors must have overlapping view cones. For each source image,
    candidates are interleaved across target sensors so same-direction views
    cannot consume the complete neighbor budget.
    """

    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape != (metadata.num_images, 3, 4):
        raise ValueError(
            f"Expected expanded extrinsic shape ({metadata.num_images},3,4), "
            f"got {extrinsic.shape}"
        )
    max_neighbors = int(max_neighbors)
    if max_neighbors <= 0:
        return np.empty((0, 2), dtype=np.int64)
    if not 0.0 < float(max_axis_angle_degrees) <= 180.0:
        raise ValueError("max_axis_angle_degrees must be in (0, 180]")

    rotations = extrinsic[:, :3, :3]
    translations = extrinsic[:, :3, 3]
    centers = np.einsum("nij,nj->ni", -rotations.transpose(0, 2, 1), translations)
    axes = rotations.transpose(0, 2, 1)[:, :, 2]
    frame_indices = np.asarray(metadata.image_frame_indices, dtype=np.int64)
    sensor_indices = np.asarray(metadata.image_sensor_indices, dtype=np.int64)
    pairs = set()

    for source_idx in range(metadata.num_images):
        per_sensor = []
        for target_sensor in range(len(metadata.sensor_names)):
            candidates = []
            for target_idx in np.where(sensor_indices == target_sensor)[0]:
                target_idx = int(target_idx)
                if frame_indices[target_idx] == frame_indices[source_idx]:
                    continue
                dot = float(np.dot(axes[source_idx], axes[target_idx]))
                angle = float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
                if angle >= float(max_axis_angle_degrees):
                    continue
                distance = float(
                    np.linalg.norm(centers[source_idx] - centers[target_idx])
                )
                frame_gap = abs(
                    int(frame_indices[source_idx]) - int(frame_indices[target_idx])
                )
                candidates.append((distance, frame_gap, angle, target_idx))
            candidates.sort()
            if candidates:
                per_sensor.append([entry[-1] for entry in candidates])

        selected = []
        rank = 0
        while len(selected) < max_neighbors and per_sensor:
            added = False
            for candidates in per_sensor:
                if rank < len(candidates):
                    selected.append(candidates[rank])
                    added = True
                    if len(selected) == max_neighbors:
                        break
            if not added:
                break
            rank += 1
        for target_idx in selected:
            pairs.add(tuple(sorted((source_idx, int(target_idx)))))
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def build_pose_audit(extrinsic, metadata):
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.shape != (metadata.num_images, 3, 4):
        raise ValueError(
            f"Expected expanded extrinsic shape ({metadata.num_images},3,4), "
            f"got {extrinsic.shape}"
        )
    rotations = extrinsic[:, :3, :3]
    translations = extrinsic[:, :3, 3]
    centers = np.einsum("nij,nj->ni", -rotations.transpose(0, 2, 1), translations)
    axes = rotations.transpose(0, 2, 1)[:, :, 2]
    frame_entries = []
    center_spreads = []
    orientation_errors = []
    sensors_per_frame = len(metadata.sensor_names)
    for frame_idx in range(metadata.num_frames):
        first = frame_idx * sensors_per_frame
        frame_centers = centers[first : first + sensors_per_frame]
        frame_axes = axes[first : first + sensors_per_frame]
        reference_center = frame_centers[metadata.center_sensor_index]
        spread = float(np.max(np.linalg.norm(frame_centers - reference_center, axis=1)))
        center_spreads.append(spread)
        sensor_entries = []
        center_axis = frame_axes[metadata.center_sensor_index]
        center_rotation = rotations[first + metadata.center_sensor_index]
        for sensor_idx, sensor_name in enumerate(metadata.sensor_names):
            measured_angle = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(np.dot(center_axis, frame_axes[sensor_idx]), -1.0, 1.0)
                    )
                )
            )
            measured_sensor_from_rig = rotations[first + sensor_idx] @ center_rotation.T
            expected_sensor_from_rig = np.asarray(
                metadata.sensor_from_rig_rotations[sensor_idx], dtype=np.float64
            )
            rotation_delta = measured_sensor_from_rig @ expected_sensor_from_rig.T
            orientation_errors.append(
                float(
                    np.rad2deg(
                        np.arccos(
                            np.clip(
                                (np.trace(rotation_delta) - 1.0) * 0.5, -1.0, 1.0
                            )
                        )
                    )
                )
            )
            sensor_entries.append(
                {
                    "sensor": sensor_name,
                    "cubemap_face": metadata.sensor_cubemap_faces[sensor_idx],
                    "yaw_degrees": float(metadata.sensor_yaws_degrees[sensor_idx]),
                    "sensor_from_rig_rotation": expected_sensor_from_rig.tolist(),
                    "image_name": metadata.image_names[first + sensor_idx],
                    "projection_center": frame_centers[sensor_idx].tolist(),
                    "viewing_direction": frame_axes[sensor_idx].tolist(),
                    "angle_from_center_degrees": measured_angle,
                }
            )
        frame_entries.append(
            {
                "frame_index": frame_idx,
                "source_frame_index": metadata.frame_source_indices[frame_idx],
                "source_frame_name": metadata.frame_names[frame_idx],
                "max_projection_center_spread": spread,
                "sensors": sensor_entries,
            }
        )
    return {
        "pose_semantics": "sensor_from_world = sensor_from_rig * rig_from_world",
        "num_frames": metadata.num_frames,
        "num_images": metadata.num_images,
        "max_projection_center_spread": float(max(center_spreads, default=0.0)),
        "max_absolute_orientation_error_degrees": float(
            max(orientation_errors, default=0.0)
        ),
        "frames": frame_entries,
    }


def center_driven_keep_indices(metadata, center_observation_counts, threshold):
    counts = np.asarray(center_observation_counts, dtype=np.int64)
    if counts.shape != (metadata.num_frames,):
        raise ValueError(
            f"Expected {metadata.num_frames} center counts, got {counts.shape}"
        )
    kept_frames = np.where(counts >= int(threshold))[0]
    if kept_frames.size == 0:
        raise ValueError("All center frames are below the observation threshold")
    kept_frame_set = set(int(idx) for idx in kept_frames)
    kept_images = [
        image_idx
        for image_idx, frame_idx in enumerate(metadata.image_frame_indices)
        if int(frame_idx) in kept_frame_set
    ]
    return kept_frames.astype(np.int64), np.asarray(kept_images, dtype=np.int64)


def filter_metadata(metadata, kept_frame_indices):
    kept_frame_indices = [int(value) for value in kept_frame_indices]
    old_to_new = {old: new for new, old in enumerate(kept_frame_indices)}
    frame_names = [metadata.frame_names[old] for old in kept_frame_indices]
    frame_source_indices = [
        metadata.frame_source_indices[old] for old in kept_frame_indices
    ]
    image_frame_indices = []
    image_sensor_indices = []
    image_names = []
    for image_idx, old_frame_idx in enumerate(metadata.image_frame_indices):
        if old_frame_idx not in old_to_new:
            continue
        image_frame_indices.append(old_to_new[old_frame_idx])
        image_sensor_indices.append(metadata.image_sensor_indices[image_idx])
        image_names.append(metadata.image_names[image_idx])
    return PanoRigMetadata(
        frame_names=frame_names,
        frame_source_indices=frame_source_indices,
        image_frame_indices=image_frame_indices,
        image_sensor_indices=image_sensor_indices,
        image_names=image_names,
        sensor_names=metadata.sensor_names,
        sensor_cubemap_faces=metadata.sensor_cubemap_faces,
        sensor_yaws_degrees=metadata.sensor_yaws_degrees,
        sensor_from_rig_rotations=metadata.sensor_from_rig_rotations,
        center_sensor_index=metadata.center_sensor_index,
        hfov_degrees=metadata.hfov_degrees,
        vfov_degrees=metadata.vfov_degrees,
    )


def _make_pycolmap_rig(pycolmap, metadata, rig_id=1):
    rig = pycolmap.Rig()
    rig.rig_id = int(rig_id)
    center_camera_id = metadata.center_sensor_index + 1
    rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, center_camera_id))
    for sensor_idx, rotation in enumerate(metadata.sensor_from_rig_rotations):
        camera_id = sensor_idx + 1
        if sensor_idx == metadata.center_sensor_index:
            continue
        matrix = np.column_stack(
            [np.asarray(rotation, dtype=np.float64), np.zeros(3, dtype=np.float64)]
        )
        rig.add_sensor(
            pycolmap.sensor_t(pycolmap.SensorType.CAMERA, camera_id),
            pycolmap.Rigid3d(matrix),
        )
    return rig


def write_rig_reconstruction(
    output_dir,
    image_names,
    image_size_hw,
    center_extrinsic,
    sensor_intrinsics,
    metadata,
    camera_model="SIMPLE_PINHOLE",
):
    import pycolmap  # noqa: PLC0415

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = (int(value) for value in image_size_hw)
    reconstruction = pycolmap.Reconstruction()
    for sensor_idx, intrinsic in enumerate(sensor_intrinsics):
        intrinsic = np.asarray(intrinsic, dtype=np.float64)
        if camera_model != "SIMPLE_PINHOLE":
            raise ValueError("Pano rig v1 supports SIMPLE_PINHOLE only")
        params = [intrinsic[0, 0], intrinsic[0, 2], intrinsic[1, 2]]
        reconstruction.add_camera(
            pycolmap.Camera(
                camera_id=sensor_idx + 1,
                model=camera_model,
                width=width,
                height=height,
                params=params,
            )
        )
    reconstruction.add_rig(_make_pycolmap_rig(pycolmap, metadata))

    center_extrinsic = np.asarray(center_extrinsic, dtype=np.float64)
    if center_extrinsic.shape != (metadata.num_frames, 3, 4):
        raise ValueError(
            f"Expected center extrinsic shape ({metadata.num_frames},3,4), "
            f"got {center_extrinsic.shape}"
        )
    for frame_idx in range(metadata.num_frames):
        frame_id = frame_idx + 1
        frame = pycolmap.Frame()
        frame.frame_id = frame_id
        frame.rig_id = 1
        frame.rig_from_world = pycolmap.Rigid3d(center_extrinsic[frame_idx])
        first_image_idx = frame_idx * len(metadata.sensor_names)
        for sensor_idx in range(len(metadata.sensor_names)):
            image_id = first_image_idx + sensor_idx + 1
            sensor_id = pycolmap.sensor_t(pycolmap.SensorType.CAMERA, sensor_idx + 1)
            frame.add_data_id(pycolmap.data_t(sensor_id, image_id))
        frame.finalize_data_ids()
        reconstruction.add_frame(frame)
        for sensor_idx in range(len(metadata.sensor_names)):
            image_idx = first_image_idx + sensor_idx
            image = pycolmap.Image(
                name=image_names[image_idx],
                camera_id=sensor_idx + 1,
                image_id=image_idx + 1,
            )
            image.frame_id = frame_id
            reconstruction.add_image(image)
    reconstruction.write(str(output_dir))
    return reconstruction


def configure_rig_database(
    database_path,
    image_names,
    image_size_hw,
    sensor_intrinsics,
    metadata,
    camera_model="SIMPLE_PINHOLE",
):
    """Replace trivial database frames with one fixed five-sensor rig."""

    import pycolmap  # noqa: PLC0415

    height, width = (int(value) for value in image_size_hw)
    database = pycolmap.Database.open(str(database_path))
    try:
        images_by_name = {image.name: image for image in database.read_all_images()}
        if set(images_by_name) != set(image_names):
            raise ValueError("Database image names do not match pano work images")
        if database.num_frames() != 0 or database.num_rigs() != 0:
            raise ValueError(
                "configure_rig_database expects a freshly merged database "
                "without frames or rigs"
            )

        for sensor_idx, intrinsic in enumerate(sensor_intrinsics):
            intrinsic = np.asarray(intrinsic, dtype=np.float64)
            camera = pycolmap.Camera(
                camera_id=sensor_idx + 1,
                model=camera_model,
                width=width,
                height=height,
                params=[intrinsic[0, 0], intrinsic[0, 2], intrinsic[1, 2]],
            )
            if database.exists_camera(camera.camera_id):
                database.update_camera(camera)
            else:
                database.write_camera(camera, use_camera_id=True)

        database.write_rig(_make_pycolmap_rig(pycolmap, metadata), use_rig_id=True)
        for frame_idx in range(metadata.num_frames):
            frame = pycolmap.Frame()
            frame.frame_id = frame_idx + 1
            frame.rig_id = 1
            first_image_idx = frame_idx * len(metadata.sensor_names)
            for sensor_idx in range(len(metadata.sensor_names)):
                image_idx = first_image_idx + sensor_idx
                image = images_by_name[image_names[image_idx]]
                sensor_id = pycolmap.sensor_t(
                    pycolmap.SensorType.CAMERA, sensor_idx + 1
                )
                frame.add_data_id(pycolmap.data_t(sensor_id, image.image_id))
            database.write_frame(frame, use_frame_id=True)

        for image_idx, name in enumerate(image_names):
            image = images_by_name[name]
            image.camera_id = metadata.image_sensor_indices[image_idx] + 1
            image.frame_id = metadata.image_frame_indices[image_idx] + 1
            database.update_image(image)
    finally:
        database.close()
