"""Fixed three-view panorama rig helpers for the pano pipeline branch.

The first implementation intentionally models only co-located perspective
views at yaw angles ``[-60, 0, +60]``.  Images are ordered frame-major:
left, center, right for every source timestamp.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


SENSOR_NAMES = ("left", "center", "right")
SENSOR_YAWS_DEGREES = (-60.0, 0.0, 60.0)
CENTER_SENSOR_INDEX = 1


@dataclass(frozen=True)
class PanoRigMetadata:
    frame_names: list[str]
    frame_source_indices: list[int]
    image_frame_indices: list[int]
    image_sensor_indices: list[int]
    image_names: list[str]
    sensor_names: tuple[str, ...] = SENSOR_NAMES
    sensor_yaws_degrees: tuple[float, ...] = SENSOR_YAWS_DEGREES
    center_sensor_index: int = CENTER_SENSOR_INDEX
    hfov_degrees: float = 90.0

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
        payload["image_order"] = "frame_major_left_center_right"
        payload["translation_model"] = "co_located_zero_baseline"
        return payload


def make_pano_rig_metadata(frame_names, hfov_degrees=90.0):
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
    )


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
        for yaw_degrees in metadata.sensor_yaws_degrees:
            sensor_from_rig = np.eye(4, dtype=np.float64)
            sensor_from_rig[:3, :3] = yaw_sensor_from_rig(yaw_degrees)
            expanded.append((sensor_from_rig @ center_w2c[frame_idx])[:3, :4])
    return np.asarray(expanded, dtype=np.float64)


def intrinsics_from_pinhole_crop(record, hfov_degrees):
    """Compute exact K after the resize and centered crop in a pyramid record."""

    source_width, source_height = record.source_size_wh
    if source_width != source_height:
        raise ValueError(
            "Pano v1 requires square source pinholes, got "
            f"{source_width}x{source_height} for {record.source_path}"
        )
    focal_source = source_width / (2.0 * np.tan(np.deg2rad(float(hfov_degrees)) / 2.0))

    def scaled(base_size_wh, crop_box):
        base_width, base_height = base_size_wh
        left, top, _right, _bottom = crop_box
        scale_x = base_width / source_width
        scale_y = base_height / source_height
        return np.array(
            [
                [focal_source * scale_x, 0.0, source_width * 0.5 * scale_x - left],
                [0.0, focal_source * scale_y, source_height * 0.5 * scale_y - top],
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
    if height != width:
        raise ValueError(
            f"Pano v1 requires square processed pinholes, got {width}x{height}"
        )
    focal = width / (2.0 * np.tan(np.deg2rad(float(hfov_degrees)) / 2.0))
    return np.array(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


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
    yaw_errors = []
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
        for sensor_idx, sensor_name in enumerate(metadata.sensor_names):
            measured_angle = float(
                np.rad2deg(
                    np.arccos(
                        np.clip(np.dot(center_axis, frame_axes[sensor_idx]), -1.0, 1.0)
                    )
                )
            )
            expected_angle = abs(float(metadata.sensor_yaws_degrees[sensor_idx]))
            yaw_error = abs(measured_angle - expected_angle)
            yaw_errors.append(yaw_error)
            sensor_entries.append(
                {
                    "sensor": sensor_name,
                    "yaw_degrees": float(metadata.sensor_yaws_degrees[sensor_idx]),
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
        "max_absolute_yaw_error_degrees": float(max(yaw_errors, default=0.0)),
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
        sensor_yaws_degrees=metadata.sensor_yaws_degrees,
        center_sensor_index=metadata.center_sensor_index,
        hfov_degrees=metadata.hfov_degrees,
    )


def _make_pycolmap_rig(pycolmap, metadata, rig_id=1):
    rig = pycolmap.Rig()
    rig.rig_id = int(rig_id)
    center_camera_id = metadata.center_sensor_index + 1
    rig.add_ref_sensor(pycolmap.sensor_t(pycolmap.SensorType.CAMERA, center_camera_id))
    for sensor_idx, yaw_degrees in enumerate(metadata.sensor_yaws_degrees):
        camera_id = sensor_idx + 1
        if sensor_idx == metadata.center_sensor_index:
            continue
        matrix = np.column_stack(
            [yaw_sensor_from_rig(yaw_degrees), np.zeros(3, dtype=np.float64)]
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
    """Replace trivial database frames with one fixed three-sensor rig."""

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
