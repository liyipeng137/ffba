import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.pano_rig import (
    build_pose_audit,
    build_rig_pose_pairs,
    center_driven_keep_indices,
    configure_rig_database,
    expand_center_extrinsics,
    filter_metadata,
    intrinsics_for_image_size,
    intrinsics_from_pinhole_crop,
    make_pano_rig_metadata,
    sensor_from_rig_rotation,
    write_rig_reconstruction,
)


def _center_poses(num_frames):
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)
    poses[:, 0, 3] = -np.arange(num_frames, dtype=np.float64)
    return poses[:, :3]


def test_expand_center_pose_matches_e2c_five_face_orientations():
    metadata = make_pano_rig_metadata(["000.png", "001.png"])
    expanded = expand_center_extrinsics(_center_poses(2), metadata)
    rotations = expanded[:, :3, :3]
    translations = expanded[:, :3, 3]
    centers = np.einsum("nij,nj->ni", -rotations.transpose(0, 2, 1), translations)
    axes = rotations.transpose(0, 2, 1)[:, :, 2]

    np.testing.assert_allclose(centers[:5], np.zeros((5, 3)), atol=1e-12)
    np.testing.assert_allclose(
        centers[5:], np.repeat([[1.0, 0.0, 0.0]], 5, axis=0), atol=1e-12
    )
    np.testing.assert_allclose(
        axes[:5],
        [[0, 0, 1], [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0]],
        atol=1e-12,
    )
    assert metadata.sensor_names == ("center", "left", "right", "up", "down")
    assert metadata.center_image_indices == [0, 5]
    assert metadata.to_dict()["image_order"] == "frame_major_center_left_right_up_down"
    audit = build_pose_audit(expanded, metadata)
    assert audit["max_projection_center_spread"] < 1e-12
    assert audit["max_absolute_orientation_error_degrees"] < 1e-12


def test_e2c_up_down_pixel_orientation_is_not_yaw_only():
    expected = {
        "center": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "left": [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
        "right": [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
        "up": [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
        "down": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    }
    for sensor_name, matrix in expected.items():
        np.testing.assert_array_equal(sensor_from_rig_rotation(sensor_name), matrix)

    camera_top = np.array([0.0, -1.0, 1.0])
    up_top_in_rig = sensor_from_rig_rotation("up").T @ camera_top
    down_top_in_rig = sensor_from_rig_rotation("down").T @ camera_top
    np.testing.assert_array_equal(up_top_in_rig, [0.0, -1.0, -1.0])
    np.testing.assert_array_equal(down_top_in_rig, [0.0, 1.0, 1.0])


def test_rig_pairs_exclude_same_frame_include_90_and_exclude_opposites():
    metadata = make_pano_rig_metadata([f"{idx:03d}.png" for idx in range(5)])
    expanded = expand_center_extrinsics(_center_poses(5), metadata)
    pairs = build_rig_pose_pairs(
        expanded, metadata, max_neighbors=10, max_axis_angle_degrees=95.0
    )
    frame_indices = np.asarray(metadata.image_frame_indices)
    sensor_indices = np.asarray(metadata.image_sensor_indices)
    assert pairs.shape[1] == 2
    assert np.all(frame_indices[pairs[:, 0]] != frame_indices[pairs[:, 1]])
    pair_sensors = {
        frozenset((sensor_indices[first], sensor_indices[second]))
        for first, second in pairs
    }
    assert frozenset((1, 2)) not in pair_sensors
    assert frozenset((3, 4)) not in pair_sensors
    center_image_idx = metadata.center_image_indices[2]
    neighbors = set(
        pairs[pairs[:, 0] == center_image_idx, 1].tolist()
        + pairs[pairs[:, 1] == center_image_idx, 0].tolist()
    )
    assert {sensor_indices[idx] for idx in neighbors} == {0, 1, 2, 3, 4}


def test_center_filter_keeps_or_drops_complete_five_face_frames():
    metadata = make_pano_rig_metadata(["000.png", "001.png", "002.png"])
    kept_frames, kept_images = center_driven_keep_indices(
        metadata, [12, 3, 20], threshold=10
    )
    np.testing.assert_array_equal(kept_frames, [0, 2])
    np.testing.assert_array_equal(kept_images, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14])

    filtered = filter_metadata(metadata, kept_frames)
    assert filtered.frame_names == ["000.png", "002.png"]
    assert filtered.frame_source_indices == [0, 2]
    assert filtered.image_frame_indices == [0] * 5 + [1] * 5
    assert filtered.image_sensor_indices == [0, 1, 2, 3, 4] * 2
    assert filtered.sensor_from_rig_rotations == metadata.sensor_from_rig_rotations


def test_intrinsics_account_for_resize_and_crop():
    record = SimpleNamespace(
        source_path="center/000.png",
        source_size_wh=(1920, 1080),
        low_base_size_wh=(480, 270),
        low_crop_box=(2, 2, 478, 268),
        high_base_size_wh=(1920, 1080),
        high_crop_box=(8, 8, 1912, 1072),
    )
    focal = 1920 / (2.0 * np.tan(np.deg2rad(110.0) / 2.0))
    low, high = intrinsics_from_pinhole_crop(record, 110.0)
    np.testing.assert_allclose(
        low,
        [[focal * 0.25, 0, 238], [0, focal * 0.25, 133], [0, 0, 1]],
    )
    np.testing.assert_allclose(
        high,
        [[focal, 0, 952], [0, focal, 532], [0, 0, 1]],
    )


def test_intrinsics_helper_preserves_non_cubemap_behavior():
    intrinsic = intrinsics_for_image_size((1080, 1920), 110.0)
    focal = 1920 / (2.0 * np.tan(np.deg2rad(110.0) / 2.0))
    np.testing.assert_allclose(
        intrinsic,
        [[focal, 0, 960], [0, focal, 540], [0, 0, 1]],
    )


def test_standard_square_cubemap_intrinsics_are_90_degrees():
    intrinsic = intrinsics_for_image_size((512, 512), 90.0)
    np.testing.assert_allclose(
        intrinsic, [[256, 0, 256], [0, 256, 256], [0, 0, 1]], atol=1e-12
    )


def test_pycolmap_rig_reconstruction_and_database_round_trip(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    metadata = make_pano_rig_metadata(["000.png", "001.png"])
    center_extrinsic = _center_poses(2)
    intrinsic = np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0, 0, 1]])
    sensor_intrinsics = [intrinsic.copy() for _ in range(5)]

    model_dir = tmp_path / "model"
    reconstruction = write_rig_reconstruction(
        model_dir,
        metadata.image_names,
        (100, 100),
        center_extrinsic,
        sensor_intrinsics,
        metadata,
    )
    reloaded = pycolmap.Reconstruction(str(model_dir))
    assert len(reconstruction.rigs) == len(reloaded.rigs) == 1
    assert len(reloaded.cameras) == 5
    assert len(reloaded.frames) == 2
    assert len(reloaded.images) == 10
    for frame_idx in range(2):
        centers = [
            reloaded.images[frame_idx * 5 + sensor_idx + 1].projection_center()
            for sensor_idx in range(5)
        ]
        np.testing.assert_allclose(
            centers, np.repeat([centers[0]], 5, axis=0), atol=1e-12
        )

    database_path = tmp_path / "database.db"
    database = pycolmap.Database.open(str(database_path))
    for camera_id in range(1, 6):
        database.write_camera(
            pycolmap.Camera(
                camera_id=camera_id,
                model="SIMPLE_PINHOLE",
                width=100,
                height=100,
                params=[50, 50, 50],
            ),
            use_camera_id=True,
        )
    for image_idx, name in enumerate(metadata.image_names):
        database.write_image(
            pycolmap.Image(
                name=name,
                camera_id=metadata.image_sensor_indices[image_idx] + 1,
                image_id=image_idx + 1,
            ),
            use_image_id=True,
        )
    database.close()

    configure_rig_database(
        database_path,
        metadata.image_names,
        (100, 100),
        sensor_intrinsics,
        metadata,
    )
    database = pycolmap.Database.open(str(database_path))
    try:
        assert database.num_rigs() == 1
        assert database.num_frames() == 2
        images = sorted(database.read_all_images(), key=lambda image: image.image_id)
        assert [image.frame_id for image in images] == [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        assert [image.camera_id for image in images] == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    finally:
        database.close()
