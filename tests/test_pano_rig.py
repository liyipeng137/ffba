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
    write_rig_reconstruction,
)


def _center_poses(num_frames):
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], num_frames, axis=0)
    poses[:, 0, 3] = -np.arange(num_frames, dtype=np.float64)
    return poses[:, :3]


def test_expand_center_pose_produces_colocated_yawed_views():
    metadata = make_pano_rig_metadata(["000.png", "001.png"])
    expanded = expand_center_extrinsics(_center_poses(2), metadata)
    rotations = expanded[:, :3, :3]
    translations = expanded[:, :3, 3]
    centers = np.einsum("nij,nj->ni", -rotations.transpose(0, 2, 1), translations)
    axes = rotations.transpose(0, 2, 1)[:, :, 2]

    np.testing.assert_allclose(centers[:3], np.zeros((3, 3)), atol=1e-12)
    np.testing.assert_allclose(
        centers[3:], np.repeat([[1.0, 0.0, 0.0]], 3, axis=0), atol=1e-12
    )
    np.testing.assert_allclose(
        axes[:3],
        [
            [-np.sqrt(3) / 2, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [np.sqrt(3) / 2, 0.0, 0.5],
        ],
        atol=1e-12,
    )
    audit = build_pose_audit(expanded, metadata)
    assert audit["max_projection_center_spread"] < 1e-12
    assert audit["max_absolute_yaw_error_degrees"] < 1e-12


def test_rig_pairs_exclude_same_frame_and_cover_adjacent_sensors():
    metadata = make_pano_rig_metadata([f"{idx:03d}.png" for idx in range(5)])
    expanded = expand_center_extrinsics(_center_poses(5), metadata)
    pairs = build_rig_pose_pairs(
        expanded,
        metadata,
        max_neighbors=6,
        max_axis_angle_degrees=85.0,
    )
    frame_indices = np.asarray(metadata.image_frame_indices)
    sensor_indices = np.asarray(metadata.image_sensor_indices)
    assert pairs.shape[1] == 2
    assert np.all(frame_indices[pairs[:, 0]] != frame_indices[pairs[:, 1]])
    assert not np.any(
        (sensor_indices[pairs[:, 0]] == 0) & (sensor_indices[pairs[:, 1]] == 2)
    )
    center_image_idx = metadata.center_image_indices[2]
    neighbors = set(
        pairs[pairs[:, 0] == center_image_idx, 1].tolist()
        + pairs[pairs[:, 1] == center_image_idx, 0].tolist()
    )
    assert {sensor_indices[idx] for idx in neighbors} == {0, 1, 2}


def test_center_filter_keeps_or_drops_complete_triplets():
    metadata = make_pano_rig_metadata(["000.png", "001.png", "002.png"])
    kept_frames, kept_images = center_driven_keep_indices(
        metadata, [12, 3, 20], threshold=10
    )
    np.testing.assert_array_equal(kept_frames, [0, 2])
    np.testing.assert_array_equal(kept_images, [0, 1, 2, 6, 7, 8])

    filtered = filter_metadata(metadata, kept_frames)
    assert filtered.frame_names == ["000.png", "002.png"]
    assert filtered.frame_source_indices == [0, 2]
    assert filtered.image_frame_indices == [0, 0, 0, 1, 1, 1]
    assert filtered.image_sensor_indices == [0, 1, 2, 0, 1, 2]


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


def test_intrinsics_for_16_by_9_image_with_110_degree_hfov():
    intrinsic = intrinsics_for_image_size((1080, 1920), 110.0)
    focal = 1920 / (2.0 * np.tan(np.deg2rad(110.0) / 2.0))
    np.testing.assert_allclose(
        intrinsic,
        [[focal, 0, 960], [0, focal, 540], [0, 0, 1]],
    )


def test_pycolmap_rig_reconstruction_and_database_round_trip(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    metadata = make_pano_rig_metadata(["000.png", "001.png"])
    center_extrinsic = _center_poses(2)
    intrinsic = np.array([[50.0, 0.0, 50.0], [0.0, 50.0, 50.0], [0, 0, 1]])
    sensor_intrinsics = [intrinsic.copy() for _ in range(3)]

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
    assert len(reloaded.cameras) == 3
    assert len(reloaded.frames) == 2
    assert len(reloaded.images) == 6
    for frame_idx in range(2):
        centers = [
            reloaded.images[frame_idx * 3 + sensor_idx + 1].projection_center()
            for sensor_idx in range(3)
        ]
        np.testing.assert_allclose(
            centers, np.repeat([centers[0]], 3, axis=0), atol=1e-12
        )

    database_path = tmp_path / "database.db"
    database = pycolmap.Database.open(str(database_path))
    for camera_id in range(1, 4):
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
        assert [image.frame_id for image in images] == [1, 1, 1, 2, 2, 2]
        assert [image.camera_id for image in images] == [1, 2, 3, 1, 2, 3]
    finally:
        database.close()
