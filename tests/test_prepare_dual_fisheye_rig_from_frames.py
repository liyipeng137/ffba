from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_dual_fisheye_rig_from_frames import (  # noqa: E402
    MANIFEST_NAME,
    SENSOR_NAMES,
    ImuRow,
    accelerometer_roll_corrections,
    detect_camera_circle,
    pinhole_intrinsics,
    prepare_dataset,
    read_gyro_rows,
    uniform_sample_positions,
)


def _imu_row(name: str, angle_degrees: float) -> ImuRow:
    angle = math.radians(angle_degrees)
    return ImuRow(
        image_name=name,
        image_key=name,
        acc_x=math.sin(angle),
        acc_y=0.0,
        acc_z=math.cos(angle),
        gyro_x=0.1,
        gyro_y=0.2,
        gyro_z=0.3,
    )


def _write_circular_image(path: Path, *, bgr: tuple[int, int, int]) -> None:
    height = width = 128
    yy, xx = np.mgrid[:height, :width]
    inside = (xx - 64) ** 2 + (yy - 60) ** 2 <= 58**2
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[inside] = np.asarray(bgr, dtype=np.uint8)
    image[inside, 1] = np.maximum(image[inside, 1], xx[inside])
    assert cv2.imwrite(str(path), image)


def test_read_gyro_rows_accepts_header_and_image_extensions(tmp_path: Path):
    path = tmp_path / "gyro.txt"
    path.write_text(
        "image_name,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n"
        "frame_000001.jpg,0.1,0.2,0.9,1,2,3\n",
        encoding="utf-8",
    )
    rows = read_gyro_rows(path)
    assert len(rows) == 1
    assert rows[0].image_name == "frame_000001.jpg"
    assert rows[0].image_key == "frame_000001"
    assert rows[0].gyro_z == pytest.approx(3.0)


def test_relative_accelerometer_roll_uses_opposite_camera_signs():
    rows = [_imu_row("f0", 5.0), _imu_row("f1", 15.0), _imu_row("f2", 25.0)]
    corrections, reference = accelerometer_roll_corrections(
        rows,
        reference_frame="f1",
        roll_offset_degrees=2.0,
        median_window=1,
    )
    assert reference == "f1"
    assert corrections["f0"]["relative_rig_roll_degrees"] == pytest.approx(-8.0)
    assert corrections["f1"]["relative_rig_roll_degrees"] == pytest.approx(2.0)
    assert corrections["f2"]["relative_rig_roll_degrees"] == pytest.approx(12.0)
    for correction in corrections.values():
        assert correction["camera2_image_rotation_degrees"] == pytest.approx(
            -correction["camera1_image_rotation_degrees"]
        )


def test_uniform_sampling_uses_bin_midpoints():
    assert uniform_sample_positions(10, 5) == [1, 3, 5, 7, 9]
    assert uniform_sample_positions(9, 3) == [1, 4, 7]


def test_camera_circle_is_clipped_to_rotation_safe_inscribed_radius(tmp_path: Path):
    paths = []
    for index in range(3):
        path = tmp_path / f"frame_{index:06d}.png"
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        cv2.circle(image, (64, 58), 70, (200, 200, 200), thickness=-1)
        assert cv2.imwrite(str(path), image)
        paths.append(path)

    circle, detections = detect_camera_circle(paths, threshold=8)
    inscribed = min(
        circle.center_x,
        circle.center_y,
        127.0 - circle.center_x,
        127.0 - circle.center_y,
    )
    assert circle.radius == pytest.approx(inscribed)
    assert max(item["radius"] for item in detections) > circle.radius


def test_90_degree_pinhole_intrinsics():
    intrinsics = pinhole_intrinsics(1024, 90.0)
    assert intrinsics == pytest.approx(
        np.asarray(
            [[512.0, 0.0, 512.0], [0.0, 512.0, 512.0], [0.0, 0.0, 1.0]]
        )
    )


def test_end_to_end_writes_synchronized_nine_view_dataset(tmp_path: Path):
    source = tmp_path / "input"
    (source / "camera1").mkdir(parents=True)
    (source / "camera2").mkdir()
    names = ["frame_000000", "frame_000001"]
    for name in names:
        _write_circular_image(source / "camera1" / f"{name}.jpg", bgr=(40, 80, 220))
        _write_circular_image(source / "camera2" / f"{name}.jpg", bgr=(220, 80, 40))
    (source / "gyro.txt").write_text(
        "\n".join(
            [
                "frame_000000,0,0,1,0.1,0.2,0.3",
                f"frame_000001,{math.sin(math.radians(10))},0,"
                f"{math.cos(math.radians(10))},0.1,0.2,0.3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    args = argparse.Namespace(
        input_dir=source,
        output_dir=output,
        num_frames=None,
        face_size=32,
        output_fov_degrees=90.0,
        fisheye_fov_degrees=200.0,
        diagonal_tilt_degrees=30.0,
        rim_margin_pixels=2.0,
        circle_black_threshold=8,
        roll_reference_frame=None,
        roll_offset_degrees=0.0,
        accel_median_window=1,
        interpolation="linear",
        save_leveled_fisheye=True,
    )
    prepare_dataset(args)

    for sensor_name in SENSOR_NAMES:
        assert [path.name for path in sorted((output / sensor_name).glob("*.png"))] == [
            "frame_000000.png",
            "frame_000001.png",
        ]
        with Image.open(output / sensor_name / "frame_000000.png") as image:
            assert image.size == (32, 32)
    for camera_name in ("camera1", "camera2"):
        assert len(list((output / f"leveled_{camera_name}").glob("*.png"))) == 2

    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["source"]["input_layout"] == "camera1/camera2/gyro.txt"
    assert manifest["source"]["selected_frame_count"] == 2
    assert manifest["projection"]["output_camera_model"] == "nominal_PINHOLE"
    assert manifest["projection"]["output_hfov_degrees"] == 90.0
    assert manifest["layout"]["sensor_order"] == list(SENSOR_NAMES)
    assert all(
        sensor["pixel_source_policy"] == "single_fisheye_only_no_blending"
        for sensor in manifest["layout"]["sensors"]
    )
    assert manifest["roll_leveling"]["gyro_columns_status"] == "stored_but_not_used"
    assert (output / "previews" / "first_frame_9view_contact_sheet.jpg").is_file()
