from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_dual_fisheye_rig_from_insv import (  # noqa: E402
    EncodedLensCalibration,
    SENSOR_NAMES,
    build_decode_command,
    choose_layout,
    choose_source_frames,
    diagonal_direction,
    parse_offset_v3,
    pinhole_intrinsics,
    project_insta360,
    scale_lens_calibration_to_stream,
    uniform_sample_positions,
    view_to_lens_rotation,
)


OFFSET_V3 = (
    "2_"
    "2.000000_4266.770_4268.000_2691.560_2704.980_"
    "-0.041_0.079_90.821_0.000000_0.000000_0.000000_"
    "0.17725553_2.05762434_-3.17907715_0.00043737_-0.00108740_"
    "10752_5376_113_"
    "2.000000_4262.700_4263.370_8074.190_2700.330_"
    "-0.070_0.091_88.608_0.000688_0.000183_-0.032500_"
    "0.19437833_2.01216817_-3.04443812_0.00073542_0.00092976_"
    "10752_5376_113_197632"
)
CROP_INFO = {
    "src_width": 5376,
    "src_height": 5376,
    "dst_width": 5312,
    "dst_height": 5312,
}


def _encoded_calibrations():
    raw_lenses, _ = parse_offset_v3(OFFSET_V3)
    return {
        index: scale_lens_calibration_to_stream(
            lens,
            lens_count=2,
            stream_width=3840,
            stream_height=3840,
            crop_info=CROP_INFO,
        )
        for index, lens in enumerate(raw_lenses)
    }


def test_uniform_sampling_and_synchronized_sensor_paths():
    assert uniform_sample_positions(10, 5) == [1, 3, 5, 7, 9]
    frames = choose_source_frames(
        [index * 0.1 for index in range(20)],
        num_frames=3,
        start_time=0.5,
        end_time=1.4,
    )
    assert [frame["source_frame_index"] for frame in frames] == [6, 9, 12]
    assert frames[0]["sensor_paths"] == {
        name: f"{name}/000000.png" for name in SENSOR_NAMES
    }


def test_parse_and_scale_x5_offset_v3():
    raw_lenses, flag = parse_offset_v3(OFFSET_V3)
    assert flag == 197632
    assert len(raw_lenses) == 2
    assert raw_lenses[0].xi == pytest.approx(2.0)
    assert raw_lenses[1].tz == pytest.approx(-0.0325)

    calibrations = _encoded_calibrations()
    assert calibrations[0].fx == pytest.approx(3084.412, abs=0.002)
    assert calibrations[0].cx == pytest.approx(1922.543, abs=0.002)
    assert calibrations[1].fx == pytest.approx(3081.470, abs=0.002)
    assert calibrations[1].cx == pytest.approx(1927.279, abs=0.002)
    assert calibrations[0].circular_support_radius_pixels > 1900


def test_90_degree_pinhole_and_center_projection():
    intrinsics = pinhole_intrinsics(1024, 90.0)
    expected = np.array(
        [[512.0, 0.0, 512.0], [0.0, 512.0, 512.0], [0.0, 0.0, 1.0]]
    )
    assert intrinsics == pytest.approx(expected)
    calibration = EncodedLensCalibration(
        lens_index=0,
        width=3840,
        height=3840,
        xi=2.0,
        fx=3000.0,
        fy=3000.0,
        cx=1920.0,
        cy=1910.0,
        k1=0.1,
        k2=0.2,
        k3=-0.3,
        p1=0.001,
        p2=-0.001,
        circular_support_radius_pixels=1900.0,
    )
    map_x, map_y, finite = project_insta360(
        np.array([[0.0, 0.0, 1.0]]), calibration
    )
    assert bool(finite[0])
    assert map_x[0] == pytest.approx(calibration.cx)
    assert map_y[0] == pytest.approx(calibration.cy)


@pytest.mark.parametrize("quadrant", ["ul", "ur", "dl", "dr"])
def test_diagonal_view_rotation_is_rigid_and_points_to_quadrant(quadrant: str):
    direction, _ = diagonal_direction(quadrant, 50.0)
    rotation = view_to_lens_rotation(direction, 7.0)
    assert rotation.T @ rotation == pytest.approx(np.eye(3), abs=1e-10)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    assert rotation[:, 2] == pytest.approx(direction)


def test_automatic_layout_keeps_90_degree_boundaries_inside_margin():
    tilt, entries, summary = choose_layout(
        _encoded_calibrations(),
        0,
        1,
        requested_tilt_degrees=None,
        minimum_tilt_degrees=30.0,
        tilt_step_degrees=0.5,
        rim_margin_pixels=64.0,
        roll_step_degrees=2.0,
    )
    assert tilt == pytest.approx(50.735610317245346)
    assert len(entries) == 8
    assert summary["selected_minimum_margin_pixels"] >= 64.0
    assert all(
        entry["minimum_margin_pixels"] >= 64.0 for entry in entries.values()
    )


def test_decode_command_stops_after_last_selected_frame(tmp_path: Path):
    command = build_decode_command(
        "ffmpeg", tmp_path / "sample.insv", 1, [3, 8, 21]
    )
    joined = " ".join(command)
    assert "-map 0:v:1" in joined
    assert "select='eq(n\\,3)+eq(n\\,8)+eq(n\\,21)'" in joined
    assert command[command.index("-frames:v") + 1] == "3"
    assert "bgr24" in command
    assert "v360" not in joined


def test_raw_baseline_magnitude_is_preserved():
    raw_lenses, _ = parse_offset_v3(OFFSET_V3)
    baseline = np.array(
        [
            raw_lenses[1].tx - raw_lenses[0].tx,
            raw_lenses[1].ty - raw_lenses[0].ty,
            raw_lenses[1].tz - raw_lenses[0].tz,
        ]
    )
    assert math.sqrt(float(baseline @ baseline)) == pytest.approx(0.0325077965)
