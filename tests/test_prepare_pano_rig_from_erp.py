from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import prepare_pano_rig_from_erp as prepare_module  # noqa: E402
from prepare_pano_rig_from_erp import (  # noqa: E402
    FACE_NAMES,
    FACE_SPECS,
    build_decode_command,
    choose_source_frames,
    default_face_size,
    prepare_dataset,
    uniform_sample_positions,
)


def _conversion_backend_available() -> bool:
    try:
        import torch  # noqa: F401
        from pytorch360convert import e2c  # noqa: F401

        return importlib.metadata.version("pytorch360convert") == "0.2.3"
    except (ImportError, importlib.metadata.PackageNotFoundError):
        return False


def test_uniform_sample_positions_uses_bin_midpoints():
    assert uniform_sample_positions(10, 5) == [1, 3, 5, 7, 9]
    assert uniform_sample_positions(9, 3) == [1, 4, 7]
    assert len(set(uniform_sample_positions(3424, 50))) == 50


def test_choose_source_frames_preserves_indices_timestamps_and_sync_names():
    frames = choose_source_frames(
        [index * 0.1 for index in range(20)],
        num_frames=3,
        start_time=0.5,
        end_time=1.4,
    )
    assert [frame["source_frame_index"] for frame in frames] == [6, 9, 12]
    assert [frame["source_timestamp_seconds"] for frame in frames] == pytest.approx(
        [0.6, 0.9, 1.2]
    )
    assert [frame["output_name"] for frame in frames] == [
        "000000.png",
        "000001.png",
        "000002.png",
    ]
    assert frames[0]["face_paths"] == {
        face_name: f"{face_name}/000000.png" for face_name in FACE_NAMES
    }


def test_standard_five_face_mapping_is_explicit():
    assert FACE_NAMES == ("center", "left", "right", "up", "down")
    assert [
        (spec["output_name"], spec["cubemap_face"], spec["e2c_stack_index"])
        for spec in FACE_SPECS
    ] == [
        ("center", "Front", 0),
        ("left", "Left", 3),
        ("right", "Right", 1),
        ("up", "Up", 4),
        ("down", "Down", 5),
    ]
    assert default_face_size(4320) == 1080


def test_decode_command_uses_ffmpeg_only_for_selected_raw_frames(tmp_path: Path):
    command = build_decode_command("ffmpeg", tmp_path / "input.mp4", [1, 3, 8])
    joined = " ".join(command)
    assert "select='eq(n\\,1)+eq(n\\,3)+eq(n\\,8)'" in joined
    assert "rawvideo" in command
    assert "rgb24" in command
    assert "v360" not in joined


def test_probe_video_uses_fast_cfr_timestamps_without_frame_enumeration(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_run_json(argv):
        calls.append(argv)
        return {
            "streams": [
                {
                    "width": 4320,
                    "height": 2160,
                    "r_frame_rate": "30000/1001",
                    "avg_frame_rate": "30000/1001",
                    "nb_frames": "3",
                    "start_time": "1.25",
                }
            ],
            "format": {"duration": "0.1"},
        }

    monkeypatch.setattr(prepare_module, "_run_json", fake_run_json)
    probe = prepare_module.probe_video(Path("input.mp4"), "ffprobe")

    assert len(calls) == 1
    assert probe["timestamp_source"] == "stream_nb_frames_and_average_fps_cfr"
    assert probe["timestamps"] == pytest.approx(
        [1.25, 1.25 + 1001 / 30000, 1.25 + 2 * 1001 / 30000]
    )


def test_missing_pytorch360convert_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def missing_version(_package_name):
        raise importlib.metadata.PackageNotFoundError("pytorch360convert")

    monkeypatch.setattr(importlib.metadata, "version", missing_version)
    with pytest.raises(
        RuntimeError,
        match=r"pytorch360convert==0\.2\.3.*python3 -m pip install",
    ):
        prepare_module._load_conversion_backend()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None
    or shutil.which("ffprobe") is None
    or not _conversion_backend_available(),
    reason="FFmpeg and pytorch360convert==0.2.3 with PyTorch are required",
)
def test_end_to_end_outputs_five_standard_cubemap_faces(tmp_path: Path):
    panorama = tmp_path / "directional.png"
    image = Image.new("RGB", (360, 180))
    image.putdata(
        [
            (
                round(x * 255 / 359),
                round(y * 255 / 179),
                32,
            )
            for y in range(180)
            for x in range(360)
        ]
    )
    image.save(panorama)
    source = tmp_path / "directional.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-loop",
            "1",
            "-framerate",
            "5",
            "-i",
            str(panorama),
            "-t",
            "1",
            "-c:v",
            "png",
            str(source),
        ],
        check=True,
    )
    output = tmp_path / "cubemap5"
    args = argparse.Namespace(
        input=source,
        output_dir=output,
        num_frames=3,
        face_size=64,
        batch_size=2,
        device="cpu",
        start_time=None,
        end_time=None,
        interpolation="nearest",
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
    )
    prepare_dataset(args)

    expected_names = ["000000.png", "000001.png", "000002.png"]
    for face_name in FACE_NAMES:
        assert [
            path.name for path in sorted((output / face_name).glob("*.png"))
        ] == expected_names
        with Image.open(output / face_name / "000000.png") as face:
            assert face.size == (64, 64)

    center_pixels = {}
    for face_name in FACE_NAMES:
        with Image.open(output / face_name / "000000.png").convert("RGB") as face:
            center_pixels[face_name] = face.getpixel((32, 32))
    assert center_pixels["left"][0] < center_pixels["center"][0]
    assert center_pixels["center"][0] < center_pixels["right"][0]
    assert center_pixels["up"][1] < center_pixels["center"][1]
    assert center_pixels["center"][1] < center_pixels["down"][1]

    manifest = json.loads((output / "pano_rig_manifest.json").read_text())
    assert manifest["output"]["hfov_degrees"] == 90.0
    assert manifest["output"]["vfov_degrees"] == 90.0
    assert manifest["output"]["width"] == manifest["output"]["height"] == 64
    assert manifest["output"]["batch_size"] == 2
    assert manifest["output"]["fx_pixels"] == 31.5
    assert manifest["output"]["fy_pixels"] == 31.5
    assert manifest["output"]["cx_pixels"] == 31.5
    assert manifest["output"]["cy_pixels"] == 31.5
    assert (
        manifest["output"]["pixel_center_convention"]
        == "pytorch360convert_linspace_endpoints"
    )
    assert manifest["cubemap"]["face_mapping"] == list(FACE_SPECS)
    assert manifest["cubemap"]["omitted_faces"][0]["cubemap_face"] == "Back"
    assert manifest["sampling"]["output_frame_count"] == 3
    assert len(manifest["frames"]) == 3
    assert all("source_frame_index" in frame for frame in manifest["frames"])
    assert all("source_timestamp_seconds" in frame for frame in manifest["frames"])
