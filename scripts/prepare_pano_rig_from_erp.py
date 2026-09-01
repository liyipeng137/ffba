#!/usr/bin/env python3
"""Uniformly sample an ERP video and prepare five standard cubemap faces.

FFmpeg is used only to probe and decode source video frames. Projection is
performed by ``pytorch360convert==0.2.3`` using its ``e2c`` function.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import subprocess
import uuid
from fractions import Fraction
from pathlib import Path
from typing import BinaryIO

from PIL import Image

MANIFEST_NAME = "pano_rig_manifest.json"
PYTORCH360CONVERT_VERSION = "0.2.3"

# pytorch360convert e2c stack order is Front, Right, Back, Left, Up, Down.
# Back is intentionally omitted for the five-face pano-rig input.
FACE_SPECS = (
    {
        "output_name": "center",
        "cubemap_face": "Front",
        "e2c_stack_index": 0,
        "look_axis": "+Z",
    },
    {
        "output_name": "left",
        "cubemap_face": "Left",
        "e2c_stack_index": 3,
        "look_axis": "-X",
    },
    {
        "output_name": "right",
        "cubemap_face": "Right",
        "e2c_stack_index": 1,
        "look_axis": "+X",
    },
    {
        "output_name": "up",
        "cubemap_face": "Up",
        "e2c_stack_index": 4,
        "look_axis": "+Y",
    },
    {
        "output_name": "down",
        "cubemap_face": "Down",
        "e2c_stack_index": 5,
        "look_axis": "-Y",
    },
)
FACE_NAMES = tuple(spec["output_name"] for spec in FACE_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Uniformly sample an equirectangular video and use "
            "pytorch360convert e2c to create synchronized center/left/right/"
            "up/down 90-degree square cubemap faces."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Input ERP video")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; it must not already exist",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="Number of source frames sampled uniformly (default: 50)",
    )
    parser.add_argument(
        "--face-size",
        type=int,
        default=None,
        help="Square face size; default is source ERP width / 4",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of decoded ERP frames passed to e2c together (default: 1)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Projection device: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Optional inclusive source start time in seconds",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Optional inclusive source end time in seconds",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "bilinear", "bicubic"),
        default="bilinear",
        help="pytorch360convert e2c sampling mode (default: bilinear)",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args()


def _run_json(argv: list[str]) -> dict:
    completed = subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def _tool_version(tool: str) -> str:
    completed = subprocess.run(
        [tool, "-version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.splitlines()[0]


def _parse_rate(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None


def _optional_float(value: object) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_video(input_path: Path, ffprobe: str) -> dict:
    """Return stream metadata and the original timestamp for every frame."""
    payload = _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=index,codec_name,width,height,pix_fmt,r_frame_rate,"
                "avg_frame_rate,nb_frames,duration,start_time:format=duration,size"
            ),
            "-of",
            "json",
            str(input_path),
        ]
    )
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one selected video stream, got {len(streams)}")
    stream = streams[0]
    width = int(stream.get("width", 0))
    height = int(stream.get("height", 0))
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid source dimensions: {width}x{height}")
    if not math.isclose(width / height, 2.0, rel_tol=0.0, abs_tol=0.01):
        raise ValueError(
            "ERP input must have an approximately 2:1 aspect ratio, got "
            f"{width}x{height}"
        )

    average_fps = _parse_rate(stream.get("avg_frame_rate"))
    nominal_fps = _parse_rate(stream.get("r_frame_rate"))
    fps = average_fps or nominal_fps
    source_start = _optional_float(stream.get("start_time")) or 0.0
    try:
        declared_frame_count = int(stream.get("nb_frames", 0))
    except (TypeError, ValueError):
        declared_frame_count = 0

    # Avoid decoding an entire CFR video solely to enumerate timestamps. A
    # stream with a declared frame count and matching nominal/average rates has
    # timestamp(frame_index) = start + frame_index / nominal_rate. If the rates
    # disagree (VFR) or the count is absent, preserve exact per-frame PTS via
    # ffprobe's slower frame enumeration fallback.
    is_cfr = (
        declared_frame_count > 0
        and nominal_fps is not None
        and (
            average_fps is None
            or math.isclose(nominal_fps, average_fps, rel_tol=1e-5, abs_tol=1e-6)
        )
    )
    if is_cfr:
        cfr_fps = average_fps or nominal_fps
        assert cfr_fps is not None
        timestamps = [
            source_start + frame_idx / cfr_fps
            for frame_idx in range(declared_frame_count)
        ]
        return {
            "stream": stream,
            "format": payload.get("format", {}),
            "timestamps": timestamps,
            "timestamp_source": "stream_nb_frames_and_average_fps_cfr",
            "fps": cfr_fps,
        }

    frame_payload = _run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(input_path),
        ]
    )
    timestamps: list[float] = []
    used_fps_fallback = False
    for frame_idx, frame in enumerate(frame_payload.get("frames", [])):
        timestamp = _optional_float(frame.get("best_effort_timestamp_time"))
        if timestamp is None:
            if fps is None:
                raise RuntimeError(
                    f"Frame {frame_idx} has no timestamp and stream FPS is unavailable"
                )
            timestamp = source_start + frame_idx / fps
            used_fps_fallback = True
        timestamps.append(timestamp)
    if not timestamps:
        raise RuntimeError("ffprobe returned no video frame timestamps")

    return {
        "stream": stream,
        "format": payload.get("format", {}),
        "timestamps": timestamps,
        "timestamp_source": (
            "ffprobe_best_effort_timestamp_time_with_fps_fallback"
            if used_fps_fallback
            else "ffprobe_best_effort_timestamp_time"
        ),
        "fps": fps,
    }


def uniform_sample_positions(item_count: int, sample_count: int) -> list[int]:
    """Return unique midpoint samples from equal-width bins."""
    if item_count <= 0:
        raise ValueError("item_count must be positive")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > item_count:
        raise ValueError(
            f"Cannot sample {sample_count} unique items from only {item_count}"
        )
    return [
        ((2 * index + 1) * item_count) // (2 * sample_count)
        for index in range(sample_count)
    ]


def choose_source_frames(
    timestamps: list[float],
    num_frames: int,
    start_time: float | None,
    end_time: float | None,
) -> list[dict]:
    if start_time is not None and start_time < 0:
        raise ValueError("--start-time must be non-negative")
    if end_time is not None and end_time < 0:
        raise ValueError("--end-time must be non-negative")
    if start_time is not None and end_time is not None and start_time > end_time:
        raise ValueError("--start-time must not exceed --end-time")
    eligible = [
        frame_idx
        for frame_idx, timestamp in enumerate(timestamps)
        if (start_time is None or timestamp >= start_time)
        and (end_time is None or timestamp <= end_time)
    ]
    if not eligible:
        raise ValueError("The requested time range contains no source video frames")
    positions = uniform_sample_positions(len(eligible), num_frames)
    return [
        {
            "output_name": f"{output_idx:06d}.png",
            "source_frame_index": eligible[position],
            "source_timestamp_seconds": timestamps[eligible[position]],
            "face_paths": {
                name: f"{name}/{output_idx:06d}.png" for name in FACE_NAMES
            },
        }
        for output_idx, position in enumerate(positions)
    ]


def default_face_size(source_width: int) -> int:
    if source_width < 4:
        raise ValueError("Source ERP width must be at least 4 pixels")
    return source_width // 4


def build_decode_command(
    ffmpeg: str,
    input_path: Path,
    source_frame_indices: list[int],
) -> list[str]:
    if not source_frame_indices:
        raise ValueError("At least one source frame index is required")
    select_expression = "+".join(
        f"eq(n\\,{frame_idx})" for frame_idx in source_frame_indices
    )
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"select='{select_expression}'",
        "-vsync",
        "0",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def _load_conversion_backend():
    install_hint = (
        "Install the required conversion backend with: "
        "python3 -m pip install 'pytorch360convert==0.2.3'"
    )
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required by pytorch360convert==0.2.3. " + install_hint
        ) from exc
    try:
        installed_version = importlib.metadata.version("pytorch360convert")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "pytorch360convert==0.2.3 is required but is not installed. "
            + install_hint
        ) from exc
    if installed_version != PYTORCH360CONVERT_VERSION:
        raise RuntimeError(
            "pytorch360convert==0.2.3 is required, but version "
            f"{installed_version} is installed. {install_hint}"
        )
    try:
        from pytorch360convert import e2c
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Could not import pytorch360convert.e2c from the required package. "
            + install_hint
        ) from exc
    return torch, e2c, installed_version


def _resolve_device(torch, requested_device: str):
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device != "cpu" and not requested_device.startswith("cuda"):
        raise ValueError("--device must be auto, cpu, cuda, or cuda:N")
    try:
        device = torch.device(requested_device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid --device value: {requested_device}") from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested_device!r} was requested, but CUDA is unavailable"
        )
    return device


def _read_exact(stream: BinaryIO, byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = stream.read(byte_count - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _tensor_to_image(face_tensor, torch) -> Image.Image:
    pixels = (
        face_tensor.detach()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(device="cpu", dtype=torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def _convert_selected_frames(
    *,
    input_path: Path,
    staging_dir: Path,
    frames: list[dict],
    source_width: int,
    source_height: int,
    face_size: int,
    batch_size: int,
    device,
    interpolation: str,
    ffmpeg: str,
    torch,
    e2c,
) -> list[str]:
    command = build_decode_command(
        ffmpeg,
        input_path,
        [frame["source_frame_index"] for frame in frames],
    )
    bytes_per_frame = source_width * source_height * 3
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    assert process.stderr is not None
    output_index = 0
    try:
        while output_index < len(frames):
            current_batch_size = min(batch_size, len(frames) - output_index)
            raw_frames: list[bytes] = []
            for _ in range(current_batch_size):
                raw = _read_exact(process.stdout, bytes_per_frame)
                if len(raw) != bytes_per_frame:
                    stderr = process.stderr.read().decode(errors="replace")
                    raise RuntimeError(
                        "FFmpeg ended before all selected frames were decoded: "
                        f"expected {bytes_per_frame} bytes, got {len(raw)}. {stderr}"
                    )
                raw_frames.append(raw)

            tensors = [
                torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                .reshape(source_height, source_width, 3)
                .permute(2, 0, 1)
                for raw in raw_frames
            ]
            erp_batch = torch.stack(tensors).to(
                device=device, dtype=torch.float32
            ) / 255.0
            with torch.inference_mode():
                cubemap = e2c(
                    erp_batch,
                    face_w=face_size,
                    mode=interpolation,
                    cube_format="stack",
                )
            expected_shape = (6, current_batch_size, 3, face_size, face_size)
            if tuple(cubemap.shape) != expected_shape:
                raise RuntimeError(
                    "Unexpected pytorch360convert e2c stack shape: expected "
                    f"{expected_shape}, got {tuple(cubemap.shape)}"
                )
            for batch_index in range(current_batch_size):
                output_name = frames[output_index + batch_index]["output_name"]
                for spec in FACE_SPECS:
                    face = cubemap[spec["e2c_stack_index"], batch_index]
                    _tensor_to_image(face, torch).save(
                        staging_dir / spec["output_name"] / output_name
                    )
            output_index += current_batch_size

        if process.stdout.read(1):
            raise RuntimeError("FFmpeg decoded more frames than were selected")
        stderr = process.stderr.read().decode(errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command, stderr=stderr)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        process.communicate()
        raise
    return command


def _verify_outputs(staging_dir: Path, frames: list[dict], face_size: int):
    expected_names = [frame["output_name"] for frame in frames]
    for face_name in FACE_NAMES:
        face_dir = staging_dir / face_name
        image_paths = sorted(face_dir.glob("*.png"))
        actual_names = [path.name for path in image_paths]
        if actual_names != expected_names:
            raise RuntimeError(
                f"Unexpected {face_name} output names: expected "
                f"{len(expected_names)}, got {len(actual_names)}"
            )
        for image_path in image_paths:
            with Image.open(image_path) as image:
                if image.size != (face_size, face_size):
                    raise RuntimeError(
                        f"Unexpected image size for {image_path}: {image.size}"
                    )


def _duration_seconds(probe: dict) -> float | None:
    return _optional_float(probe["format"].get("duration")) or _optional_float(
        probe["stream"].get("duration")
    )


def prepare_dataset(args: argparse.Namespace) -> Path:
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"ERP video not found: {input_path}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {output_dir}"
        )
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    torch, e2c, package_version = _load_conversion_backend()
    device = _resolve_device(torch, args.device)
    probe = probe_video(input_path, args.ffprobe)
    stream = probe["stream"]
    source_width = int(stream["width"])
    source_height = int(stream["height"])
    face_size = (
        default_face_size(source_width)
        if args.face_size is None
        else args.face_size
    )
    if face_size <= 0:
        raise ValueError("--face-size must be positive")
    frames = choose_source_frames(
        probe["timestamps"], args.num_frames, args.start_time, args.end_time
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = output_dir.parent / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    try:
        for face_name in FACE_NAMES:
            (staging_dir / face_name).mkdir(parents=True, exist_ok=False)
        print(
            f"Preparing {len(frames)} synchronized cubemap5 frames from {input_path}",
            flush=True,
        )
        print(
            f"Faces: {face_size}x{face_size}, FOV=90x90 degrees, "
            f"device={device}, batch_size={args.batch_size}",
            flush=True,
        )
        decode_command = _convert_selected_frames(
            input_path=input_path,
            staging_dir=staging_dir,
            frames=frames,
            source_width=source_width,
            source_height=source_height,
            face_size=face_size,
            batch_size=args.batch_size,
            device=device,
            interpolation=args.interpolation,
            ffmpeg=args.ffmpeg,
            torch=torch,
            e2c=e2c,
        )
        _verify_outputs(staging_dir, frames, face_size)

        focal_pixels = face_size / 2.0
        manifest = {
            "schema_version": 2,
            "source": {
                "path": str(input_path),
                "codec": stream.get("codec_name"),
                "width": source_width,
                "height": source_height,
                "pixel_format": stream.get("pix_fmt"),
                "average_fps": probe["fps"],
                "source_frame_count": len(probe["timestamps"]),
                "timestamp_source": probe["timestamp_source"],
                "duration_seconds": _duration_seconds(probe),
            },
            "sampling": {
                "strategy": "uniform_midpoint_bins_over_source_frames",
                "requested_frame_count": args.num_frames,
                "output_frame_count": len(frames),
                "start_time_seconds": args.start_time,
                "end_time_seconds": args.end_time,
            },
            "output": {
                "face_size": face_size,
                "width": face_size,
                "height": face_size,
                "image_format": "png",
                "hfov_degrees": 90.0,
                "vfov_degrees": 90.0,
                "fx_pixels": focal_pixels,
                "fy_pixels": focal_pixels,
                "cx_pixels": face_size / 2.0,
                "cy_pixels": face_size / 2.0,
                "interpolation": args.interpolation,
                "device": str(device),
                "batch_size": args.batch_size,
            },
            "cubemap": {
                "projection": "standard_perspective_cubemap",
                "e2c_full_stack_order": [
                    "Front", "Right", "Back", "Left", "Up", "Down"
                ],
                "face_mapping": list(FACE_SPECS),
                "omitted_faces": [
                    {
                        "cubemap_face": "Back",
                        "e2c_stack_index": 2,
                        "look_axis": "-Z",
                    }
                ],
                "orientation_convention": (
                    "pytorch360convert e2c coordinates: Front=+Z, Right=+X, "
                    "Back=-Z, Left=-X, Up=+Y, Down=-Y; no post-rotation or "
                    "flip is applied"
                ),
            },
            "frames": frames,
            "tools": {
                "ffmpeg": _tool_version(args.ffmpeg),
                "ffprobe": _tool_version(args.ffprobe),
                "ffmpeg_role": "video probe/decode/frame selection only",
                "ffmpeg_decode_command": decode_command,
                "projection": "pytorch360convert.e2c",
                "pytorch360convert": package_version,
                "torch": torch.__version__,
            },
        }
        (staging_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        )
        staging_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    print(f"Prepared cubemap5 dataset: {output_dir}", flush=True)
    print(f"Manifest: {output_dir / MANIFEST_NAME}", flush=True)
    return output_dir


def main():
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
