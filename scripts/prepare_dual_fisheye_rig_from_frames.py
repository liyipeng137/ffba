#!/usr/bin/env python3
"""Create nine approximate pinhole views from paired fisheye still images.

Only this input layout is supported::

    input_dir/
      camera1/<images>
      camera2/<matching images>
      gyro.txt

``gyro.txt`` is a headerless CSV with rows of the form::

    image_name,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z

This is intentionally a calibration-free preparation path. It detects each
circular image support, assumes an equidistant fisheye with configurable FOV,
and composes the validated accelerometer roll correction into a single remap.
The output FOV is therefore nominal rather than metrically calibrated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


SENSOR_NAMES = (
    "front_center",
    "front_ul",
    "front_ur",
    "front_dl",
    "front_dr",
    "back_ul",
    "back_ur",
    "back_dl",
    "back_dr",
)
DIAGONAL_SIGNS = {
    "ul": (-1.0, -1.0),
    "ur": (1.0, -1.0),
    "dl": (-1.0, 1.0),
    "dr": (1.0, 1.0),
}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MANIFEST_NAME = "dual_fisheye_frames_rig_manifest.json"
REPORT_NAME = "prepare_report.json"


@dataclass(frozen=True)
class ImuRow:
    image_name: str
    image_key: str
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


@dataclass(frozen=True)
class FisheyeCircle:
    center_x: float
    center_y: float
    radius: float
    detection_threshold: int


@dataclass(frozen=True)
class SensorGeometry:
    name: str
    source_camera: str
    role: str
    quadrant: str | None
    tilt_degrees: float
    azimuth_degrees: float
    view_to_leveled_lens_rotation: tuple[tuple[float, ...], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create accelerometer-leveled, approximate 90-degree nine-view "
            "images from camera1/camera2/gyro.txt."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; it must not already exist",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Uniformly sample this many rows; default: process every row",
    )
    parser.add_argument("--face-size", type=int, default=1024)
    parser.add_argument("--output-fov-degrees", type=float, default=90.0)
    parser.add_argument(
        "--fisheye-fov-degrees",
        type=float,
        default=200.0,
        help="Assumed total equidistant fisheye FOV (default: 200)",
    )
    parser.add_argument(
        "--diagonal-tilt-degrees",
        type=float,
        default=40.0,
        help="Tilt of each diagonal view from its lens axis (default: 40)",
    )
    parser.add_argument("--rim-margin-pixels", type=float, default=16.0)
    parser.add_argument("--circle-black-threshold", type=int, default=8)
    parser.add_argument(
        "--roll-reference-frame",
        default=None,
        help="Image name/stem used as zero roll; default: first gyro row",
    )
    parser.add_argument("--roll-offset-degrees", type=float, default=0.0)
    parser.add_argument(
        "--accel-median-window",
        type=int,
        default=1,
        help="Odd median window for accelerometer roll (default: 1)",
    )
    parser.add_argument(
        "--interpolation",
        choices=("nearest", "linear", "cubic", "lanczos"),
        default="linear",
    )
    parser.add_argument(
        "--save-leveled-fisheye",
        action="store_true",
        help="Also save selected roll-leveled circular fisheye images",
    )
    return parser.parse_args()


def normalize_image_key(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Empty image name in gyro.txt")
    path = Path(value)
    if path.name != value or value in {".", ".."}:
        raise ValueError(f"gyro.txt image name must be a basename: {value!r}")
    return path.stem if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES else value


def read_gyro_rows(path: Path) -> list[ImuRow]:
    if not path.is_file():
        raise FileNotFoundError(f"gyro.txt not found: {path}")
    rows: list[ImuRow] = []
    keys = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for line_number, values in enumerate(csv.reader(stream), start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if line_number == 1 and values[0].strip().lower() in {
                "image_name",
                "filename",
                "图片名",
            }:
                continue
            if len(values) != 7:
                raise ValueError(
                    f"{path}:{line_number}: expected 7 CSV fields, got {len(values)}"
                )
            image_name = values[0].strip()
            image_key = normalize_image_key(image_name)
            try:
                numbers = [float(value) for value in values[1:]]
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: non-numeric IMU value"
                ) from exc
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError(f"{path}:{line_number}: non-finite IMU value")
            if image_key in keys:
                raise ValueError(f"Duplicate gyro.txt image key: {image_key}")
            keys.add(image_key)
            rows.append(ImuRow(image_name, image_key, *numbers))
    if not rows:
        raise ValueError(f"No IMU rows found in {path}")
    return rows


def index_camera_images(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Camera directory not found: {directory}")
    images: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            continue
        if path.stem in images:
            raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
        images[path.stem] = path.resolve()
    if not images:
        raise ValueError(f"No supported images found in {directory}")
    return images


def uniform_sample_positions(item_count: int, sample_count: int) -> list[int]:
    if item_count <= 0 or sample_count <= 0:
        raise ValueError("item_count and sample_count must be positive")
    if sample_count > item_count:
        raise ValueError(
            f"Cannot sample {sample_count} unique rows from only {item_count}"
        )
    return [
        ((2 * index + 1) * item_count) // (2 * sample_count)
        for index in range(sample_count)
    ]


def select_rows(rows: list[ImuRow], num_frames: int | None) -> list[ImuRow]:
    if num_frames is None:
        return list(rows)
    positions = uniform_sample_positions(len(rows), num_frames)
    return [rows[position] for position in positions]


def _median_filter(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0 or window % 2 == 0:
        raise ValueError("--accel-median-window must be a positive odd integer")
    if window == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.asarray(
        [np.median(padded[index : index + window]) for index in range(len(values))],
        dtype=np.float64,
    )


def accelerometer_roll_corrections(
    rows: list[ImuRow],
    *,
    reference_frame: str | None,
    roll_offset_degrees: float,
    median_window: int,
) -> tuple[dict[str, dict], str]:
    raw_angles = np.asarray(
        [math.atan2(row.acc_x, row.acc_z) for row in rows], dtype=np.float64
    )
    unwrapped = np.degrees(np.unwrap(raw_angles))
    filtered = _median_filter(unwrapped, median_window)
    reference_key = (
        normalize_image_key(reference_frame) if reference_frame else rows[0].image_key
    )
    try:
        reference_index = next(
            index for index, row in enumerate(rows) if row.image_key == reference_key
        )
    except StopIteration as exc:
        raise ValueError(
            f"Roll reference frame is absent from gyro.txt: {reference_key}"
        ) from exc
    common = filtered - filtered[reference_index] + float(roll_offset_degrees)
    corrections = {
        row.image_key: {
            "raw_accel_xz_angle_degrees": float(unwrapped[index]),
            "filtered_accel_xz_angle_degrees": float(filtered[index]),
            "relative_rig_roll_degrees": float(common[index]),
            "camera1_image_rotation_degrees": float(common[index]),
            "camera2_image_rotation_degrees": float(-common[index]),
        }
        for index, row in enumerate(rows)
    }
    return corrections, reference_key


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return image


def detect_fisheye_circle(image: np.ndarray, threshold: int) -> FisheyeCircle:
    if not 0 <= threshold <= 255:
        raise ValueError("--circle-black-threshold must be in [0, 255]")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a BGR image, got shape {image.shape}")
    support = (np.max(image, axis=2) > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("Could not detect non-black fisheye support")
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    height, width = image.shape[:2]
    if area < 0.25 * width * height:
        raise RuntimeError(
            "Detected fisheye support is too small; adjust --circle-black-threshold"
        )
    (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
    if radius <= 0.25 * min(width, height):
        raise RuntimeError("Detected fisheye circle radius is implausibly small")
    return FisheyeCircle(
        center_x=float(center_x),
        center_y=float(center_y),
        radius=float(radius),
        detection_threshold=int(threshold),
    )


def detect_camera_circle(
    paths: list[Path], threshold: int
) -> tuple[FisheyeCircle, list[dict]]:
    if not paths:
        raise ValueError("At least one image is required for circle detection")
    positions = sorted({0, len(paths) // 2, len(paths) - 1})
    detections = [
        detect_fisheye_circle(_read_image(paths[index]), threshold)
        for index in positions
    ]
    center_x = float(np.median([item.center_x for item in detections]))
    center_y = float(np.median([item.center_y for item in detections]))
    detected_radius = float(np.median([item.radius for item in detections]))
    height, width = _read_image(paths[0]).shape[:2]
    inscribed_radius = min(
        center_x,
        center_y,
        width - 1.0 - center_x,
        height - 1.0 - center_y,
    )
    result = FisheyeCircle(
        center_x=center_x,
        center_y=center_y,
        # A fitted support circle can extend outside a cropped square JPEG.
        # Sampling an inscribed circle keeps every source coordinate valid
        # after the accelerometer-derived roll rotation.
        radius=min(detected_radius, inscribed_radius),
        detection_threshold=threshold,
    )
    details = [
        {"source_path": str(paths[position]), **asdict(detection)}
        for position, detection in zip(positions, detections, strict=True)
    ]
    return result, details


def pinhole_intrinsics(face_size: int, fov_degrees: float) -> np.ndarray:
    if face_size <= 0:
        raise ValueError("--face-size must be positive")
    if not 0.0 < fov_degrees < 180.0:
        raise ValueError("--output-fov-degrees must be in (0, 180)")
    focal = face_size / (2.0 * math.tan(math.radians(fov_degrees) / 2.0))
    principal = face_size / 2.0
    return np.asarray(
        [[focal, 0.0, principal], [0.0, focal, principal], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def diagonal_direction(
    quadrant: str, tilt_degrees: float
) -> tuple[np.ndarray, float]:
    if quadrant not in DIAGONAL_SIGNS:
        raise ValueError(f"Unknown diagonal quadrant: {quadrant}")
    x_sign, y_sign = DIAGONAL_SIGNS[quadrant]
    tilt = math.radians(tilt_degrees)
    azimuth = math.atan2(y_sign, x_sign)
    direction = np.asarray(
        [
            math.sin(tilt) * math.cos(azimuth),
            math.sin(tilt) * math.sin(azimuth),
            math.cos(tilt),
        ],
        dtype=np.float64,
    )
    return _normalize(direction), math.degrees(azimuth)


def view_to_lens_rotation(direction: np.ndarray) -> np.ndarray:
    forward = _normalize(np.asarray(direction, dtype=np.float64))
    reference_down = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(reference_down, forward)
    if np.linalg.norm(right) < 1e-8:
        reference_down = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        right = np.cross(reference_down, forward)
    right = _normalize(right)
    down = _normalize(np.cross(forward, right))
    rotation = np.column_stack([right, down, forward])
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
        raise RuntimeError("View rotation is not orthonormal")
    return rotation


def make_sensor_geometries(
    diagonal_tilt_degrees: float,
) -> list[SensorGeometry]:
    if not 0.0 < diagonal_tilt_degrees < 90.0:
        raise ValueError("--diagonal-tilt-degrees must be in (0, 90)")
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(3))
    sensors = [
        SensorGeometry(
            name="front_center",
            source_camera="camera1",
            role="feed_forward_reference",
            quadrant=None,
            tilt_degrees=0.0,
            azimuth_degrees=0.0,
            view_to_leveled_lens_rotation=identity,
        )
    ]
    for prefix, source_camera in (("front", "camera1"), ("back", "camera2")):
        for quadrant in DIAGONAL_SIGNS:
            direction, azimuth = diagonal_direction(
                quadrant, diagonal_tilt_degrees
            )
            rotation = view_to_lens_rotation(direction)
            sensors.append(
                SensorGeometry(
                    name=f"{prefix}_{quadrant}",
                    source_camera=source_camera,
                    role="rig_supplementary",
                    quadrant=quadrant,
                    tilt_degrees=diagonal_tilt_degrees,
                    azimuth_degrees=azimuth,
                    view_to_leveled_lens_rotation=tuple(
                        tuple(float(value) for value in row)
                        for row in rotation
                    ),
                )
            )
    if tuple(sensor.name for sensor in sensors) != SENSOR_NAMES:
        raise RuntimeError("Internal sensor order does not match SENSOR_NAMES")
    return sensors


def build_approximate_equidistant_remap(
    *,
    face_size: int,
    output_fov_degrees: float,
    fisheye_fov_degrees: float,
    circle: FisheyeCircle,
    source_size: tuple[int, int],
    geometry: SensorGeometry,
    rim_margin_pixels: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    if not 90.0 < fisheye_fov_degrees <= 360.0:
        raise ValueError("--fisheye-fov-degrees must be in (90, 360]")
    if rim_margin_pixels < 0.0 or rim_margin_pixels >= circle.radius:
        raise ValueError(
            "--rim-margin-pixels must be non-negative and below radius"
        )
    intrinsics = pinhole_intrinsics(face_size, output_fov_degrees)
    coordinates = np.arange(face_size, dtype=np.float64) + 0.5
    normalized_x = (coordinates - intrinsics[0, 2]) / intrinsics[0, 0]
    normalized_y = (coordinates - intrinsics[1, 2]) / intrinsics[1, 1]
    grid_x, grid_y = np.meshgrid(normalized_x, normalized_y)
    rays_view = np.stack(
        [grid_x, grid_y, np.ones_like(grid_x)], axis=-1
    )
    rays_view /= np.linalg.norm(rays_view, axis=-1, keepdims=True)
    rotation = np.asarray(geometry.view_to_leveled_lens_rotation)
    rays_lens = rays_view @ rotation.T

    z = np.clip(rays_lens[..., 2], -1.0, 1.0)
    theta = np.arccos(z)
    planar = np.hypot(rays_lens[..., 0], rays_lens[..., 1])
    source_radius = (
        theta
        * circle.radius
        / math.radians(fisheye_fov_degrees / 2.0)
    )
    scale = np.divide(
        source_radius,
        planar,
        out=np.zeros_like(source_radius),
        where=planar > 1e-12,
    )
    map_x = circle.center_x + rays_lens[..., 0] * scale
    map_y = circle.center_y + rays_lens[..., 1] * scale
    width, height = source_size
    valid_radius = circle.radius - rim_margin_pixels
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (theta <= math.radians(fisheye_fov_degrees / 2.0))
        & (source_radius <= valid_radius)
        & (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    stats = {
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "total_pixel_count": int(valid.size),
        "valid_pixel_ratio": float(np.mean(valid)),
        "maximum_lens_angle_degrees": float(np.degrees(np.max(theta))),
        "maximum_source_radius_pixels": float(np.max(source_radius)),
        "valid_source_radius_pixels": float(valid_radius),
    }
    return (
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        valid.astype(np.uint8) * 255,
        stats,
    )


def compose_image_rotation_into_remap(
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    circle: FisheyeCircle,
    image_rotation_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Map leveled-image coordinates directly back to the raw source."""
    source_to_leveled = cv2.getRotationMatrix2D(
        (circle.center_x, circle.center_y),
        image_rotation_degrees,
        1.0,
    )
    leveled_to_source = cv2.invertAffineTransform(source_to_leveled)
    source_x = (
        leveled_to_source[0, 0] * map_x
        + leveled_to_source[0, 1] * map_y
        + leveled_to_source[0, 2]
    )
    source_y = (
        leveled_to_source[1, 0] * map_x
        + leveled_to_source[1, 1] * map_y
        + leveled_to_source[1, 2]
    )
    return source_x.astype(np.float32), source_y.astype(np.float32)


def _interpolation_flag(value: str) -> int:
    return {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }[value]


def _write_preview(output_dir: Path, output_name: str, face_size: int) -> Path:
    tile_size = min(320, face_size)
    label_height = 28
    canvas = Image.new(
        "RGB", (tile_size * 3, (tile_size + label_height) * 3)
    )
    draw = ImageDraw.Draw(canvas)
    for index, sensor_name in enumerate(SENSOR_NAMES):
        row, column = divmod(index, 3)
        path = output_dir / sensor_name / output_name
        with Image.open(path).convert("RGB") as image:
            image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
            x = column * tile_size
            y = row * (tile_size + label_height)
            canvas.paste(image, (x, y))
            draw.rectangle(
                (
                    x,
                    y + tile_size,
                    x + tile_size,
                    y + tile_size + label_height,
                ),
                fill="black",
            )
            draw.text(
                (x + 6, y + tile_size + 6), sensor_name, fill="white"
            )
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    path = preview_dir / "first_frame_9view_contact_sheet.jpg"
    canvas.save(path, quality=92)
    return path


def _validate_source_sizes(
    rows: list[ImuRow],
    camera_images: dict[str, dict[str, Path]],
) -> tuple[int, int]:
    expected = None
    for row in rows:
        for camera_name in ("camera1", "camera2"):
            path = camera_images[camera_name][row.image_key]
            with Image.open(path) as image:
                size = image.size
            if size[0] != size[1]:
                raise ValueError(
                    f"Expected square fisheye image, got {size}: {path}"
                )
            if expected is None:
                expected = size
            elif size != expected:
                raise ValueError(
                    f"All images must share one size; got {size} vs "
                    f"{expected}: {path}"
                )
    assert expected is not None
    return expected


def prepare_dataset(args: argparse.Namespace) -> Path:
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a new path: {output_dir}"
        )

    rows = read_gyro_rows(input_dir / "gyro.txt")
    selected_rows = select_rows(rows, args.num_frames)
    camera_images = {
        "camera1": index_camera_images(input_dir / "camera1"),
        "camera2": index_camera_images(input_dir / "camera2"),
    }
    for row in rows:
        for camera_name in ("camera1", "camera2"):
            if row.image_key not in camera_images[camera_name]:
                raise FileNotFoundError(
                    f"Missing {camera_name} image for gyro row: "
                    f"{row.image_name}"
                )
    source_size = _validate_source_sizes(rows, camera_images)
    corrections, reference_key = accelerometer_roll_corrections(
        rows,
        reference_frame=args.roll_reference_frame,
        roll_offset_degrees=args.roll_offset_degrees,
        median_window=args.accel_median_window,
    )

    circles = {}
    circle_details = {}
    for camera_name in ("camera1", "camera2"):
        paths = [
            camera_images[camera_name][row.image_key]
            for row in selected_rows
        ]
        (
            circles[camera_name],
            circle_details[camera_name],
        ) = detect_camera_circle(paths, args.circle_black_threshold)

    sensors = make_sensor_geometries(args.diagonal_tilt_degrees)
    base_remaps = {}
    masks = {}
    remap_stats = {}
    for sensor in sensors:
        map_x, map_y, mask, stats = build_approximate_equidistant_remap(
            face_size=args.face_size,
            output_fov_degrees=args.output_fov_degrees,
            fisheye_fov_degrees=args.fisheye_fov_degrees,
            circle=circles[sensor.source_camera],
            source_size=source_size,
            geometry=sensor,
            rim_margin_pixels=args.rim_margin_pixels,
        )
        base_remaps[sensor.name] = (map_x, map_y)
        masks[sensor.name] = mask
        remap_stats[sensor.name] = stats

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = (
        output_dir.parent
        / f".{output_dir.name}.tmp-{uuid.uuid4().hex}"
    )
    try:
        for sensor_name in SENSOR_NAMES:
            (staging_dir / sensor_name).mkdir(
                parents=True, exist_ok=False
            )
        (staging_dir / "masks").mkdir(parents=True, exist_ok=False)
        if args.save_leveled_fisheye:
            (staging_dir / "leveled_camera1").mkdir(
                parents=True, exist_ok=False
            )
            (staging_dir / "leveled_camera2").mkdir(
                parents=True, exist_ok=False
            )
        for sensor_name, mask in masks.items():
            path = staging_dir / "masks" / f"{sensor_name}.png"
            if not cv2.imwrite(str(path), mask):
                raise RuntimeError(f"Failed to write mask: {path}")

        interpolation = _interpolation_flag(args.interpolation)
        frame_payloads = []
        print(
            f"Preparing {len(selected_rows)} synchronized nine-view groups",
            flush=True,
        )
        for output_index, row in enumerate(selected_rows):
            sources = {
                camera_name: _read_image(
                    camera_images[camera_name][row.image_key]
                )
                for camera_name in ("camera1", "camera2")
            }
            frame_corrections = corrections[row.image_key]
            output_name = f"{row.image_key}.png"

            if args.save_leveled_fisheye:
                for camera_name in ("camera1", "camera2"):
                    angle = frame_corrections[
                        f"{camera_name}_image_rotation_degrees"
                    ]
                    circle = circles[camera_name]
                    matrix = cv2.getRotationMatrix2D(
                        (circle.center_x, circle.center_y), angle, 1.0
                    )
                    leveled = cv2.warpAffine(
                        sources[camera_name],
                        matrix,
                        source_size,
                        flags=interpolation,
                        borderMode=cv2.BORDER_CONSTANT,
                    )
                    path = (
                        staging_dir
                        / f"leveled_{camera_name}"
                        / output_name
                    )
                    if not cv2.imwrite(str(path), leveled):
                        raise RuntimeError(
                            f"Failed to write leveled fisheye: {path}"
                        )

            sensor_paths = {}
            for sensor in sensors:
                camera_name = sensor.source_camera
                image_rotation = frame_corrections[
                    f"{camera_name}_image_rotation_degrees"
                ]
                map_x, map_y = compose_image_rotation_into_remap(
                    *base_remaps[sensor.name],
                    circle=circles[camera_name],
                    image_rotation_degrees=image_rotation,
                )
                output = cv2.remap(
                    sources[camera_name],
                    map_x,
                    map_y,
                    interpolation=interpolation,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                output[masks[sensor.name] == 0] = 0
                path = staging_dir / sensor.name / output_name
                if not cv2.imwrite(str(path), output):
                    raise RuntimeError(f"Failed to write output: {path}")
                sensor_paths[sensor.name] = (
                    f"{sensor.name}/{output_name}"
                )

            frame_payloads.append(
                {
                    "output_index": output_index,
                    "image_key": row.image_key,
                    "output_name": output_name,
                    "camera1_source": str(
                        camera_images["camera1"][row.image_key]
                    ),
                    "camera2_source": str(
                        camera_images["camera2"][row.image_key]
                    ),
                    "imu": asdict(row),
                    "roll": frame_corrections,
                    "sensor_paths": sensor_paths,
                }
            )
            if (
                (output_index + 1) % 10 == 0
                or output_index + 1 == len(selected_rows)
            ):
                print(
                    f"Processed {output_index + 1}/{len(selected_rows)}",
                    flush=True,
                )

        preview_path = _write_preview(
            staging_dir,
            frame_payloads[0]["output_name"],
            args.face_size,
        )

        pinhole_k = pinhole_intrinsics(
            args.face_size, args.output_fov_degrees
        ).tolist()
        manifest = {
            "schema_version": 1,
            "source": {
                "input_dir": str(input_dir),
                "input_layout": "camera1/camera2/gyro.txt",
                "gyro_path": str(input_dir / "gyro.txt"),
                "gyro_row_count": len(rows),
                "selected_frame_count": len(selected_rows),
                "source_width": source_size[0],
                "source_height": source_size[1],
                "camera1_semantics": "front_physical_fisheye",
                "camera2_semantics": "back_physical_fisheye",
            },
            "roll_leveling": {
                "method": "accelerometer_relative_roll_only",
                "angle_formula": (
                    "unwrap(degrees(atan2(acc_x, acc_z)))"
                ),
                "reference_frame": reference_key,
                "roll_offset_degrees": args.roll_offset_degrees,
                "accel_median_window": args.accel_median_window,
                "camera1_image_rotation_sign": 1,
                "camera2_image_rotation_sign": -1,
                "gyro_columns_status": "stored_but_not_used",
                "absolute_gravity_alignment_status": (
                    "relative_to_reference_frame_not_absolutely_calibrated"
                ),
            },
            "projection": {
                "calibration_status": (
                    "approximate_no_lens_calibration"
                ),
                "model": "assumed_equidistant_circular_fisheye",
                "assumed_fisheye_fov_degrees": (
                    args.fisheye_fov_degrees
                ),
                "output_camera_model": "nominal_PINHOLE",
                "output_width": args.face_size,
                "output_height": args.face_size,
                "output_hfov_degrees": args.output_fov_degrees,
                "output_vfov_degrees": args.output_fov_degrees,
                "output_intrinsics": pinhole_k,
                "output_distortion": [],
                "diagonal_tilt_degrees": (
                    args.diagonal_tilt_degrees
                ),
                "rim_margin_pixels": args.rim_margin_pixels,
                "interpolation": args.interpolation,
                "sampling_passes_per_output": 1,
                "detected_circles": {
                    camera_name: {
                        "selected": asdict(circles[camera_name]),
                        "detections": circle_details[camera_name],
                    }
                    for camera_name in ("camera1", "camera2")
                },
            },
            "layout": {
                "name": (
                    "approximate_dual_fisheye_8_plus_front_center"
                ),
                "sensor_order": list(SENSOR_NAMES),
                "sensors": [
                    {
                        **asdict(sensor),
                        "mask_path": f"masks/{sensor.name}.png",
                        "remap_stats": remap_stats[sensor.name],
                        "pixel_source_policy": (
                            "single_fisheye_only_no_blending"
                        ),
                    }
                    for sensor in sensors
                ],
            },
            "frames": frame_payloads,
            "warnings": [
                (
                    "Fisheye intrinsics and distortion are unavailable; "
                    "output FOV is nominal."
                ),
                (
                    "Roll is relative to the selected reference frame, "
                    "not absolute gravity."
                ),
                (
                    "Per-frame gyro values are not integrated because "
                    "timestamps are unavailable."
                ),
                (
                    "Residual distortion is expected, especially in "
                    "diagonal views near the rim."
                ),
            ],
        }
        (staging_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report = {
            "source": str(input_dir),
            "output": str(output_dir),
            "frames": len(selected_rows),
            "views_per_frame": len(SENSOR_NAMES),
            "face_size": args.face_size,
            "nominal_output_fov_degrees": (
                args.output_fov_degrees
            ),
            "assumed_fisheye_fov_degrees": (
                args.fisheye_fov_degrees
            ),
            "diagonal_tilt_degrees": (
                args.diagonal_tilt_degrees
            ),
            "minimum_sensor_valid_pixel_ratio": min(
                item["valid_pixel_ratio"]
                for item in remap_stats.values()
            ),
            "sensor_remap_stats": remap_stats,
            "preview": str(
                output_dir / preview_path.relative_to(staging_dir)
            ),
        }
        (staging_dir / REPORT_NAME).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        staging_dir.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(
        f"Prepared approximate nine-view dataset: {output_dir}",
        flush=True,
    )
    print(f"Manifest: {output_dir / MANIFEST_NAME}", flush=True)
    print(
        "Preview: "
        f"{output_dir / 'previews' / 'first_frame_9view_contact_sheet.jpg'}",
        flush=True,
    )
    return output_dir


def main():
    prepare_dataset(parse_args())


if __name__ == "__main__":
    main()
