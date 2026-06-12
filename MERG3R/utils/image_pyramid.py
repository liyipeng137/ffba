import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass
class ImagePyramidRecord:
    source_path: str
    relative_path: str
    low_path: str
    high_path: str
    source_size_wh: tuple[int, int]
    low_base_size_wh: tuple[int, int]
    low_size_wh: tuple[int, int]
    low_crop_box: tuple[int, int, int, int]
    high_base_size_wh: tuple[int, int]
    high_size_wh: tuple[int, int]
    high_crop_box: tuple[int, int, int, int]
    low_to_high_scale_xy: tuple[float, float]


@dataclass
class ImagePyramidResult:
    low_dir: Path
    high_dir: Path
    manifest_path: Path
    records: list[ImagePyramidRecord]


def iter_image_files(images_dir, recursive=False):
    images_dir = Path(images_dir)
    iterator = images_dir.rglob("*") if recursive else images_dir.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )


def load_image_tensors_from_dir(
    images_dir,
    device="cpu",
    subsample=1,
    num_images=-1,
    recursive=False,
):
    image_paths = iter_image_files(images_dir, recursive=recursive)
    if num_images != -1:
        image_paths = image_paths[:num_images]
    image_paths = image_paths[::subsample]
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    tensors = []
    shapes = set()
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append(tensor)
        shapes.add(tuple(tensor.shape[-2:]))

    if len(shapes) != 1:
        raise ValueError(
            "Pipeline images must have a common shape. " f"Got shapes: {sorted(shapes)}"
        )

    images = torch.stack(tensors).to(device)
    image_names = [str(path) for path in image_paths]
    print(
        f"[IMAGE_PYRAMID] Loaded {len(image_names)} images from {images_dir}: {images.shape}"
    )
    return images, image_names


def center_crop_to_multiple(width, height, multiple):
    if multiple <= 0:
        raise ValueError("multiple must be > 0")
    new_width = width - (width % multiple)
    new_height = height - (height % multiple)
    if new_width <= 0 or new_height <= 0:
        raise ValueError(
            f"Image too small to crop to a multiple of {multiple}: " f"{width}x{height}"
        )

    dx = width - new_width
    dy = height - new_height
    left = dx // 2
    top = dy // 2
    right = width - (dx - left)
    bottom = height - (dy - top)
    return left, top, right, bottom


def scale_box(box, scale):
    left, top, right, bottom = box
    return left * scale, top * scale, right * scale, bottom * scale


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image.save(path, quality=95, subsampling=0, optimize=True)
    elif ext == ".png":
        image.save(path, optimize=True)
    else:
        image.save(path)


def build_two_resolution_image_pyramid(
    images_dir,
    output_dir,
    stage1_downscale_n=4,
    multiple=14,
    stage2_scale_factor=None,
    recursive=False,
):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    low_dir = output_dir / "low"
    high_dir = output_dir / "high"
    low_dir.mkdir(parents=True, exist_ok=True)
    high_dir.mkdir(parents=True, exist_ok=True)

    stage1_downscale_n = int(stage1_downscale_n)
    if stage1_downscale_n <= 0:
        raise ValueError("stage1_downscale_n must be >= 1")
    if stage2_scale_factor is None:
        stage2_scale_factor = stage1_downscale_n
    stage2_scale_factor = int(stage2_scale_factor)
    if stage2_scale_factor <= 0:
        raise ValueError("stage2_scale_factor must be >= 1")

    records = []
    for source_path in iter_image_files(images_dir, recursive=recursive):
        relative_path = source_path.relative_to(images_dir)
        low_path = low_dir / relative_path
        high_path = high_dir / relative_path

        with Image.open(source_path) as image_raw:
            image = ImageOps.exif_transpose(image_raw).convert("RGB")
            width0, height0 = image.size

            low_width_base = max(1, width0 // stage1_downscale_n)
            low_height_base = max(1, height0 // stage1_downscale_n)
            low_base = image.resize(
                (low_width_base, low_height_base),
                resample=Image.Resampling.LANCZOS,
            )
            low_crop_box = center_crop_to_multiple(
                low_width_base,
                low_height_base,
                multiple,
            )
            low_image = low_base.crop(low_crop_box)
            save_image(low_image, low_path)

            high_width_base = low_width_base * stage2_scale_factor
            high_height_base = low_height_base * stage2_scale_factor
            high_base = image.resize(
                (high_width_base, high_height_base),
                resample=Image.Resampling.LANCZOS,
            )
            high_crop_box = scale_box(low_crop_box, stage2_scale_factor)
            high_image = high_base.crop(high_crop_box)
            save_image(high_image, high_path)

        low_width = low_crop_box[2] - low_crop_box[0]
        low_height = low_crop_box[3] - low_crop_box[1]
        high_width = high_crop_box[2] - high_crop_box[0]
        high_height = high_crop_box[3] - high_crop_box[1]
        records.append(
            ImagePyramidRecord(
                source_path=str(source_path),
                relative_path=str(relative_path),
                low_path=str(low_path),
                high_path=str(high_path),
                source_size_wh=(int(width0), int(height0)),
                low_base_size_wh=(int(low_width_base), int(low_height_base)),
                low_size_wh=(int(low_width), int(low_height)),
                low_crop_box=tuple(int(v) for v in low_crop_box),
                high_base_size_wh=(int(high_width_base), int(high_height_base)),
                high_size_wh=(int(high_width), int(high_height)),
                high_crop_box=tuple(int(v) for v in high_crop_box),
                low_to_high_scale_xy=(
                    float(high_width) / float(low_width),
                    float(high_height) / float(low_height),
                ),
            )
        )

    if not records:
        raise ValueError(f"No images found in {images_dir}")

    manifest_path = output_dir / "image_pyramid_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "images_dir": str(images_dir),
                "low_dir": str(low_dir),
                "high_dir": str(high_dir),
                "stage1_downscale_n": int(stage1_downscale_n),
                "stage2_scale_factor": int(stage2_scale_factor),
                "multiple": int(multiple),
                "same_fov_low_high": True,
                "records": [asdict(record) for record in records],
            },
            f,
            indent=2,
        )

    return ImagePyramidResult(
        low_dir=low_dir,
        high_dir=high_dir,
        manifest_path=manifest_path,
        records=records,
    )


def load_matching_high_images(high_dir, low_image_names, low_dir, device="cpu"):
    high_dir = Path(high_dir).resolve()
    low_dir = Path(low_dir).resolve()
    tensors = []
    high_paths = []
    shapes = set()
    for low_name in low_image_names:
        low_path = Path(low_name).resolve()
        relative_path = low_path.relative_to(low_dir)
        high_path = high_dir / relative_path
        if not high_path.exists():
            raise FileNotFoundError(f"Missing high-resolution match: {high_path}")
        with Image.open(high_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append(tensor)
        high_paths.append(str(high_path))
        shapes.add(tuple(tensor.shape[-2:]))

    if len(shapes) != 1:
        raise ValueError(
            "High-resolution pipeline images must have a common shape. "
            f"Got shapes: {sorted(shapes)}"
        )
    return torch.stack(tensors).to(device), high_paths


def scale_intrinsics_low_to_high(
    intrinsics, low_image_names, low_dir, manifest_records
):
    intrinsic_np = np.asarray(intrinsics, dtype=np.float32).copy()
    if intrinsic_np.ndim == 2:
        intrinsic_np = np.repeat(intrinsic_np[None], len(low_image_names), axis=0)
    if intrinsic_np.ndim != 3 or intrinsic_np.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected intrinsics shape (3, 3) or (N, 3, 3), "
            f"got {intrinsic_np.shape}"
        )
    if intrinsic_np.shape[0] != len(low_image_names):
        raise ValueError(
            f"Expected {len(low_image_names)} intrinsics, got {intrinsic_np.shape[0]}"
        )

    low_dir = Path(low_dir).resolve()
    records_by_relative = {
        Path(record.relative_path): record for record in manifest_records
    }
    for idx, low_name in enumerate(low_image_names):
        relative_path = Path(low_name).resolve().relative_to(low_dir)
        record = records_by_relative[relative_path]
        scale_x, scale_y = record.low_to_high_scale_xy
        intrinsic_np[idx, 0, 0] *= scale_x
        intrinsic_np[idx, 0, 2] *= scale_x
        intrinsic_np[idx, 1, 1] *= scale_y
        intrinsic_np[idx, 1, 2] *= scale_y
    return intrinsic_np
