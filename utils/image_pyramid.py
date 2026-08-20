import json
from concurrent.futures import ThreadPoolExecutor
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
    low_path: str | None
    high_path: str | None
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


@dataclass
class InMemoryImagePyramidResult:
    low_images: torch.Tensor | None
    high_images: torch.Tensor
    image_names: list[str]
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


def _build_image_pyramid_record(
    source_path,
    images_dir,
    low_dir,
    high_dir,
    stage1_downscale_n,
    stage2_scale_factor,
    multiple,
):
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
    return ImagePyramidRecord(
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


def _build_image_pyramid_tensors_record(
    source_path,
    images_dir,
    stage1_downscale_n,
    stage2_scale_factor,
    multiple,
):
    relative_path = source_path.relative_to(images_dir)

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

        high_width_base = low_width_base * stage2_scale_factor
        high_height_base = low_height_base * stage2_scale_factor
        high_base = image.resize(
            (high_width_base, high_height_base),
            resample=Image.Resampling.LANCZOS,
        )
        high_crop_box = scale_box(low_crop_box, stage2_scale_factor)
        high_image = high_base.crop(high_crop_box)

        low_array = np.asarray(low_image, dtype=np.float32) / 255.0
        high_array = np.asarray(high_image, dtype=np.float32) / 255.0

    low_width = low_crop_box[2] - low_crop_box[0]
    low_height = low_crop_box[3] - low_crop_box[1]
    high_width = high_crop_box[2] - high_crop_box[0]
    high_height = high_crop_box[3] - high_crop_box[1]
    record = ImagePyramidRecord(
        source_path=str(source_path),
        relative_path=str(relative_path),
        low_path=None,
        high_path=None,
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
    low_tensor = torch.from_numpy(low_array).permute(2, 0, 1)
    high_tensor = torch.from_numpy(high_array).permute(2, 0, 1)
    return low_tensor, high_tensor, record


def build_two_resolution_image_tensors(
    images_dir,
    output_dir,
    stage1_downscale_n=4,
    multiple=14,
    stage2_scale_factor=None,
    recursive=False,
    num_workers=16,
    subsample=1,
    num_images=-1,
):
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage1_downscale_n = int(stage1_downscale_n)
    if stage1_downscale_n <= 0:
        raise ValueError("stage1_downscale_n must be >= 1")
    if stage2_scale_factor is None:
        stage2_scale_factor = stage1_downscale_n
    stage2_scale_factor = int(stage2_scale_factor)
    if stage2_scale_factor <= 0:
        raise ValueError("stage2_scale_factor must be >= 1")
    num_workers = int(num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1")
    subsample = int(subsample)
    if subsample <= 0:
        raise ValueError("subsample must be >= 1")

    image_paths = iter_image_files(images_dir, recursive=recursive)
    if num_images != -1:
        image_paths = image_paths[:num_images]
    image_paths = image_paths[::subsample]
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    worker_count = min(num_workers, len(image_paths))

    def worker(source_path):
        return _build_image_pyramid_tensors_record(
            source_path,
            images_dir,
            stage1_downscale_n,
            stage2_scale_factor,
            multiple,
        )

    if worker_count == 1:
        built = [worker(source_path) for source_path in image_paths]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            built = list(executor.map(worker, image_paths))

    low_tensors = [item[0] for item in built]
    high_tensors = [item[1] for item in built]
    records = [item[2] for item in built]
    low_shapes = {tuple(tensor.shape[-2:]) for tensor in low_tensors}
    high_shapes = {tuple(tensor.shape[-2:]) for tensor in high_tensors}
    if len(low_shapes) != 1 or len(high_shapes) != 1:
        raise ValueError(
            "Pipeline images must have common low/high shapes. "
            f"Got low={sorted(low_shapes)}, high={sorted(high_shapes)}"
        )

    low_images = torch.stack(low_tensors)
    high_images = torch.stack(high_tensors)
    image_names = [str(path) for path in image_paths]
    manifest_path = output_dir / "image_pyramid_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "images_dir": str(images_dir),
                "materialization": "memory",
                "stage1_downscale_n": int(stage1_downscale_n),
                "stage2_scale_factor": int(stage2_scale_factor),
                "multiple": int(multiple),
                "num_workers": int(worker_count),
                "same_fov_low_high": True,
                "records": [asdict(record) for record in records],
            },
            f,
            indent=2,
        )

    print(
        "[IMAGE_PYRAMID] Built in-memory two-resolution tensors: "
        f"images={len(records)}, workers={worker_count}, "
        f"low_shape={tuple(low_images.shape)}, "
        f"high_shape={tuple(high_images.shape)}"
    )
    return InMemoryImagePyramidResult(
        low_images=low_images,
        high_images=high_images,
        image_names=image_names,
        manifest_path=manifest_path,
        records=records,
    )


def build_two_resolution_image_pyramid(
    images_dir,
    output_dir,
    stage1_downscale_n=4,
    multiple=14,
    stage2_scale_factor=None,
    recursive=False,
    num_workers=16,
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
    num_workers = int(num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1")

    image_paths = iter_image_files(images_dir, recursive=recursive)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    worker_count = min(num_workers, len(image_paths))
    if worker_count == 1:
        records = [
            _build_image_pyramid_record(
                source_path,
                images_dir,
                low_dir,
                high_dir,
                stage1_downscale_n,
                stage2_scale_factor,
                multiple,
            )
            for source_path in image_paths
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            records = list(
                executor.map(
                    lambda source_path: _build_image_pyramid_record(
                        source_path,
                        images_dir,
                        low_dir,
                        high_dir,
                        stage1_downscale_n,
                        stage2_scale_factor,
                        multiple,
                    ),
                    image_paths,
                )
            )

    print(
        "[IMAGE_PYRAMID] Built two-resolution images: "
        f"images={len(records)}, workers={worker_count}, low_dir={low_dir}, "
        f"high_dir={high_dir}"
    )

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
                "num_workers": int(worker_count),
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


def _load_matching_high_image(low_name, high_dir, low_dir):
    low_path = Path(low_name).resolve()
    relative_path = low_path.relative_to(low_dir)
    high_path = high_dir / relative_path
    if not high_path.exists():
        raise FileNotFoundError(f"Missing high-resolution match: {high_path}")
    with Image.open(high_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor, str(high_path)


def load_matching_high_images(
    high_dir,
    low_image_names,
    low_dir,
    device="cpu",
    num_workers=16,
):
    high_dir = Path(high_dir).resolve()
    low_dir = Path(low_dir).resolve()
    low_image_names = list(low_image_names)
    if not low_image_names:
        raise ValueError("low_image_names is empty")

    num_workers = int(num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1")
    worker_count = min(num_workers, len(low_image_names))

    if worker_count == 1:
        loaded = [
            _load_matching_high_image(low_name, high_dir, low_dir)
            for low_name in low_image_names
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            loaded = list(
                executor.map(
                    lambda low_name: _load_matching_high_image(
                        low_name,
                        high_dir,
                        low_dir,
                    ),
                    low_image_names,
                )
            )

    tensors = [item[0] for item in loaded]
    high_paths = [item[1] for item in loaded]
    shapes = {tuple(tensor.shape[-2:]) for tensor in tensors}

    if len(shapes) != 1:
        raise ValueError(
            "High-resolution pipeline images must have a common shape. "
            f"Got shapes: {sorted(shapes)}"
        )
    print(
        "[IMAGE_PYRAMID] Loaded matching high images: "
        f"images={len(high_paths)}, workers={worker_count}, shape={tensors[0].shape}"
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


def scale_intrinsics_with_pyramid_records(intrinsics, manifest_records):
    records = list(manifest_records)
    intrinsic_np = np.asarray(intrinsics, dtype=np.float32).copy()
    if intrinsic_np.ndim == 2:
        intrinsic_np = np.repeat(intrinsic_np[None], len(records), axis=0)
    if intrinsic_np.ndim != 3 or intrinsic_np.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected intrinsics shape (3, 3) or (N, 3, 3), "
            f"got {intrinsic_np.shape}"
        )
    if intrinsic_np.shape[0] != len(records):
        raise ValueError(
            f"Expected {len(records)} intrinsics, got {intrinsic_np.shape[0]}"
        )

    for idx, record in enumerate(records):
        scale_x, scale_y = record.low_to_high_scale_xy
        intrinsic_np[idx, 0, 0] *= scale_x
        intrinsic_np[idx, 0, 2] *= scale_x
        intrinsic_np[idx, 1, 1] *= scale_y
        intrinsic_np[idx, 1, 2] *= scale_y
    return intrinsic_np
