#!/usr/bin/env python3
"""Prepare high/low resolution image sets with VGGT-Omega preprocessing.

For each input image, this script applies the same image-space preprocessing as
`MERG3R/feedforward/vggt_omega/utils/load_fn.py`:

1. EXIF-transpose and convert to RGB, compositing alpha over white.
2. Center-crop extreme aspect ratios into [0.5, 2.0].
3. Resize the low-resolution image with VGGT-Omega's `balanced` or `max_size`
   target-shape rule, so low width/height are multiples of the patch size.
4. Resize the high-resolution image to exactly `scale_n` times the low size.
5. Optionally pad all images in each output set to a common size, matching the
   VGGT-Omega loader behavior for mixed image shapes. In scaled mode, the high
   common size is exactly `scale_n` times the low common size.

The high and low outputs are generated from the same cropped source image, so
they preserve the same scene field-of-view and only differ in sampling density
and optional per-set padding. With the default `--scale-n 2`, every high image
can be downsampled by exactly 2x in both axes to recover the low image size.

Example:
  python scripts/preprocess_vggt_omega_images.py \
    --images ./images \
    --high-out ./images_vggt_omega_high \
    --low-out ./images_vggt_omega_low \
    --low-resolution 512 \
    --scale-n 2
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


@dataclass
class ImagePlan:
    rel_path: str
    original_size: Tuple[int, int]
    crop_box: Tuple[int, int, int, int]
    cropped_size: Tuple[int, int]
    high_size: Tuple[int, int]
    low_size: Tuple[int, int]


def _iter_images(images_dir: Path, recursive: bool) -> Iterable[Path]:
    iterator = images_dir.rglob("*") if recursive else images_dir.iterdir()
    for path in iterator:
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")


def _supported_aspect_crop_box(
    width: int,
    height: int,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> Tuple[int, int, int, int]:
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return left, 0, left + crop_width, height

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return 0, top, width, top + crop_height

    return 0, 0, width, height


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def _balanced_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> Tuple[int, int]:
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _max_size_target_shape(aspect_ratio: float, image_resolution: int, patch_size: int) -> Tuple[int, int]:
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = _round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = _round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def _target_shape(
    width: int,
    height: int,
    image_resolution: int,
    patch_size: int,
    mode: str,
) -> Tuple[int, int]:
    if image_resolution <= 0:
        raise ValueError("image_resolution must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be divisible by patch_size")

    aspect_ratio = height / max(width, 1)
    if mode == "balanced":
        return _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
    if mode == "max_size":
        return _max_size_target_shape(aspect_ratio, image_resolution, patch_size)
    raise ValueError("mode must be either 'balanced' or 'max_size'")


def _save_image(image: Image.Image, output_path: Path) -> None:
    _ensure_dir(output_path.parent)
    image.save(output_path)


def _pad_to_size(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    target_w, target_h = target_size
    width, height = image.size
    if width == target_w and height == target_h:
        return image
    if width > target_w or height > target_h:
        raise ValueError(f"Cannot pad {width}x{height} into smaller target {target_w}x{target_h}")

    pad_left = (target_w - width) // 2
    pad_top = (target_h - height) // 2
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    canvas.paste(image, (pad_left, pad_top))
    return canvas


def _build_plans(
    files: list[Path],
    images_dir: Path,
    low_resolution: int,
    patch_size: int,
    mode: str,
    scale_n: int,
) -> list[ImagePlan]:
    if scale_n <= 0:
        raise ValueError("scale_n must be positive")

    plans = []
    for path in files:
        with _load_rgb_image(path) as image:
            width, height = image.size
            crop_box = _supported_aspect_crop_box(width, height)
            cropped_w = crop_box[2] - crop_box[0]
            cropped_h = crop_box[3] - crop_box[1]
            low_h, low_w = _target_shape(cropped_w, cropped_h, low_resolution, patch_size, mode)
            high_w = low_w * scale_n
            high_h = low_h * scale_n

        plans.append(
            ImagePlan(
                rel_path=str(path.relative_to(images_dir)),
                original_size=(width, height),
                crop_box=crop_box,
                cropped_size=(cropped_w, cropped_h),
                high_size=(high_w, high_h),
                low_size=(low_w, low_h),
            )
        )
    return plans


def _common_size(plans: list[ImagePlan], attr: str) -> Tuple[int, int]:
    sizes = [getattr(plan, attr) for plan in plans]
    max_w = max(width for width, _ in sizes)
    max_h = max(height for _, height in sizes)
    return max_w, max_h


def _process_images(
    files: list[Path],
    images_dir: Path,
    high_out: Path,
    low_out: Path,
    plans: list[ImagePlan],
    high_common_size: Tuple[int, int] | None,
    low_common_size: Tuple[int, int] | None,
    quiet: bool,
) -> None:
    plan_by_rel = {plan.rel_path: plan for plan in plans}

    for path in files:
        rel = path.relative_to(images_dir)
        plan = plan_by_rel[str(rel)]
        out_high = high_out / rel.with_suffix(".png")
        out_low = low_out / rel.with_suffix(".png")

        with _load_rgb_image(path) as image:
            cropped = image.crop(plan.crop_box)
            high = cropped.resize(plan.high_size, Image.Resampling.BICUBIC)
            low = cropped.resize(plan.low_size, Image.Resampling.BICUBIC)

            if high_common_size is not None:
                high = _pad_to_size(high, high_common_size)
            if low_common_size is not None:
                low = _pad_to_size(low, low_common_size)

            _save_image(high, out_high)
            _save_image(low, out_low)

        if not quiet:
            print(
                f"OK  {rel} | crop {plan.cropped_size[0]}x{plan.cropped_size[1]} "
                f"-> high {out_high.name} {high.size[0]}x{high.size[1]} "
                f"-> low {out_low.name} {low.size[0]}x{low.size[1]}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create VGGT-Omega high/low image sets.")
    parser.add_argument("--images", required=True, type=Path, help="Input images directory")
    parser.add_argument("--high-out", required=True, type=Path, help="Output directory for high-res images")
    parser.add_argument("--low-out", required=True, type=Path, help="Output directory for low-res images")
    parser.add_argument("--low-resolution", default=512, type=int, help="VGGT-Omega low image_resolution")
    parser.add_argument(
        "--scale-n",
        default=2,
        type=int,
        help="High-res size is exactly low-res size multiplied by this factor.",
    )
    parser.add_argument("--patch-size", default=16, type=int, help="VGGT-Omega patch size")
    parser.add_argument(
        "--mode",
        default="balanced",
        choices=["balanced", "max_size"],
        help="VGGT-Omega resize mode. Defaults to balanced.",
    )
    parser.add_argument(
        "--no-pad-to-common",
        action="store_true",
        help="Disable VGGT-Omega-style per-set padding to the common max shape.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--manifest", type=Path, help="Optional JSON manifest output path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-file logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir: Path = args.images
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Input directory not found: {images_dir}")

    files = sorted(_iter_images(images_dir, args.recursive))
    if not files:
        raise SystemExit(f"No images found in {images_dir} (supported: {sorted(SUPPORTED_EXTS)})")

    _ensure_dir(args.high_out)
    _ensure_dir(args.low_out)

    plans = _build_plans(
        files,
        images_dir,
        low_resolution=args.low_resolution,
        patch_size=args.patch_size,
        mode=args.mode,
        scale_n=args.scale_n,
    )

    pad_to_common = not args.no_pad_to_common
    low_common_size = _common_size(plans, "low_size") if pad_to_common else None
    high_common_size = (
        (low_common_size[0] * args.scale_n, low_common_size[1] * args.scale_n)
        if pad_to_common
        else None
    )

    _process_images(
        files,
        images_dir,
        args.high_out,
        args.low_out,
        plans,
        high_common_size,
        low_common_size,
        quiet=args.quiet,
    )

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.high_out.parent / "vggt_omega_preprocess_manifest.json"
    _ensure_dir(manifest_path.parent)
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "patch_size": args.patch_size,
                "low_resolution": args.low_resolution,
                "scale_n": args.scale_n,
                "pad_to_common": pad_to_common,
                "high_common_size": high_common_size,
                "low_common_size": low_common_size,
                "num_images": len(plans),
                "images": [asdict(plan) for plan in plans],
            },
            f,
            indent=2,
        )

    print(
        f"Done. TOTAL={len(plans)} high_out={args.high_out} low_out={args.low_out} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None
    main()
