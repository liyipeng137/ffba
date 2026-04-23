#!/usr/bin/env python3
"""Prepare two-resolution image sets with consistent FOV.

Per input image (original resolution):
1) Stage-1 (low-res): resize to 1/N of the original size (integer floor), then center-crop minimally
   so that width and height are multiples of `--multiple` (default: 14). Save to `--stage1-out`.

2) Stage-2 (high-res): generate an image whose size is exactly 2x the Stage-1 *pre-crop* size,
   then apply a crop box that is exactly 2x the Stage-1 crop box. This guarantees Stage-2 has
   the same field-of-view as Stage-1 but with 2x linear resolution. Save to `--stage2-out`.

Why this works:
- Stage-1 is: I1 = crop(resize(I0, s1), box1), s1 = 1/N
- Stage-2 is: I2 = crop(resize(I0, s2), box2), s2 = 2*s1, box2 = 2*box1

Example:
  python crop_resize_image.py \
    --images ./images \
    --stage1-out ./images_stage1_half_crop14 \
    --stage2-out ./images_stage2_2x_stage1 

Notes:
- Cropping is centered and minimal (only removes the smallest border necessary).
- EXIF orientation is respected (images are transposed before processing).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from PIL import Image, ImageOps


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}


def _iter_images(images_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        for p in images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p
    else:
        for p in images_dir.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                yield p


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _unit(v):
    import numpy as np

    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v
    return v / n


def _center_crop_to_multiple(w: int, h: int, multiple: int) -> Tuple[int, int, int, int]:
    """Return (left, top, right, bottom) crop box with minimal centered crop.

    Ensures output width/height are multiples of `multiple`.
    """
    if multiple <= 0:
        raise ValueError("multiple must be > 0")

    new_w = w - (w % multiple)
    new_h = h - (h % multiple)

    if new_w <= 0 or new_h <= 0:
        raise ValueError(
            f"Image too small to crop to a multiple of {multiple}: {w}x{h}."
        )

    dx = w - new_w
    dy = h - new_h

    left = dx // 2
    right = w - (dx - left)  # distribute odd pixel to the left side

    top = dy // 2
    bottom = h - (dy - top)

    return left, top, right, bottom


def _scale_box(box: Tuple[int, int, int, int], k: int) -> Tuple[int, int, int, int]:
    l, t, r, b = box
    return l * k, t * k, r * k, b * k


def _update_intrinsics(
    fx0: float, fy0: float, cx0: float, cy0: float,
    w0: int, h0: int,
    w_new: int, h_new: int,
    crop_box: Tuple[int, int, int, int],
) -> Tuple[float, float, float, float]:
    """更新内参：先 resize 再 crop。
    
    Args:
        fx0, fy0, cx0, cy0: 原始内参
        w0, h0: 原始尺寸
        w_new, h_new: resize 后尺寸
        crop_box: (left, top, right, bottom) 裁剪框
    
    Returns:
        (fx_new, fy_new, cx_new, cy_new)
    """
    left, top, _, _ = crop_box
    
    # Resize: 内参按比例缩放
    scale_x = w_new / w0
    scale_y = h_new / h0
    fx_resized = fx0 * scale_x
    fy_resized = fy0 * scale_y
    cx_resized = cx0 * scale_x
    cy_resized = cy0 * scale_y
    
    # Crop: 焦距不变，主点减去裁剪偏移
    fx_new = fx_resized
    fy_new = fy_resized
    cx_new = cx_resized - left
    cy_new = cy_resized - top
    
    return fx_new, fy_new, cx_new, cy_new


def _set_avg_intrinsics(data: Dict, sums: Dict[str, float], count: int, stage_name: str) -> None:
    """在 transforms 顶层写入平均内参统计（与 frames 同级）。"""
    data.pop("avg_intrinsics", None)
    if count <= 0:
        print(f"警告: {stage_name} 没有可用帧用于计算平均内参，跳过写入 avg_intrinsics")
        return

    data["avg_intrinsics"] = {
        "fl_x": float(sums["fl_x"] / count),
        "fl_y": float(sums["fl_y"] / count),
        "cx": float(sums["cx"] / count),
        "cy": float(sums["cy"] / count),
        "w": float(sums["w"] / count),
        "h": float(sums["h"] / count),
        "num_frames_used": int(count),
    }


def _save_image(img: Image.Image, out_path: Path) -> None:
    _ensure_dir(out_path.parent)

    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        img.save(out_path, quality=95, subsampling=0, optimize=True)
    elif ext == ".png":
        img.save(out_path, optimize=True)
    else:
        img.save(out_path)


def _process_transforms_json(
    transforms_path: Path,
    images_root: Path,
    transform_infos: Dict[str, Dict],
    stage1_out_root: Path,
    stage2_out_root: Path,
) -> None:
    """读取 transforms.json，更新内参，保存到两个阶段的输出目录。
    
    Args:
        transforms_path: 原始 transforms.json 路径
        images_root: 图片根目录
        transform_infos: {相对路径: transform_info} 映射
        stage1_out_root: Stage-1 输出目录
        stage2_out_root: Stage-2 输出目录
    """
    with open(transforms_path, 'r') as f:
        data = json.load(f)
    
    # 检测是否有全局内参
    has_global_intrinsics = all(k in data for k in ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h'])
    
    # 准备两个输出的 transforms.json
    data_stage1 = json.loads(json.dumps(data))  # 深拷贝
    data_stage2 = json.loads(json.dumps(data))
    stage1_sums = {"fl_x": 0.0, "fl_y": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
    stage2_sums = {"fl_x": 0.0, "fl_y": 0.0, "cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0}
    stage1_count = 0
    stage2_count = 0
    
    # 处理每个 frame
    for i, frame in enumerate(data['frames']):
        file_path = frame['file_path']
        # 规范化路径（去除前导 ./）
        file_path_normalized = file_path.lstrip('./')
        
        # 匹配 transform_info
        if file_path_normalized not in transform_infos:
            print(f"警告: transforms.json 中的 {file_path} 未找到对应的处理结果，跳过")
            continue
        
        info = transform_infos[file_path_normalized]
        
        # 获取原始内参（优先使用 per-frame，其次使用全局）
        if 'fl_x' in frame:
            fx0 = frame['fl_x']
            fy0 = frame['fl_y']
            cx0 = frame['cx']
            cy0 = frame['cy']
            w0 = frame['w']
            h0 = frame['h']
        elif has_global_intrinsics:
            fx0 = data['fl_x']
            fy0 = data['fl_y']
            cx0 = data['cx']
            cy0 = data['cy']
            w0 = data['w']
            h0 = data['h']
        else:
            print(f"警告: {file_path} 没有内参信息，跳过")
            continue
        
        # 验证尺寸一致性
        if info['w0'] != w0 or info['h0'] != h0:
            print(f"警告: {file_path} 的尺寸不一致 (transforms: {w0}x{h0}, 实际: {info['w0']}x{info['h0']})")
        
        # 计算 Stage-1 内参
        fx1, fy1, cx1, cy1 = _update_intrinsics(
            fx0, fy0, cx0, cy0,
            info['w0'], info['h0'],
            info['w1'], info['h1'],
            info['box1'],
        )
        w1_final = info['box1'][2] - info['box1'][0]
        h1_final = info['box1'][3] - info['box1'][1]
        
        # 计算 Stage-2 内参
        fx2, fy2, cx2, cy2 = _update_intrinsics(
            fx0, fy0, cx0, cy0,
            info['w0'], info['h0'],
            info['w2'], info['h2'],
            info['box2'],
        )
        w2_final = info['box2'][2] - info['box2'][0]
        h2_final = info['box2'][3] - info['box2'][1]
        
        # 更新 Stage-1 frame
        data_stage1['frames'][i].update({
            'w': w1_final,
            'h': h1_final,
            'fl_x': fx1,
            'fl_y': fy1,
            'cx': cx1,
            'cy': cy1,
        })
        stage1_sums["fl_x"] += float(fx1)
        stage1_sums["fl_y"] += float(fy1)
        stage1_sums["cx"] += float(cx1)
        stage1_sums["cy"] += float(cy1)
        stage1_sums["w"] += float(w1_final)
        stage1_sums["h"] += float(h1_final)
        stage1_count += 1
        
        # 更新 Stage-2 frame
        data_stage2['frames'][i].update({
            'w': w2_final,
            'h': h2_final,
            'fl_x': fx2,
            'fl_y': fy2,
            'cx': cx2,
            'cy': cy2,
        })
        stage2_sums["fl_x"] += float(fx2)
        stage2_sums["fl_y"] += float(fy2)
        stage2_sums["cx"] += float(cx2)
        stage2_sums["cy"] += float(cy2)
        stage2_sums["w"] += float(w2_final)
        stage2_sums["h"] += float(h2_final)
        stage2_count += 1
    
    # 如果有全局内参，删除它（因为每个 frame 都有自己的内参了）
    if has_global_intrinsics:
        for key in ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h']:
            data_stage1.pop(key, None)
            data_stage2.pop(key, None)

    # 写入与 frames 同级的平均内参统计
    _set_avg_intrinsics(data_stage1, stage1_sums, stage1_count, stage_name="Stage-1")
    _set_avg_intrinsics(data_stage2, stage2_sums, stage2_count, stage_name="Stage-2")
    
    # 保存
    out_path1 = stage1_out_root / 'transforms.json'
    out_path2 = stage2_out_root / 'transforms.json'
    
    _ensure_dir(out_path1.parent)
    _ensure_dir(out_path2.parent)
    
    with open(out_path1, 'w') as f:
        json.dump(data_stage1, f, indent=4)
    print(f"已保存 Stage-1 transforms.json 到: {out_path1}")
    
    with open(out_path2, 'w') as f:
        json.dump(data_stage2, f, indent=4)
    print(f"已保存 Stage-2 transforms.json 到: {out_path2}")


def process_one(
    in_path: Path,
    images_root: Path,
    stage1_out_root: Path,
    stage2_out_root: Path,
    stage1_downscale_n: int,
    multiple: int,
) -> Tuple[bool, str, Optional[Dict]]:
    """处理单张图片，返回成功状态、消息和变换参数。
    
    Returns:
        (success, message, transform_info)
        transform_info: {
            'w0': int, 'h0': int,
            'w1': int, 'h1': int, 'box1': tuple,
            'w2': int, 'h2': int, 'box2': tuple,
        }
    """
    rel = in_path.relative_to(images_root)
    out1 = stage1_out_root / rel
    out2 = stage2_out_root / rel

    try:
        with Image.open(in_path) as im0:
            im = ImageOps.exif_transpose(im0)
            w0, h0 = im.size

            # Stage-1 resize: 1/N of original (integer floor).
            n = int(stage1_downscale_n)
            if n <= 0:
                raise ValueError("stage1_downscale_n must be >= 1")
            w1 = max(1, w0 // n)
            h1 = max(1, h0 // n)
            im1_base = im.resize((w1, h1), resample=Image.Resampling.LANCZOS)

            # Stage-1 crop to multiple.
            box1 = _center_crop_to_multiple(w1, h1, multiple)
            im1 = im1_base.crop(box1)
            _save_image(im1, out1)

            # Stage-2: 2x Stage-1 pre-crop size, crop box scaled by 2.
            k = n
            w2 = w1 * k
            h2 = h1 * k
            im2_base = im.resize((w2, h2), resample=Image.Resampling.LANCZOS)
            box2 = _scale_box(box1, k)
            im2 = im2_base.crop(box2)
            _save_image(im2, out2)

            transform_info = {
                'w0': w0, 'h0': h0,
                'w1': w1, 'h1': h1, 'box1': box1,
                'w2': w2, 'h2': h2, 'box2': box2,
            }

            msg = (
                f"OK  {rel} | orig {w0}x{h0} -> stage1_base {w1}x{h1} crop {im1.size[0]}x{im1.size[1]} "
                f"-> stage2_base {w2}x{h2} crop {im2.size[0]}x{im2.size[1]}"
            )
            return True, msg, transform_info

    except Exception as e:
        return False, f"ERR {rel} | {type(e).__name__}: {e}", None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create Stage-1 and Stage-2 image sets.")
    p.add_argument("--images", required=True, type=Path, help="Input images directory")
    p.add_argument(
        "--stage1-out",
        required=True,
        type=Path,
        help="Output directory for Stage-1 images (half-res then crop-to-multiple).",
    )
    p.add_argument(
        "--stage2-out",
        required=True,
        type=Path,
        help="Output directory for Stage-2 images (2x Stage-1 linear resolution; consistent FOV).",
    )
    p.add_argument(
        "--stage1-downscale-n",
        default=2,
        type=int,
        help="Stage-1 downscale factor N (Stage-1 base size = original size / N). Default: 2 (half-res).",
    )
    p.add_argument(
        "--multiple",
        default=14,
        type=int,
        help="Stage-1 crop so that width and height are multiples of this value (default: 14).",
    )
    p.add_argument(
        "--transforms-json",
        type=Path,
        help="Optional: transforms.json path. If provided, will update intrinsics and save to output directories.",
    )
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument("--quiet", action="store_true", help="Suppress per-file logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    images_dir: Path = args.images
    if not images_dir.exists() or not images_dir.is_dir():
        raise SystemExit(f"Input directory not found: {images_dir}")

    _ensure_dir(args.stage1_out)
    _ensure_dir(args.stage2_out)

    files = list(_iter_images(images_dir, args.recursive))
    if not files:
        raise SystemExit(
            f"No images found in {images_dir} (supported: {sorted(SUPPORTED_EXTS)})"
        )

    ok = 0
    err = 0
    transform_infos = {}  # {相对路径: transform_info}
    
    for f in files:
        success, msg, info = process_one(
            f,
            images_root=images_dir,
            stage1_out_root=args.stage1_out,
            stage2_out_root=args.stage2_out,
            stage1_downscale_n=args.stage1_downscale_n,
            multiple=args.multiple,
        )
        if success:
            ok += 1
            if info:
                rel = f.relative_to(images_dir)
                transform_infos[str(rel)] = info
        else:
            err += 1
        if not args.quiet:
            print(msg)

    print(f"Done. OK={ok} ERR={err} TOTAL={ok + err}")
    
    # 处理 transforms.json（如果提供）
    if args.transforms_json:
        if not args.transforms_json.exists():
            print(f"警告: transforms.json 不存在: {args.transforms_json}")
        else:
            print(f"\n处理 transforms.json...")
            _process_transforms_json(
                args.transforms_json,
                images_dir,
                transform_infos,
                args.stage1_out,
                args.stage2_out,
            )


if __name__ == "__main__":
    # Work around PIL's large-image protection if needed.
    Image.MAX_IMAGE_PIXELS = None
    main()
