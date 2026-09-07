import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from utils.image_pyramid import SUPPORTED_IMAGE_EXTS


NERFSTUDIO_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class NerfstudioPriorData:
    images: torch.Tensor
    retrieval_images: torch.Tensor | None
    image_names: list[str]
    image_size_hw: tuple[int, int]
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    image_ids: np.ndarray
    audit: dict


def convert_nerfstudio_c2w_to_opencv_w2c(transform_matrix):
    c2w_nerfstudio = np.asarray(transform_matrix, dtype=np.float64)
    if c2w_nerfstudio.shape != (4, 4):
        raise ValueError(
            "Nerfstudio transform_matrix must have shape (4, 4), "
            f"got {c2w_nerfstudio.shape}"
        )
    if not np.isfinite(c2w_nerfstudio).all():
        raise ValueError("Nerfstudio transform_matrix contains non-finite values")
    if not np.allclose(c2w_nerfstudio[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(
            "Nerfstudio transform_matrix must end with [0, 0, 0, 1]"
        )

    rotation = c2w_nerfstudio[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("Nerfstudio transform_matrix rotation is not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if determinant <= 0.0 or not np.isclose(determinant, 1.0, atol=1e-3):
        raise ValueError(
            "Nerfstudio transform_matrix rotation must have determinant +1, "
            f"got {determinant}"
        )

    c2w_opencv = c2w_nerfstudio @ NERFSTUDIO_TO_OPENCV
    return np.linalg.inv(c2w_opencv)[:3, :4]


def _frame_value(frame, root, key):
    if key in frame:
        return frame[key]
    if key in root:
        return root[key]
    raise ValueError(f"Nerfstudio frame is missing required field {key!r}")


def _resolve_frame_path(transforms_dir, raw_path):
    raw_path = Path(str(raw_path))
    candidate = raw_path if raw_path.is_absolute() else transforms_dir / raw_path
    candidate = candidate.resolve()
    if candidate.is_file():
        return candidate
    if candidate.suffix:
        raise FileNotFoundError(f"Nerfstudio frame image does not exist: {candidate}")
    for suffix in sorted(SUPPORTED_IMAGE_EXTS):
        with_suffix = candidate.with_suffix(suffix)
        if with_suffix.is_file():
            return with_suffix
    raise FileNotFoundError(f"Nerfstudio frame image does not exist: {candidate}")


def _retrieval_size(width, height, long_side, patch_multiple):
    scale = float(long_side) / float(max(width, height))
    target_width = max(
        patch_multiple,
        int(round(width * scale / patch_multiple)) * patch_multiple,
    )
    target_height = max(
        patch_multiple,
        int(round(height * scale / patch_multiple)) * patch_multiple,
    )
    return target_width, target_height


def load_nerfstudio_prior(
    transforms_json,
    dataset_dir,
    *,
    subsample=1,
    num_images=-1,
    num_workers=16,
    retrieval_long_side=512,
    retrieval_patch_multiple=16,
):
    transforms_path = Path(transforms_json).resolve()
    dataset_path = Path(dataset_dir).resolve()
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Nerfstudio transforms file not found: {transforms_path}")
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset image directory not found: {dataset_path}")

    subsample = int(subsample)
    num_images = int(num_images)
    num_workers = int(num_workers)
    retrieval_long_side = int(retrieval_long_side)
    retrieval_patch_multiple = int(retrieval_patch_multiple)
    if subsample <= 0:
        raise ValueError("subsample must be >= 1")
    if num_images == 0 or num_images < -1:
        raise ValueError("num_images must be -1 or >= 1")
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1")
    if retrieval_long_side <= 0 or retrieval_patch_multiple <= 0:
        raise ValueError("DINO retrieval image dimensions must be positive")

    with open(transforms_path) as handle:
        root = json.load(handle)
    frames = root.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Nerfstudio transforms.json must contain a non-empty frames list")

    indexed_frames = list(enumerate(frames))
    if num_images != -1:
        indexed_frames = indexed_frames[:num_images]
    indexed_frames = indexed_frames[::subsample]
    if not indexed_frames:
        raise ValueError("No Nerfstudio frames remain after num_images/subsample")

    selected = []
    seen_paths = set()
    transforms_dir = transforms_path.parent
    for original_index, frame in indexed_frames:
        if not isinstance(frame, dict):
            raise ValueError(f"Nerfstudio frame {original_index} is not an object")
        if "file_path" not in frame:
            raise ValueError(f"Nerfstudio frame {original_index} has no file_path")
        image_path = _resolve_frame_path(transforms_dir, frame["file_path"])
        try:
            image_path.relative_to(dataset_path)
        except ValueError as exc:
            raise ValueError(
                f"Nerfstudio frame {original_index} resolves outside --dataset: "
                f"{image_path}"
            ) from exc
        if image_path in seen_paths:
            raise ValueError(f"Duplicate Nerfstudio frame image: {image_path}")
        seen_paths.add(image_path)

        width = int(_frame_value(frame, root, "w"))
        height = int(_frame_value(frame, root, "h"))
        fl_x = float(_frame_value(frame, root, "fl_x"))
        fl_y = float(_frame_value(frame, root, "fl_y"))
        if width <= 0 or height <= 0 or fl_x <= 0.0 or fl_y <= 0.0:
            raise ValueError(
                f"Nerfstudio frame {original_index} has invalid image intrinsics"
            )
        w2c = convert_nerfstudio_c2w_to_opencv_w2c(frame["transform_matrix"])
        selected.append(
            {
                "original_index": int(original_index),
                "image_path": image_path,
                "width": width,
                "height": height,
                "fl_x": fl_x,
                "fl_y": fl_y,
                "w2c": w2c,
            }
        )

    source_sizes = {(item["width"], item["height"]) for item in selected}
    if len(source_sizes) != 1:
        raise ValueError(
            "Prior-pose mode currently requires a common image size; "
            f"got {sorted(source_sizes)}"
        )
    width, height = next(iter(source_sizes))
    retrieval_width, retrieval_height = _retrieval_size(
        width,
        height,
        retrieval_long_side,
        retrieval_patch_multiple,
    )

    def load_one(item):
        with Image.open(item["image_path"]) as image_raw:
            image = ImageOps.exif_transpose(image_raw).convert("RGB")
            if image.size != (item["width"], item["height"]):
                raise ValueError(
                    f"Image size for {item['image_path']} is {image.size}, expected "
                    f"{(item['width'], item['height'])} from transforms.json"
                )
            retrieval = image.resize(
                (retrieval_width, retrieval_height),
                resample=Image.Resampling.LANCZOS,
            )
            image_array = np.asarray(image, dtype=np.float32).copy() / 255.0
            retrieval_array = np.asarray(retrieval, dtype=np.float32).copy() / 255.0
        return (
            torch.from_numpy(image_array).permute(2, 0, 1),
            torch.from_numpy(retrieval_array).permute(2, 0, 1),
        )

    worker_count = min(num_workers, len(selected))
    if worker_count == 1:
        loaded = [load_one(item) for item in selected]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            loaded = list(executor.map(load_one, selected))

    images = torch.stack([item[0] for item in loaded])
    retrieval_images = torch.stack([item[1] for item in loaded])
    focal_values = np.asarray(
        [(item["fl_x"] + item["fl_y"]) * 0.5 for item in selected],
        dtype=np.float64,
    )
    shared_focal = float(focal_values.mean())
    intrinsic_one = np.asarray(
        [
            [shared_focal, 0.0, width * 0.5],
            [0.0, shared_focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    intrinsic = np.repeat(intrinsic_one[None], len(selected), axis=0)
    extrinsic = np.stack([item["w2c"] for item in selected]).astype(np.float32)
    camera_centers = np.stack(
        [
            -pose[:3, :3].T @ pose[:3, 3]
            for pose in extrinsic.astype(np.float64)
        ]
    )
    steps = np.linalg.norm(np.diff(camera_centers, axis=0), axis=1)
    audit = {
        "source": "nerfstudio_transforms",
        "transforms_json": str(transforms_path),
        "dataset": str(dataset_path),
        "camera_model_input": root.get("camera_model"),
        "camera_model_output": "SIMPLE_PINHOLE",
        "coordinate_conversion": {
            "input": "Nerfstudio/OpenGL camera-to-world",
            "output": "OpenCV/COLMAP world-to-camera",
            "camera_axis_right_multiply": NERFSTUDIO_TO_OPENCV.tolist(),
        },
        "frame_count_input": int(len(frames)),
        "frame_count_selected": int(len(selected)),
        "selected_original_indices": [item["original_index"] for item in selected],
        "subsample": subsample,
        "num_images": num_images,
        "image_names": [str(item["image_path"]) for item in selected],
        "image_size_wh": [width, height],
        "image_order": "transforms_json_frames",
        "image_pyramid": False,
        "intrinsics_policy": "shared_mean_focal_centered_principal_point",
        "focal_input": {
            "min": float(focal_values.min()),
            "median": float(np.median(focal_values)),
            "mean": shared_focal,
            "max": float(focal_values.max()),
            "std": float(focal_values.std()),
        },
        "shared_intrinsic": intrinsic_one.tolist(),
        "dino_retrieval_size_wh": [retrieval_width, retrieval_height],
        "ignored_fields": ["depth_file_path", "distortion_parameters"],
        "trajectory": {
            "camera_center_min": camera_centers.min(axis=0).tolist(),
            "camera_center_max": camera_centers.max(axis=0).tolist(),
            "step_min": float(steps.min()) if steps.size else 0.0,
            "step_median": float(np.median(steps)) if steps.size else 0.0,
            "step_p90": float(np.percentile(steps, 90)) if steps.size else 0.0,
            "step_max": float(steps.max()) if steps.size else 0.0,
        },
    }
    return NerfstudioPriorData(
        images=images,
        retrieval_images=retrieval_images,
        image_names=[str(item["image_path"]) for item in selected],
        image_size_hw=(height, width),
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        image_ids=np.arange(len(selected), dtype=np.int64),
        audit=audit,
    )
