import argparse
import json
import os
from typing import Tuple

import numpy as np


def _get_value(frame, data, key):
    if key in frame:
        return frame[key]
    if key in data:
        return data[key]
    raise KeyError(f"Missing '{key}' in frame and top-level transforms.json")


def _infer_camera_model(data, override):
    if override:
        return override.lower()
    model = str(data.get("camera_model", "opengl")).lower()
    if "opencv" in model:
        return "opencv"
    if "opengl" in model:
        return "opengl"
    return "opengl"


def _load_transforms(path, camera_model_override=None) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], str]:
    with open(path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        raise ValueError("No frames found in transforms.json")

    camera_model = _infer_camera_model(data, camera_model_override)

    poses = []
    intrinsics = []
    sizes = set()

    for idx, frame in enumerate(frames):
        mat = np.array(frame.get("transform_matrix"), dtype=np.float32)
        if mat.shape != (4, 4):
            raise ValueError(f"Frame {idx} has invalid transform_matrix shape: {mat.shape}")

        w = int(_get_value(frame, data, "w"))
        h = int(_get_value(frame, data, "h"))
        sizes.add((h, w))

        fl_x = float(_get_value(frame, data, "fl_x"))
        fl_y = float(_get_value(frame, data, "fl_y"))
        cx = float(_get_value(frame, data, "cx"))
        cy = float(_get_value(frame, data, "cy"))

        K = np.array(
            [
                [fl_x, 0.0, cx],
                [0.0, fl_y, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        poses.append(mat)
        intrinsics.append(K)

    if len(sizes) != 1:
        raise ValueError(f"Inconsistent image sizes across frames: {sorted(sizes)}")

    image_size = sizes.pop()  # (H, W)
    poses = np.stack(poses, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    return poses, intrinsics, image_size, camera_model


def _convert_opengl_to_opencv_c2w(poses):
    poses = np.array(poses, copy=True)
    poses[:, :3, 1:3] *= -1.0
    return poses


def main():
    parser = argparse.ArgumentParser(
        description="Encode transforms.json (OpenGL/OpenCV c2w) into conditions .npz"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to transforms.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output conditions .npz",
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default=None,
        choices=["opengl", "opencv"],
        help="Override camera_model detection in transforms.json",
    )
    parser.add_argument(
        "--no_depths",
        action="store_true",
        help="Do not write depths to the .npz (example_mm expects depths key).",
    )
    args = parser.parse_args()

    poses, intrinsics, (h, w), camera_model = _load_transforms(
        args.input, camera_model_override=args.camera_model
    )

    if camera_model == "opengl":
        poses = _convert_opengl_to_opencv_c2w(poses)

    depths = None
    if not args.no_depths:
        depths = np.zeros((poses.shape[0], h, w), dtype=np.float32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if depths is None:
        np.savez(args.output, poses=poses, intrinsics=intrinsics)
    else:
        np.savez(args.output, poses=poses, intrinsics=intrinsics, depths=depths)

    print(f"Saved conditions to: {args.output}")
    print(f"Frames: {poses.shape[0]}, Image size: {w}x{h}, Camera model: {camera_model}")


if __name__ == "__main__":
    main()
