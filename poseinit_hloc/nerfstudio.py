import json
from pathlib import Path

import numpy as np

from .camera import Camera, focal2fov, load_rgb_tensor


def _frame_value(frame: dict, meta: dict, key: str, default=None):
    value = frame.get(key, meta.get(key, default))
    if value is None:
        raise ValueError(f"Missing required camera field '{key}' in transforms.json")
    return value


class NerfStudioDataset:
    def __init__(self, source_path: str | Path, transform_name: str = "transforms.json"):
        self.source_path = Path(source_path)
        self.transform_path = self.source_path / transform_name
        self.all_cameras = self._load_cameras()

    def __len__(self) -> int:
        return len(self.all_cameras)

    def __iter__(self):
        return iter(self.all_cameras)

    def __getitem__(self, index: int) -> Camera:
        return self.all_cameras[index]

    def _load_cameras(self) -> list[Camera]:
        if not self.transform_path.exists():
            raise FileNotFoundError(f"Missing transforms file: {self.transform_path}")

        with open(self.transform_path, "r") as f:
            meta = json.load(f)

        frames = meta.get("frames", [])
        if not frames:
            raise ValueError(f"No frames found in {self.transform_path}")

        # get global intrinsics
        global_fx = float(meta.get("fl_x", None))
        global_fy = float(meta.get("fl_y", None))
        global_cx = float(meta.get("cx", None))
        global_cy = float(meta.get("cy", None))
        has_global_intrinsics = False
        if all(x is not None for x in [global_fx, global_fy, global_cx, global_cy]):
            has_global_intrinsics = True
            print(f"Using global intrinsics: fx={global_fx}, fy={global_fy}, cx={global_cx}, cy={global_cy}")

        cameras = []
        for frame in frames:
            image_rel = frame.get("file_path")
            if not image_rel:
                raise ValueError("Every frame must contain file_path")
            image_path = self.source_path / image_rel
            if not image_path.exists():
                raise FileNotFoundError(f"Missing frame image: {image_path}")

            image = load_rgb_tensor(image_path)
            image_height, image_width = image.shape[:2]
            width = int(_frame_value(frame, meta, "w", image_width))
            height = int(_frame_value(frame, meta, "h", image_height))

            if has_global_intrinsics:
                fx = global_fx
                fy = global_fy
                cx = global_cx
                cy = global_cy
            else:
                print(f"Using frame-specific intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                fx = float(_frame_value(frame, "fl_x"))
                fy = float(_frame_value(frame, "fl_y"))
                cx = float(_frame_value(frame, "cx"))
                cy = float(_frame_value(frame, "cy"))

            c2w = np.array(frame["transform_matrix"], dtype=np.float64)
            c2w[:, 1:3] *= -1
            extrinsics = np.linalg.inv(c2w)
            R = np.transpose(extrinsics[:3, :3])
            T = extrinsics[:3, 3]

            cameras.append(
                Camera(
                    R=R,
                    T=T,
                    FoVx=focal2fov(fx, width),
                    FoVy=focal2fov(fy, height),
                    image_width=width,
                    image_height=height,
                    principal_point_ndc=np.array([cx / width, cy / height], dtype=np.float64),
                    image_path=image_path,
                    image_name=str(image_rel),
                    image=image,
                )
            )

        return sorted(cameras, key=lambda camera: camera.image_name)
