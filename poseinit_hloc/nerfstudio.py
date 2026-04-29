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
            fx = float(_frame_value(frame, meta, "fl_x"))
            fy = float(frame.get("fl_y", meta.get("fl_y", fx)))
            cx = float(frame.get("cx", meta.get("cx", width / 2)))
            cy = float(frame.get("cy", meta.get("cy", height / 2)))

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
