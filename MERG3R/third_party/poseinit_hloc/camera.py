import dataclasses
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def focal2fov(focal: float, pixels: int) -> float:
    return 2 * math.atan(pixels / (2 * focal))


def fov2focal(fov: float, pixels: int) -> float:
    return pixels / (2 * math.tan(fov / 2))


def get_world_to_view(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    Rt = np.zeros((4, 4), dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def resize_image_tensor(tensor_image: torch.Tensor, resolution: tuple[int, int]) -> torch.Tensor:
    if tensor_image.max() <= 1.0:
        tensor_image = tensor_image * 255.0
    pil_image = Image.fromarray(tensor_image.byte().cpu().numpy())
    resized = pil_image.resize(resolution)
    return torch.from_numpy(np.array(resized)).float() / 255.0


@dataclasses.dataclass
class Camera:
    R: np.ndarray
    T: np.ndarray
    FoVx: float
    FoVy: float
    image_width: int
    image_height: int
    principal_point_ndc: np.ndarray
    image_path: Path
    image_name: str
    image: torch.Tensor

    @property
    def extrinsics(self) -> torch.Tensor:
        return torch.tensor(get_world_to_view(self.R, self.T)).contiguous()

    @property
    def intrinsics(self) -> torch.Tensor:
        fx = fov2focal(self.FoVx, self.image_width)
        fy = fov2focal(self.FoVy, self.image_height)
        cx = self.image_width * self.principal_point_ndc[0]
        cy = self.image_height * self.principal_point_ndc[1]
        return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)

    def downsample_scale(self, scale: float) -> "Camera":
        if scale <= 0:
            raise ValueError("resolution scale must be positive")
        resolution = (round(self.image_width / scale), round(self.image_height / scale))
        self.image = resize_image_tensor(self.image, resolution)[..., :3].clamp(0.0, 1.0)
        self.image_width, self.image_height = resolution
        return self


def load_rgb_tensor(image_path: Path) -> torch.Tensor:
    if str(image_path).lower().endswith((".heic", ".heif")):
        try:
            from pillow_heif import register_heif_opener
            from PIL import ImageOps

            register_heif_opener()
            with Image.open(image_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        except ImportError as exc:
            raise ImportError("pillow_heif is required to read HEIC/HEIF images") from exc
    else:
        pil_img = Image.open(image_path).convert("RGB")
    return torch.from_numpy(np.array(pil_img)).float() / 255.0
