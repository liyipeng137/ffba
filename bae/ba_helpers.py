import torch
import torch.nn as nn
from bae.autograd.function import TrackingTensor, map_transform
from bae.utils.ba import rotate_euler, rotate_quat

USE_QUATERNIONS = True

@map_transform
def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    if USE_QUATERNIONS:
        points_proj = rotate_quat(points, camera_params[..., :7])
    else:
        points_proj = rotate_euler(points, camera_params[..., 3:6])
        points_proj = points_proj + camera_params[..., :3]
    points_proj = -points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)
    f = camera_params[..., -3].unsqueeze(-1)
    k1 = camera_params[..., -2].unsqueeze(-1)
    k2 = camera_params[..., -1].unsqueeze(-1)
    
    n = torch.sum(points_proj**2, axis=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    points_proj = points_proj * r * f

    return points_proj


@map_transform
def project_colmap(points, camera_params, intrinsics):
    """Project onto image plane with shared PINHOLE intrinsics."""
    if USE_QUATERNIONS:
        points_proj = rotate_quat(points, camera_params[..., :7])
    else:
        points_proj = rotate_euler(points, camera_params[..., 3:6])
        points_proj = points_proj + camera_params[..., :3]

    points_proj = points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)
    fx = intrinsics[..., 0].unsqueeze(-1)
    fy = intrinsics[..., 1].unsqueeze(-1)
    cx = intrinsics[..., 2].unsqueeze(-1)
    cy = intrinsics[..., 3].unsqueeze(-1)
    u = fx * points_proj[..., 0].unsqueeze(-1) + cx
    v = fy * points_proj[..., 1].unsqueeze(-1) + cy

    return torch.cat([u, v], dim=-1)


class Reproj(nn.Module):
    def __init__(self, camera_params, points_3d, load_colmap=False, intrinsics=None, optimize_intrinsics=False):
        super().__init__()
        self.pose = nn.Parameter(TrackingTensor(camera_params))
        self.points_3d = nn.Parameter(TrackingTensor(points_3d))
        self.pose.trim_SE3_grad = True
        self.load_colmap = load_colmap
        self.optimize_intrinsics = optimize_intrinsics

        if load_colmap:
            if intrinsics is None:
                raise ValueError("intrinsics must be provided for COLMAP mode")
            if intrinsics.dim() == 1:
                intrinsics = intrinsics.unsqueeze(0)
            if intrinsics.shape[-1] != 4:
                raise ValueError("intrinsics must have shape [4] or [1, 4]")

            if optimize_intrinsics:
                self.shared_intr = nn.Parameter(TrackingTensor(intrinsics))
            else:
                self.register_buffer("shared_intr", intrinsics)
        else:
            self.shared_intr = None

    def forward(self, points_2d, camera_indices=None, point_indices=None):
        if isinstance(points_2d, dict):
            input_dict = points_2d
            points_2d = input_dict["points_2d"]
            camera_indices = input_dict["camera_indices"]
            point_indices = input_dict["point_indices"]

        camera_params = self.pose
        points_3d = self.points_3d

        if self.load_colmap and self.shared_intr is not None:
            zero_indices = torch.zeros_like(camera_indices)
            intrinsics_batched = self.shared_intr[zero_indices]
            points_proj = project_colmap(
                points_3d[point_indices],
                camera_params[camera_indices],
                intrinsics_batched,
            )
        else:
            points_proj = project(points_3d[point_indices], camera_params[camera_indices])

        loss = points_proj - points_2d
        return loss

def least_square_error(
    camera_params,
    points_3d,
    camera_indices,
    point_indices,
    points_2d,
    load_colmap=False,
    intrinsics=None,
    optimize_intrinsics=False,
):
    model = Reproj(
        camera_params,
        points_3d,
        load_colmap=load_colmap,
        intrinsics=intrinsics,
        optimize_intrinsics=optimize_intrinsics,
    )
    loss = model(points_2d, camera_indices, point_indices)
    return torch.sum(loss**2, dim=-1).mean()
