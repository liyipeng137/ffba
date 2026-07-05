"""Geometry helpers shared by the MERG3R pipeline."""

import numpy as np
import torch


def to_homogeneous_extrinsic(extrinsic):
    extrinsic = np.asarray(extrinsic)
    if extrinsic.shape == (4, 4):
        return extrinsic.astype(np.float64)
    if extrinsic.shape != (3, 4):
        raise ValueError(
            f"Expected extrinsic shape (3, 4) or (4, 4), got {extrinsic.shape}"
        )
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = extrinsic
    return out


def depth_to_cam_coords_points(depth_map, intrinsic):
    height, width = depth_map.shape
    intrinsic = np.asarray(intrinsic)
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map
    return np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)


def depth_to_world_coords_points(depth_map, extrinsic, intrinsic, eps=1e-8):
    depth_map = np.asarray(depth_map).squeeze()
    point_mask = depth_map > eps
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)
    cam_to_world = np.linalg.inv(to_homogeneous_extrinsic(extrinsic))
    world_coords_points = (
        np.dot(cam_coords_points, cam_to_world[:3, :3].T) + cam_to_world[:3, 3]
    ).astype(np.float32)
    return world_coords_points, cam_coords_points, point_mask


def unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam):
    """Unproject z-depth maps to world-coordinate point maps."""
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.detach().cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.detach().cpu().numpy()

    depth_map = np.asarray(depth_map)
    extrinsics_cam = np.asarray(extrinsics_cam)
    intrinsics_cam = np.asarray(intrinsics_cam)
    if depth_map.ndim == 5 and depth_map.shape[0] == 1:
        depth_map = depth_map[0]
    if depth_map.ndim == 2:
        depth_map = depth_map[None]
    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]
    if extrinsics_cam.ndim == 2:
        extrinsics_cam = extrinsics_cam[None]
    if intrinsics_cam.ndim == 2:
        intrinsics_cam = intrinsics_cam[None]

    world_points = []
    for frame_idx in range(depth_map.shape[0]):
        world, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx],
            extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx],
        )
        world_points.append(world)
    return np.stack(world_points, axis=0)


def depth_intrinsics_to_local_points(depth, intrinsic):
    """Back-project z-depth into camera-local XYZ coordinates."""
    if not isinstance(depth, torch.Tensor):
        depth = torch.as_tensor(depth)
    if not isinstance(intrinsic, torch.Tensor):
        intrinsic = torch.as_tensor(intrinsic, dtype=depth.dtype, device=depth.device)
    else:
        intrinsic = intrinsic.to(device=depth.device, dtype=depth.dtype)

    if depth.ndim == 5:
        if depth.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for depth, got shape {tuple(depth.shape)}"
            )
        depth = depth.squeeze(0)
    if depth.ndim == 4:
        if depth.shape[-1] != 1:
            raise ValueError(
                f"Expected last depth dim 1, got shape {tuple(depth.shape)}"
            )
        depth = depth[..., 0]
    if depth.ndim != 3:
        raise ValueError(f"Expected depth shape (N,H,W), got {tuple(depth.shape)}")

    if intrinsic.ndim == 4:
        if intrinsic.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1 for intrinsic, got shape {tuple(intrinsic.shape)}"
            )
        intrinsic = intrinsic.squeeze(0)

    n, height, width = depth.shape
    if intrinsic.ndim == 2:
        intrinsic = intrinsic.unsqueeze(0).expand(n, -1, -1)
    if intrinsic.shape[0] != n:
        raise ValueError(
            f"Intrinsic frame count {intrinsic.shape[0]} does not match depth {n}"
        )

    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = x.unsqueeze(0).expand(n, -1, -1)
    y = y.unsqueeze(0).expand(n, -1, -1)

    fx = intrinsic[:, 0, 0].view(n, 1, 1)
    fy = intrinsic[:, 1, 1].view(n, 1, 1)
    cx = intrinsic[:, 0, 2].view(n, 1, 1)
    cy = intrinsic[:, 1, 2].view(n, 1, 1)

    x_cam = (x - cx) * depth / fx
    y_cam = (y - cy) * depth / fy
    return torch.stack((x_cam, y_cam, depth), dim=-1).to(
        dtype=torch.float32, device="cpu"
    )


def compute_depth(points, extrin):
    points_homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points_camera = torch.einsum("nij,nhwj->nhwi", extrin, points_homogeneous)
    return points_camera[..., 2]


@torch.no_grad()
def estimate_intrinsics_and_depth(points: torch.Tensor):
    """Estimate pinhole intrinsics and z-depth from camera-coordinate points."""
    batch, height, width, _ = points.shape
    device = points.device
    u_grid = (
        torch.arange(width, device=device, dtype=torch.float)
        .view(1, 1, width)
        .expand(batch, height, width)
    )
    v_grid = (
        torch.arange(height, device=device, dtype=torch.float)
        .view(1, height, 1)
        .expand(batch, height, width)
    )

    x_coord = points[..., 0]
    y_coord = points[..., 1]
    depth = points[..., 2]
    valid = torch.isfinite(depth) & (depth > 1e-6)

    intrinsic = torch.zeros((batch, 3, 3), dtype=points.dtype, device=device)
    intrinsic[:, 2, 2] = 1.0

    for idx in range(batch):
        mask = valid[idx]
        if mask.sum().item() < 4:
            intrinsic[idx] = torch.full(
                (3, 3), float("nan"), dtype=points.dtype, device=device
            )
            intrinsic[idx, 2, 2] = 1.0
            continue

        a_u = (x_coord[idx][mask] / depth[idx][mask]).unsqueeze(1)
        a_v = (y_coord[idx][mask] / depth[idx][mask]).unsqueeze(1)
        mat_u = torch.cat([a_u, torch.ones_like(a_u)], dim=1)
        mat_v = torch.cat([a_v, torch.ones_like(a_v)], dim=1)

        u = u_grid[idx][mask].unsqueeze(1)
        v = v_grid[idx][mask].unsqueeze(1)

        sol_u = torch.linalg.lstsq(mat_u, u).solution.squeeze(1)
        sol_v = torch.linalg.lstsq(mat_v, v).solution.squeeze(1)

        intrinsic[idx, 0, 0] = sol_u[0]
        intrinsic[idx, 0, 2] = sol_u[1]
        intrinsic[idx, 1, 1] = sol_v[0]
        intrinsic[idx, 1, 2] = sol_v[1]

    return intrinsic, depth


def convert_to_homogeneous_matrix(input: torch.Tensor):
    if len(input.shape) == 2:
        return torch.cat(
            [input, torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(input.device)]
        )

    if len(input.shape) == 3:
        batch_size = input.shape[0]
        homo_part = torch.stack(
            [torch.tensor([0, 0, 0, 1]).reshape(1, 4) for _ in range(batch_size)]
        ).to(input.device)
        return torch.cat([input, homo_part], dim=1)

    raise ValueError("Input shape incorrect for homogeneous matrix.")


def remove_homogeneous_row(matrix: torch.Tensor) -> torch.Tensor:
    if len(matrix.shape) == 2:
        if matrix.shape != (4, 4):
            raise ValueError("Expected shape (4, 4) for single matrix.")
        return matrix[:3]

    if len(matrix.shape) == 3:
        if matrix.shape[1:] != (4, 4):
            raise ValueError("Expected shape (B, 4, 4) for batched matrix.")
        return matrix[:, :3]

    raise ValueError("Invalid input shape.")


# Backward-compatible private name used by older local code.
_to_homogeneous_extrinsic = to_homogeneous_extrinsic
