import einops
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def rotation_matrix_to_quaternion(
    R: np.ndarray | torch.Tensor,
    w_first: bool = True,
) -> np.ndarray:
    """Convert a rotation matrix to a quaternion.

    Args:
        R: Rotation matrix of shape ``(3, 3)`` or batched ``(..., 3, 3)``.
            ``torch.Tensor`` inputs are moved to CPU and converted to numpy.
        w_first: If ``True``, return ``[w, x, y, z]``; otherwise
            ``[x, y, z, w]``.

    Returns:
        Quaternion array of shape ``(4,)`` or ``(..., 4)``.
    """
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()

    if w_first:
        return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
    else:
        return Rotation.from_matrix(R).as_quat()


def quaternion_to_rotation_matrix(
    q: np.ndarray | torch.Tensor,
    w_first: bool = True,
) -> np.ndarray:
    """Convert a single quaternion to a rotation matrix.

    Only handles a single (non-batched) quaternion because of the explicit
    component indexing.

    Args:
        q: Quaternion of shape ``(4,)``. ``torch.Tensor`` inputs are moved to
            CPU and converted to numpy.
        w_first: If ``True``, ``q`` is interpreted as ``[w, x, y, z]``;
            otherwise as ``[x, y, z, w]``.

    Returns:
        Rotation matrix of shape ``(3, 3)``.
    """
    if isinstance(q, torch.Tensor):
        q = q.cpu().numpy()

    if w_first:
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    else:
        return Rotation.from_quat(q).as_matrix()


def restore_identity(extrinsics: torch.Tensor) -> torch.Tensor:
    """Compose out the first view's pose so that view 0 is identity.

    For each batch, applies the inverse of ``extrinsics[:, 0]`` on the right
    of every view, leaving view 0 as the identity transform.

    Args:
        extrinsics: ``(B, N, 4, 4)`` or ``(B, N, 3, 4)``. Mutated in place.

    Returns:
        The same ``extrinsics`` tensor, mutated in place.
    """
    B, N, _, _ = extrinsics.shape

    extrinsics_0 = extrinsics[:, 0:1].clone()
    extrinsics[:, :, :3, :3] = extrinsics[:, :, :3, :3] @ extrinsics_0[
        :, :, :3, :3
    ].transpose(-1, -2).expand(-1, N, -1, -1)
    extrinsics[:, :, :3, 3:] = extrinsics[:, :, :3, 3:] - extrinsics[
        :, :, :3, :3
    ] @ extrinsics_0[:, 0:1, :3, 3:].expand(-1, N, -1, -1)

    return extrinsics


def bilinear_interpolate_value(
    value_map: torch.Tensor,
    coords: torch.Tensor,
    align_corners: bool = False,
) -> torch.Tensor:
    """Sample N points from each value map with bilinear interpolation.

    Args:
        value_map: ``(B, C, H, W)``.
        coords: ``(B, N, 2)`` pixel coordinates ``(x, y)``.
        align_corners: Forwarded to ``F.grid_sample`` and used to scale
            ``coords`` consistently.

    Returns:
        Sampled values of shape ``(B, N, C)``.
    """
    assert value_map.dim() == 4, "bad dimension for the value map"

    sizes = value_map.shape[2:]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)],
            device=coords.device,
        )
    else:
        coords = coords * torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device
        )

    coords -= 1

    B, N, _ = coords.shape
    sampled = F.grid_sample(
        value_map, coords.unsqueeze(-2), align_corners=align_corners
    ).permute(0, 2, 3, 1)

    return sampled.view(B, N, -1)


def project(rays: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Project camera-space rays to pixel coordinates.

    Negative-depth points are projected with a sign flip so the returned
    coordinates remain finite and signed; callers can detect such points by
    checking the sign of the input ray's z component.

    Args:
        rays: ``(B, N, 3)`` camera-space rays.
        intrinsics: ``(B, 3, 3)`` pinhole intrinsics.

    Returns:
        Pixel coordinates of shape ``(B, N, 2)``.
    """
    assert rays.dim() == 3, "bad dimension for the coordinates"

    uv_homogeneous = torch.einsum("b t j, b n j -> b n t", intrinsics, rays)
    scale = torch.where(uv_homogeneous[..., 2:] < 0, -1, 1)

    uv = (
        uv_homogeneous[..., :2]
        / torch.clamp(torch.abs(uv_homogeneous[..., 2:]), min=1e-3)
        * scale
    )

    return uv


def unproject(uvs: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Unproject pixel coordinates to unit-depth camera-space rays.

    Args:
        uvs: ``(B, N, 2)`` pixel coordinates.
        intrinsics: ``(B, 3, 3)`` pinhole intrinsics.

    Returns:
        Camera-space rays of shape ``(B, N, 3)`` with z = 1.
    """
    uv_homogeneous = torch.cat([uvs, torch.ones_like(uvs)[..., :1]], dim=-1)

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    intrinsics_inv = torch.zeros_like(intrinsics)
    intrinsics_inv[:, 0, 0] = 1.0 / fx
    intrinsics_inv[:, 1, 1] = 1.0 / fy
    intrinsics_inv[:, 0, 2] = -cx / fx
    intrinsics_inv[:, 1, 2] = -cy / fy
    intrinsics_inv[:, 2, 2] = 1.0

    rays = torch.einsum("b t j, b n j -> b n t", intrinsics_inv, uv_homogeneous)

    return rays


# Note: the coordinates of camera_points should correspond to extrinsics
def project_tracks(
    camera_points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsic: torch.Tensor,
    angle_threshold: float = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project per-view camera points into every other view in the batch.

    Validity is decided by the angle between the world-space direction and
    the principal axis: a point is invalid in view ``j`` if it is farther than
    ``angle_threshold`` degrees behind view ``j``'s principal axis.

    Args:
        camera_points: ``(B, H, W, 3)`` (3D inputs are unsqueezed to 4D
            ``(B, 1, H, W, 3)``) — points expressed in the source camera frame.
        extrinsics: ``(B, N, 4, 4)`` or ``(B, N, 3, 4)`` world-from-camera (or
            camera-from-world; must match the camera frame ``camera_points``
            is expressed in).
        intrinsic: ``(B, N, 3, 3)`` pinhole intrinsics per view.
        angle_threshold: Tolerance in degrees for the back-of-camera test.

    Returns:
        Tuple of:
            * ``track_virtual`` ``(B, N, K, 2)`` — projected pixel coords.
            * ``tracks_points`` ``(B, K, 3)`` — flattened source camera points.
            * ``valid_mask`` ``(B, N, K)`` — per-view validity flags.
            * ``is_negative`` ``(B, N, K)`` — points behind the source view.
    """
    B, N = extrinsics.shape[:2]

    if camera_points.dim() == 3:
        camera_points = camera_points.unsqueeze(1)

    world_points_j = torch.einsum(
        "b h w c, b n d c -> b n h w d", camera_points, extrinsics[:, :, :3, :3]
    ) + extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(3)

    image_points_j = project(
        einops.rearrange(
            world_points_j.to(torch.float64), "b n h w d -> (b n) (h w) d"
        ),
        einops.rearrange(intrinsic.to(torch.float64), "b n d c -> (b n) d c"),
    ).to(world_points_j.dtype)  # (B*N, H*W, 2)

    # Do not consider points outside the image as invalid. Instead, check
    # the depth of the points; if the angle to the principle point is too
    # large, then we consider it as invalid.
    world_points_j_normalized = world_points_j / torch.clamp(
        world_points_j.norm(dim=-1, keepdim=True), min=1e-6
    )
    invalid_i2j = (
        world_points_j_normalized[..., 2] < np.sin(np.deg2rad(angle_threshold))
    ).reshape(B, N, -1)  # (B, N, K)
    invalid_j2i = (
        world_points_j_normalized[..., 2] > -np.sin(np.deg2rad(angle_threshold))
    ).reshape(B, N, -1)  # (B, N, K)

    # Use these poins as the tracks
    track_virtual = image_points_j.reshape(B, N, -1, 2)  # (B, N, grid_num, 2)

    return (
        track_virtual,
        camera_points.flatten(1, 2),
        ~(invalid_i2j * invalid_j2i),
        ~invalid_j2i,
    )
