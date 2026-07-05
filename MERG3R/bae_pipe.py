import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation

_LOCAL_BAE_ROOT = Path(__file__).resolve().parent / "third_party" / "bae"
_LOCAL_BAE_PKG_ROOT = _LOCAL_BAE_ROOT / "bae"


def _is_relative_to(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _ensure_local_bae_on_path():
    if not (_LOCAL_BAE_ROOT / "ba_colmap.py").is_file():
        raise FileNotFoundError(f"BAE directory not found: {_LOCAL_BAE_ROOT}")
    bae_root_str = str(_LOCAL_BAE_ROOT)
    if not sys.path or sys.path[0] != bae_root_str:
        sys.path.insert(0, bae_root_str)


_ensure_local_bae_on_path()

import pypose as pp  # noqa: E402
import torch.nn as nn  # noqa: E402
from pypose.autograd.function import psjac  # noqa: E402

import bae  # noqa: E402

if not _is_relative_to(Path(bae.__file__), _LOCAL_BAE_PKG_ROOT):
    raise ImportError(
        "Imported the wrong BAE package: "
        f"{Path(bae.__file__).resolve()}. Expected it under {_LOCAL_BAE_PKG_ROOT}."
    )

from bae.optim import LM  # noqa: E402
from bae.utils.pysolvers import PCG  # noqa: E402


@psjac
def project_colmap(points, camera_params, intrinsics):
    """Project COLMAP world points with shared PINHOLE intrinsics."""
    points_proj = pp.SE3(camera_params[..., :7]).Act(points)
    points_proj = points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)

    fx = intrinsics[..., 0].unsqueeze(-1)
    fy = intrinsics[..., 1].unsqueeze(-1)
    cx = intrinsics[..., 2].unsqueeze(-1)
    cy = intrinsics[..., 3].unsqueeze(-1)
    u = fx * points_proj[..., 0].unsqueeze(-1) + cx
    v = fy * points_proj[..., 1].unsqueeze(-1) + cy
    return torch.cat([u, v], dim=-1)


class ColmapResidual(nn.Module):
    def __init__(self, camera_params, points_3d, intrinsics, optimize_intrinsics=True):
        super().__init__()
        if intrinsics is None:
            raise ValueError("intrinsics must be provided for COLMAP mode")
        if intrinsics.dim() == 1:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.shape[-1] != 4:
            raise ValueError("intrinsics must have shape [4] or [1, 4]")

        self.pose = pp.Parameter(camera_params, sjac=True)
        self.points_3d = pp.Parameter(points_3d, sjac=True)
        self.pose.trim_SE3_grad = True

        if optimize_intrinsics:
            self.shared_intr = pp.Parameter(intrinsics, sjac=True)
        else:
            self.register_buffer("shared_intr", intrinsics)

    def forward(self, points_2d, camera_indices=None, point_indices=None):
        if isinstance(points_2d, dict):
            input_dict = points_2d
            points_2d = input_dict["points_2d"]
            camera_indices = input_dict["camera_indices"]
            point_indices = input_dict["point_indices"]

        zero_indices = torch.zeros_like(camera_indices)
        intrinsics_batched = self.shared_intr[zero_indices]
        points_proj = project_colmap(
            self.points_3d[point_indices],
            self.pose[camera_indices],
            intrinsics_batched,
        )
        return points_proj - points_2d


def _normalize_extrinsics(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(
            f"Expected extrinsic shape (N, 3, 4) or (N, 4, 4), got {extrinsic.shape}"
        )

    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic[:, :3, :4]
    return extrinsic


def _extrinsics_to_camera_params(extrinsic):
    extrinsic = _normalize_extrinsics(extrinsic)
    camera_params = np.zeros((extrinsic.shape[0], 7), dtype=np.float64)
    camera_params[:, :3] = extrinsic[:, :3, 3]
    camera_params[:, 3:7] = Rotation.from_matrix(extrinsic[:, :3, :3]).as_quat()
    return camera_params


def _camera_params_to_extrinsics(camera_params):
    camera_params = np.asarray(camera_params, dtype=np.float64)
    extrinsic = np.tile(np.eye(4, dtype=np.float64), (camera_params.shape[0], 1, 1))
    extrinsic[:, :3, :3] = Rotation.from_quat(camera_params[:, 3:7]).as_matrix()
    extrinsic[:, :3, 3] = camera_params[:, :3]
    return extrinsic.astype(np.float32)


def _mean_pinhole_intrinsics(intrinsic):
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.ndim != 3 or intrinsic.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsic shape (N, 3, 3), got {intrinsic.shape}")
    return np.array(
        [
            np.mean(intrinsic[:, 0, 0]),
            np.mean(intrinsic[:, 1, 1]),
            np.mean(intrinsic[:, 0, 2]),
            np.mean(intrinsic[:, 1, 2]),
        ],
        dtype=np.float64,
    )


def _pinhole_to_matrix(intrinsics_4):
    fx, fy, cx, cy = np.asarray(intrinsics_4, dtype=np.float64).reshape(4)
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _frame_valid_mask(valid_track_mask, frame_idx, count):
    if valid_track_mask is None:
        return np.ones(count, dtype=bool)

    mask = valid_track_mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if isinstance(mask, list):
        frame_mask = np.asarray(mask[frame_idx])
    else:
        frame_mask = np.asarray(mask[frame_idx])

    if frame_mask.size < count:
        padded = np.zeros(count, dtype=bool)
        padded[: frame_mask.size] = frame_mask.astype(bool)
        return padded
    return frame_mask[:count].astype(bool)


def _build_bae_observations(track, points_id, points_3d, valid_track_mask=None):
    points_3d = np.asarray(points_3d, dtype=np.float64)
    points_2d = []
    camera_indices = []
    compact_point_indices = []
    used_camera_ids = []
    used_point_ids = []
    camera_id_to_compact = {}
    point_id_to_compact = {}

    for camera_idx, (track_i, point_ids_i) in enumerate(zip(track, points_id)):
        track_i = np.asarray(track_i)
        point_ids_i = np.asarray(point_ids_i)
        if track_i.size == 0 or point_ids_i.size == 0:
            continue

        track_i = track_i.reshape(-1, 2)
        point_ids_i = point_ids_i.reshape(-1).astype(np.int64)
        count = min(len(track_i), len(point_ids_i))
        if count == 0:
            continue

        mask_i = _frame_valid_mask(valid_track_mask, camera_idx, count)
        for obs_idx in np.flatnonzero(mask_i):
            point_id = int(point_ids_i[obs_idx])
            point_2d = track_i[obs_idx]
            if point_id < 0 or point_id >= len(points_3d):
                continue
            if not (
                np.isfinite(point_2d).all() and np.isfinite(points_3d[point_id]).all()
            ):
                continue

            compact_camera_id = camera_id_to_compact.get(camera_idx)
            if compact_camera_id is None:
                compact_camera_id = len(used_camera_ids)
                camera_id_to_compact[camera_idx] = compact_camera_id
                used_camera_ids.append(camera_idx)

            compact_id = point_id_to_compact.get(point_id)
            if compact_id is None:
                compact_id = len(used_point_ids)
                point_id_to_compact[point_id] = compact_id
                used_point_ids.append(point_id)

            points_2d.append(point_2d.astype(np.float64))
            camera_indices.append(compact_camera_id)
            compact_point_indices.append(compact_id)

    if not points_2d:
        raise ValueError("No valid observations available for BAE refinement")

    return {
        "points_2d": np.asarray(points_2d, dtype=np.float64),
        "camera_indices": np.asarray(camera_indices, dtype=np.int64),
        "point_indices": np.asarray(compact_point_indices, dtype=np.int64),
        "used_camera_ids": np.asarray(used_camera_ids, dtype=np.int64),
        "used_point_ids": np.asarray(used_point_ids, dtype=np.int64),
        "points_3d": points_3d[np.asarray(used_point_ids, dtype=np.int64)],
    }


def run_bae_refinement(
    predictions,
    track,
    points_id,
    valid_track_mask=None,
    iters=20,
    device="cuda",
    optimize_intrinsics=False,
):
    _ensure_local_bae_on_path()

    observations = _build_bae_observations(
        track,
        points_id,
        predictions["points"],
        valid_track_mask=valid_track_mask,
    )

    all_camera_params = _extrinsics_to_camera_params(predictions["extrinsic"])
    used_camera_ids = observations["used_camera_ids"]
    dropped_camera_ids = np.setdiff1d(
        np.arange(all_camera_params.shape[0]), used_camera_ids
    )
    if len(dropped_camera_ids) > 0:
        print(
            "[BAE] Warning: dropping "
            f"{len(dropped_camera_ids)} frames with no valid observations: "
            f"{dropped_camera_ids.tolist()}"
        )

    camera_params = torch.tensor(
        all_camera_params[used_camera_ids],
        dtype=torch.float64,
        device=device,
    )
    points_3d = torch.tensor(
        observations["points_3d"], dtype=torch.float64, device=device
    )
    intrinsics = torch.tensor(
        _mean_pinhole_intrinsics(predictions["intrinsic"]),
        dtype=torch.float64,
        device=device,
    )
    input_dict = {
        "points_2d": torch.tensor(
            observations["points_2d"], dtype=torch.float64, device=device
        ),
        "camera_indices": torch.tensor(
            observations["camera_indices"], dtype=torch.int64, device=device
        ),
        "point_indices": torch.tensor(
            observations["point_indices"], dtype=torch.int64, device=device
        ),
    }

    model = ColmapResidual(
        camera_params.clone(),
        points_3d.clone(),
        intrinsics=intrinsics,
        optimize_intrinsics=optimize_intrinsics,
    ).to(device)

    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    solver = PCG(tol=1e-4, maxiter=250)
    optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

    with torch.no_grad():
        initial_loss = torch.sum(model(input_dict) ** 2, dim=-1).mean().item()

    start = perf_counter()
    loss = None
    for idx in range(iters):
        loss = optimizer.step(input_dict)
        print(f"[BAE] Iteration {idx} loss {loss.item()} time {perf_counter() - start}")

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        ending_loss = torch.sum(model(input_dict) ** 2, dim=-1).mean().item()

    optimized_camera_params = model.pose.detach().cpu().numpy()
    optimized_points = model.points_3d.detach().cpu().numpy()
    if optimize_intrinsics:
        optimized_intrinsics = model.shared_intr.detach().cpu().numpy().reshape(4)
    else:
        optimized_intrinsics = intrinsics.detach().cpu().numpy().reshape(4)

    full_points = np.asarray(predictions["points"], dtype=np.float32).copy()
    full_points[observations["used_point_ids"]] = optimized_points.astype(np.float32)

    shared_K = _pinhole_to_matrix(optimized_intrinsics)
    optimized_extrinsics = _camera_params_to_extrinsics(optimized_camera_params)
    original_extrinsics = np.asarray(predictions["extrinsic"], dtype=np.float32).copy()
    if original_extrinsics.shape[-2:] == (4, 4):
        original_extrinsics[used_camera_ids] = optimized_extrinsics
        predictions["extrinsic"] = original_extrinsics
    else:
        original_extrinsics[used_camera_ids] = optimized_extrinsics[:, :3, :4]
        predictions["extrinsic"] = original_extrinsics
    predictions["points"] = full_points
    predictions["intrinsic"] = np.repeat(
        shared_K[None], predictions["extrinsic"].shape[0], axis=0
    )

    return {
        "initial_loss": initial_loss,
        "ending_loss": ending_loss,
        "num_observations": int(len(observations["points_2d"])),
        "num_cameras": int(len(used_camera_ids)),
        "num_dropped_cameras": int(len(dropped_camera_ids)),
        "num_points": int(len(observations["used_point_ids"])),
        "time": perf_counter() - start,
    }
