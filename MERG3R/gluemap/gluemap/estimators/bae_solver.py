import importlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pycolmap
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class BaeProblemData:
    camera_params: np.ndarray
    points_3d: np.ndarray
    intrinsics: np.ndarray
    points_2d: np.ndarray
    camera_indices: np.ndarray
    point_indices: np.ndarray
    is_negative: np.ndarray
    is_virtual: np.ndarray
    image_ids: list[int]
    real_point_ids: list[int]
    virtual_point_ids: list[int]
    camera_id: int
    camera_model: str
    bae_root: Path
    skipped: dict


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ensure_bae_runtime():
    merg3r_root = Path(__file__).resolve().parents[3]
    bae_root = merg3r_root / "bae"
    if not (bae_root / "ba_colmap.py").is_file():
        raise FileNotFoundError(f"Could not find MERG3R BAE at {bae_root}")

    bae_root_str = str(bae_root)
    if not sys.path or sys.path[0] != bae_root_str:
        sys.path.insert(0, bae_root_str)

    bae_pkg = importlib.import_module("bae")
    bae_pkg_path = Path(bae_pkg.__file__).resolve()
    expected_pkg_root = (bae_root / "bae").resolve()
    if not _is_relative_to(bae_pkg_path, expected_pkg_root):
        raise ImportError(
            "Imported the wrong BAE package: "
            f"{bae_pkg_path}. Expected it under {expected_pkg_root}."
        )

    ba_colmap = importlib.import_module("ba_colmap")
    ba_colmap_path = Path(ba_colmap.__file__).resolve()
    if not _is_relative_to(ba_colmap_path, bae_root.resolve()):
        raise ImportError(
            "Imported the wrong ba_colmap module: "
            f"{ba_colmap_path}. Expected it under {bae_root}."
        )

    import pypose as pp  # noqa: PLC0415
    from bae.optim import LM  # noqa: PLC0415
    from bae.utils.pysolvers import PCG  # noqa: PLC0415
    from pypose.autograd.function import psjac  # noqa: PLC0415

    return SimpleNamespace(
        bae_root=bae_root,
        ba_colmap=ba_colmap,
        LM=LM,
        PCG=PCG,
        pp=pp,
        psjac=psjac,
    )


def _pose_params_pycolmap_to_bae(pose_params):
    pose_params = np.asarray(pose_params, dtype=np.float64).reshape(7)
    return np.concatenate([pose_params[4:7], pose_params[0:4]], axis=0)


def _pose_params_bae_to_pycolmap(pose_params):
    pose_params = np.asarray(pose_params, dtype=np.float64).reshape(7)
    quat = pose_params[3:7]
    quat_norm = np.linalg.norm(quat)
    if not np.isfinite(quat_norm) or quat_norm <= 0:
        raise ValueError(f"Invalid optimized quaternion: {quat}")
    quat = quat / quat_norm
    return np.concatenate([quat, pose_params[0:3]], axis=0)


def _rigid3d_from_pycolmap_params(pose_params):
    pose_params = _pose_params_bae_to_pycolmap(pose_params)
    qx, qy, qz, qw, tx, ty, tz = pose_params
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    translation = np.array([tx, ty, tz], dtype=np.float64)
    return pycolmap.Rigid3d(pycolmap.Rotation3d(rotation), translation)


def _point_xyz(point3D):
    xyz = np.asarray(point3D.xyz, dtype=np.float64).reshape(-1)
    if xyz.shape[0] < 3:
        raise ValueError(
            f"Invalid point3D.xyz shape: {np.asarray(point3D.xyz).shape}"
        )
    return xyz[:3]


def _shared_camera(reconstruction):
    image_camera_ids = {
        image.camera_id for image in reconstruction.images.values()
    }
    if len(image_camera_ids) != 1:
        raise ValueError(
            "BAE backend currently requires all images to share one camera, "
            f"got camera_ids={sorted(image_camera_ids)}"
        )
    camera_id = next(iter(image_camera_ids))
    if camera_id not in reconstruction.cameras:
        raise ValueError(f"Shared camera_id={camera_id} is missing")
    return camera_id, reconstruction.cameras[camera_id]


def _intrinsics_from_camera(camera):
    model_name = camera.model_name
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if model_name == "SIMPLE_PINHOLE":
        if params.shape[0] != 3:
            raise ValueError(f"SIMPLE_PINHOLE expects 3 params, got {params}")
        return params.astype(np.float64), model_name
    if model_name == "PINHOLE":
        if params.shape[0] != 4:
            raise ValueError(f"PINHOLE expects 4 params, got {params}")
        return params.astype(np.float64), model_name
    raise ValueError(
        "BAE backend currently supports SIMPLE_PINHOLE/PINHOLE only, got "
        f"{model_name}"
    )


def _collect_real_observations(reconstruction, real_point_ids, real_point_idx):
    observations = []
    skipped = {
        "real_missing_image": 0,
        "real_bad_point2d_idx": 0,
        "real_nonfinite_point": 0,
        "real_nonfinite_xy": 0,
    }
    for point3D_id, point3D in reconstruction.points3D.items():
        try:
            xyz = _point_xyz(point3D)
        except ValueError:
            skipped["real_nonfinite_point"] += 1
            continue
        if not np.isfinite(xyz).all():
            skipped["real_nonfinite_point"] += 1
            continue
        point_idx = len(real_point_ids)
        real_point_ids.append(point3D_id)
        real_point_idx[point3D_id] = point_idx
        for elem in point3D.track.elements:
            image_id = elem.image_id
            point2D_idx = elem.point2D_idx
            if image_id not in reconstruction.images:
                skipped["real_missing_image"] += 1
                continue
            image = reconstruction.images[image_id]
            if point2D_idx >= len(image.points2D):
                skipped["real_bad_point2d_idx"] += 1
                continue
            xy = np.asarray(image.points2D[point2D_idx].xy, dtype=np.float64)
            if not np.isfinite(xy).all():
                skipped["real_nonfinite_xy"] += 1
                continue
            observations.append((image_id, point_idx, xy, False, False))
    return observations, skipped


def _collect_virtual_observations(
    reconstruction,
    virtual_reconstruction,
    negative_depth_observations,
    real_point_count,
    virtual_point_ids,
):
    if virtual_reconstruction is None:
        return [], {
            "virtual_missing_image": 0,
            "virtual_missing_ref_image": 0,
            "virtual_bad_point2d_idx": 0,
            "virtual_empty_xyz": 0,
            "virtual_nonfinite_point": 0,
            "virtual_nonfinite_xy": 0,
        }

    name_to_ref_id = {
        img.name: img_id for img_id, img in reconstruction.images.items()
    }
    observations = []
    skipped = {
        "virtual_missing_image": 0,
        "virtual_missing_ref_image": 0,
        "virtual_bad_point2d_idx": 0,
        "virtual_empty_xyz": 0,
        "virtual_nonfinite_point": 0,
        "virtual_nonfinite_xy": 0,
    }
    for point3D_id, point3D in virtual_reconstruction.points3D.items():
        xyz = _point_xyz(point3D)
        if np.all(xyz == 0):
            skipped["virtual_empty_xyz"] += 1
            continue
        if not np.isfinite(xyz).all():
            skipped["virtual_nonfinite_point"] += 1
            continue

        point_idx = real_point_count + len(virtual_point_ids)
        virtual_point_ids.append(point3D_id)
        for elem in point3D.track.elements:
            virtual_image_id = elem.image_id
            point2D_idx = elem.point2D_idx
            if virtual_image_id not in virtual_reconstruction.images:
                skipped["virtual_missing_image"] += 1
                continue
            image = virtual_reconstruction.images[virtual_image_id]
            ref_id = name_to_ref_id.get(image.name)
            if ref_id is None:
                skipped["virtual_missing_ref_image"] += 1
                continue
            if point2D_idx >= len(image.points2D):
                skipped["virtual_bad_point2d_idx"] += 1
                continue
            xy = np.asarray(image.points2D[point2D_idx].xy, dtype=np.float64)
            if not np.isfinite(xy).all():
                skipped["virtual_nonfinite_xy"] += 1
                continue
            is_negative = (
                virtual_image_id in negative_depth_observations
                and point2D_idx
                in negative_depth_observations[virtual_image_id]
            )
            observations.append((ref_id, point_idx, xy, True, is_negative))
    return observations, skipped


def _build_bae_problem(
    reconstruction,
    virtual_reconstruction,
    negative_depth_observations,
    bae_root,
    include_virtual=True,
):
    camera_id, camera = _shared_camera(reconstruction)
    intrinsics, camera_model = _intrinsics_from_camera(camera)

    real_point_ids = []
    real_point_idx = {}
    real_obs, real_skipped = _collect_real_observations(
        reconstruction, real_point_ids, real_point_idx
    )
    virtual_point_ids = []
    virtual_reconstruction_for_bae = (
        virtual_reconstruction if include_virtual else None
    )
    virtual_obs, virtual_skipped = _collect_virtual_observations(
        reconstruction,
        virtual_reconstruction_for_bae,
        negative_depth_observations,
        len(real_point_ids),
        virtual_point_ids,
    )
    observations = real_obs + virtual_obs
    if not observations:
        raise ValueError("No observations available for BAE bundle adjustment")

    used_image_ids = sorted({obs[0] for obs in observations})
    image_id_to_compact = {
        image_id: idx for idx, image_id in enumerate(used_image_ids)
    }
    camera_params = np.stack(
        [
            _pose_params_pycolmap_to_bae(
                reconstruction.frames[image_id].rig_from_world.params
            )
            for image_id in used_image_ids
        ],
        axis=0,
    ).astype(np.float64)

    real_points = [
        _point_xyz(reconstruction.points3D[point3D_id])
        for point3D_id in real_point_ids
    ]
    virtual_points = (
        [
            _point_xyz(virtual_reconstruction_for_bae.points3D[point3D_id])
            for point3D_id in virtual_point_ids
        ]
        if virtual_reconstruction_for_bae is not None
        else []
    )
    points_3d = np.asarray(real_points + virtual_points, dtype=np.float64)
    if points_3d.size == 0:
        raise ValueError("No 3D points available for BAE bundle adjustment")

    points_2d = np.stack([obs[2] for obs in observations], axis=0).astype(
        np.float64
    )
    camera_indices = np.asarray(
        [image_id_to_compact[obs[0]] for obs in observations], dtype=np.int64
    )
    point_indices = np.asarray(
        [obs[1] for obs in observations], dtype=np.int64
    )
    is_virtual = np.asarray([obs[3] for obs in observations], dtype=bool)
    is_negative = np.asarray([obs[4] for obs in observations], dtype=bool)

    skipped = {**real_skipped, **virtual_skipped}
    return BaeProblemData(
        camera_params=camera_params,
        points_3d=points_3d,
        intrinsics=intrinsics,
        points_2d=points_2d,
        camera_indices=camera_indices,
        point_indices=point_indices,
        is_negative=is_negative,
        is_virtual=is_virtual,
        image_ids=used_image_ids,
        real_point_ids=real_point_ids,
        virtual_point_ids=virtual_point_ids,
        camera_id=camera_id,
        camera_model=camera_model,
        bae_root=bae_root,
        skipped=skipped,
    )


def _make_bae_model(runtime, camera_model, optimize_intrinsics):
    pp = runtime.pp
    psjac = runtime.psjac

    @psjac
    def reprojection_residual_pinhole(
        points,
        camera_params,
        intrinsics,
        points_2d,
        point_sign,
    ):
        points_cam = pp.SE3(camera_params[..., :7]).Act(points)
        points_cam = points_cam * point_sign
        z = points_cam[..., 2:3]
        valid = z > 1e-12
        z_safe = torch.where(valid, z, torch.ones_like(z))

        fx = intrinsics[..., 0:1]
        fy = intrinsics[..., 1:2]
        cx = intrinsics[..., 2:3]
        cy = intrinsics[..., 3:4]
        x = fx * points_cam[..., 0:1] / z_safe + cx
        y = fy * points_cam[..., 1:2] / z_safe + cy
        residual = torch.cat([x, y], dim=-1) - points_2d
        return torch.where(
            valid.expand_as(residual), residual, torch.zeros_like(residual)
        )

    @psjac
    def reprojection_residual_simple_pinhole(
        points,
        camera_params,
        focal,
        principal_point,
        points_2d,
        point_sign,
    ):
        points_cam = pp.SE3(camera_params[..., :7]).Act(points)
        points_cam = points_cam * point_sign
        z = points_cam[..., 2:3]
        valid = z > 1e-12
        z_safe = torch.where(valid, z, torch.ones_like(z))

        f = focal[..., 0:1]
        cx = principal_point[..., 0:1]
        cy = principal_point[..., 1:2]
        x = f * points_cam[..., 0:1] / z_safe + cx
        y = f * points_cam[..., 1:2] / z_safe + cy
        residual = torch.cat([x, y], dim=-1) - points_2d
        return torch.where(
            valid.expand_as(residual), residual, torch.zeros_like(residual)
        )

    if camera_model == "SIMPLE_PINHOLE":
        residual_fn = reprojection_residual_simple_pinhole
        expected_intrinsics_dim = 3
    elif camera_model == "PINHOLE":
        residual_fn = reprojection_residual_pinhole
        expected_intrinsics_dim = 4
    else:
        raise ValueError(f"Unsupported BAE camera model: {camera_model}")

    class GluemapBaeResidual(nn.Module):
        def __init__(self, camera_params, points_3d, intrinsics):
            super().__init__()
            self.pose = pp.Parameter(camera_params, sjac=True)
            self.points_3d = pp.Parameter(points_3d, sjac=True)
            self.pose.trim_SE3_grad = True
            if intrinsics.dim() == 1:
                intrinsics = intrinsics.unsqueeze(0)
            if intrinsics.shape[-1] != expected_intrinsics_dim:
                raise ValueError(
                    f"{camera_model} expects intrinsics dim "
                    f"{expected_intrinsics_dim}, got {intrinsics.shape}"
                )
            if optimize_intrinsics:
                if camera_model != "SIMPLE_PINHOLE":
                    raise ValueError(
                        "BAE intrinsic optimization currently only supports "
                        "SIMPLE_PINHOLE"
                    )
                self.shared_focal = pp.Parameter(
                    intrinsics[..., 0:1], sjac=True
                )
                self.register_buffer(
                    "shared_principal_point", intrinsics[..., 1:3]
                )
            else:
                self.register_buffer("shared_intr", intrinsics)

        def forward(self, input_dict=None, **kwargs):
            if input_dict is None:
                input_dict = kwargs
            elif isinstance(input_dict, dict):
                input_dict = {**input_dict, **kwargs}
            else:
                raise TypeError(
                    "GluemapBaeResidual.forward expects a dict input or "
                    f"keyword tensors, got {type(input_dict)!r}"
                )
            points_2d = input_dict["points_2d"]
            camera_indices = input_dict["camera_indices"]
            point_indices = input_dict["point_indices"]
            point_sign = input_dict["point_sign"]

            zero_indices = torch.zeros_like(camera_indices)
            if camera_model == "SIMPLE_PINHOLE":
                if optimize_intrinsics:
                    focal = self.shared_focal[zero_indices]
                    principal_point = self.shared_principal_point[zero_indices]
                else:
                    intrinsics = self.shared_intr[zero_indices]
                    focal = intrinsics[..., 0:1]
                    principal_point = intrinsics[..., 1:3]
                return residual_fn(
                    self.points_3d[point_indices],
                    self.pose[camera_indices],
                    focal,
                    principal_point,
                    points_2d,
                    point_sign,
                )

            intrinsics = self.shared_intr[zero_indices]
            return residual_fn(
                self.points_3d[point_indices],
                self.pose[camera_indices],
                intrinsics,
                points_2d,
                point_sign,
            )

        def optimized_intrinsics(self):
            if camera_model == "SIMPLE_PINHOLE":
                if optimize_intrinsics:
                    focal = self.shared_focal.detach().as_subclass(
                        torch.Tensor
                    )
                    principal_point = (
                        self.shared_principal_point.detach().as_subclass(
                            torch.Tensor
                        )
                    )
                    return torch.cat(
                        [focal, principal_point],
                        dim=-1,
                    )
                return self.shared_intr
            return self.shared_intr

    return GluemapBaeResidual


def _loss_value(model, input_dict):
    with torch.no_grad():
        residual = model(input_dict)
        if hasattr(residual, "tensor"):
            residual = residual.tensor()
        return torch.sum(residual**2, dim=-1).mean().item()


def _camera_centers_from_params(camera_params):
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    camera_params = np.asarray(camera_params, dtype=np.float64)
    rotations = Rotation.from_quat(camera_params[:, 3:7]).as_matrix()
    translations = camera_params[:, :3]
    return np.einsum(
        "nij,nj->ni",
        -np.transpose(rotations, (0, 2, 1)),
        translations,
    )


def _rotation_angle_deltas_deg(before, after):
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    before_r = Rotation.from_quat(before[:, 3:7])
    after_r = Rotation.from_quat(after[:, 3:7])
    delta = after_r * before_r.inv()
    return np.rad2deg(delta.magnitude())


def _pose_drift_summary(before, after):
    before_centers = _camera_centers_from_params(before)
    after_centers = _camera_centers_from_params(after)
    before_centered = before_centers - before_centers.mean(
        axis=0, keepdims=True
    )
    after_centered = after_centers - after_centers.mean(axis=0, keepdims=True)
    before_scale = float(np.sqrt(np.mean(np.sum(before_centered**2, axis=1))))
    after_scale = float(np.sqrt(np.mean(np.sum(after_centered**2, axis=1))))
    translations_delta = np.linalg.norm(after[:, :3] - before[:, :3], axis=1)
    rotations_delta = _rotation_angle_deltas_deg(before, after)
    return {
        "camera_center_centroid_shift": float(
            np.linalg.norm(
                after_centers.mean(axis=0) - before_centers.mean(axis=0)
            )
        ),
        "camera_center_scale_before": before_scale,
        "camera_center_scale_after": after_scale,
        "camera_center_scale_ratio": (
            float(after_scale / before_scale) if before_scale > 0 else None
        ),
        "translation_delta_mean": float(np.mean(translations_delta)),
        "translation_delta_max": float(np.max(translations_delta)),
        "rotation_delta_deg_mean": float(np.mean(rotations_delta)),
        "rotation_delta_deg_max": float(np.max(rotations_delta)),
    }


def _intrinsics_as_list(intrinsics):
    return [
        float(value)
        for value in np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    ]


def _format_float_list(values):
    return "[" + ", ".join(f"{value:.6g}" for value in values) + "]"


def _relative_translation_from_bae_poses(pose1, pose2):
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    pose1 = np.asarray(pose1, dtype=np.float64).reshape(7)
    pose2 = np.asarray(pose2, dtype=np.float64).reshape(7)
    rotation1 = Rotation.from_quat(pose1[3:7]).as_matrix()
    rotation2 = Rotation.from_quat(pose2[3:7]).as_matrix()
    translation1 = pose1[:3]
    translation2 = pose2[:3]
    return translation1 - rotation1 @ rotation2.T @ translation2


def _point_gauge_label(problem, point_idx):
    num_real = len(problem.real_point_ids)
    if point_idx < num_real:
        return {
            "point_index": int(point_idx),
            "source": "real",
            "point3D_id": int(problem.real_point_ids[point_idx]),
        }
    virtual_idx = point_idx - num_real
    return {
        "point_index": int(point_idx),
        "source": "virtual",
        "point3D_id": int(problem.virtual_point_ids[virtual_idx]),
    }


def _select_three_gauge_points(points_3d, candidate_indices, eps=1e-9):
    selected = []
    for point_idx in candidate_indices:
        candidate = selected + [point_idx]
        candidate_points = np.asarray(points_3d[candidate], dtype=np.float64)
        rank = np.linalg.matrix_rank(candidate_points.T, tol=eps)
        if rank > len(selected):
            selected.append(point_idx)
        if len(selected) == 3:
            break
    return selected


def _apply_three_points_gauge(problem, point_fixed_mask, summary, reason=None):
    candidate_indices = np.unique(problem.point_indices)
    selected = _select_three_gauge_points(
        problem.points_3d, candidate_indices
    )
    for point_idx in selected:
        point_fixed_mask[point_idx, :] = True

    summary["applied"] = (
        "three_points" if len(selected) == 3 else "partial_three_points"
    )
    summary["fallback_reason"] = reason
    summary["fixed_points"] = [
        _point_gauge_label(problem, point_idx) for point_idx in selected
    ]
    if len(selected) < 3:
        summary["warning"] = (
            "Failed to find three linearly independent points for BAE gauge "
            "fix."
        )


def _build_bae_gauge_fix(problem, fix_gauge):
    mode = (fix_gauge or "none").lower().replace("-", "_")
    valid_modes = {"none", "two_cams", "two_cams_full", "three_points"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unknown BAE gauge fix '{fix_gauge}', expected one of "
            f"{sorted(valid_modes)}"
        )

    pose_fixed_mask = np.zeros(
        (problem.camera_params.shape[0], 6), dtype=bool
    )
    point_fixed_mask = np.zeros((problem.points_3d.shape[0], 3), dtype=bool)
    summary = {
        "requested": mode,
        "applied": "none",
        "fixed_images": [],
        "fixed_points": [],
        "translation_fixed_dim": None,
        "baseline": None,
        "fallback_reason": None,
        "num_fixed_pose_dofs": 0,
        "num_fixed_point_dofs": 0,
    }

    if mode == "none":
        return pose_fixed_mask, point_fixed_mask, summary

    if mode == "three_points":
        _apply_three_points_gauge(problem, point_fixed_mask, summary)
    else:
        if problem.camera_params.shape[0] < 2:
            _apply_three_points_gauge(
                problem,
                point_fixed_mask,
                summary,
                reason="fewer than two cameras",
            )
        else:
            image1_idx = 0
            image2_idx = None
            baseline = None
            fixed_dim = None
            for candidate_idx in range(1, problem.camera_params.shape[0]):
                candidate_baseline = _relative_translation_from_bae_poses(
                    problem.camera_params[image1_idx],
                    problem.camera_params[candidate_idx],
                )
                max_abs = float(np.max(np.abs(candidate_baseline)))
                if max_abs > 1e-9:
                    image2_idx = candidate_idx
                    baseline = candidate_baseline
                    fixed_dim = int(np.argmax(np.abs(candidate_baseline)))
                    break

            if image2_idx is None:
                _apply_three_points_gauge(
                    problem,
                    point_fixed_mask,
                    summary,
                    reason="two-camera baseline is degenerate",
                )
            else:
                pose_fixed_mask[image1_idx, :] = True
                if mode == "two_cams_full":
                    pose_fixed_mask[image2_idx, :] = True
                    summary["applied"] = "two_cams_full"
                else:
                    pose_fixed_mask[image2_idx, fixed_dim] = True
                    summary["applied"] = "two_cams"
                summary["fixed_images"] = [
                    {
                        "image_id": int(problem.image_ids[image1_idx]),
                        "camera_index": int(image1_idx),
                        "fixed_pose_tangent_dofs": [0, 1, 2, 3, 4, 5],
                    },
                    {
                        "image_id": int(problem.image_ids[image2_idx]),
                        "camera_index": int(image2_idx),
                        "fixed_pose_tangent_dofs": (
                            [0, 1, 2, 3, 4, 5]
                            if mode == "two_cams_full"
                            else [int(fixed_dim)]
                        ),
                    },
                ]
                summary["translation_fixed_dim"] = int(fixed_dim)
                summary["baseline"] = [
                    float(value) for value in baseline.reshape(-1)
                ]

    summary["num_fixed_pose_dofs"] = int(pose_fixed_mask.sum())
    summary["num_fixed_point_dofs"] = int(point_fixed_mask.sum())
    return pose_fixed_mask, point_fixed_mask, summary


def _attach_fixed_dof_mask(parameter, mask, device):
    if mask is None or not bool(np.any(mask)):
        return
    parameter.fixed_dof_mask = torch.tensor(
        mask, dtype=torch.bool, device=device
    )


def _camera_params_from_intrinsics(camera_model, intrinsics):
    intrinsics = np.asarray(intrinsics, dtype=np.float64).reshape(-1)
    if camera_model == "SIMPLE_PINHOLE":
        if intrinsics.shape[0] != 3:
            raise ValueError(
                f"SIMPLE_PINHOLE expects [f, cx, cy], got {intrinsics}"
            )
        return intrinsics.astype(np.float64)
    if camera_model == "PINHOLE":
        if intrinsics.shape[0] != 4:
            raise ValueError(
                f"PINHOLE expects [fx, fy, cx, cy], got {intrinsics}"
            )
        return intrinsics.astype(np.float64)
    raise ValueError(f"Unsupported camera model: {camera_model}")


def _write_optimized_reconstruction(
    reconstruction,
    virtual_reconstruction,
    problem,
    optimized_camera_params,
    optimized_points,
    optimized_intrinsics=None,
):
    for idx, image_id in enumerate(problem.image_ids):
        reconstruction.frames[image_id].rig_from_world = (
            _rigid3d_from_pycolmap_params(optimized_camera_params[idx])
        )

    num_real = len(problem.real_point_ids)
    for idx, point3D_id in enumerate(problem.real_point_ids):
        reconstruction.points3D[point3D_id].xyz = optimized_points[idx].astype(
            np.float64
        )

    if optimized_intrinsics is not None:
        reconstruction.cameras[problem.camera_id].params = (
            _camera_params_from_intrinsics(
                problem.camera_model, optimized_intrinsics
            )
        )

    if virtual_reconstruction is not None:
        for offset, point3D_id in enumerate(problem.virtual_point_ids):
            point_idx = num_real + offset
            virtual_reconstruction.points3D[point3D_id].xyz = optimized_points[
                point_idx
            ].astype(np.float64)

        source_by_name = {
            img.name: (img_id, img)
            for img_id, img in reconstruction.images.items()
        }
        for target_id, target_img in virtual_reconstruction.images.items():
            if target_img.name not in source_by_name:
                continue
            src_id, _src_img = source_by_name[target_img.name]
            virtual_reconstruction.frames[target_id].rig_from_world = (
                reconstruction.frames[src_id].rig_from_world
            )
        for camera_id, camera in reconstruction.cameras.items():
            if camera_id in virtual_reconstruction.cameras:
                virtual_reconstruction.cameras[camera_id].params = camera.params


def bundle_adjustment_bae(
    reconstruction: pycolmap.Reconstruction,
    virtual_reconstruction: pycolmap.Reconstruction | None,
    negative_depth_observations: dict[int, set[int]],
    max_num_iterations: int = 20,
    device: str = "cuda",
    optimize_intrinsics: bool = False,
    real_only: bool = False,
    fix_gauge: str = "two_cams",
):
    runtime = _ensure_bae_runtime()
    problem = _build_bae_problem(
        reconstruction,
        virtual_reconstruction,
        negative_depth_observations,
        runtime.bae_root,
        include_virtual=not real_only,
    )
    if optimize_intrinsics and problem.camera_model != "SIMPLE_PINHOLE":
        raise ValueError(
            "BAE intrinsic optimization currently only supports "
            f"SIMPLE_PINHOLE, got {problem.camera_model}"
        )

    logger.info(
        "BAE bundle adjustment: "
        f"{len(problem.real_point_ids)} real tracks, "
        f"{len(problem.virtual_point_ids)} virtual tracks, "
        f"{int((~problem.is_virtual).sum())} real obs, "
        f"{int(problem.is_virtual.sum())} virtual obs, "
        f"{int(problem.is_negative.sum())} negative obs, "
        f"optimize_intrinsics={optimize_intrinsics}, "
        f"real_only={real_only}, "
        f"fix_gauge={fix_gauge}"
    )

    torch_device = torch.device(device)
    dtype = torch.float64
    input_dict = {
        "points_2d": torch.tensor(
            problem.points_2d, dtype=dtype, device=torch_device
        ),
        "camera_indices": torch.tensor(
            problem.camera_indices, dtype=torch.int64, device=torch_device
        ),
        "point_indices": torch.tensor(
            problem.point_indices, dtype=torch.int64, device=torch_device
        ),
        "point_sign": torch.tensor(
            np.where(problem.is_negative, -1.0, 1.0)[:, None],
            dtype=dtype,
            device=torch_device,
        ),
    }
    camera_params = torch.tensor(
        problem.camera_params, dtype=dtype, device=torch_device
    )
    points_3d = torch.tensor(
        problem.points_3d, dtype=dtype, device=torch_device
    )
    intrinsics = torch.tensor(
        problem.intrinsics, dtype=dtype, device=torch_device
    )

    model_cls = _make_bae_model(
        runtime, problem.camera_model, optimize_intrinsics
    )
    model = model_cls(camera_params.clone(), points_3d.clone(), intrinsics).to(
        torch_device
    )
    pose_fixed_mask, point_fixed_mask, gauge_summary = _build_bae_gauge_fix(
        problem, fix_gauge
    )
    _attach_fixed_dof_mask(model.pose, pose_fixed_mask, torch_device)
    _attach_fixed_dof_mask(
        model.points_3d, point_fixed_mask, torch_device
    )
    logger.info(
        "BAE gauge fix: "
        f"requested={gauge_summary['requested']}, "
        f"applied={gauge_summary['applied']}, "
        f"fixed_pose_dofs={gauge_summary['num_fixed_pose_dofs']}, "
        f"fixed_point_dofs={gauge_summary['num_fixed_point_dofs']}, "
        f"fixed_images={gauge_summary['fixed_images']}, "
        f"fixed_points={gauge_summary['fixed_points']}, "
        f"translation_fixed_dim={gauge_summary['translation_fixed_dim']}, "
        f"baseline={gauge_summary['baseline']}, "
        f"fallback_reason={gauge_summary['fallback_reason']}"
    )
    strategy = runtime.pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    solver = runtime.PCG(tol=1e-4, maxiter=250)
    optimizer = runtime.LM(model, strategy=strategy, solver=solver, reject=30)

    initial_loss = _loss_value(model, input_dict)
    t0 = time.time()
    last_loss = initial_loss
    for idx in range(int(max_num_iterations)):
        last_loss_tensor = optimizer.step(input_dict)
        if hasattr(last_loss_tensor, "item"):
            last_loss = float(last_loss_tensor.item())
        else:
            last_loss = float(last_loss_tensor)
        logger.info(
            "BAE iteration "
            f"{idx + 1}/{max_num_iterations}: loss={last_loss:.6e}"
        )

    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.time() - t0
    ending_loss = _loss_value(model, input_dict)

    optimized_camera_params = model.pose.detach().cpu().numpy()
    optimized_points = model.points_3d.detach().cpu().numpy()
    optimized_intrinsics = (
        model.optimized_intrinsics().detach().cpu().numpy().reshape(-1)
    )
    intrinsics_initial = _intrinsics_as_list(problem.intrinsics)
    intrinsics_final = _intrinsics_as_list(optimized_intrinsics)
    pose_drift = _pose_drift_summary(
        problem.camera_params, optimized_camera_params
    )
    _write_optimized_reconstruction(
        reconstruction,
        virtual_reconstruction,
        problem,
        optimized_camera_params,
        optimized_points,
        optimized_intrinsics if optimize_intrinsics else None,
    )

    summary = {
        "backend": "bae",
        "bae_root": str(problem.bae_root),
        "device": str(torch_device),
        "num_iterations": int(max_num_iterations),
        "num_cameras": int(problem.camera_params.shape[0]),
        "num_points_real": int(len(problem.real_point_ids)),
        "num_points_virtual": int(len(problem.virtual_point_ids)),
        "num_observations_real": int((~problem.is_virtual).sum()),
        "num_observations_virtual": int(problem.is_virtual.sum()),
        "num_observations_negative": int(problem.is_negative.sum()),
        "initial_loss": float(initial_loss),
        "last_optimizer_loss": float(last_loss),
        "ending_loss": float(ending_loss),
        "seconds": float(seconds),
        "optimize_intrinsics": bool(optimize_intrinsics),
        "real_only": bool(real_only),
        "intrinsics_initial": intrinsics_initial,
        "intrinsics_final": intrinsics_final,
        "fix_gauge": gauge_summary["applied"] not in {"none"},
        "gauge_fix": gauge_summary,
        "loss": "plain_squared",
        "camera_model": problem.camera_model,
        "skipped": problem.skipped,
        "pose_drift": pose_drift,
    }
    logger.info(
        "BAE bundle adjustment done: "
        f"loss {initial_loss:.6e} -> {ending_loss:.6e}, "
        f"time={seconds:.2f}s, real_only={real_only}"
    )
    logger.info(
        "BAE intrinsics: "
        f"model={problem.camera_model}, "
        f"optimize={optimize_intrinsics}, "
        f"initial={_format_float_list(intrinsics_initial)}, "
        f"final={_format_float_list(intrinsics_final)}"
    )
    logger.info(
        "BAE pose drift: "
        "center_shift="
        f"{pose_drift['camera_center_centroid_shift']:.6g}, "
        f"scale_before={pose_drift['camera_center_scale_before']:.6g}, "
        f"scale_after={pose_drift['camera_center_scale_after']:.6g}, "
        f"scale_ratio={pose_drift['camera_center_scale_ratio']}, "
        f"translation_delta_mean={pose_drift['translation_delta_mean']:.6g}, "
        f"translation_delta_max={pose_drift['translation_delta_max']:.6g}, "
        f"rotation_delta_deg_mean={pose_drift['rotation_delta_deg_mean']:.6g}, "
        f"rotation_delta_deg_max={pose_drift['rotation_delta_deg_max']:.6g}"
    )
    return reconstruction, virtual_reconstruction, summary
