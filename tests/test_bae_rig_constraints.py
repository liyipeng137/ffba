import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "gluemap"))
os.environ.setdefault("BAE_USE_PYPOSE_AMBIENT_GRAD", "1")

import torch  # noqa: E402

from bae.autograd import graph as autograd_graph  # noqa: E402
from gluemap.estimators import bae_solver  # noqa: E402
from utils.gluemap_refine_core import build_bae_rig_config  # noqa: E402


def _yaw_transform(degrees):
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
    return transform


def _rig_transform(frame_idx):
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = float(frame_idx)
    return transform


class _Pose:
    def __init__(self, matrix):
        self._matrix = np.asarray(matrix, dtype=np.float64)
        from scipy.spatial.transform import Rotation

        quat = Rotation.from_matrix(self._matrix[:3, :3]).as_quat()
        self.params = np.concatenate([quat, self._matrix[:3, 3]])

    def matrix(self):
        return self._matrix[:3, :4]


class _Image:
    def __init__(self, image_id, name, camera_id, frame_id, pose):
        self.image_id = image_id
        self.name = name
        self.camera_id = camera_id
        self.frame_id = frame_id
        self.points2D = [SimpleNamespace(xy=np.array([320.0, 240.0]))]
        self._pose = _Pose(pose)

    def cam_from_world(self):
        return self._pose


def _fake_reconstruction(num_frames=2, num_sensors=5, native_rig=True):
    sensor_transforms = [
        _yaw_transform(degrees)
        for degrees in np.linspace(-80.0, 80.0, num_sensors)
    ]
    images = {}
    frames = {}
    image_names = []
    image_frame_indices = []
    image_sensor_indices = []
    image_id = 1
    for frame_idx in range(num_frames):
        rig_pose = _rig_transform(frame_idx)
        shared_frame_id = frame_idx + 1
        if native_rig:
            frames[shared_frame_id] = SimpleNamespace(
                rig_from_world=_Pose(rig_pose)
            )
        for sensor_idx, sensor_pose in enumerate(sensor_transforms):
            frame_id = shared_frame_id if native_rig else image_id
            if not native_rig:
                frames[frame_id] = SimpleNamespace(
                    rig_from_world=_Pose(sensor_pose @ rig_pose)
                )
            name = f"frame_{frame_idx:03d}_sensor_{sensor_idx}.png"
            images[image_id] = _Image(
                image_id,
                name,
                camera_id=sensor_idx + 1,
                frame_id=frame_id,
                pose=sensor_pose @ rig_pose,
            )
            image_names.append(name)
            image_frame_indices.append(frame_idx)
            image_sensor_indices.append(sensor_idx)
            image_id += 1

    elements = [
        SimpleNamespace(image_id=image_id, point2D_idx=0)
        for image_id in images
    ]
    points3d = {
        1: SimpleNamespace(
            xyz=np.array([0.0, 0.0, 5.0]),
            track=SimpleNamespace(elements=elements),
        ),
        2: SimpleNamespace(
            xyz=np.array([1.0, 1.0, 6.0]),
            track=SimpleNamespace(elements=elements),
        ),
        3: SimpleNamespace(
            xyz=np.array([-1.0, 2.0, 7.0]),
            track=SimpleNamespace(elements=elements),
        ),
    }
    cameras = {
        sensor_idx + 1: SimpleNamespace(
            model_name="SIMPLE_PINHOLE",
            params=np.array([500.0, 320.0, 240.0]),
        )
        for sensor_idx in range(num_sensors)
    }
    reconstruction = SimpleNamespace(
        images=images,
        frames=frames,
        cameras=cameras,
        points3D=points3d,
    )
    config = build_bae_rig_config(
        image_names,
        image_frame_indices,
        image_sensor_indices,
        sensor_transforms,
    )
    return reconstruction, config, sensor_transforms


def _build_problem(native_rig=True):
    reconstruction, config, sensor_transforms = _fake_reconstruction(
        native_rig=native_rig
    )
    problem = bae_solver._build_bae_problem(
        reconstruction,
        virtual_reconstruction=None,
        negative_depth_observations={},
        bae_root=REPO_ROOT / "third_party" / "bae" / "bae",
        include_virtual=False,
        rig_config=config,
    )
    return reconstruction, config, sensor_transforms, problem


def test_five_camera_rig_uses_one_pose_block_per_frame():
    _reconstruction, _config, _sensors, problem = _build_problem()

    assert problem.rig_enabled is True
    assert problem.camera_params.shape == (2, 7)
    assert problem.intrinsics.shape == (5, 3)
    assert problem.camera_ids == [1, 2, 3, 4, 5]
    np.testing.assert_array_equal(
        problem.image_pose_indices,
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    )
    expected_per_track = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    np.testing.assert_array_equal(
        problem.pose_indices,
        np.tile(expected_per_track, len(problem.real_point_ids)),
    )


def test_fixed_sensor_relatives_survive_rig_pose_update_and_writeback():
    reconstruction, _config, sensor_transforms, problem = _build_problem(
        native_rig=False
    )
    before_relatives = [
        sensor @ np.linalg.inv(sensor_transforms[2])
        for sensor in sensor_transforms
    ]
    optimized = problem.camera_params.copy()
    optimized[0, :3] += [0.4, -0.2, 0.1]
    optimized[1, :3] += [-0.3, 0.5, 0.2]

    bae_solver._write_optimized_reconstruction(
        reconstruction,
        virtual_reconstruction=None,
        problem=problem,
        optimized_camera_params=optimized,
        optimized_points=problem.points_3d,
    )

    for frame_idx in range(2):
        first = frame_idx * 5 + 1
        expanded = [
            reconstruction.frames[first + sensor_idx].rig_from_world.matrix()
            for sensor_idx in range(5)
        ]
        expanded = [
            bae_solver._homogeneous_transform(pose, "expanded")
            for pose in expanded
        ]
        center_inv = np.linalg.inv(expanded[2])
        for sensor_idx, pose in enumerate(expanded):
            np.testing.assert_allclose(
                pose @ center_inv,
                before_relatives[sensor_idx],
                atol=1e-10,
            )


def test_rig_sparse_jacobian_has_frame_not_image_pose_columns():
    _reconstruction, _config, _sensors, problem = _build_problem()
    runtime = bae_solver._ensure_bae_runtime()
    model_cls = bae_solver._make_bae_model(
        runtime,
        problem.camera_model,
        optimize_intrinsics=False,
        rig_enabled=True,
    )
    dtype = torch.float64
    model = model_cls(
        torch.tensor(problem.camera_params, dtype=dtype),
        torch.tensor(problem.points_3d, dtype=dtype),
        torch.tensor(problem.intrinsics, dtype=dtype),
        sensor_from_rig=torch.tensor(problem.sensor_from_rig, dtype=dtype),
    )
    assert "sensor_from_rig" in dict(model.named_buffers())
    assert "sensor_from_rig" not in dict(model.named_parameters())
    sensor_before = model.sensor_from_rig.detach().clone()
    input_dict = {
        "points_2d": torch.tensor(problem.points_2d, dtype=dtype),
        "camera_indices": torch.tensor(problem.camera_indices),
        "pose_indices": torch.tensor(problem.pose_indices),
        "intrinsics_indices": torch.tensor(problem.intrinsics_indices),
        "point_indices": torch.tensor(problem.point_indices),
        "point_sign": torch.ones((len(problem.points_2d), 1), dtype=dtype),
    }

    residual = model(input_dict)
    pose_jacobian = autograd_graph.jacobian(
        residual, [model.pose, model.points_3d]
    )[0]

    assert pose_jacobian.shape == (2 * len(problem.points_2d), 6 * 2)
    torch.testing.assert_close(model.sensor_from_rig, sensor_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BAE LM requires CUDA")
def test_bae_iteration_does_not_modify_fixed_sensor_transforms():
    _reconstruction, _config, _sensors, problem = _build_problem()
    runtime = bae_solver._ensure_bae_runtime()
    device = torch.device("cuda")
    dtype = torch.float64
    model_cls = bae_solver._make_bae_model(
        runtime,
        problem.camera_model,
        optimize_intrinsics=False,
        rig_enabled=True,
    )
    model = model_cls(
        torch.tensor(problem.camera_params, dtype=dtype, device=device),
        torch.tensor(problem.points_3d, dtype=dtype, device=device),
        torch.tensor(problem.intrinsics, dtype=dtype, device=device),
        sensor_from_rig=torch.tensor(
            problem.sensor_from_rig, dtype=dtype, device=device
        ),
    ).to(device)
    pose_mask, point_mask, _summary = bae_solver._build_bae_gauge_fix(
        problem, "two_cams"
    )
    bae_solver._attach_fixed_dof_mask(model.pose, pose_mask, device)
    bae_solver._attach_fixed_dof_mask(model.points_3d, point_mask, device)
    input_dict = {
        "points_2d": torch.tensor(
            problem.points_2d, dtype=dtype, device=device
        ),
        "camera_indices": torch.tensor(
            problem.camera_indices, device=device
        ),
        "pose_indices": torch.tensor(problem.pose_indices, device=device),
        "intrinsics_indices": torch.tensor(
            problem.intrinsics_indices, device=device
        ),
        "point_indices": torch.tensor(problem.point_indices, device=device),
        "point_sign": torch.ones(
            (len(problem.points_2d), 1), dtype=dtype, device=device
        ),
    }
    before = model.sensor_from_rig.detach().clone()
    optimizer = runtime.LM(
        model,
        strategy=runtime.pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4),
        solver=runtime.PCG(tol=1e-4, maxiter=50),
        reject=5,
    )

    optimizer.step(input_dict)
    torch.cuda.synchronize()

    torch.testing.assert_close(model.sensor_from_rig, before, rtol=0.0, atol=0.0)


def test_two_camera_gauge_is_selected_over_two_rig_frames():
    _reconstruction, _config, _sensors, problem = _build_problem()

    pose_mask, point_mask, summary = bae_solver._build_bae_gauge_fix(
        problem, "two_cams"
    )

    assert pose_mask.shape == (2, 6)
    assert point_mask.shape == (3, 3)
    assert summary["num_fixed_pose_dofs"] == 7
    assert [entry["pose_index"] for entry in summary["fixed_frames"]] == [0, 1]
    assert [entry["frame_id"] for entry in summary["fixed_frames"]] == [0, 1]


def test_nonrig_model_keeps_independent_image_pose_parameters():
    reconstruction, _config, _sensors = _fake_reconstruction(
        num_frames=2, num_sensors=1, native_rig=False
    )
    problem = bae_solver._build_bae_problem(
        reconstruction,
        virtual_reconstruction=None,
        negative_depth_observations={},
        bae_root=REPO_ROOT / "third_party" / "bae" / "bae",
        include_virtual=False,
        rig_config=None,
    )
    runtime = bae_solver._ensure_bae_runtime()
    model_cls = bae_solver._make_bae_model(
        runtime,
        problem.camera_model,
        optimize_intrinsics=False,
        rig_enabled=False,
    )
    dtype = torch.float64
    model = model_cls(
        torch.tensor(problem.camera_params, dtype=dtype),
        torch.tensor(problem.points_3d, dtype=dtype),
        torch.tensor(problem.intrinsics, dtype=dtype),
    )
    input_dict = {
        "points_2d": torch.tensor(problem.points_2d, dtype=dtype),
        "camera_indices": torch.tensor(problem.camera_indices),
        "pose_indices": torch.tensor(problem.pose_indices),
        "intrinsics_indices": torch.tensor(problem.intrinsics_indices),
        "point_indices": torch.tensor(problem.point_indices),
        "point_sign": torch.ones((len(problem.points_2d), 1), dtype=dtype),
    }
    residual = model(input_dict).tensor().detach()
    points_cam = runtime.pp.SE3(
        model.pose[input_dict["camera_indices"]]
    ).Act(model.points_3d[input_dict["point_indices"]])
    intrinsics = model.shared_intr[input_dict["intrinsics_indices"]]
    expected_xy = torch.stack(
        [
            intrinsics[:, 0] * points_cam[:, 0] / points_cam[:, 2]
            + intrinsics[:, 1],
            intrinsics[:, 0] * points_cam[:, 1] / points_cam[:, 2]
            + intrinsics[:, 2],
        ],
        dim=-1,
    )
    expected = expected_xy - input_dict["points_2d"]
    expected = torch.where(
        (points_cam[:, 2:3] > 1e-12).expand_as(expected),
        expected,
        torch.zeros_like(expected),
    )

    assert problem.rig_enabled is False
    assert model.pose.shape == (2, 7)
    np.testing.assert_array_equal(problem.pose_indices, problem.camera_indices)
    torch.testing.assert_close(residual, expected)
