from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import bae.autograd.graph as autograd_graph  # noqa: E402
from ba_colmap import ColmapResidual  # noqa: E402
from datapipes.colmap_loader import (  # noqa: E402
    read_colmap_data,
    save_colmap_cameras,
    save_colmap_result,
)


def _write_colmap_text_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "cameras.txt").write_text(
        "# Camera list\n"
        "1 PINHOLE 640 480 100.0 100.0 320.0 240.0\n",
        encoding="utf-8",
    )
    (model_dir / "images.txt").write_text(
        "# Image list\n"
        "7 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 image0001.png\n"
        "320.0 240.0 11 420.0 240.0 12\n",
        encoding="utf-8",
    )
    (model_dir / "points3D.txt").write_text(
        "# Point list\n"
        "11 0.0 0.0 1.0 255 0 0 0.0 7 0\n"
        "12 1.0 0.0 1.0 0 255 0 0.0 7 1\n",
        encoding="utf-8",
    )


def test_colmap_text_loader_and_zero_residual(tmp_path: Path):
    model_dir = tmp_path / "sparse"
    _write_colmap_text_model(model_dir)

    data = read_colmap_data(input_dir=str(model_dir))
    assert data["camera_params"].shape == (1, 7)
    assert data["points_3d"].shape == (2, 3)
    assert torch.equal(data["camera_index_of_observations"], torch.tensor([0, 0]))
    assert torch.equal(data["point_index_of_observations"], torch.tensor([0, 1]))
    assert torch.allclose(
        data["intrinsics"],
        torch.tensor([100.0, 100.0, 320.0, 240.0], dtype=torch.float64),
    )

    model = ColmapResidual(
        data["camera_params"].clone(),
        data["points_3d"].clone(),
        data["intrinsics"].clone(),
    )
    residual = model(
        data["points_2d"],
        data["camera_index_of_observations"],
        data["point_index_of_observations"],
    )
    assert torch.allclose(residual, torch.zeros_like(residual))

    j_pose, j_points, j_intr = autograd_graph.jacobian(
        residual,
        [model.pose, model.points_3d, model.shared_intr],
    )
    assert j_pose.shape == (4, 6)
    assert j_points.shape == (4, 6)
    assert j_intr.shape == (4, 4)
    assert torch.equal(j_intr.col_indices(), torch.tensor([0, 0], dtype=j_intr.col_indices().dtype))


def test_colmap_text_save_roundtrip(tmp_path: Path):
    model_dir = tmp_path / "sparse"
    out_dir = tmp_path / "optimized"
    _write_colmap_text_model(model_dir)

    data = read_colmap_data(input_dir=str(model_dir))
    out_dir.mkdir()
    save_colmap_result(
        str(out_dir / "images_optimized.txt"),
        str(out_dir / "points3D_optimized.txt"),
        data,
        data["camera_params"],
        data["points_3d"],
    )
    save_colmap_cameras(
        str(out_dir / "cameras_optimized.txt"),
        data,
        data["intrinsics"],
    )

    saved = read_colmap_data(
        str(out_dir / "cameras_optimized.txt"),
        str(out_dir / "images_optimized.txt"),
        str(out_dir / "points3D_optimized.txt"),
    )
    assert torch.allclose(saved["camera_params"], data["camera_params"])
    assert torch.allclose(saved["points_3d"], data["points_3d"])
    assert torch.allclose(saved["points_2d"], data["points_2d"])
    assert torch.allclose(saved["intrinsics"], data["intrinsics"])
