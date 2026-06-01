"""Unit tests for gluemap.estimators.intrinsics_averaging.intrinsics_averaging.

The function takes a list of per-cluster (B, N_i, 3, 3) intrinsics tensors,
groups image indices into camera "buckets" via ``intrinsics_mapping``, and
returns one (B, 3, 3) median tensor per bucket (or ``None`` for empty
buckets). When ``camera_model`` starts with ``SIMPLE``, fx and fy are first
forced equal to (fx + fy) / 2 before the median.
"""

import numpy as np
import torch

from gluemap.estimators.intrinsics_averaging import intrinsics_averaging


def _K(fx, fy, cx=320.0, cy=240.0):
    return torch.tensor(
        [[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]], dtype=torch.float64
    )


def test_identical_intrinsics_returns_input():
    K = _K(600.0, 620.0)
    # One cluster with 3 images, all sharing the same K. (B=1, N=3, 3, 3).
    intrinsics_all = [torch.stack([K[0]] * 3, dim=0).unsqueeze(0)]
    communities = [[0, 1, 2]]
    intrinsics_mapping = {0: 0, 1: 0, 2: 0}

    out = intrinsics_averaging(
        intrinsics_all, communities, intrinsics_mapping, camera_model="PINHOLE"
    )

    assert len(out) == 1
    np.testing.assert_allclose(out[0].numpy(), K.numpy(), atol=1e-12)


def test_outlier_focal_rejected_by_pinhole_median():
    inlier = _K(600.0, 600.0)
    outlier = _K(10000.0, 10000.0)
    Ks = [inlier[0]] * 5 + [outlier[0]] * 4  # 5 inliers, 4 outliers
    intrinsics_all = [torch.stack(Ks, dim=0).unsqueeze(0)]
    communities = [list(range(9))]
    intrinsics_mapping = {i: 0 for i in range(9)}

    out = intrinsics_averaging(
        intrinsics_all, communities, intrinsics_mapping, camera_model="PINHOLE"
    )

    np.testing.assert_allclose(out[0][0, 0, 0].item(), 600.0, atol=1e-12)
    np.testing.assert_allclose(out[0][0, 1, 1].item(), 600.0, atol=1e-12)


def test_simple_camera_focal_is_average_of_fx_fy():
    # Per-image (fx, fy) before averaging — for SIMPLE the function forces
    # fx=fy=(fx+fy)/2 BEFORE the median.
    pairs = [(300.0, 400.0), (310.0, 390.0), (290.0, 410.0)]
    Ks = [_K(fx, fy)[0] for fx, fy in pairs]
    intrinsics_all = [torch.stack(Ks, dim=0).unsqueeze(0)]
    communities = [[0, 1, 2]]
    intrinsics_mapping = {0: 0, 1: 0, 2: 0}

    out = intrinsics_averaging(
        intrinsics_all,
        communities,
        intrinsics_mapping,
        camera_model="SIMPLE_PINHOLE",
    )

    # All three images get fx=fy=(fx+fy)/2 = 350.0; median is 350.0.
    np.testing.assert_allclose(out[0][0, 0, 0].item(), 350.0, atol=1e-12)
    np.testing.assert_allclose(out[0][0, 1, 1].item(), 350.0, atol=1e-12)
    # SIMPLE keeps fx==fy.
    assert out[0][0, 0, 0].item() == out[0][0, 1, 1].item()


def test_pinhole_keeps_fx_fy_separate():
    # Distinct fx and fy medians per axis under PINHOLE.
    fxs = [600.0, 610.0, 620.0]
    fys = [400.0, 410.0, 420.0]
    Ks = [_K(fx, fy)[0] for fx, fy in zip(fxs, fys, strict=False)]
    intrinsics_all = [torch.stack(Ks, dim=0).unsqueeze(0)]
    communities = [[0, 1, 2]]
    intrinsics_mapping = {0: 0, 1: 0, 2: 0}

    out = intrinsics_averaging(
        intrinsics_all, communities, intrinsics_mapping, camera_model="PINHOLE"
    )
    np.testing.assert_allclose(out[0][0, 0, 0].item(), 610.0, atol=1e-12)
    np.testing.assert_allclose(out[0][0, 1, 1].item(), 410.0, atol=1e-12)


def test_multiple_buckets_routed_independently():
    # Two camera buckets routed via intrinsics_mapping; assert each bucket's
    # median is independent of the other.
    K_bucket0 = _K(500.0, 500.0)
    K_bucket1 = _K(800.0, 800.0)
    Ks = [K_bucket0[0], K_bucket0[0], K_bucket1[0], K_bucket1[0]]
    intrinsics_all = [torch.stack(Ks, dim=0).unsqueeze(0)]
    communities = [[0, 1, 2, 3]]
    # Images 0, 1 -> bucket 0; images 2, 3 -> bucket 1.
    intrinsics_mapping = {0: 0, 1: 0, 2: 1, 3: 1}

    out = intrinsics_averaging(
        intrinsics_all, communities, intrinsics_mapping, camera_model="PINHOLE"
    )

    assert len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), K_bucket0.numpy(), atol=1e-12)
    np.testing.assert_allclose(out[1].numpy(), K_bucket1.numpy(), atol=1e-12)


def test_empty_bucket_returns_none():
    # Bucket 1 has no contributors; should be None in the output list.
    K = _K(500.0, 500.0)
    intrinsics_all = [K.unsqueeze(0)]  # (B=1, N=1, 3, 3)
    communities = [[0]]
    intrinsics_mapping = {0: 0, 1: 1}  # bucket 1 declared but empty

    out = intrinsics_averaging(
        intrinsics_all, communities, intrinsics_mapping, camera_model="PINHOLE"
    )
    assert len(out) == 2
    np.testing.assert_allclose(out[0].numpy(), K.numpy(), atol=1e-12)
    assert out[1] is None
