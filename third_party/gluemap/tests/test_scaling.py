"""Unit tests for gluemap.math.scaling coordinate transformations."""

import copy

import numpy as np
import torch

from gluemap.math.scaling import (
    keep_inframes,
    rescale_intrinsics,
    rescale_tracks,
    rescale_tracks_single,
    standardize_query_points,
)


def _make_K(fx, fy, cx, cy):
    K = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    K[0, 0, 0] = fx
    K[0, 1, 1] = fy
    K[0, 0, 2] = cx
    K[0, 1, 2] = cy
    return K


def test_rescale_intrinsics_round_trip():
    K_orig = [_make_K(600.0, 620.0, 320.0, 240.0).clone()]
    scales = [(0.5, 0.5, 10.0, 12.0)]

    K_work = copy.deepcopy(K_orig)
    K_work = rescale_intrinsics(K_work, scales, inverse=False)
    K_work = rescale_intrinsics(K_work, scales, inverse=True)

    np.testing.assert_allclose(K_work[0].numpy(), K_orig[0].numpy(), atol=1e-12)


def test_rescale_intrinsics_skips_none():
    K = [None, _make_K(600.0, 600.0, 320.0, 240.0).clone()]
    scales = [(0.5, 0.5, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0)]
    out = rescale_intrinsics(K, scales, inverse=False)
    assert out[0] is None
    # Second entry: fx scaled from 600 to 1200.
    assert out[1][0, 0, 0].item() == 1200.0


def test_rescale_intrinsics_known_values():
    K = [_make_K(600.0, 600.0, 320.0, 240.0).clone()]
    scales = [(2.0, 4.0, 10.0, 20.0)]
    out = rescale_intrinsics(K, scales, inverse=False)
    # Forward: cx -= x_offset; row 0 /= x_scale -> fx_new = 600/2 = 300,
    # cx_new = (320-10)/2 = 155.
    np.testing.assert_allclose(out[0][0, 0, 0].item(), 300.0)
    np.testing.assert_allclose(out[0][0, 0, 2].item(), 155.0)
    np.testing.assert_allclose(out[0][0, 1, 1].item(), 150.0)
    np.testing.assert_allclose(out[0][0, 1, 2].item(), (240.0 - 20.0) / 4.0)


def test_keep_inframes_boundary_behavior():
    H, W = 100, 200
    image_shape_ori = {0: (H, W)}
    # tracks shape (1, n_inner, K, 2): n_inner=1, K=4 sample points.
    tracks = torch.tensor(
        [[[[0.0, 0.0], [W - 1, H - 1], [W, H - 1], [-0.5, 50.0]]]],
        dtype=torch.float32,
    )
    vis = torch.ones(1, 1, 4, dtype=torch.float32)
    predictions_dict = {
        "indexes": [[0]],
        "tracks": [tracks],
        "vis": [vis],
    }
    keep_inframes(predictions_dict, image_shape_ori, indexes=[0])

    vis_out = predictions_dict["vis"][0][0, 0]
    # (0,0) inside, (W-1,H-1) inside, (W,H-1) outside (x>=W),
    # (-0.5,50) outside (x<0).
    assert vis_out[0].item() == 1.0
    assert vis_out[1].item() == 1.0
    assert vis_out[2].item() == 0.0
    assert vis_out[3].item() == 0.0


def test_standardize_then_rescale_single_round_trip():
    rng = np.random.default_rng(0)
    pts_orig = torch.from_numpy(rng.uniform(-50, 50, size=(20, 2)))
    image_change = (1.5, 2.0, 10.0, -5.0)

    pts = pts_orig.clone()
    pts = standardize_query_points(pts, image_change)
    pts = rescale_tracks_single(pts, image_change)
    np.testing.assert_allclose(pts.numpy(), pts_orig.numpy(), atol=1e-12)


def test_rescale_single_then_standardize_round_trip():
    rng = np.random.default_rng(1)
    pts_orig = torch.from_numpy(rng.uniform(-50, 50, size=(1, 30, 2)))
    image_change = (0.7, 1.3, 4.0, -2.0)

    pts = pts_orig.clone()
    pts = rescale_tracks_single(pts, image_change)
    pts = standardize_query_points(pts, image_change)
    np.testing.assert_allclose(pts.numpy(), pts_orig.numpy(), atol=1e-12)


def test_rescale_tracks_per_image_scales():
    # Two images with different (sx, sy, x_shift, y_shift): ensure the per-image
    # transform is applied to that image's track entry, not a global one.
    image_changes_full = {
        0: (2.0, 2.0, 0.0, 0.0),
        1: (10.0, 10.0, 0.0, 0.0),
    }
    pts_img0 = torch.tensor([[10.0, 20.0]], dtype=torch.float64)
    pts_img1 = torch.tensor([[10.0, 20.0]], dtype=torch.float64)
    tracks = torch.stack([pts_img0, pts_img1], dim=0).unsqueeze(
        0
    )  # shape (1, 2, 1, 2)
    predictions_dict = {
        "indexes": {0: [0, 1]},
        "tracks": {0: tracks.clone()},
    }
    out = rescale_tracks(
        predictions_dict, image_changes_full, indexes=[0], reverse=False
    )

    # reverse=False uses rescale_tracks_single: t[i] = (t[i] - shift)/scale.
    # img 0: (10,20) -> (5,10). img 1: (10,20) -> (1,2).
    np.testing.assert_allclose(
        out["tracks"][0][0, 0, 0].numpy(), [5.0, 10.0], atol=1e-12
    )
    np.testing.assert_allclose(
        out["tracks"][0][0, 1, 0].numpy(), [1.0, 2.0], atol=1e-12
    )
