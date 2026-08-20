import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = ModuleType("torch")
    torch_stub.no_grad = lambda: lambda function: function
    sys.modules["torch"] = torch_stub

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import gluemap_refine_core as ref


def _scalar_snap_prior_tracks_to_features(
    tracks,
    features,
    snap_threshold=1.0,
    keep_unsnapped=True,
):
    keypoint_trees = []
    for feats in features:
        keypoints = np.asarray(feats["keypoints"], dtype=np.float32)
        keypoint_trees.append(
            ref.cKDTree(keypoints) if keypoints.shape[0] > 0 else None
        )

    snapped_tracks = []
    stats = {
        "enabled": True,
        "snap_threshold": float(snap_threshold),
        "keep_unsnapped": bool(keep_unsnapped),
        "input_tracks": int(len(tracks)),
        "input_observations": 0,
        "output_tracks": 0,
        "output_observations": 0,
        "snapped_observations": 0,
        "unsnapped_kept_observations": 0,
        "dropped_observations": 0,
        "center_observations": 0,
        "center_snapped_observations": 0,
        "center_unsnapped_kept_observations": 0,
        "center_dropped_observations": 0,
        "neighbor_observations": 0,
        "neighbor_snapped_observations": 0,
        "neighbor_unsnapped_kept_observations": 0,
        "neighbor_dropped_observations": 0,
        "snap_distance_mean": 0.0,
        "snap_distance_max": 0.0,
    }
    snap_distances = []

    for track in tracks:
        snapped_obs = []
        seen_images = set()
        for obs_idx, (image_idx, xy) in enumerate(track):
            image_idx = int(image_idx)
            if image_idx in seen_images:
                continue
            seen_images.add(image_idx)
            stats["input_observations"] += 1
            prefix = "center" if obs_idx == 0 else "neighbor"
            stats[f"{prefix}_observations"] += 1

            xy = np.asarray(xy, dtype=np.float32)
            tree = keypoint_trees[image_idx]
            if tree is None:
                if keep_unsnapped:
                    snapped_obs.append((image_idx, xy))
                    stats["unsnapped_kept_observations"] += 1
                    stats[f"{prefix}_unsnapped_kept_observations"] += 1
                else:
                    stats["dropped_observations"] += 1
                    stats[f"{prefix}_dropped_observations"] += 1
                continue

            distance, keypoint_idx = tree.query(xy, k=1)
            if float(distance) <= snap_threshold:
                snapped_xy = features[image_idx]["keypoints"][int(keypoint_idx)].astype(
                    np.float32
                )
                snapped_obs.append((image_idx, snapped_xy))
                stats["snapped_observations"] += 1
                stats[f"{prefix}_snapped_observations"] += 1
                snap_distances.append(float(distance))
            elif keep_unsnapped:
                snapped_obs.append((image_idx, xy))
                stats["unsnapped_kept_observations"] += 1
                stats[f"{prefix}_unsnapped_kept_observations"] += 1
            else:
                stats["dropped_observations"] += 1
                stats[f"{prefix}_dropped_observations"] += 1

        if len(snapped_obs) >= 2:
            snapped_tracks.append(snapped_obs)
            stats["output_observations"] += len(snapped_obs)

    stats["output_tracks"] = int(len(snapped_tracks))
    if snap_distances:
        stats["snap_distance_mean"] = float(np.mean(snap_distances))
        stats["snap_distance_max"] = float(np.max(snap_distances))
    return snapped_tracks, stats


def _assert_tracks_equal(actual, expected):
    assert len(actual) == len(expected)
    for actual_track, expected_track in zip(actual, expected):
        assert len(actual_track) == len(expected_track)
        for (actual_image, actual_xy), (expected_image, expected_xy) in zip(
            actual_track,
            expected_track,
        ):
            assert actual_image == expected_image
            np.testing.assert_array_equal(actual_xy, expected_xy)


@pytest.mark.parametrize("keep_unsnapped", [True, False])
@pytest.mark.parametrize("snap_threshold", [0.0, 0.5, 1.0])
def test_batched_snap_matches_scalar_reference(keep_unsnapped, snap_threshold):
    features = [
        {"keypoints": np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)},
        {"keypoints": np.empty((0, 2), dtype=np.float32)},
        {"keypoints": np.array([[10.0, 10.0], [12.0, 12.0]], dtype=np.float32)},
    ]
    tracks = [
        [
            (0, np.array([0.25, 0.0])),
            (1, np.array([2.0, 2.0])),
            (2, np.array([10.5, 10.0])),
            (0, np.array([5.0, 5.0])),
        ],
        [(2, np.array([12.0, 12.0])), (0, np.array([100.0, 100.0]))],
        [(1, np.array([3.0, 3.0])), (2, np.array([50.0, 50.0]))],
        [(0, np.array([5.0, 5.0])), (0, np.array([0.0, 0.0]))],
        [],
    ]

    expected_tracks, expected_stats = _scalar_snap_prior_tracks_to_features(
        tracks,
        features,
        snap_threshold=snap_threshold,
        keep_unsnapped=keep_unsnapped,
    )
    actual_tracks, actual_stats = ref.snap_prior_tracks_to_features(
        tracks,
        features,
        snap_threshold=snap_threshold,
        keep_unsnapped=keep_unsnapped,
    )

    _assert_tracks_equal(actual_tracks, expected_tracks)
    assert actual_stats == expected_stats

    actual_keypoints, actual_matches, actual_merge_stats = (
        ref.tracks_to_keypoints_and_matches(
            actual_tracks,
            len(features),
            match_topology="star",
        )
    )
    expected_keypoints, expected_matches, expected_merge_stats = (
        ref.tracks_to_keypoints_and_matches(
            expected_tracks,
            len(features),
            match_topology="star",
        )
    )
    for actual, expected in zip(actual_keypoints, expected_keypoints):
        np.testing.assert_array_equal(actual, expected)
    assert actual_matches.keys() == expected_matches.keys()
    for pair in actual_matches:
        np.testing.assert_array_equal(actual_matches[pair], expected_matches[pair])
    assert actual_merge_stats == expected_merge_stats


def test_batched_snap_matches_scalar_reference_on_random_tracks():
    rng = np.random.default_rng(7)
    features = [
        {"keypoints": rng.uniform(0, 100, size=(count, 2)).astype(np.float32)}
        for count in (50, 0, 30, 80, 1)
    ]
    tracks = []
    for _ in range(200):
        track = []
        for _ in range(int(rng.integers(0, 8))):
            track.append(
                (
                    int(rng.integers(0, len(features))),
                    rng.uniform(0, 100, size=2).astype(np.float32),
                )
            )
        tracks.append(track)

    expected_tracks, expected_stats = _scalar_snap_prior_tracks_to_features(
        tracks,
        features,
        snap_threshold=3.0,
        keep_unsnapped=True,
    )
    actual_tracks, actual_stats = ref.snap_prior_tracks_to_features(
        tracks,
        features,
        snap_threshold=3.0,
        keep_unsnapped=True,
    )

    _assert_tracks_equal(actual_tracks, expected_tracks)
    assert actual_stats == expected_stats


def test_batched_snap_matches_scalar_reference_for_equidistant_keypoints():
    features = [
        {"keypoints": np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)},
        {"keypoints": np.array([[5.0, 5.0]], dtype=np.float32)},
    ]
    tracks = [
        [(0, np.array([1.0, 0.0])), (1, np.array([5.0, 5.0]))],
        [(0, np.array([1.0, 0.0])), (1, np.array([5.0, 5.0]))],
    ]

    expected_tracks, expected_stats = _scalar_snap_prior_tracks_to_features(
        tracks,
        features,
        snap_threshold=1.0,
    )
    actual_tracks, actual_stats = ref.snap_prior_tracks_to_features(
        tracks,
        features,
        snap_threshold=1.0,
    )

    _assert_tracks_equal(actual_tracks, expected_tracks)
    assert actual_stats == expected_stats
