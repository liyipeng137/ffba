import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def test_final_bae_huber_delta_defaults_to_legacy_behavior():
    args = SimpleNamespace(
        bae_huber_delta=1.0,
        num_refinement_iterations=3,
    )

    deltas = [
        ref._bae_huber_delta_for_iteration(args, outer_iter)
        for outer_iter in range(args.num_refinement_iterations)
    ]

    assert deltas == [1.0, 1.0, 1.0]


def test_final_bae_huber_delta_only_applies_to_final_iteration():
    args = SimpleNamespace(
        bae_huber_delta=1.0,
        final_bae_huber_delta=2.0,
        num_refinement_iterations=3,
    )

    deltas = [
        ref._bae_huber_delta_for_iteration(args, outer_iter)
        for outer_iter in range(args.num_refinement_iterations)
    ]

    assert deltas == [1.0, 1.0, 2.0]


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


def _budget_track(
    point3d_id,
    image_ids,
    angular_error_p90,
    max_triangulation_angle_deg,
    source="p_only",
):
    return {
        "point3D_id": point3d_id,
        "image_ids": tuple(image_ids),
        "track_length": len(image_ids),
        "angular_error_p90": angular_error_p90,
        "max_triangulation_angle_deg": max_triangulation_angle_deg,
        "source": source,
    }


def test_bae_budget_plan_is_source_agnostic_and_accepts_whole_track_overshoot():
    records = [
        _budget_track(1, [1, 2], 0.8, 1.0, source="p_only"),
        _budget_track(2, [1, 2], 0.9, 20.0, source="s_only"),
        _budget_track(3, [1, 2, 3], 0.95, 1.0, source="p_only"),
    ]

    plan = ref._plan_bae_track_deletions(
        records,
        image_observations={1: 3, 2: 3, 3: 1},
        max_observations=5,
        min_observations_per_image=0,
    )

    assert plan["reached_budget"] is True
    assert plan["remaining_observations"] == 5
    assert [record["point3D_id"] for record in plan["deleted_records"]] == [2]

    overshoot = ref._plan_bae_track_deletions(
        [
            _budget_track(10, [1, 2, 3], 0.5, 10.0),
            _budget_track(11, [1, 2, 3], 0.4, 10.0),
        ],
        image_observations={1: 2, 2: 2, 3: 2},
        max_observations=4,
        min_observations_per_image=0,
    )
    assert overshoot["reached_budget"] is True
    assert overshoot["remaining_observations"] == 3


def test_bae_budget_plan_uses_small_parallax_after_equal_error():
    records = [
        _budget_track(1, [1, 2], 0.8, 10.0),
        _budget_track(2, [1, 2], 0.8, 2.0),
        _budget_track(3, [1, 2, 3], 0.9, 1.0),
    ]

    plan = ref._plan_bae_track_deletions(
        records,
        image_observations={1: 3, 2: 3, 3: 1},
        max_observations=5,
        min_observations_per_image=0,
    )

    assert [record["point3D_id"] for record in plan["deleted_records"]] == [2]


def test_bae_budget_plan_reports_unreachable_floor_without_mutating_records():
    records = [
        _budget_track(1, [1, 2], 0.9, 1.0),
        _budget_track(2, [1, 3], 0.8, 1.0),
        _budget_track(3, [2, 3], 0.7, 1.0),
    ]

    plan = ref._plan_bae_track_deletions(
        records,
        image_observations={1: 2, 2: 2, 3: 2},
        max_observations=2,
        min_observations_per_image=1,
    )

    assert plan["reached_budget"] is False
    assert plan["remaining_observations"] == 4
    assert [record["point3D_id"] for record in plan["deleted_records"]] == [1]
    assert [record["point3D_id"] for record in records] == [1, 2, 3]


def test_track_max_triangulation_angle_degrees():
    angle = ref._track_max_triangulation_angle_degrees(
        point_xyz=np.zeros(3),
        image_ids=[1, 2],
        camera_centers={
            1: np.array([-1.0, 0.0, 0.0]),
            2: np.array([0.0, -1.0, 0.0]),
        },
    )

    assert angle == pytest.approx(90.0)


def _ordered_candidate(
    neighbor,
    *,
    overlap=0.8,
    grid_coverage=0.7,
    visible_ratio=0.9,
    motion=0.08,
    eligible=True,
    temporal=True,
):
    return {
        "image_index": neighbor,
        "candidate_sources": ["temporal"] if temporal else ["dino"],
        "selection_eligible": eligible,
        "projected_overlap": overlap,
        "projected_grid_coverage": grid_coverage,
        "projected_visible_ratio": visible_ratio,
        "projected_depth_valid_ratio": visible_ratio,
        "motion_median_normalized": motion,
    }


def test_projected_overlap_candidates_include_ordered_motion_signals():
    extrinsic = np.zeros((2, 3, 4), dtype=np.float64)
    extrinsic[:, :3, :3] = np.eye(3)
    intrinsics = np.repeat(
        np.array([[[2.0, 0.0, 1.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]]),
        2,
        axis=0,
    )

    groups, stats, details = ref.build_projected_overlap_groups(
        pairs=np.array([[0, 1]], dtype=np.int64),
        extrinsic=extrinsic,
        intrinsics=intrinsics,
        depth=np.ones((2, 4, 4), dtype=np.float32),
        depth_conf=None,
        retrieval_sim_matrix=np.eye(2, dtype=np.float64),
        max_neighbors=None,
        rotation_threshold=30.0,
        dino_candidates=1,
        max_samples=16,
        reprojection_threshold=1.0,
        temporal_window=1,
        min_projected_overlap=0.5,
        min_projected_grid_coverage=0.5,
        min_projected_visible_ratio=0.5,
    )

    assert groups == [[0, 1], [1, 0]]
    assert stats["max_neighbors"] is None
    assert details[0][0]["motion_median_normalized"] == pytest.approx(0.0)
    assert details[0][0]["parallax_median_deg"] == pytest.approx(0.0)
    assert details[0][0]["consistent_grid_mask"].bit_count() == 16


def test_ordered_sift_pair_selection_uses_budget_without_ineligible_fill():
    candidate_details = {
        0: [
            _ordered_candidate(1),
            _ordered_candidate(2, temporal=False),
            _ordered_candidate(4, eligible=False),
        ],
        1: [_ordered_candidate(2), _ordered_candidate(3, temporal=False)],
        2: [_ordered_candidate(3), _ordered_candidate(4, temporal=False)],
        3: [_ordered_candidate(4)],
        4: [],
    }

    pairs, stats = ref.select_ordered_sift_pairs(
        candidate_details,
        num_images=5,
        pair_budget=5,
    )

    assert pairs.shape == (5, 2)
    assert (0, 4) not in {tuple(pair) for pair in pairs.tolist()}
    assert stats["selected_pairs"] == 5
    assert stats["connected_components"] == 1
    assert stats["zero_degree_images"] == 0

    budgeted_pairs, budgeted_stats = ref.select_ordered_sift_pairs(
        candidate_details,
        num_images=5,
        pair_budget=2,
    )
    assert budgeted_pairs.shape == (2, 2)
    assert budgeted_stats["backbone_truncated_by_budget"] is True


def test_ordered_vggsfm_groups_are_sift_aware_and_variable_k():
    candidate_details = {
        0: [
            _ordered_candidate(1),
            _ordered_candidate(2),
            _ordered_candidate(3),
        ],
        1: [_ordered_candidate(0)],
        2: [],
        3: [],
    }
    sift_pair_quality = {
        (0, 1): {"matches": 512, "grid_coverage_min": 0.8},
        (0, 2): {"matches": 8, "grid_coverage_min": 0.05},
    }

    groups, stats = ref.build_ordered_vggsfm_groups(
        candidate_details,
        sift_pair_quality,
        num_images=4,
        neighbor_slot_budget=4,
        sift_deficit_weight=1.0,
        hard_max_neighbors=3,
        legacy_neighbors_per_center=1,
    )

    groups_by_center = {group[0]: group[1:] for group in groups}
    assert groups_by_center[0][0] == 2
    assert len(groups_by_center[0]) == 3
    assert stats["selected_neighbor_slots"] == 4
    assert stats["centers_exceeding_legacy_k"] == 1
    assert stats["max_selected_neighbors"] == 3

    cost_limited_groups, cost_limited_stats = ref.build_ordered_vggsfm_groups(
        candidate_details,
        sift_pair_quality,
        num_images=4,
        neighbor_slot_budget=4,
        sift_deficit_weight=1.0,
        hard_max_neighbors=3,
        legacy_neighbors_per_center=1,
        group_cost_budget=13,
    )
    assert sum(len(group) ** 2 for group in cost_limited_groups) <= 13
    assert cost_limited_stats["group_cost_proxy"] <= 13


def test_sift_pair_quality_prefers_verified_matches(monkeypatch, tmp_path):
    class FakeDatabase:
        def read_all_images(self):
            return [
                SimpleNamespace(image_id=1, name="a.jpg"),
                SimpleNamespace(image_id=2, name="b.jpg"),
            ]

        def read_keypoints(self, image_id):
            del image_id
            return np.array(
                [[0.5, 0.5, 1.0, 0.0], [4.5, 4.5, 1.0, 0.0], [7.5, 7.5, 1.0, 0.0]],
                dtype=np.float32,
            )

        def read_all_matches(self):
            return [42], [np.array([[0, 0], [1, 1], [2, 2]], dtype=np.uint32)]

        def read_two_view_geometry(self, image_id1, image_id2):
            assert (image_id1, image_id2) == (1, 2)
            return SimpleNamespace(
                inlier_matches=np.array([[0, 0], [2, 2]], dtype=np.uint32)
            )

        def close(self):
            pass

    database = FakeDatabase()
    fake_pycolmap = SimpleNamespace(
        Database=SimpleNamespace(open=lambda _path: database),
        pair_id_to_image_pair=lambda _pair_id: (1, 2),
    )
    monkeypatch.setattr(ref, "_lazy_import_pycolmap", lambda: fake_pycolmap)

    quality = ref.summarize_database_pair_quality(
        tmp_path / "database.db",
        ["a.jpg", "b.jpg"],
        (8, 8),
    )

    assert quality[(0, 1)]["matches"] == 2
    assert quality[(0, 1)]["raw_matches"] == 3
    assert quality[(0, 1)]["grid_coverage_min"] == pytest.approx(2 / 64)


def test_remap_groups_drops_removed_centers_and_neighbors():
    groups = [[0, 1, 2, 3], [1, 0], [2, 0, 3], [3, 2]]

    remapped = ref.remap_groups(groups, {0: 0, 2: 1, 3: 2})

    assert remapped == [[0, 1, 2], [1, 0, 2], [2, 1]]


def test_bae_budget_pruning_deletes_selected_track_in_place():
    class FakeReconstruction:
        def __init__(self, points3d, images):
            self.points3D = points3d
            self.images = images

        def delete_point3D(self, point3d_id):
            del self.points3D[point3d_id]

    def image_at(center):
        pose = SimpleNamespace(
            rotation=SimpleNamespace(matrix=lambda: np.eye(3)),
            translation=-np.asarray(center, dtype=np.float64),
        )
        return SimpleNamespace(cam_from_world=lambda: pose)

    def point(xyz, observations):
        elements = [
            SimpleNamespace(image_id=image_id, point2D_idx=point2d_idx)
            for image_id, point2d_idx in observations
        ]
        return SimpleNamespace(
            xyz=np.asarray(xyz, dtype=np.float64),
            track=SimpleNamespace(elements=elements),
        )

    reconstruction = FakeReconstruction(
        points3d={
            1: point([0.0, 0.0, 5.0], [(1, 0), (2, 0)]),
            2: point([0.0, 0.0, 5.0], [(1, 1), (2, 1)]),
            3: point([0.0, 0.0, 5.0], [(1, 1), (2, 1), (3, 1)]),
        },
        images={
            1: image_at([-1.0, 0.0, 0.0]),
            2: image_at([1.0, 0.0, 0.0]),
            3: image_at([0.0, 1.0, 0.0]),
        },
    )
    features = [{"keypoints": np.zeros((1, 2), dtype=np.float32)} for _ in range(3)]
    angular_errors = {
        1: [(1, 0, 0.9), (2, 0, 0.9)],
        2: [(1, 1, 0.8), (2, 1, 0.8)],
        3: [(1, 1, 0.95), (2, 1, 0.95), (3, 1, 0.95)],
    }

    stats = ref.prune_reconstruction_for_bae_observation_budget(
        reconstruction,
        features,
        max_observations=5,
        angular_errors_per_track=angular_errors,
        min_observations_per_image=0,
    )

    assert stats["applied"] is True
    assert stats["before"]["observations"] == 7
    assert stats["after"]["observations"] == 5
    assert stats["deleted_by_source"]["s_only"] == {
        "tracks": 1,
        "observations": 2,
    }
    assert set(reconstruction.points3D) == {2, 3}
