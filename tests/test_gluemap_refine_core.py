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


def test_joint_selection_is_invariant_to_sift_labels():
    """Real C++ selection must prune redundant SIFT and preserve source audits."""
    pytest.importorskip("pygluemap")
    ref._ensure_gluemap_imports()

    class Reconstruction:
        def __init__(self, swap_sources):
            self.images = {i: None for i in (1, 2, 3)}
            self.points3D = {
                point_id: SimpleNamespace(track=SimpleNamespace(elements=[
                    SimpleNamespace(image_id=i, point2D_idx=(point_id + 2 * swap_sources) % 4)
                    for i in self.images
                ]))
                for point_id in range(4)
            }

        def delete_point3D(self, point_id):
            del self.points3D[point_id]

    features = [{"keypoints": np.zeros((2, 2))} for _ in range(3)]
    retained = []
    for swap_sources in (False, True):
        reconstruction = Reconstruction(swap_sources)
        stats, pairs = ref.run_select_tracks(
            reconstruction, features, min_num_support_abs=0, return_pair_count=True,
        )
        assert stats["before"] == {"total": 4, "s": 2, "non_s": 2, "mixed": 0}
        assert stats["after"]["total"] == 1
        assert stats["after"]["s"] + stats["after"]["non_s"] == 1
        assert stats["removed_points3D"] == 3
        assert sorted(pairs.values()) == [1, 1, 1]
        retained.append(set(reconstruction.points3D))
    assert retained[0] == retained[1]


def test_sift_candidate_pairs_keep_pose_and_add_temporal_recall():
    combined, stats = ref.build_sift_candidate_pairs(
        np.array([[0, 1], [2, 1]], dtype=np.int64),
        num_images=4,
        temporal_window=2,
    )

    np.testing.assert_array_equal(
        combined,
        np.array(
            [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]],
            dtype=np.int64,
        ),
    )
    assert stats == {
        "legacy_pose_pair_count": 2,
        "temporal_pair_count": 5,
        "temporal_overlap_pair_count": 2,
        "temporal_new_pair_count": 3,
        "sift_candidate_pair_count": 5,
    }


def test_sift_grid_coverage_requires_multiple_inliers_per_cell():
    keypoints = np.array(
        [
            [10.0, 10.0],
            [11.0, 11.0],
            [60.0, 10.0],
            [10.0, 60.0],
            [60.0, 60.0],
        ]
    )

    coverage, occupied = ref._grid_coverage_for_matched_keypoints(
        keypoints,
        np.arange(5),
        image_size_hw=(100, 100),
        grid_size=2,
        min_inliers_per_cell=2,
    )

    assert occupied == 1
    assert coverage == pytest.approx(0.25)


def test_sift_first_center_selection_uses_support_and_max_gap():
    selection = ref.select_sift_first_centers(
        num_images=6,
        valid_edges=np.array(
            [[0, 1], [2, 3], [4, 5]],
            dtype=np.int64,
        ),
        max_center_gap=2,
    )

    assert selection["selected_centers"] == [0, 2, 4]
    assert selection["owner"] == [0, 0, 2, 2, 4, 4]
    assert [frame["reason"] for frame in selection["frames"]] == [
        "first_frame",
        "skipped_supported",
        "max_gap",
        "skipped_supported",
        "max_gap",
        "skipped_supported",
    ]


def test_sift_first_center_selection_promotes_unsupported_frame():
    selection = ref.select_sift_first_centers(
        num_images=5,
        valid_edges=np.array([[0, 1], [3, 4]], dtype=np.int64),
        max_center_gap=3,
    )

    assert selection["selected_centers"] == [0, 2, 3]
    assert selection["owner"] == [0, 0, 2, 3, 3]
    assert selection["frames"][2]["reason"] == "insufficient_sift_support"


def test_sift_schedule_threshold_sweep_reports_predicted_center_count():
    records = [
        {
            "pair": [0, 1],
            "inlier_count": 100,
            "source_grid_coverage": 0.20,
            "target_grid_coverage": 0.20,
        },
        {
            "pair": [2, 3],
            "inlier_count": 70,
            "source_grid_coverage": 0.15,
            "target_grid_coverage": 0.15,
        },
    ]

    simulations = ref.simulate_sift_schedule_thresholds(
        records,
        num_images=4,
        max_center_gap=2,
        inlier_thresholds=(64, 96),
        coverage_thresholds=(0.15,),
    )

    assert simulations == [
        {
            "min_pair_inliers": 64,
            "min_grid_coverage": 0.15,
            "valid_schedule_pair_count": 2,
            "selected_center_count": 2,
            "skipped_center_count": 2,
            "selected_center_ratio": 0.5,
        },
        {
            "min_pair_inliers": 96,
            "min_grid_coverage": 0.15,
            "valid_schedule_pair_count": 1,
            "selected_center_count": 3,
            "skipped_center_count": 1,
            "selected_center_ratio": 0.75,
        },
    ]


def test_three_layer_groups_force_owned_and_bridges_before_projected_fill():
    details = {
        0: [
            {"image_index": 2},
            {"image_index": 3},
            {"image_index": 4},
        ],
        2: [
            {"image_index": 1},
            {"image_index": 5},
        ],
        4: [
            {"image_index": 2},
            {"image_index": 1},
            {"image_index": 0},
        ],
    }

    groups, stats = ref.build_three_layer_vggsfm_groups(
        selected_centers=[0, 2, 4],
        owner=[0, 0, 2, 2, 4, 4],
        valid_sift_edges=np.array([[0, 1], [0, 2], [2, 3], [2, 4], [4, 5]]),
        projected_candidate_details=details,
        num_images=6,
        max_neighbors=3,
    )

    assert groups == [
        [0, 1, 2, 3],
        [2, 3, 0, 4],
        [4, 5, 2, 1],
    ]
    assert stats["overflow_groups"] == 0
    assert stats["layer_member_counts"] == {
        "owned_frame": 3,
        "adjacent_center_bridge": 4,
        "projected_overlap_fill": 2,
    }
    assert stats["group_records"][0]["member_sources"] == {
        "1": "owned_frame",
        "2": "adjacent_center_bridge",
        "3": "projected_overlap_fill",
    }


def test_three_layer_groups_reject_non_center_owner():
    with pytest.raises(ValueError, match="Every owner must be a selected center"):
        ref.build_three_layer_vggsfm_groups(
            selected_centers=[0, 2],
            owner=[0, 1, 2],
            valid_sift_edges=[],
            projected_candidate_details={},
            num_images=3,
            max_neighbors=2,
        )


def test_sift_pose_dino_candidates_prioritize_valid_sift_evidence():
    extrinsic = np.repeat(np.eye(4, dtype=np.float64)[None, :3], 5, axis=0)
    retrieval = np.zeros((5, 5), dtype=np.float64)
    retrieval[0, 4] = 0.99
    records = [
        {
            "pair": [0, 1],
            "inlier_count": 100,
            "source_grid_coverage": 0.25,
            "target_grid_coverage": 0.25,
            "valid_schedule_edge": True,
        },
        {
            "pair": [0, 2],
            "inlier_count": 80,
            "source_grid_coverage": 0.40,
            "target_grid_coverage": 0.40,
            "valid_schedule_edge": True,
        },
        {
            "pair": [0, 3],
            "inlier_count": 200,
            "source_grid_coverage": 0.50,
            "target_grid_coverage": 0.10,
            "valid_schedule_edge": False,
        },
    ]

    details, stats = ref.build_sift_pose_dino_candidate_details(
        selected_centers=[0],
        sift_pair_records=records,
        pose_pairs=np.array([[0, 3]], dtype=np.int64),
        temporal_pairs=np.array([[0, 1]], dtype=np.int64),
        retrieval_sim_matrix=retrieval,
        extrinsic=extrinsic,
        rotation_threshold=30.0,
        dino_candidates=1,
    )

    assert [item["image_index"] for item in details[0]] == [2, 1, 3, 4]
    assert stats["strategy"] == "sift_pose_dino"
    assert stats["valid_sift_candidate_count"]["mean"] == 2.0


def test_three_layer_groups_support_depth_free_fill_provenance():
    groups, stats = ref.build_three_layer_vggsfm_groups(
        selected_centers=[0, 2],
        owner=[0, 0, 2],
        valid_sift_edges=np.array([[0, 1]], dtype=np.int64),
        projected_candidate_details={
            0: [{"image_index": 2}],
            2: [{"image_index": 0}],
        },
        num_images=3,
        max_neighbors=3,
        fill_source_name="sift_pose_dino_fill",
        strategy_name="sift_first_three_layer_sift_pose_dino",
    )

    assert groups == [[0, 1, 2], [2, 0]]
    assert stats["strategy"] == "sift_first_three_layer_sift_pose_dino"
    assert stats["layer_member_counts"]["sift_pose_dino_fill"] == 2


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
