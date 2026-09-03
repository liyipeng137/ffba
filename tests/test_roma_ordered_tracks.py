import numpy as np

from utils.roma_ordered_tracks import (
    _TrackRecord,
    build_ordered_tracking_plan,
    compute_stage_a_temporal_geometry,
    rescue_short_tracks_for_frame_coverage,
    spatial_prefilter_indices,
    spatial_select_indices,
)


def test_spatial_selection_round_robins_across_image_cells():
    points = np.array(
        [
            [5, 5],
            [6, 6],
            [55, 5],
            [56, 6],
            [5, 55],
            [6, 56],
            [55, 55],
            [56, 56],
        ],
        dtype=np.float32,
    )
    scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])

    selected = spatial_select_indices(
        points,
        scores,
        4,
        (64, 64),
        radius=3,
        grid_size=2,
    )

    assert selected.tolist() == [0, 2, 4, 6]


def test_spatial_prefilter_preserves_low_score_sparse_cell():
    dense_cluster = np.stack(
        [np.linspace(1, 30, 100), np.full(100, 5.0)], axis=-1
    ).astype(np.float32)
    sparse_cell = np.array([[55.0, 55.0]], dtype=np.float32)
    points = np.concatenate([dense_cluster, sparse_cell], axis=0)
    scores = np.concatenate([np.linspace(1.0, 0.5, 100), [0.01]])

    selected = spatial_prefilter_indices(
        points,
        scores,
        max_points=4,
        image_size_wh=(64, 64),
        grid_size=2,
        oversample=2,
    )

    assert 100 in selected


def test_spatial_selection_fills_cells_missing_existing_tracks_first():
    points = np.array(
        [[5, 5], [10, 10], [40, 10], [45, 10]], dtype=np.float32
    )
    existing = np.array(
        [[5, 5], [8, 8], [12, 12], [16, 16]], dtype=np.float32
    )

    selected = spatial_select_indices(
        points,
        np.ones(4),
        max_points=1,
        image_size_wh=(64, 64),
        radius=0,
        grid_size=2,
        existing_points=existing,
    )

    assert selected.tolist() == [2]


def _record(observations):
    record = _TrackRecord()
    for image_idx, xy in observations.items():
        record.observations[image_idx] = np.asarray(xy, dtype=np.float32)
        record.certainty[image_idx] = 1.0
        record.covariance[image_idx] = np.eye(2, dtype=np.float32)
        record.provenance[image_idx] = "birth" if image_idx == min(observations) else "sequential"
    return record


def test_short_track_rescue_only_fills_weak_frame_coverage():
    records = [
        _record({0: [5, 5], 1: [6, 5], 2: [7, 5]}),
        _record({2: [10, 10], 3: [11, 10]}),
        _record({2: [50, 10], 3: [51, 10]}),
        _record({2: [10, 50], 3: [11, 50]}),
        _record({3: [50, 50]}),
    ]

    rescued, stats = rescue_short_tracks_for_frame_coverage(
        records,
        {0},
        num_images=4,
        min_observations=2,
        image_size_wh=(64, 64),
        grid_size=2,
        radius=3,
    )

    assert len(rescued) == 2
    assert rescued <= {1, 2, 3}
    assert 4 not in rescued
    assert stats["rescued_observations"] == 4
    assert stats["deficient_frames_after"] == [0, 1]


def test_ordered_plan_keeps_backbone_and_adds_two_distinct_direct_anchors():
    adjacent = [
        {"motion_median_normalized": value}
        for value in (0.025, 0.025, 0.025, 0.025, 0.025)
    ]
    geometry = {
        (source, target): {
            "projected_overlap": 0.8,
            "projected_grid_coverage": 0.75,
            "parallax_median_deg": 1.0 + 0.1 * (target - source),
        }
        for target in range(1, 6)
        for source in range(max(0, target - 5), target - 1)
    }

    plan = build_ordered_tracking_plan(
        adjacent,
        geometry,
        6,
        max_anchor_gap=5,
        continuity_motion_target=0.05,
        geometry_motion_target=0.10,
    )

    sequential = {(edge.source, edge.target) for edge in plan if edge.kind == "sequential"}
    assert sequential == {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)}
    target_five_direct = [edge for edge in plan if edge.target == 5 and edge.kind != "sequential"]
    assert {edge.kind for edge in target_five_direct} == {
        "continuity_direct",
        "geometry_direct",
    }
    assert len({edge.source for edge in target_five_direct}) == 2


def test_stage_a_geometry_reuses_pose_depth_and_intrinsics():
    extrinsic = np.repeat(np.eye(4, dtype=np.float64)[None], 3, axis=0)
    intrinsic = np.repeat(
        np.array(
            [[50.0, 0.0, 31.5], [0.0, 50.0, 23.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )[None],
        3,
        axis=0,
    )
    depth = np.full((3, 48, 64), 5.0, dtype=np.float32)

    geometry = compute_stage_a_temporal_geometry(
        extrinsic,
        intrinsic,
        depth,
        max_gap=2,
        max_samples=64,
    )

    # Adjacent pairs are the unconditional RoMa backbone; Stage A geometry is
    # only needed to vet non-adjacent direct anchors.
    assert set(geometry) == {(0, 2)}
    assert geometry[(0, 2)]["projected_overlap"] == 1.0
    assert geometry[(0, 2)]["projected_grid_coverage"] == 1.0
    assert geometry[(0, 2)]["parallax_median_deg"] == 0.0
