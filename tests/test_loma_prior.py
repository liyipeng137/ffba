import argparse
import ast
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from utils import loma_prior as loma


def test_candidates_keep_pose_temporal_and_one_way_dino_without_nonfinite_fill():
    similarity = np.full((5, 5), np.nan)
    similarity[0, 4] = 0.9
    similarity[4, 2] = 0.8
    similarity[2, 1] = np.inf  # invalid retrieval must not become a candidate
    records, stats = loma.build_loma_candidate_pairs(
        [[0, 1]], [[1, 2], [2, 3]], similarity, 30
    )
    assert [r["pair"] for r in records] == [[0, 1], [0, 4], [1, 2], [2, 3], [2, 4]]
    assert records[1]["dino_retrieval"] == [
        {"source": 0, "target": 4, "rank": 1, "similarity": 0.9}
    ]
    assert stats["prior_only_pairs"] == 2
    assert stats["sift_intersection_pairs"] == 3
    assert stats["zero_degree_images"] == []
    assert stats["components"] == [[0, 1, 2, 3, 4]]


def test_candidates_are_not_truncated_to_old_group_k_and_ties_are_stable():
    records, _ = loma.build_loma_candidate_pairs([], [], np.ones((40, 40)), 30)
    zero_neighbors = [r["pair"][1] for r in records if r["pair"][0] == 0]
    assert len(zero_neighbors) == 39  # incoming choices retained, no degree=30 cap
    outgoing = [
        d["target"] for r in records for d in r["dino_retrieval"] if d["source"] == 0
    ]
    assert outgoing == list(range(1, 31))
    single, _ = loma.build_loma_candidate_pairs([], [], np.ones((1, 1)))
    assert single == []


def test_sift_annotation_does_not_remove_strong_weak_or_untried_pairs():
    config = {"min_pair_inliers": 128, "min_grid_coverage": 0.2}
    schedule = {
        "config": config,
        "pairs": [
            {
                "pair": [0, 1],
                "inlier_count": 200,
                "source_grid_coverage": 0.3,
                "target_grid_coverage": 0.4,
            },
            {
                "pair": [1, 2],
                "inlier_count": 0,
                "source_grid_coverage": 0,
                "target_grid_coverage": 0,
            },
        ],
    }
    candidates = [{"pair": [0, 1]}, {"pair": [0, 2]}, {"pair": [1, 2]}]
    result = loma.annotate_loma_pairs_with_sift(candidates, [[0, 1], [1, 2]], schedule)
    assert [r["sift_support"] for r in result] == [
        "sufficient",
        "untried",
        "insufficient",
    ]
    assert result[1]["sift"] is None
    assert len(result[2]["sift_insufficient_reasons"]) == 3
    with pytest.raises(ValueError, match="exactly the executed"):
        loma.annotate_loma_pairs_with_sift(candidates, [[0, 1]], schedule)


def test_pixel_centers_roundtrip_without_half_pixel_shift_or_clamping():
    h, w = 1920, 1440
    pixels = np.array(
        [[0.5, 0.5], [1439.5, 1919.5], [210.25, 913.75]], dtype=np.float32
    )
    normalized = 2 * pixels / [w, h] - 1
    np.testing.assert_allclose(
        loma.normalized_to_work_pixels(normalized, h, w), pixels, atol=1e-4
    )


def _selection_record(
    i, j, support="untried", temporal=False, inliers=100, coverage=0.1
):
    return {
        "pair": [i, j],
        "sources": ["temporal"] if temporal else ["dino"],
        "sift_support": support,
        "sift": None
        if support == "untried"
        else {
            "inlier_count": inliers,
            "source_grid_coverage": coverage,
            "target_grid_coverage": coverage,
        },
    }


def test_selection_preserves_temporal_zero_quotas_and_all_mode():
    pool = [_selection_record(0, 1, temporal=True), _selection_record(0, 2)]
    similarity = np.ones((3, 3))
    records, stats = loma.select_loma_pairs(
        pool,
        similarity,
        sufficient_neighbors=0,
        insufficient_neighbors=0,
        untried_neighbors=0,
    )
    assert [r["pair"] for r in records if r["selected"]] == [[0, 1]]
    assert not records[1]["executed"] and "raw_matches" not in records[1]
    assert stats["graph"]["components"] == [[0, 1], [2]]
    all_records, _ = loma.select_loma_pairs(pool, similarity, mode="all")
    assert all(r["selected"] for r in all_records)
    assert all("selected" not in r for r in pool)  # input audit stays immutable
    with pytest.raises(ValueError, match="nonnegative"):
        loma.select_loma_pairs(pool, similarity, insufficient_neighbors=-1)


def test_selection_limits_each_class_without_refill_or_incoming_degree_cap():
    n = 31
    pool = [
        _selection_record(
            0, j, ("sufficient", "insufficient", "untried")[(j - 1) // 10]
        )
        for j in range(1, n)
    ]
    records, stats = loma.select_loma_pairs(pool, np.ones((n, n)))
    assert stats["directed_choices_per_image"][0] == {
        "sufficient": 3,
        "insufficient": 5,
        "untried": 5,
    }
    assert stats["directed_choices_per_image"][1] == {
        "sufficient": 1,
        "insufficient": 0,
        "untried": 0,
    }
    assert stats["graph"]["degree"][0] == 30  # all leaf choices are retained
    assert all(r["selected"] for r in records)
    for r in records:
        assert any(choice.get("source") == r["pair"][1] for choice in r["selection"])


def test_selection_ranks_reliable_support_and_plausible_weak_edges_with_soft_diversity():
    pool = [
        _selection_record(0, 1, "sufficient", inliers=200, coverage=0.3),
        _selection_record(0, 2, "sufficient", inliers=300, coverage=0.4),
        _selection_record(0, 3, "insufficient", inliers=0),
        _selection_record(0, 4, "insufficient", inliers=90),
        _selection_record(0, 5, "untried"),
        _selection_record(0, 6, "untried"),
        _selection_record(0, 9, "untried"),
    ]
    sim = np.zeros((10, 10))
    sim[0, [1, 2, 3, 4, 5, 6, 9]] = [0.9, 0.8, 0.1, 0.9, 0.9, 0.8, 0.7]
    records, _ = loma.select_loma_pairs(
        pool,
        sim,
        sufficient_neighbors=1,
        insufficient_neighbors=1,
        untried_neighbors=2,
        temporal_window=1,
    )
    choices = [
        (r["pair"][1], c["reason"], c["rank"])
        for r in records
        for c in r["selection"]
        if c.get("source") == 0
    ]
    assert choices == [
        (2, "sufficient", 1),
        (4, "insufficient", 1),
        (6, "untried", 1),
        (9, "untried", 2),
    ]
    # If all candidates are adjacent, diversity must not prevent quota filling.
    records, _ = loma.select_loma_pairs(
        pool, sim, untried_neighbors=3, temporal_window=100
    )
    assert (
        sum(
            c.get("source") == 0 and c["reason"] == "untried"
            for r in records
            for c in r["selection"]
        )
        == 3
    )


def test_selection_is_deterministic_and_within_pool_and_pair_bound():
    n = 50
    rng = np.random.default_rng(13)
    sim = rng.normal(size=(n, n))
    pool = [
        _selection_record(
            i,
            j,
            ("sufficient", "insufficient", "untried")[(i + j) % 3],
            temporal=j - i <= 2,
        )
        for i in range(n)
        for j in range(i + 1, n)
    ]
    a, stats = loma.select_loma_pairs(pool, sim)
    b, _ = loma.select_loma_pairs(list(reversed(pool)), sim)
    assert {tuple(r["pair"]): r["selection"] for r in a} == {
        tuple(r["pair"]): r["selection"] for r in b
    }
    assert stats["num_selected_pairs"] <= stats["temporal_pairs"] + n * 13
    assert stats["graph"]["num_components"] == 1
    assert stats["num_skipped_pairs"] > 0


def _synthetic_features():
    rng = np.random.default_rng(19)
    points = rng.uniform([-1.2, -0.8, 4], [1.2, 0.8, 8], (100, 3))
    intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
    features = []
    for center in (0, 0.35, 0.8):
        local = points - [center, 0, 0]
        projected = local @ intrinsic.T
        features.append(
            {
                "keypoints": (projected[:, :2] / projected[:, 2:]).astype(np.float32),
                "image_size_hw": (480, 640),
            }
        )
    return features, intrinsic


class SyntheticBackend:
    metadata = {"test_backend": True}

    def __init__(self, features):
        self.features = features
        self.extractions = []
        self.matches = []

    def synchronize(self):
        pass

    def extract(self, path):
        i = int(Path(path).stem)
        self.extractions.append(i)
        return self.features[i]

    def match(self, feature0, feature1):
        self.matches.append((id(feature0), id(feature1)))
        ids = np.arange(len(feature0["keypoints"]), dtype=np.uint32)
        return np.column_stack((ids, ids)), np.ones(len(ids))

    def match_batch(self, pairs):
        return [self.match(a, b) for a, b in pairs]

    def extract_batched(self, paths, batch_size, workers):
        for i in reversed(range(len(paths))):
            yield i, self.extract(paths[i])


def test_real_geometry_db_roundtrip_and_dropped_frame_remap(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    features, intrinsic = _synthetic_features()
    backend = SyntheticBackend(features)
    records = [
        {"pair": [0, 1], "sift_support": "sufficient"},
        {"pair": [0, 2], "sift_support": "untried"},
        {"pair": [1, 2], "sift_support": "insufficient"},
    ]
    result = loma.run_loma_prior(
        [f"{i}.png" for i in range(3)],
        (480, 640),
        [intrinsic] * 3,
        records,
        backend=backend,
    )
    assert backend.extractions == [0, 1, 2]
    assert len(backend.matches) == 3
    assert result.stats["num_verified_pairs"] == 3
    np.testing.assert_array_equal(result.observation_counts(), [100, 100, 100])
    assert result.stats["unique_matched_observations"] == 300  # not 600 pair endpoints
    json.dumps(result.stats)  # nested native geometry options must serialize
    subset = result.subset([0, 2])
    assert set(subset.geometries) == {(0, 1)}
    assert subset.pair_records[0]["original_pair"] == [0, 2]
    db_path = tmp_path / "prior.db"
    loma.write_loma_database(db_path, ["0.png", "2.png"], (480, 640), intrinsic, subset)
    db = pycolmap.Database.open(str(db_path))
    try:
        assert db.num_images() == 2
        np.testing.assert_array_equal(db.read_keypoints(2), features[2]["keypoints"])
        np.testing.assert_array_equal(
            db.read_matches(1, 2), subset.geometries[(0, 1)].inlier_matches
        )
        assert len(db.read_two_view_geometry(1, 2).inlier_matches) == 100
    finally:
        db.close()


def test_empty_matches_remain_a_valid_empty_prior(tmp_path):
    pytest.importorskip("pycolmap")
    features, intrinsic = _synthetic_features()
    backend = SyntheticBackend(features)
    backend.match = lambda *_: (np.empty((0, 2), dtype=np.uint32), np.empty(0))
    result = loma.run_loma_prior(
        ["0.png", "1.png"],
        (480, 640),
        [intrinsic] * 2,
        [{"pair": [0, 1], "sift_support": "untried"}],
        backend=backend,
    )
    assert result.stats["num_pairs"] == 1
    assert result.stats["num_verified_pairs"] == 0
    assert result.stats["verified_graph"]["components"] == [[0], [1]]
    assert result.stats["unique_matched_observations"] == 0
    loma.write_loma_database(
        tmp_path / "empty.db", ["0.png", "1.png"], (480, 640), intrinsic, result
    )


def test_real_triangulation_builds_three_view_tracks(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    features, intrinsic = _synthetic_features()
    records = [
        {"pair": pair, "sift_support": "untried"} for pair in ([0, 1], [0, 2], [1, 2])
    ]
    result = loma.run_loma_prior(
        [f"{i}.png" for i in range(3)],
        (480, 640),
        [intrinsic] * 3,
        records,
        backend=SyntheticBackend(features),
    )
    db_path = tmp_path / "triangulate.db"
    names = [f"{i}.png" for i in range(3)]
    loma.write_loma_database(db_path, names, (480, 640), intrinsic, result)
    reconstruction = pycolmap.Reconstruction()
    camera = loma._camera(pycolmap, intrinsic, (480, 640))
    reconstruction.add_camera(camera)
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(camera.sensor_id)
    reconstruction.add_rig(rig)
    for i, center in enumerate((0, 0.35, 0.8)):
        image = pycolmap.Image(
            name=names[i],
            camera_id=1,
            image_id=i + 1,
            keypoints=features[i]["keypoints"].astype(float),
        )
        image.frame_id = i + 1
        frame = pycolmap.Frame()
        frame.frame_id = i + 1
        frame.rig_id = 1
        frame.add_data_id(image.data_id)
        frame.rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(), np.array([-center, 0, 0])
        )
        reconstruction.add_frame(frame)
        reconstruction.add_image(image)
        reconstruction.register_frame(i + 1)
    options = pycolmap.IncrementalPipelineOptions()
    options.extract_colors = False
    options.ba_global_max_refinements = 0
    triangulated = pycolmap.triangulate_points(
        reconstruction, db_path, tmp_path, tmp_path / "triangulated", options=options
    )
    assert triangulated.num_reg_images() == 3
    assert triangulated.num_points3D() == 100
    assert all(point.track.length() == 3 for point in triangulated.points3D.values())
    audit = loma.summarize_final_tracks(triangulated, {1: 0, 2: 0, 3: 0}, result)
    assert audit["track_length_histogram"] == {3: 100}
    assert audit["by_sift_support_length_histogram"]["untried"] == {3: 100}


def _load_definitions(filename, names, namespace):
    """Exercise actual config functions without unrelated CUDA model imports."""
    path = Path(__file__).resolve().parents[1] / filename
    tree = ast.parse(path.read_text())
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


def _config_namespace():
    ns = {
        "__name__": __name__,
        "dataclass": dataclass,
        "SimpleNamespace": SimpleNamespace,
        "Path": Path,
        "np": np,
        "CAMERA_MODEL": "SIMPLE_PINHOLE",
        "S_DATABASE_MODE": "sift",
        "QUERY_SOURCE": "aliked",
        "GROUP_STRATEGY": "pose",
        "TRACKER_INPUT": "1024",
    }
    return _load_definitions(
        "utils/gluemap_spv_refine.py",
        {"GluemapSpvRefineConfig", "_make_refine_args"},
        ns,
    )


def test_provider_config_preserves_vggsfm_and_disallows_hidden_loma_cap():
    ns = _config_namespace()
    Config = ns["GluemapSpvRefineConfig"]
    make = ns["_make_refine_args"]
    legacy = make(Config(path_tracker="tracker", bae_max_observations=2_000_000))
    assert legacy.prior_provider == "vggsfm"
    assert legacy.bae_max_observations == 2_000_000
    assert legacy.build_virtual_tracks
    loma_args = make(
        Config(path_tracker="not-used", prior_provider="loma", ba_backend="bae")
    )
    assert loma_args.bae_max_observations == 0
    assert loma_args.loma_pair_selection == "sift_guided"
    assert (
        loma_args.loma_sufficient_neighbors,
        loma_args.loma_insufficient_neighbors,
        loma_args.loma_untried_neighbors,
    ) == (3, 5, 5)
    assert not loma_args.build_virtual_tracks
    settings = dict(
        loma_match_batch_size=8,
        loma_extract_batch_size=2,
        loma_preprocess_workers=4,
        loma_geometry_workers=4,
        loma_feature_cache="cuda",
    )
    configured = make(
        Config(
            path_tracker="",
            prior_provider="loma",
            ba_backend="bae",
            device="cuda",
            **settings,
        )
    )
    assert {name: getattr(configured, name) for name in settings} == settings
    with pytest.raises(ValueError, match="preprocess_workers"):
        make(
            Config(
                path_tracker="",
                prior_provider="loma",
                ba_backend="bae",
                loma_preprocess_workers=-1,
            )
        )
    with pytest.raises(ValueError, match="no observation cap"):
        make(
            Config(
                path_tracker="",
                prior_provider="loma",
                ba_backend="bae",
                bae_max_observations=2_000_000,
            )
        )
    with pytest.raises(ValueError, match="requires --ba_backend bae"):
        make(Config(path_tracker="", prior_provider="loma"))


def test_cli_default_provider_and_intrinsics_can_be_disabled(monkeypatch):
    ns = {"argparse": argparse, "PIPELINE_GROUP_STRATEGY": "pose"}
    _load_definitions("run_merg3r_gluemap_pipeline.py", {"parse_args"}, ns)
    monkeypatch.setattr(
        "sys.argv", ["pipeline", "--dataset", "images", "--output_dir", "out"]
    )
    defaults = ns["parse_args"]()
    assert defaults.prior_provider == "vggsfm"
    assert defaults.bae_optimize_intrinsics is True
    assert (
        defaults.loma_match_batch_size,
        defaults.loma_extract_batch_size,
        defaults.loma_preprocess_workers,
        defaults.loma_geometry_workers,
        defaults.loma_feature_cache,
    ) == (1, 1, 0, 1, "cpu")
    monkeypatch.setattr(
        "sys.argv",
        [
            "pipeline",
            "--dataset",
            "images",
            "--output_dir",
            "out",
            "--prior_provider",
            "loma",
            "--no-bae_optimize_intrinsics",
        ],
    )
    args = ns["parse_args"]()
    assert args.loma_dino_candidates == 30
    assert args.loma_pair_selection == "sift_guided"
    assert (
        args.loma_sufficient_neighbors,
        args.loma_insufficient_neighbors,
        args.loma_untried_neighbors,
    ) == (3, 5, 5)
    assert args.bae_max_observations == 0
    assert args.bae_optimize_intrinsics is False


@pytest.mark.parametrize(
    "has_depth,drop_middle,prune_pairs",
    [(False, False, False), (True, True, False), (False, False, True)],
)
def test_refinement_routes_loma_through_sift_merge_and_shared_backend(
    tmp_path, monkeypatch, has_depth, drop_middle, prune_pairs
):
    """Actual orchestration/DB merge, synthetic inference and stubbed BA solver."""
    pycolmap = pytest.importorskip("pycolmap")
    feature_data, intrinsic = _synthetic_features()
    names = [f"{i}.png" for i in range(3)]
    sift_features = [
        {"keypoints": feature["keypoints"][:10]} for feature in feature_data
    ]
    ns = _config_namespace()
    core_ns = {"np": np}
    _load_definitions(
        "utils/gluemap_refine_core.py",
        {
            "canonicalize_pair_array",
            "build_temporal_pairs",
            "build_sift_candidate_pairs",
        },
        core_ns,
    )
    ref = SimpleNamespace(
        **{key: value for key, value in core_ns.items() if callable(value)}
    )
    ref._ensure_gluemap_imports = lambda: None
    ref._lazy_import_pycolmap = lambda: pycolmap
    events = []
    sift_stats = {
        "observations_per_image": [10] * 3,
        "num_keypoints_total": 30,
        "num_pairs": 3,
        "num_matches": 30,
    }

    def prepare(*args):
        events.append("sift")
        loma.write_loma_database(
            tmp_path / "database_sift.db",
            names,
            (480, 640),
            intrinsic,
            loma.LoMaPriorResult([x["keypoints"] for x in sift_features], {}, [], {}),
        )
        return sift_features, dict(sift_stats)

    ref.prepare_sift_database_for_refine = prepare
    ref.analyze_sift_schedule_graph = lambda *_args, **_kwargs: {
        "config": {"min_pair_inliers": 128, "min_grid_coverage": 0.2},
        "pairs": [
            {
                "pair": pair,
                "inlier_count": 0,
                "source_grid_coverage": 0,
                "target_grid_coverage": 0,
            }
            for pair in _args[2].tolist()
        ],
        "verified_pair_count": 0,
        "valid_schedule_pair_count": 0,
    }
    ref.simulate_sift_schedule_thresholds = Mock(
        side_effect=AssertionError("No LoMa threshold sweep")
    )
    ref.select_sift_first_centers = Mock(
        side_effect=AssertionError("No LoMa center selection")
    )
    ref.run_vggsfm_prior_tracks = Mock(
        side_effect=AssertionError("No VGGSfM inference")
    )
    ref.format_count_summary = lambda label, values: label
    kept = [0, 2] if drop_middle else [0, 1, 2]
    selected_names = [names[i] for i in kept]
    selected_features = [sift_features[i] for i in kept]
    final_pairs = np.array(
        [[i, j] for i in range(len(kept)) for j in range(i + 1, len(kept))]
    )

    def frame_filter(
        image_names,
        images,
        poses,
        features,
        pairs,
        tracks,
        s_counts,
        p_counts,
        threshold,
        enabled,
    ):
        np.testing.assert_array_equal(p_counts, [100, 100, 100])
        assert tracks == []  # no synthetic two-view tracks used for counts
        return (
            selected_names,
            images[kept],
            poses[kept],
            selected_features,
            final_pairs,
            [],
            {
                "min_frame_observations": threshold,
                "kept_indices": kept,
                "dropped_indices": [1] if drop_middle else [],
                "dropped_names": [names[1]] if drop_middle else [],
            },
        )

    ref.filter_low_coverage_frames = frame_filter
    ref.average_intrinsics_with_gluemap = lambda *_: (
        np.array([intrinsic]),
        [intrinsic],
        {i: 0 for i in range(len(kept))},
    )
    ref.summarize_intrinsics = lambda *_: {}
    ref.save_intrinsics_artifacts = lambda *_: None
    ref.write_coarse_reconstruction = lambda *_: None

    def filter_sift(*_args):
        loma.write_loma_database(
            tmp_path / "database_sift.db",
            selected_names,
            (480, 640),
            intrinsic,
            loma.LoMaPriorResult(
                [x["keypoints"] for x in selected_features], {}, [], {}
            ),
        )
        return selected_features, {**sift_stats, "num_keypoints_total": 10 * len(kept)}

    ref.filter_sift_database_for_refine = filter_sift
    ref.build_s_keypoint_count = lambda *_: {i + 1: 10 for i in range(len(kept))}
    original_run = loma.run_loma_prior

    def run(*args, **kwargs):
        events.append("loma")
        if has_depth:
            assert {
                key: kwargs[key]
                for key in (
                    "match_batch_size",
                    "extract_batch_size",
                    "preprocess_workers",
                    "geometry_workers",
                    "feature_cache",
                )
            } == dict(
                match_batch_size=2,
                extract_batch_size=2,
                preprocess_workers=2,
                geometry_workers=2,
                feature_cache="cpu",
            )
        if prune_pairs:
            assert [r["pair"] for r in args[3]] == [[0, 1], [1, 2]]
        return original_run(*args, **kwargs, backend=SyntheticBackend(feature_data))

    monkeypatch.setattr(loma, "run_loma_prior", run)
    native_merge = {
        "pycolmap": pycolmap,
        "os": os,
        "np": np,
        "logger": logging.getLogger(__name__),
    }
    _load_definitions(
        "third_party/gluemap/gluemap/utils/colmap.py",
        {"merge_colmap_databases"},
        native_merge,
    )
    module = ModuleType("gluemap.utils.colmap")
    module.merge_colmap_databases = native_merge["merge_colmap_databases"]
    monkeypatch.setitem(sys.modules, "gluemap.utils.colmap", module)

    def refine(args, _pycolmap, output, image_names, *_args):
        events.append("refine")
        assert args.prior_provider == "loma" and args.bae_max_observations == 0
        assert not args.bae_optimize_intrinsics
        assert image_names == selected_names
        db = pycolmap.Database.open(str(output / "database_merged.db"))
        try:
            assert db.num_images() == len(kept)
            for i, old in enumerate(kept):
                combined = db.read_keypoints(i + 1)
                assert len(combined) == 110
                np.testing.assert_array_equal(
                    combined[10:], feature_data[old]["keypoints"]
                )
            assert (db.read_matches(1, 2) >= 10).all()  # SIFT-first offsets preserved
            if prune_pairs:
                assert not db.exists_matches(1, 3)
        finally:
            db.close()
        return (
            pycolmap.Reconstruction(),
            None,
            {
                "final": {
                    "real": {"points3D": 0},
                    "virtual": {"points3D": 0},
                    "real_by_source": {"s_only": 0, "p_only": 0, "mixed": 0},
                }
            },
        )

    ref.run_merg3r_augmented_refinement_loop = refine
    ns.update(
        ref=ref,
        time=time,
        json=json,
        _save_work_images=lambda *_args, **_kwargs: (tmp_path, names),
        export_prediction_depth_maps=lambda *_args, **_kwargs: {"test_export": True},
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    _load_definitions(
        "utils/gluemap_spv_refine.py",
        {
            "GluemapSpvRefineResult",
            "_debug",
            "_write_json",
            "run_gluemap_spv_refinement",
        },
        ns,
    )
    state = SimpleNamespace(
        high_images=np.zeros((3, 3, 480, 640)),
        high_image_size_hw=(480, 640),
        low_image_size_hw=(480, 640),
        intrinsic_high=np.array([intrinsic] * 3),
        intrinsic_low=np.array([intrinsic] * 3),
        extrinsic=np.array([np.eye(4)[:3]] * 3),
        pairs=np.array([[0, 1]]),
        raw_depth=np.ones((3, 2, 2)) if has_depth else None,
        raw_depth_conf=None,
        retrieval_sim_matrix=np.ones((3, 3)),
        low_image_names=names,
        high_image_names=names,
        image_pyramid=None,
    )
    config = ns["GluemapSpvRefineConfig"](
        path_tracker="DO_NOT_LOAD",
        prior_provider="loma",
        ba_backend="bae",
        device="cpu",
        vggsfm_group_strategy="projected_overlap",
        vggsfm_schedule_mode="sift_first_sparse",
        vggsfm_max_center_gap=0,
        bae_optimize_intrinsics=False,
        sift_temporal_window=1 if prune_pairs else 2,
        loma_untried_neighbors=0 if prune_pairs else 5,
        loma_match_batch_size=2 if has_depth else 1,
        loma_extract_batch_size=2 if has_depth else 1,
        loma_preprocess_workers=2 if has_depth else 0,
        loma_geometry_workers=2 if has_depth else 1,
    )
    result = ns["run_gluemap_spv_refinement"](state, tmp_path, config)
    assert events == ["sift", "loma", "refine"]
    assert result.image_names == selected_names
    assert result.stats["prior_provider"] == "loma"
    assert result.stats["vggsfm_schedule_mode"] is None
    assert result.stats["depth_export"]["enabled"] == has_depth
    assert not (tmp_path / "database_vggsfm_prior.db").exists()
    assert not (tmp_path / "vggsfm_schedule.json").exists()
    assert (tmp_path / "prior_loma_pairs.json").exists()
    if prune_pairs:
        audit = json.loads((tmp_path / "prior_loma_pairs.json").read_text())
        assert audit["selection"]["num_selected_pairs"] == 2
        skipped = next(r for r in audit["pairs"] if r["pair"] == [0, 2])
        assert not skipped["selected"] and not skipped["executed"]
        assert "inlier_matches" not in skipped
    assert json.loads((tmp_path / "refine_stats.json").read_text())["loma"][
        "final_tracks"
    ]
