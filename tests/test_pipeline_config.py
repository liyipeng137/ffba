from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from ffba.config import DEFAULT_CONFIG, flatten, load_config, parse_args
from ffba.matching.scheduling import (
    build_scheduling_order,
    resolve_group_strategy,
    select_sift_first_centers,
)
from ffba.matching.sift import build_sift_candidate_pairs


@pytest.mark.parametrize("mode", ["standard", "lite"])
@pytest.mark.parametrize("input_order", ["ordered", "unordered"])
@pytest.mark.parametrize("prior_pose", [False, True])
def test_eight_workflows_share_one_config(mode, input_order, prior_pose):
    argv = [
        "--dataset",
        "images",
        "--output_dir",
        "out",
        "--mode",
        mode,
        "--input_order",
        input_order,
    ]
    if prior_pose:
        argv += ["--prior_transforms_json", "transforms.json"]
    args = parse_args(argv)
    assert args.prior_provider == ("loma" if mode == "lite" else "vggsfm")
    assert args.ba_backend == "bae"
    assert args.vggsfm_schedule_mode == "sift_first_sparse"
    assert args.sequence_type == "shortest_path"
    assert args.alignment_type == "weighted_iterative"
    assert args.sift_temporal_window == (2 if input_order == "ordered" else 0)
    assert bool(args.prior_transforms_json) == prior_pose


def test_overrides_do_not_mutate_defaults_and_cli_wins(tmp_path):
    path = tmp_path / "override.yaml"
    path.write_text(
        "pipeline:\n  mode: standard\nprior:\n  loma:\n    match_batch_size: 3\nbae:\n  optimize_intrinsics: false\n"
    )
    args = parse_args(
        [
            "--dataset",
            "images",
            "--output_dir",
            "out",
            "--config",
            str(path),
            "--mode",
            "lite",
        ]
    )
    assert args.mode == "lite"
    assert args.loma_match_batch_size == 3
    assert not args.bae_optimize_intrinsics
    assert flatten(load_config())["loma_match_batch_size"] == 2


@pytest.mark.parametrize(
    "text",
    [
        "ba_backend: ceres\n",
        "prior:\n  loma:\n    match_batch_size: 0\n",
        "prior:\n  vggsfm:\n    group_strategy: pose\n",
        "sift:\n  temporal_window: true\n",
        'bae:\n  optimize_intrinsics: "false"\n',
        "bae:\n  huber_delta: .nan\n",
        "pipeline:\n  mode: lite\nbae:\n  max_observations: 100\n",
        "pipeline:\n  mode: lite\n  mode: standard\n",
    ],
)
def test_invalid_config_fails_early(tmp_path, text):
    path = tmp_path / "invalid.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_config(path)


def test_nullable_final_delta_and_unknown_repository_field(tmp_path, monkeypatch):
    path = tmp_path / "override.yaml"
    path.write_text("bae:\n  final_huber_delta: null\n")
    assert flatten(load_config(path))["final_bae_huber_delta"] is None
    original = yaml.safe_load(DEFAULT_CONFIG.read_text())
    original["runtime"]["misspelled_workers"] = 3
    path.write_text(yaml.safe_dump(original))
    monkeypatch.setattr("ffba.config.DEFAULT_CONFIG", path)
    with pytest.raises(ValueError, match="schema mismatch"):
        load_config()


def test_dino_order_keeps_image_identity_and_valid_owners():
    similarity = np.array(
        [
            [0.0, 0.1, 0.9, 0.2],
            [0.1, 0, 0.2, 0.8],
            [0.9, 0.2, 0, 0.7],
            [0.2, 0.8, 0.7, 0],
        ]
    )
    original = similarity.copy()
    order = build_scheduling_order(similarity, "unordered")
    assert sorted(order) == list(range(4))
    assert order != list(range(4))
    assert order == build_scheduling_order(similarity, "unordered")
    np.testing.assert_array_equal(similarity, original)
    edges = [[order[i], order[i + 1]] for i in range(3)]
    selected = select_sift_first_centers(4, edges, 2, order=order)
    assert selected["selected_centers"] == order[::2]
    assert selected["owner"][order[1]] == order[0]
    assert selected["owner"][order[3]] == order[2]
    for record in selected["frames"]:
        assert record["owner"] == selected["owner"][record["image_index"]]
    ordered = select_sift_first_centers(4, [[0, 1], [1, 2], [2, 3]], 2)
    assert ordered == select_sift_first_centers(
        4, [[0, 1], [1, 2], [2, 3]], 2, order=[0, 1, 2, 3]
    )
    with pytest.raises(ValueError, match="permutation"):
        select_sift_first_centers(4, [], 2, order=[0, 0, 2, 3])


def test_unordered_never_promotes_dino_neighbors_to_temporal_pairs():
    args = parse_args(
        ["--dataset", "images", "--output_dir", "out", "--input_order", "unordered"]
    )
    pairs, stats = build_sift_candidate_pairs([[0, 3]], 4, args.sift_temporal_window)
    np.testing.assert_array_equal(pairs, [[0, 3]])
    assert stats["temporal_pair_count"] == 0


@pytest.mark.parametrize(
    "requested,has_depth,effective",
    [
        ("projected_overlap", True, "projected_overlap"),
        ("projected_overlap", False, "sift_pose_dino"),
        ("sift_pose_dino", True, "sift_pose_dino"),
        ("sift_pose_dino", False, "sift_pose_dino"),
    ],
)
def test_depth_fallback_is_explicit(requested, has_depth, effective):
    result = resolve_group_strategy(requested, has_depth)
    assert result["requested_strategy"] == requested
    assert result["effective_strategy"] == effective
    assert bool(result["fallback_reason"]) == (requested != effective)


def test_help_does_not_require_torch():
    root = Path(__file__).resolve().parents[1]
    code = "import sys,runpy; sys.modules['torch']=None; sys.argv=['pipeline','--help']; runpy.run_path('run.py',run_name='__main__')"
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--mode" in result.stdout
    assert "--ba_backend" not in result.stdout
