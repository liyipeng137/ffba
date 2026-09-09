"""Compare the formal controller against the previous BAE filtering contract."""

import ast
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import logging
from dataclasses import dataclass

import pytest


def load_controller(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("final_round", [False, True])
@pytest.mark.parametrize("filter_rounds", [1, 3, 5])
def test_bae_controller_matches_existing_filter_schedule(
    monkeypatch, final_round, filter_rounds
):
    root = Path(__file__).resolve().parents[1]
    solver = ModuleType("gluemap.estimators.bae_solver")
    calls = []

    def solve(reconstruction, virtual, negative, **kwargs):
        calls.append(("solve", kwargs))
        return reconstruction, virtual, {"successful": True}

    solver.bundle_adjustment_bae = solve
    monkeypatch.setitem(sys.modules, "gluemap.estimators.bae_solver", solver)

    def filter_(reconstruction, error_type, threshold, min_length, **kwargs):
        calls.append(("filter", threshold, min_length))
        # Enough removal to have triggered the old Ceres re-BA path.
        return 100, 10

    namespace = dict(
        __name__=__name__,
        dataclass=dataclass,
        pycolmap=SimpleNamespace(Reconstruction=object),
        logger=logging.getLogger("test"),
        logging=logging,
        filter_reconstruction_by_reprojection_error=filter_,
        ReprojectionErrorType=SimpleNamespace(NORMALIZED="normalized"),
    )
    previous = load_controller(
        root / "third_party/gluemap/gluemap/controllers/augmented_bundle_adjustment.py",
        {"IterativeBAOptions", "iterative_bundle_adjustment"},
        dict(namespace),
    )
    current = load_controller(
        root / "ffba/reconstruction/bae.py",
        {"IterativeBAOptions", "iterative_bundle_adjustment"},
        dict(namespace),
    )
    common = dict(
        max_filter_iterations=filter_rounds,
        normalized_reproj_threshold=0.01,
        min_track_length=2,
        bae_device="cuda",
        bae_max_iterations=20,
        bae_optimize_intrinsics=True,
        bae_fix_gauge="two_cams",
        bae_robust_loss="huber",
        bae_huber_delta=2.0,
        run_post_ba_filter=final_round,
    )
    reconstruction = SimpleNamespace(
        points3D={1: SimpleNamespace(track=SimpleNamespace(elements=[1, 2, 3]))}
    )
    old_options = previous["IterativeBAOptions"](
        **common, ba_backend="bae", allow_re_ba_after_filter=False
    )
    new_options = current["IterativeBAOptions"](**common)
    previous["iterative_bundle_adjustment"](reconstruction, None, {}, old_options)
    expected = list(calls)
    calls.clear()
    current["iterative_bundle_adjustment"](
        reconstruction, None, {}, options=new_options
    )
    assert calls == expected
    assert len([c for c in calls if c[0] == "solve"]) == 1
    assert new_options.last_ba_summary == old_options.last_ba_summary
    with pytest.raises(ValueError, match="real tracks only"):
        current["iterative_bundle_adjustment"](
            reconstruction, object(), {}, options=new_options
        )
