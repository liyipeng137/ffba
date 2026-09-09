import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pycolmap")
spec = importlib.util.spec_from_file_location(
    "ffba_track_selection_metrics",
    Path(__file__).resolve().parents[1] / "scripts" / "evaluate_track_selection.py",
)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
compare_cameras = metrics.compare_cameras
epipolar_errors = metrics.epipolar_errors
load_database = metrics.load_database


def test_fixed_matches_follow_source_image_names_when_ids_reverse(tmp_path):
    def write_db(filename, names, keypoints, matches):
        with sqlite3.connect(tmp_path / filename) as db:
            db.execute("CREATE TABLE images (image_id INTEGER, name TEXT)")
            db.execute("CREATE TABLE keypoints (image_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
            db.execute("CREATE TABLE two_view_geometries (pair_id INTEGER, rows INTEGER, cols INTEGER, data BLOB)")
            db.executemany("INSERT INTO images VALUES (?,?)", names.items())
            for i, xy in keypoints.items():
                xy = np.array(xy, dtype=np.float32)
                db.execute("INSERT INTO keypoints VALUES (?,?,?,?)", (i, len(xy), 2, xy.tobytes()))
            matches = np.array(matches, dtype=np.uint32)
            db.execute("INSERT INTO two_view_geometries VALUES (?,?,?,?)",
                       (2147483649, len(matches), 2, matches.tobytes()))

    write_db("database_sift.db", {1: "b", 2: "a"},
             {1: [[0, 1]], 2: [[0, 0], [1, 0]]}, [[0, 1]])
    write_db("database_loma_prior.db", {1: "a", 2: "b"},
             {1: [[2, 0]], 2: [[1, 1]]}, [[0, 0]])
    write_db("database_merged.db", {1: "a", 2: "b"},
             {1: [[0, 0], [1, 0], [2, 0]], 2: [[0, 1], [1, 1]]}, [[0, 1], [0, 2]])
    _, _, _, pairs, audit = load_database(tmp_path)
    assert audit["out_of_bounds_matches"] == 1
    assert audit["database_sift.db_pairs_requiring_column_swap"] == 1
    assert [(i, j) for i, j, _, _ in pairs] == [(1, 2), (1, 2)]
    np.testing.assert_array_equal(pairs[0][2], [[1, 0]])
    np.testing.assert_array_equal(pairs[1][2], [[2, 1]])


def test_fixed_match_epipolar_error_detects_pose_change():
    xyz = np.array([[.2, .3, 3.], [1.2, -.4, 4.]])
    cameras = {
        i: {"R": np.eye(3), "t": np.array([-float(i - 1), 0., 0.]), "K": np.eye(3)}
        for i in (1, 2)
    }
    xy = {}
    for i, camera in cameras.items():
        projected = xyz + camera["t"]
        xy[i] = projected[:, :2] / projected[:, 2:]
    pairs = [(1, 2, np.array([[0, 0], [1, 1]]), np.array([0, 1]))]
    before = epipolar_errors(cameras, xy, pairs, {(1, 2)})
    assert before["by_source"]["sift"]["max"] == pytest.approx(0, abs=1e-12)
    assert before["by_source"]["prior"]["max"] == pytest.approx(0, abs=1e-12)
    cameras[2]["t"][1] += .1
    after = epipolar_errors(cameras, xy, pairs, {(1, 2)})
    assert after["by_source"]["sift"]["min"] > 0.1
    assert after["by_source"]["prior"]["min"] > 0.1


def test_camera_comparison_removes_global_similarity():
    angle = .3
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    centers = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.]])
    reference = {i: {"C": c, "R": np.eye(3), "K": np.eye(3)}
                 for i, c in enumerate(centers)}
    current = {i: {"C": 2 * rotation @ c + [3, 4, 5],
                   "R": rotation.T, "K": np.eye(3)}
               for i, c in enumerate(centers)}
    metrics = compare_cameras(reference, current)
    assert metrics["similarity_scale"] == pytest.approx(.5)
    assert metrics["aligned_center_shift"]["max"] < 1e-12
    assert metrics["aligned_rotation_shift_deg"]["max"] < 1e-5
