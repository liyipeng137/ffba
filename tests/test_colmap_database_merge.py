"""Real SQLite/PyCOLMAP regressions without unrelated GPU tracker imports."""

import ast
import logging
import os
from pathlib import Path

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")


@pytest.fixture
def merge_databases():
    # Execute the production functions, isolating only the module's GPU imports.
    path = Path(__file__).resolve().parents[1] / "third_party/gluemap/gluemap/utils/colmap.py"
    functions = {"_remap_matches_to_output_pair", "merge_colmap_databases"}
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
    namespace = {"np": np, "pycolmap": pycolmap, "os": os,
                 "logger": logging.getLogger(__name__)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["merge_colmap_databases"]


def write_source(path, ids, keypoints, matches):
    db = pycolmap.Database.open(str(path))
    try:
        camera = pycolmap.Camera(model="SIMPLE_PINHOLE", width=100, height=100,
                                 params=[100, 50, 50])
        camera_id = db.write_camera(camera)
        for name, image_id in ids.items():
            image = pycolmap.Image(name=name, image_id=image_id, camera_id=camera_id)
            db.write_image(image, use_image_id=True)
            db.write_keypoints(image_id, keypoints[name])
        for (a, b), correspondences in matches.items():
            correspondences = np.asarray(correspondences, dtype=np.uint32)
            db.write_matches(ids[a], ids[b], correspondences)
            geometry = pycolmap.TwoViewGeometry()
            geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
            geometry.inlier_matches = correspondences[:1].copy()
            db.write_two_view_geometry(ids[a], ids[b], geometry)
    finally:
        db.close()


@pytest.mark.parametrize("primary_first", [False, True])
@pytest.mark.parametrize("reverse_secondary_ids", [False, True])
def test_merge_preserves_match_endpoints_with_feature_offsets(
    tmp_path, merge_databases, primary_first, reverse_secondary_ids,
):
    primary_ids = {"a": 3, "b": 7, "c": 11}
    secondary_ids = {"a": 2, "b": 1, "c": 5} if reverse_secondary_ids else {"a": 1, "b": 2, "c": 5}
    primary_xy = {
        name: np.column_stack((np.arange(count) + base, np.full(count, base))).astype(np.float32)
        for name, count, base in [("a", 2, 10), ("b", 5, 20), ("c", 3, 30)]
    }
    secondary_xy = {
        name: np.column_stack((np.arange(count) + base, np.full(count, base))).astype(np.float32)
        for name, count, base in [("a", 4, 40), ("b", 2, 50), ("c", 3, 60)]
    }
    primary_matches = {("a", "b"): [[1, 3], [0, 4]], ("b", "c"): [[4, 2]]}
    secondary_matches = {("a", "b"): [[3, 0], [2, 1]], ("b", "c"): [[1, 2]]}
    primary = tmp_path / "primary.db"
    secondary = tmp_path / "secondary.db"
    output = tmp_path / "merged.db"
    write_source(primary, primary_ids, primary_xy, primary_matches)
    write_source(secondary, secondary_ids, secondary_xy, secondary_matches)
    merge_databases(str(primary), str(secondary), str(output), primary_first)

    db = pycolmap.Database.open(str(output))
    try:
        for name, image_id in primary_ids.items():
            blocks = [primary_xy[name], secondary_xy[name]]
            if not primary_first:
                blocks.reverse()
            np.testing.assert_array_equal(db.read_keypoints(image_id), np.vstack(blocks))
        for a, b in primary_matches:
            expected_matches, expected_inliers = [], []
            for xy, pairs, offsets in [
                (primary_xy, primary_matches, secondary_xy if not primary_first else None),
                (secondary_xy, secondary_matches, primary_xy if primary_first else None),
            ]:
                matches = np.asarray(pairs[a, b], dtype=np.uint32)
                offset = np.array([len(offsets[a]), len(offsets[b])] if offsets else [0, 0], dtype=np.uint32)
                expected_matches.extend((matches + offset).tolist())
                expected_inliers.extend((matches[:1] + offset).tolist())
                # Explicit physical endpoints catch even in-bounds column swaps.
                merged_a, merged_b = db.read_keypoints(primary_ids[a]), db.read_keypoints(primary_ids[b])
                np.testing.assert_array_equal(merged_a[matches[:, 0] + offset[0]], xy[a][matches[:, 0]])
                np.testing.assert_array_equal(merged_b[matches[:, 1] + offset[1]], xy[b][matches[:, 1]])
            actual = db.read_matches(primary_ids[a], primary_ids[b])
            geometry = db.read_two_view_geometry(primary_ids[a], primary_ids[b])
            assert sorted(actual.tolist()) == sorted(expected_matches)
            assert sorted(geometry.inlier_matches.tolist()) == sorted(expected_inliers)
    finally:
        db.close()

    # Merging must not change either source's feature indices/correspondences.
    for path, ids, pairs in [(primary, primary_ids, primary_matches),
                             (secondary, secondary_ids, secondary_matches)]:
        db = pycolmap.Database.open(str(path))
        try:
            for (a, b), matches in pairs.items():
                np.testing.assert_array_equal(db.read_matches(ids[a], ids[b]), matches)
        finally:
            db.close()
