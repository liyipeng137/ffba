"""Compare frozen-input BAE replays; metrics are diagnostics, not ground truth.

Uses identical verified source-database matches for epipolar checks, including matches
not retained in a final model. They are NOT independent held-out observations.
"""

import argparse
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pycolmap


def distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"count": len(values), "finite": 0}
    return {"count": len(values), "finite": len(finite), "mean": float(finite.mean()),
            **dict(zip(["min", "p10", "median", "p90", "p95", "max"],
                       np.quantile(finite, [0, .1, .5, .9, .95, 1]).tolist()))}


def load_database(source):
    with sqlite3.connect(f"file:{source / 'database_sift.db'}?mode=ro", uri=True) as db:
        sift = dict(db.execute("SELECT images.name, keypoints.rows FROM images "
                               "JOIN keypoints ON images.image_id=keypoints.image_id"))
    with sqlite3.connect(f"file:{source / 'database_merged.db'}?mode=ro", uri=True) as db:
        names = dict(db.execute("SELECT image_id,name FROM images"))
        keypoints = {i: np.frombuffer(data, np.float32).reshape(rows, cols)[:, :2].copy()
                     for i, rows, cols, data in db.execute(
                         "SELECT image_id,rows,cols,data FROM keypoints")}
        merged_audit = {"pairs": 0, "matches": 0, "out_of_bounds_matches": 0,
                        "pairs_with_out_of_bounds_matches": 0}
        for pair_id, rows, cols, data in db.execute(
                "SELECT pair_id,rows,cols,data FROM two_view_geometries "
                "WHERE rows>0 ORDER BY pair_id"):
            i, j = divmod(pair_id, 2147483647)
            matches = np.frombuffer(data, np.uint32).reshape(rows, cols)
            invalid = (matches[:, 0] >= len(keypoints[i])) | (matches[:, 1] >= len(keypoints[j]))
            merged_audit["pairs"] += 1
            merged_audit["matches"] += len(matches)
            merged_audit["out_of_bounds_matches"] += int(invalid.sum())
            merged_audit["pairs_with_out_of_bounds_matches"] += int(invalid.any())
    pairs = []
    name_to_id = {name: i for i, name in names.items()}
    rng = np.random.default_rng(20260908)
    for source_code, filename in enumerate(["database_sift.db", "database_loma_prior.db"]):
        with sqlite3.connect(f"file:{source / filename}?mode=ro", uri=True) as db:
            source_names = dict(db.execute("SELECT image_id,name FROM images"))
            for i, rows, cols, data in db.execute("SELECT image_id,rows,cols,data FROM keypoints"):
                name = source_names[i]
                xy = np.frombuffer(data, np.float32).reshape(rows, cols)[:, :2]
                offset = sift[name] if source_code else 0
                np.testing.assert_array_equal(xy, keypoints[name_to_id[name]][offset:offset + rows])
            reversed_pairs = 0
            for pair_id, rows, cols, data in db.execute(
                    "SELECT pair_id,rows,cols,data FROM two_view_geometries "
                    "WHERE rows>0 ORDER BY pair_id"):
                a, b = divmod(pair_id, 2147483647)
                name_a, name_b = source_names[a], source_names[b]
                i, j = name_to_id[name_a], name_to_id[name_b]
                matches = np.frombuffer(data, np.uint32).reshape(rows, cols)
                matches = matches[rng.choice(rows, min(rows, 64), replace=False)].copy()
                if source_code:
                    matches += np.array([sift[name_a], sift[name_b]], dtype=np.uint32)
                if i > j:
                    i, j = j, i
                    matches = matches[:, ::-1]
                    reversed_pairs += 1
                if (matches[:, 0] >= len(keypoints[i])).any() or (matches[:, 1] >= len(keypoints[j])).any():
                    raise ValueError(f"Invalid source matches in {filename}, pair {pair_id}")
                pairs.append((i, j, matches, np.full(len(matches), source_code)))
            merged_audit[f"{filename}_pairs_requiring_column_swap"] = reversed_pairs
    return names, keypoints, {i: sift[n] for i, n in names.items()}, pairs, merged_audit


def camera_arrays(reconstruction, names):
    by_name = {im.name: im for im in reconstruction.images.values()}
    result = {}
    for i, name in names.items():
        image = by_name.get(name)
        if image is None or not image.has_pose:
            continue
        matrix = image.cam_from_world().matrix()
        camera = reconstruction.cameras[image.camera_id]
        result[i] = {
            "R": matrix[:, :3], "t": matrix[:, 3],
            "C": -matrix[:, :3].T @ matrix[:, 3],
            "K": camera.calibration_matrix(), "size": (camera.width, camera.height),
        }
    return result


def epipolar_errors(cameras, keypoints, pairs, valid_pairs):
    bearings = {}
    for i, camera in cameras.items():
        xy = keypoints[i]
        b = np.column_stack((xy, np.ones(len(xy)))) @ np.linalg.inv(camera["K"]).T
        bearings[i] = b / np.linalg.norm(b, axis=1, keepdims=True)
    buckets = [[], [], []]
    per_image = {i: [] for i in cameras}
    per_image_source = {source: {i: [] for i in cameras} for source in (0, 1)}
    for i, j, matches, sources in pairs:
        if (i, j) not in valid_pairs or i not in cameras or j not in cameras:
            continue
        a, b = cameras[i], cameras[j]
        relative_r = b["R"] @ a["R"].T
        relative_t = b["t"] - relative_r @ a["t"]
        x, y = bearings[i][matches[:, 0]], bearings[j][matches[:, 1]]
        ex = np.cross(relative_t, x @ relative_r.T)
        ety = np.cross(y, relative_t) @ relative_r
        numerator = np.abs(np.einsum("ij,ij->i", y, ex))
        error = .5 * (np.arcsin(np.clip(numerator / np.maximum(np.linalg.norm(ex, axis=1), 1e-15), 0, 1))
                      + np.arcsin(np.clip(numerator / np.maximum(np.linalg.norm(ety, axis=1), 1e-15), 0, 1)))
        error = np.rad2deg(error)
        for source in range(3):
            values = error[sources == source].tolist()
            buckets[source].extend(values)
            if source in per_image_source:
                per_image_source[source][i].extend(values)
                per_image_source[source][j].extend(values)
        per_image[i].extend(error.tolist())
        per_image[j].extend(error.tolist())
    return {
        "by_source": {k: distribution(v) for k, v in zip(["sift", "prior", "mixed"], buckets)},
        "per_image": {i: distribution(v) for i, v in per_image.items()},
        "per_image_by_source": {
            name: {i: distribution(v) for i, v in per_image_source[source].items()}
            for source, name in enumerate(["sift", "prior"])
        },
    }


def model_coverage(reconstruction, cameras, names, keypoints, sift_counts):
    observations = {i: [] for i in names}
    parent = {i: i for i in names}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lengths = []
    point_xyz = []
    angle_sample = []
    step = max(1, len(reconstruction.points3D) // 20000)
    for point_index, point in enumerate(reconstruction.points3D.values()):
        elements = list(point.track.elements)
        length = len(elements)
        lengths.append(length)
        point_xyz.append(point.xyz)
        for e in elements:
            observations[e.image_id].append((e.point2D_idx, length))
            parent[root(e.image_id)] = root(elements[0].image_id)
        if point_index % step == 0 and length >= 2:
            rays = np.array([point.xyz - cameras[e.image_id]["C"] for e in elements])
            rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-15)
            angle_sample.append(np.rad2deg(np.arccos(np.clip((rays @ rays.T).min(), -1, 1))))
    per_image = []
    for i, rows in observations.items():
        if i not in cameras:
            continue
        a = np.asarray(rows, dtype=int).reshape(-1, 2)
        xy = keypoints[i][a[:, 0]]
        w, h = cameras[i]["size"]
        cells = np.floor(xy / [w, h] * 8).astype(int).clip(0, 7)
        keys = cells[:, 1] * 8 + cells[:, 0]
        grid = np.bincount(keys, minlength=64)
        grid_long = np.bincount(keys[a[:, 1] >= 3], minlength=64)
        per_image.append({
            "image_id": i, "name": names[i], "observations": len(a),
            "observations_ge3": int((a[:, 1] >= 3).sum()),
            "sift_observations": int((a[:, 0] < sift_counts[i]).sum()),
            "grid_coverage": float((grid >= 2).mean()),
            "grid_coverage_ge3": float((grid_long >= 2).mean()),
            "grid_counts": grid.tolist(),
        })
    summary = {key: distribution([row[key] for row in per_image]) for key in
               ["observations", "observations_ge3", "grid_coverage", "grid_coverage_ge3"]}
    hist = dict(zip(*np.unique(lengths, return_counts=True)))
    component_roots, component_sizes = np.unique([root(i) for i in names], return_counts=True)
    return {
        "registered_images": len(cameras), "points": len(lengths),
        "observations": int(sum(lengths)), "track_length": distribution(lengths),
        "track_length_histogram": {int(k): int(v) for k, v in hist.items()},
        "max_triangulation_angle_sample_deg": distribution(angle_sample),
        "per_image_summary": summary, "per_image": per_image,
        "track_graph_components": len(component_roots),
        "track_graph_component_sizes": sorted(component_sizes.tolist(), reverse=True),
        "zero_observation_images": [row["name"] for row in per_image if not row["observations"]],
        "under_64_observation_images": [row["name"] for row in per_image if row["observations"] < 64],
    }, np.asarray(point_xyz)


def compare_cameras(reference, current):
    ids = sorted(reference.keys() & current.keys())
    x = np.array([current[i]["C"] for i in ids])
    y = np.array([reference[i]["C"] for i in ids])
    xc, yc = x - x.mean(0), y - y.mean(0)
    u, singular, vt = np.linalg.svd(yc.T @ xc / len(x))
    signs = np.array([1., 1., np.linalg.det(u @ vt)])
    rotation = (u * signs) @ vt
    scale = np.dot(singular, signs) / np.mean(np.sum(xc ** 2, axis=1))
    aligned = scale * xc @ rotation.T + y.mean(0)
    translation = np.linalg.norm(aligned - y, axis=1)
    scene_radius = np.quantile(np.linalg.norm(yc, axis=1), .9)
    angular = []
    focal = []
    for i in ids:
        delta = current[i]["R"] @ rotation.T @ reference[i]["R"].T
        angular.append(np.rad2deg(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
        focal.append(100 * (current[i]["K"][0, 0] / reference[i]["K"][0, 0] - 1))
    return {"similarity_scale": float(scale), "reference_p90_camera_radius": float(scene_radius),
            "aligned_center_shift": distribution(translation),
            "aligned_center_shift_percent_radius": distribution(100 * translation / scene_radius),
            "aligned_rotation_shift_deg": distribution(angular),
            "focal_change_percent": distribution(focal),
            "per_image": [{"image_id": i, "center_shift": float(d), "rotation_deg": float(a)}
                          for i, d, a in zip(ids, translation, angular)]}


def plot_diagnostics(results, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    colors = {"baseline": "#475569", "no_two_view": "#d97706", "all_sources": "#0284c7"}
    for variant, result in results["runs"].items():
        rows = sorted(result["coverage"]["per_image"], key=lambda r: r["image_id"])
        x = [r["image_id"] - 1 for r in rows]
        axes[0, 0].plot(x, [r["observations"] for r in rows], color=colors[variant],
                        label=variant, lw=.8, alpha=.8)
        axes[0, 1].plot(x, [r["grid_coverage_ge3"] for r in rows], color=colors[variant],
                        label=variant, lw=.8, alpha=.8)
        epi = result["fixed_match_epipolar_deg"]["per_image_by_source"]["sift"]
        ids = sorted(epi, key=int)
        axes[1, 0].plot([int(i) - 1 for i in ids], [epi[i].get("p90", np.nan) for i in ids],
                        color=colors[variant], label=variant, lw=.8, alpha=.8)
        if variant != "baseline":
            poses = result["camera_change_from_baseline"]["per_image"]
            axes[1, 1].plot([r["image_id"] - 1 for r in poses], [r["rotation_deg"] for r in poses],
                            color=colors[variant], label=variant, lw=.8)
    titles = ["Final observations", "8x8 coverage: tracks with >=3 views",
              "Fixed SIFT matches: per-image epipolar P90 (deg)",
              "Aligned rotation change from baseline (deg)"]
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("Image index")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle("789 images: track-selection diagnostics (no independent ground truth)")
    fig.savefig(output / "quality_diagnostics.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    cli = parser.parse_args()
    names, keypoints, sift_counts, pairs, merged_audit = load_database(cli.source_run)
    coarse = pycolmap.Reconstruction(str(cli.source_run / "coarse"))
    initial = camera_arrays(coarse, names)
    valid_pairs = {(i, j) for i, j, _, _ in pairs
                   if np.linalg.norm(initial[i]["C"] - initial[j]["C"]) > 1e-6}
    results = {"method": __doc__, "sample_seed": 20260908, "max_matches_per_pair": 64,
               "sampled_source_pairs": len(pairs), "nondegenerate_unique_pairs": len(valid_pairs),
               "merged_database_audit": merged_audit,
               "sampled_matches": sum(len(m) for _, _, m, _ in pairs), "runs": {}}
    reference = None
    for variant in ["baseline", "no_two_view", "all_sources"]:
        start = time.perf_counter()
        folder = cli.experiment / variant
        reconstruction = pycolmap.Reconstruction(str(folder / "refined_gluemap_aba"))
        cameras = camera_arrays(reconstruction, names)
        if reference is None:
            reference = cameras
        coverage, xyz = model_coverage(reconstruction, cameras, names, keypoints, sift_counts)
        row = {"coverage": coverage,
               "fixed_match_epipolar_deg": epipolar_errors(cameras, keypoints, pairs, valid_pairs),
               "camera_change_from_baseline": compare_cameras(reference, cameras)}
        results["runs"][variant] = row
        np.savez_compressed(folder / "quality_geometry.npz", xyz=xyz,
                            centers=np.array([cameras[i]["C"] for i in sorted(cameras)]))
        (folder / "quality_metrics.json").write_text(json.dumps(row, indent=2))
        print(f"Evaluated {variant}: {time.perf_counter() - start:.1f}s", flush=True)
        del reconstruction
    (cli.experiment / "quality_comparison.json").write_text(json.dumps(results, indent=2))
    plot_diagnostics(results, cli.experiment)


if __name__ == "__main__":
    main()
