#!/usr/bin/env python3
"""
Rank frames that still cause layering after per-frame depth correction.

Idea
----
Layering = several frames place the same surface at different world positions.
For every refined 3D point observed by >= --min_obs frames, unproject each
observing frame's (corrected) FF depth at the observation pixel into the world,
take the median as the consensus, and find the *outlier* frame (the one whose
unprojection is far from consensus and clearly separated from the rest). Tally
per frame how often it is the outlier. Frames with a high outlier fraction are
the ones that most disagree with their neighbors -> candidates to skip in TSDF.

This is anchor-based, so it sees layering on textured (BA-point) regions. It is
blind to textureless / non-anchor regions -- but those are exactly the ones a
"skip the frame" fix cannot cleanly repair anyway, and the TSDF skip-and-check
loop reveals anything this misses (layering that persists after skipping all
flagged frames is non-anchor / depth-quality, not a removable-frame problem).

Run this on the CORRECTED depth (e.g. pred_depth_scaled) to find frames that
*still* layer after correction.

Outputs (under --output):
  per_frame_outliers.csv   stem, observed, outlier_count, outlier_fraction, ...
  layering_frames.txt      top candidate stems, one per line (feed tsdf
                           --exclude_frames)
  summary.json

Reuses the COLMAP / depth IO from diagnose_depth_pose_consistency.py.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from diagnose_depth_pose_consistency import (  # noqa: E402
    camera_center,
    intrinsics_matrix,
    load_ff_depth,
    qvec2rotmat,
    read_colmap_model,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--refined", required=True, help="Post-BA COLMAP model dir")
    ap.add_argument("--depth", required=True,
                    help="Corrected depth dir to test (e.g. pred_depth_scaled)")
    ap.add_argument("--output", default="layering_frames", help="Output dir")
    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=50.0)
    ap.add_argument("--min_obs", type=int, default=3,
                    help="Min observing frames for a point to attribute an outlier")
    ap.add_argument("--outlier_rel", type=float, default=0.03,
                    help="Outlier if its world offset / depth exceeds this")
    ap.add_argument("--sep_ratio", type=float, default=2.0,
                    help="Outlier must be this much farther than the 2nd-worst frame")
    ap.add_argument("--min_observed", type=int, default=50,
                    help="Min consensus-points a frame must appear in to be ranked")
    ap.add_argument("--top_k", type=int, default=20,
                    help="How many top frames to write to layering_frames.txt")
    ap.add_argument("--min_fraction", type=float, default=0.0,
                    help="Also require outlier_fraction >= this for layering_frames.txt")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cameras, images, points = read_colmap_model(args.refined)
    if not points:
        raise RuntimeError("Refined model has no points3D.")

    # Per-image cache: K^-1, R, t, depth, low/high scale.
    cache = {}
    for img_id, im in images.items():
        cam = cameras[im["camera_id"]]
        stem = Path(im["name"]).stem
        depth = load_ff_depth(args.depth, stem)
        if depth is None:
            continue
        Hd, Wd = depth.shape[:2]
        cache[img_id] = dict(
            stem=stem, Kinv=np.linalg.inv(intrinsics_matrix(cam)),
            R=qvec2rotmat(im["qvec"]), t=im["tvec"],
            depth=depth, Hd=Hd, Wd=Wd,
            sx=Wd / cam["width"], sy=Hd / cam["height"],
        )
    print(f"[LAYER] images with depth={len(cache)}, points3D={len(points)}")

    observed = {iid: 0 for iid in cache}
    outlier = {iid: 0 for iid in cache}
    n_points_used = 0

    for p in points.values():
        ids, world, depths = [], [], []
        for img_id, p2d in zip(p["track_image_ids"].tolist(),
                               p["track_p2d_idx"].tolist(), strict=False):
            c = cache.get(int(img_id))
            if c is None:
                continue
            xy = images[int(img_id)]["xys"][int(p2d)]
            u = int(round(float(xy[0]) * c["sx"]))
            v = int(round(float(xy[1]) * c["sy"]))
            if not (0 <= u < c["Wd"] and 0 <= v < c["Hd"]):
                continue
            d = float(c["depth"][v, u])
            if not (args.depth_min < d < args.depth_max):
                continue
            ray = c["Kinv"] @ np.array([float(xy[0]), float(xy[1]), 1.0])
            X_world = c["R"].T @ (d * ray - c["t"])
            ids.append(int(img_id))
            world.append(X_world)
            depths.append(d)

        if len(ids) < args.min_obs:
            continue
        n_points_used += 1
        world = np.stack(world, 0)
        consensus = np.median(world, axis=0)
        rel = np.linalg.norm(world - consensus, axis=1) / max(np.mean(depths), 1e-6)

        for iid in ids:
            observed[iid] += 1

        order = np.argsort(rel)[::-1]
        worst, worst_rel = order[0], rel[order[0]]
        second_rel = rel[order[1]] if len(order) > 1 else 0.0
        if worst_rel >= args.outlier_rel and \
                worst_rel >= args.sep_ratio * max(second_rel, 1e-9):
            outlier[ids[worst]] += 1

    rows = []
    for iid, c in cache.items():
        obs = observed[iid]
        oc = outlier[iid]
        frac = (oc / obs) if obs > 0 else 0.0
        rows.append(dict(stem=c["stem"], image_id=iid, observed=obs,
                         outlier_count=oc, outlier_fraction=round(frac, 4)))
    # Rank by outlier fraction (then count), among frames with enough support.
    ranked = sorted(
        [r for r in rows if r["observed"] >= args.min_observed],
        key=lambda r: (r["outlier_fraction"], r["outlier_count"]), reverse=True,
    )
    rows.sort(key=lambda r: r["stem"])

    with open(out / "per_frame_outliers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "image_id", "observed",
                                          "outlier_count", "outlier_fraction"])
        w.writeheader()
        w.writerows(rows)

    candidates = [r["stem"] for r in ranked
                  if r["outlier_fraction"] >= args.min_fraction][: args.top_k]
    with open(out / "layering_frames.txt", "w") as f:
        f.write("\n".join(candidates) + ("\n" if candidates else ""))

    with open(out / "summary.json", "w") as f:
        json.dump({
            "inputs": {"refined": args.refined, "depth": args.depth},
            "points_used": n_points_used,
            "params": {"min_obs": args.min_obs, "outlier_rel": args.outlier_rel,
                       "sep_ratio": args.sep_ratio, "min_observed": args.min_observed},
            "top_candidates": ranked[: args.top_k],
        }, f, indent=2, default=str)

    print(f"[LAYER] points used (>= {args.min_obs} obs) = {n_points_used}")
    print(f"[LAYER] top {min(args.top_k, len(ranked))} layering candidates "
          f"(stem  outlier_frac  outlier/observed):")
    for r in ranked[: args.top_k]:
        print(f"   {r['stem']}   {r['outlier_fraction']:.3f}   "
              f"{r['outlier_count']}/{r['observed']}")
    print(f"[LAYER] wrote {len(candidates)} stems -> {out / 'layering_frames.txt'}")


if __name__ == "__main__":
    main()
