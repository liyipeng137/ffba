#!/usr/bin/env python3
"""
Rank/flag frames that cause layering after per-frame depth correction (v2).

vs find_layering_frames.py
--------------------------
Same anchor-based core (per shared 3D point, find the frame whose corrected-depth
unprojection is the outlier vs the consensus of the OTHER observing frames), but
the selection is no longer a fixed `--top_k`. Instead:

  * Iterative removal: each round flags the statistically anomalous frames, then
    REMOVES them from the consensus and re-scores. This matters because bad
    frames pollute the median consensus -- after removing the worst, the next
    tier becomes visible. Rounds run until convergence.
  * Adaptive threshold: a frame is flagged when its outlier_fraction exceeds
    `median + k_sigma * robust_sigma` of the *remaining* frames' fractions. This
    auto-adapts to each dataset's baseline noise (no fixed count).
  * Absolute floor: AND outlier_fraction > `--abs_floor`, so we only flag frames
    that are the outlier on a *meaningful* fraction of points (avoids over-
    flagging borderline frames -> avoids over-skipping / holes).
  * Safety cap: never remove more than `--max_frac` of all frames.
  * Per-round cap: `--per_round_cap` limits flags per round, spreading removal
    over more rounds (fewer in round 1) so the consensus re-cleans between
    removals (can reduce false positives).

Limitations (unchanged): anchor-based, so it sees layering on textured regions
only. The mesh skip-and-check / leave-one-out render remains the ground truth and
the way to catch non-anchor cases and decide whole-frame-skip vs per-pixel mask.

Run on the CORRECTED depth (e.g. pred_depth_scaled).

The detection core (`build_detector_cache`, `build_point_obs`, `iterative_detect`)
is importable so `correct_and_flag_depth.py` can reuse it on in-memory depth.

Outputs (under --output):
  per_frame_outliers.csv   stem, image_id, flagged_round, observed,
                           outlier_count, outlier_fraction
  layering_frames.txt      all flagged stems (feed tsdf --exclude_frames)
  summary.json             per-round diagnostics + stop reason
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from diagnose_depth_pose_consistency import (  # noqa: E402
    intrinsics_matrix,
    load_ff_depth,
    qvec2rotmat,
    read_colmap_model,
)


# --------------------------------------------------------------------------- #
# Detection core (importable)
# --------------------------------------------------------------------------- #

def build_detector_cache(cameras, images, depth_for_stem):
    """Per-image cache for the detector.

    depth_for_stem(stem) -> np.ndarray | None. Lets the caller supply depth from
    disk (find_layering) or from memory (correct_and_flag).
    """
    cache = {}
    for img_id, im in images.items():
        cam = cameras[im["camera_id"]]
        stem = Path(im["name"]).stem
        depth = depth_for_stem(stem)
        if depth is None:
            continue
        Hd, Wd = depth.shape[:2]
        cache[img_id] = dict(
            stem=stem, Kinv=np.linalg.inv(intrinsics_matrix(cam)),
            R=qvec2rotmat(im["qvec"]), t=im["tvec"],
            depth=depth, Hd=Hd, Wd=Wd,
            sx=Wd / cam["width"], sy=Hd / cam["height"],
        )
    return cache


def build_point_obs(cache, images, points, min_obs, depth_min, depth_max):
    """Precompute per-point world unprojections once; rounds just re-filter."""
    point_obs = []
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
            if not (depth_min < d < depth_max):
                continue
            ray = c["Kinv"] @ np.array([float(xy[0]), float(xy[1]), 1.0])
            ids.append(int(img_id))
            world.append(c["R"].T @ (d * ray - c["t"]))
            depths.append(d)
        if len(ids) >= min_obs:
            point_obs.append((np.asarray(ids, dtype=np.int64),
                              np.asarray(world, dtype=np.float64),
                              np.asarray(depths, dtype=np.float64)))
    return point_obs


def _score_round(point_obs, active_flag, min_obs, outlier_rel, sep_ratio):
    """One scoring pass over all points using only currently-active frames."""
    observed = defaultdict(int)
    outlier = defaultdict(int)
    n_points_used = 0
    for ids, world, depths in point_obs:
        mask = active_flag[ids]
        if int(mask.sum()) < min_obs:
            continue
        ids_a = ids[mask]
        world_a = world[mask]
        depths_a = depths[mask]
        n_points_used += 1

        consensus = np.median(world_a, axis=0)
        rel = np.linalg.norm(world_a - consensus, axis=1) / max(
            float(depths_a.mean()), 1e-6
        )
        for iid in ids_a.tolist():
            observed[iid] += 1

        order = np.argsort(rel)[::-1]
        worst_rel = float(rel[order[0]])
        second_rel = float(rel[order[1]]) if order.size > 1 else 0.0
        if worst_rel >= outlier_rel and worst_rel >= sep_ratio * max(second_rel, 1e-9):
            outlier[int(ids_a[order[0]])] += 1
    return observed, outlier, n_points_used


def iterative_detect(point_obs, cache, n_frames, *, min_obs, outlier_rel,
                     sep_ratio, min_observed, k_sigma, abs_floor, max_frac,
                     per_round_cap, max_rounds, log_prefix="[LAYERv2]"):
    """Iterative adaptive-threshold outlier-frame flagging.

    Returns (flagged:set, record:dict, rounds_info:list, stop_reason:str, cap:int).
    """
    max_id = max(cache) if cache else 0
    flagged = set()
    record = {iid: dict(stem=c["stem"], flagged_round="", observed=0,
                        outlier_count=0, outlier_fraction=0.0)
              for iid, c in cache.items()}
    rounds_info = []
    stop_reason = "max_rounds"
    cap = int(max_frac * n_frames)

    for r in range(1, max_rounds + 1):
        active_flag = np.zeros(max_id + 1, dtype=bool)
        for iid in cache:
            if iid not in flagged:
                active_flag[iid] = True

        observed, outlier, n_pts = _score_round(
            point_obs, active_flag, min_obs, outlier_rel, sep_ratio
        )

        fracs = {}
        for iid in cache:
            if iid in flagged:
                continue
            obs = observed.get(iid, 0)
            oc = outlier.get(iid, 0)
            frac = (oc / obs) if obs > 0 else 0.0
            record[iid].update(observed=obs, outlier_count=oc,
                               outlier_fraction=round(frac, 5))
            if obs >= min_observed:
                fracs[iid] = frac

        if not fracs:
            stop_reason = "no_scored_frames"
            rounds_info.append(dict(round=r, points_used=n_pts, scored_frames=0,
                                    median_frac=None, robust_sigma=None,
                                    threshold=None, newly=[]))
            break

        vals = np.array(list(fracs.values()))
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        robust_sigma = 1.4826 * mad
        # Too few frames -> distribution stats unreliable; rely on abs_floor.
        thr = (abs_floor if vals.size < 8
               else max(med + k_sigma * robust_sigma, abs_floor))

        newly = sorted((iid for iid, f in fracs.items() if f > thr),
                       key=lambda i: fracs[i], reverse=True)
        if per_round_cap > 0:
            newly = newly[:per_round_cap]

        rounds_info.append(dict(
            round=r, points_used=n_pts, scored_frames=int(vals.size),
            median_frac=round(med, 5), robust_sigma=round(robust_sigma, 5),
            threshold=round(thr, 5),
            newly=[dict(stem=cache[i]["stem"], outlier_fraction=round(fracs[i], 5),
                        outlier_count=record[i]["outlier_count"],
                        observed=record[i]["observed"]) for i in newly],
        ))
        print(f"{log_prefix} round {r}: points={n_pts}, scored={vals.size}, "
              f"median={med:.4f}, sigma={robust_sigma:.4f}, thr={thr:.4f}, "
              f"newly={len(newly)}")

        if not newly:
            stop_reason = "converged_no_new"
            break
        if len(flagged) + len(newly) > cap:
            allowed = cap - len(flagged)
            if allowed <= 0:
                stop_reason = "safety_cap_max_frac"
                break
            newly = newly[:allowed]
            stop_reason = "safety_cap_max_frac"
            for iid in newly:
                record[iid]["flagged_round"] = r
                flagged.add(iid)
            break
        for iid in newly:
            record[iid]["flagged_round"] = r
            flagged.add(iid)

    return flagged, record, rounds_info, stop_reason, cap


def write_detection_outputs(out, cache, flagged, record, rounds_info,
                            stop_reason, cap, params):
    """Write per_frame_outliers.csv, layering_frames.txt, summary.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_frame_outliers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "image_id", "flagged_round",
                                          "observed", "outlier_count",
                                          "outlier_fraction"])
        w.writeheader()
        for iid, rec in record.items():
            w.writerow({"image_id": iid, **{k: rec[k] for k in
                        ["stem", "flagged_round", "observed",
                         "outlier_count", "outlier_fraction"]}})

    flagged_sorted = sorted(
        flagged,
        key=lambda i: (record[i]["flagged_round"], -record[i]["outlier_fraction"]),
    )
    flagged_stems = [cache[i]["stem"] for i in flagged_sorted]
    with open(out / "layering_frames.txt", "w") as f:
        f.write("\n".join(flagged_stems) + ("\n" if flagged_stems else ""))

    summary = dict(
        params=params, n_frames=len(cache),
        num_flagged=len(flagged), max_allowed=cap, stop_reason=stop_reason,
        flagged_frames=[dict(stem=cache[i]["stem"],
                             flagged_round=record[i]["flagged_round"],
                             outlier_fraction=record[i]["outlier_fraction"],
                             outlier_count=record[i]["outlier_count"],
                             observed=record[i]["observed"])
                        for i in flagged_sorted],
        rounds=rounds_info,
    )
    return summary, flagged_sorted, flagged_stems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def add_detector_args(ap):
    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=50.0)
    ap.add_argument("--min_obs", type=int, default=3,
                    help="Min observing (active) frames to attribute an outlier")
    ap.add_argument("--outlier_rel", type=float, default=0.03,
                    help="Outlier if its world offset / depth exceeds this")
    ap.add_argument("--sep_ratio", type=float, default=2.0,
                    help="Outlier must be this much farther than the 2nd-worst frame")
    ap.add_argument("--min_observed", type=int, default=50,
                    help="Min consensus-points for a frame to be scored/ranked")
    ap.add_argument("--k_sigma", type=float, default=3.0,
                    help="Flag if outlier_fraction > median + k_sigma*robust_sigma")
    ap.add_argument("--abs_floor", type=float, default=0.07,
                    help="AND outlier_fraction must exceed this absolute floor")
    ap.add_argument("--max_frac", type=float, default=0.15,
                    help="Safety cap: never flag more than this fraction of frames")
    ap.add_argument("--per_round_cap", type=int, default=10,
                    help="Max frames to flag per round (0 = unlimited)")
    ap.add_argument("--max_rounds", type=int, default=20)


def detector_params(args):
    return {k: getattr(args, k) for k in
            ["min_obs", "outlier_rel", "sep_ratio", "min_observed", "k_sigma",
             "abs_floor", "max_frac", "per_round_cap", "max_rounds"]}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--refined", required=True, help="Post-BA COLMAP model dir")
    ap.add_argument("--depth", required=True,
                    help="Corrected depth dir to test (e.g. pred_depth_scaled)")
    ap.add_argument("--output", default="layering_frames_v2", help="Output dir")
    add_detector_args(ap)
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cameras, images, points = read_colmap_model(args.refined)
    if not points:
        raise RuntimeError("Refined model has no points3D.")

    cache = build_detector_cache(
        cameras, images, lambda stem: load_ff_depth(args.depth, stem)
    )
    n_frames = len(cache)
    print(f"[LAYERv2] images with depth={n_frames}, points3D={len(points)}")

    point_obs = build_point_obs(cache, images, points, args.min_obs,
                                args.depth_min, args.depth_max)
    print(f"[LAYERv2] points with >= {args.min_obs} obs = {len(point_obs)}")

    flagged, record, rounds_info, stop_reason, cap = iterative_detect(
        point_obs, cache, n_frames, **detector_params(args)
    )

    summary, flagged_sorted, flagged_stems = write_detection_outputs(
        out, cache, flagged, record, rounds_info, stop_reason, cap,
        detector_params(args),
    )
    summary["inputs"] = {"refined": args.refined, "depth": args.depth}
    summary["points_used"] = len(point_obs)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[LAYERv2] stop={stop_reason}, flagged {len(flagged)}/{n_frames} "
          f"(cap {cap}) over {len(rounds_info)} rounds")
    for i in flagged_sorted:
        rec = record[i]
        print(f"   r{rec['flagged_round']}  {rec['stem']}  "
              f"frac={rec['outlier_fraction']:.3f}  "
              f"{rec['outlier_count']}/{rec['observed']}")
    print(f"[LAYERv2] wrote {len(flagged_stems)} stems -> {out/'layering_frames.txt'}")


if __name__ == "__main__":
    main()
