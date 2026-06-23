#!/usr/bin/env python3
"""
Correct feed-forward depth to post-BA poses, then flag the untrusted frames.

One pass that merges:
  1. apply_per_frame_scale.py  -> per-frame depth correction (scalar by default)
     using the refined sparse anchors. Writes the corrected depth (3 formats).
  2. find_layering_frames_v2.py -> iterative adaptive-threshold detection of the
     frames that STILL cause layering after correction. Writes layering_frames.txt.

The detector runs on the just-corrected depth held in memory (no disk round-trip).

Inputs
------
--refined : post-BA COLMAP model dir (cameras/images/points3D) -- anchors + poses
--coarse  : pre-BA COLMAP model dir (optional) -- global-scale cross-check
--depth   : original FF depth dir (uses depth_npy/, falls back to depth_u16/)
--output  : output dir

Outputs (under --output)
------------------------
  depth_scaled/depth_npy|depth_u16|depth_vis/   corrected depth (3 formats)
  layering_frames.txt                            untrusted frames (tsdf --exclude_frames)
  per_frame_scale.csv                            per-frame correction model
  per_frame_outliers.csv                         per-frame layering score
  summary.json                                   correction + detection summary

Reuses leaf functions from apply_per_frame_scale.py and find_layering_frames_v2.py.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def camera_center(qvec, tvec):
    R = qvec2rotmat(qvec)
    return -R.T @ tvec


def umeyama_sim3(src, dst):
    """Best similarity (scale s, rotation R, translation t) mapping src->dst.
    Returns s, R, t, and per-point residual (meters in dst space)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    sc = src - mu_s
    dc = dst - mu_d
    cov = (dc.T @ sc) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / src.shape[0]
    s = float(np.trace(np.diag(D) @ S) / var_s)
    t = mu_d - s * R @ mu_s
    mapped = (s * (R @ src.T).T) + t
    resid = np.linalg.norm(mapped - dst, axis=1)
    return s, R, t, resid

def load_ff_depth(depth_dir, stem):
    """Load feed-forward depth (meters, float32) for a frame stem.
    Prefers depth_npy/<stem>.npy, falls back to depth_u16/<stem>.png (/1000)."""
    npy = Path(depth_dir) / "depth_npy" / f"{stem}.npy"
    if npy.exists():
        return np.load(npy).astype(np.float32)
    png = Path(depth_dir) / "depth_u16" / f"{stem}.png"
    if png.exists():
        from PIL import Image
        return np.asarray(Image.open(png), dtype=np.float32) / 1000.0
    return None

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from correct_depth_scale import (  # noqa: E402
    apply_model,
    collect_frame_anchors,
    fit_frame_model,
    save_depth_three_formats,
)
from diagnose_depth_pose_consistency import (  # noqa: E402
    read_colmap_model,
)
from correct_find_layer_depth import (  # noqa: E402
    add_detector_args,
    build_detector_cache,
    build_point_obs,
    detector_params,
    iterative_detect,
    write_detection_outputs,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--refined", required=True,
                    help="Post-BA COLMAP model dir (cameras/images/points3D)")
    ap.add_argument("--coarse", default="",
                    help="Pre-BA COLMAP model dir (optional): global-scale cross-check")
    ap.add_argument("--depth", required=True, help="Original FF depth dir")
    ap.add_argument("--output", required=True, help="Output dir")
    # correction params (mirror apply_per_frame_scale.py)
    ap.add_argument("--mode", choices=["scalar", "affine", "inv_affine"],
                    default="scalar")
    ap.add_argument("--min_anchors", type=int, default=8)
    ap.add_argument("--ratio_clip", type=float, default=4.0)
    ap.add_argument("--clamp", type=float, default=1.5,
                    help="Frames with scalar s_i outside [global/clamp, global*clamp] "
                         "fall back to global scale")
    ap.add_argument("--min_depth_span", type=float, default=0.30,
                    help="affine/inv_affine ill-posed below this -> per-frame scalar")
    # detector params (--depth_min/--depth_max are shared by both stages)
    add_detector_args(ap)
    args = ap.parse_args()

    out = Path(args.output)
    depth_out = out / "depth_scaled"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[C&F] Reading refined model: {args.refined}")
    cameras, images, points = read_colmap_model(args.refined)
    if not points:
        raise RuntimeError("Refined model has no points3D.")
    print(f"[C&F] images={len(images)}, points3D={len(points)}, mode={args.mode}")

    # ----- Pass 1: fit per-frame correction model ----- #
    fitted = {}
    for im in images.values():
        stem = Path(im["name"]).stem
        depth = load_ff_depth(args.depth, stem)
        if depth is None:
            fitted[stem] = None
            continue
        d, z = collect_frame_anchors(
            im, cameras[im["camera_id"]], depth, points,
            args.depth_min, args.depth_max,
        )
        model, s, n, status, scv, mcv = fit_frame_model(
            d, z, args.mode, args.ratio_clip, args.min_anchors, args.min_depth_span
        )
        fitted[stem] = dict(model=model, scalar_s=s, n=n, status=status)

    reliable = [f["scalar_s"] for f in fitted.values()
                if f is not None and f["scalar_s"] is not None]
    global_scale = float(np.median(reliable)) if reliable else None

    sim3_scale = None
    if args.coarse:
        try:
            _, c_images, _ = read_colmap_model(args.coarse)
            ref_c = {Path(im["name"]).stem: camera_center(im["qvec"], im["tvec"])
                     for im in images.values()}
            coa_c = {Path(im["name"]).stem: camera_center(im["qvec"], im["tvec"])
                     for im in c_images.values()}
            common = sorted(set(ref_c) & set(coa_c))
            if len(common) >= 3:
                sim3_scale = float(umeyama_sim3(
                    np.stack([coa_c[n] for n in common]),
                    np.stack([ref_c[n] for n in common]))[0])
        except Exception as e:  # noqa: BLE001
            print(f"[C&F] Sim3 cross-check skipped: {e}")

    if global_scale is None:
        global_scale = sim3_scale if sim3_scale is not None else 1.0
    print(f"[C&F] global scale = {global_scale:.5f}"
          + (f"  | Sim3 = {sim3_scale:.5f}" if sim3_scale is not None else ""))
    lo, hi = global_scale / args.clamp, global_scale * args.clamp
    global_model = ("scalar", global_scale)

    # ----- Pass 2: apply, write corrected depth, keep in memory ----- #
    corrected = {}   # stem -> corrected depth (for the detector)
    rows = []
    counts = {"per_frame": 0, "affine_illposed_scalar": 0,
              "fallback_few_anchors": 0, "fallback_clamp": 0, "no_depth": 0}
    for im in images.values():
        stem = Path(im["name"]).stem
        f = fitted[stem]
        depth = load_ff_depth(args.depth, stem)
        if depth is None or f is None:
            counts["no_depth"] += 1
            rows.append(dict(stem=stem, n_anchors=0, status="no_depth",
                             scalar_s="", model=""))
            continue

        if f["model"] is None:
            model, status = global_model, "fallback_few_anchors"
            counts["fallback_few_anchors"] += 1
        elif not (lo <= f["scalar_s"] <= hi):
            model, status = global_model, "fallback_clamp"
            counts["fallback_clamp"] += 1
        else:
            model, status = f["model"], f["status"]
            counts["affine_illposed_scalar" if status == "affine_illposed_scalar"
                   else "per_frame"] += 1

        depth_c = apply_model(depth, model)
        corrected[stem] = depth_c
        save_depth_three_formats(depth_c, stem, depth_out)
        rows.append(dict(
            stem=stem, n_anchors=f["n"], status=status,
            scalar_s=("" if f["scalar_s"] is None else round(float(f["scalar_s"]), 6)),
            model="|".join(str(round(float(x), 6)) for x in model[1:])))

    rows.sort(key=lambda r: r["stem"])
    with open(out / "per_frame_scale.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["stem", "n_anchors", "status",
                                           "scalar_s", "model"])
        w.writeheader()
        w.writerows(rows)
    print(f"[C&F] corrected depth -> {depth_out}/depth_npy|depth_u16|depth_vis  "
          f"(per_frame={counts['per_frame']}, "
          f"fallback={counts['fallback_few_anchors']+counts['fallback_clamp']})")

    # ----- Detect frames that still layer (on corrected depth in memory) ----- #
    cache = build_detector_cache(cameras, images, lambda stem: corrected.get(stem))
    n_frames = len(cache)
    point_obs = build_point_obs(cache, images, points, args.min_obs,
                                args.depth_min, args.depth_max)
    print(f"[C&F] detector: frames={n_frames}, points>= {args.min_obs}obs={len(point_obs)}")

    flagged, record, rounds_info, stop_reason, cap = iterative_detect(
        point_obs, cache, n_frames, log_prefix="[C&F]", **detector_params(args)
    )
    det_summary, flagged_sorted, flagged_stems = write_detection_outputs(
        out, cache, flagged, record, rounds_info, stop_reason, cap,
        detector_params(args),
    )

    # ----- Combined summary ----- #
    summary = {
        "inputs": {"refined": args.refined, "coarse": args.coarse or None,
                   "depth": args.depth, "output": str(out)},
        "correction": {
            "mode": args.mode,
            "global_scale": global_scale,
            "sim3_scale_coarse_to_refined": sim3_scale,
            "clamp_range": [lo, hi],
            "counts": counts,
        },
        "detection": det_summary,
        "untrusted_frames": flagged_stems,
    }
    with open(out / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, default=str)

    print(f"\n[C&F] DONE")
    print(f"  corrected depth   : {depth_out}")
    print(f"  untrusted frames  : {len(flagged_stems)}/{n_frames} "
          f"(stop={stop_reason}) -> {out/'layering_frames.txt'}")
    for i in flagged_sorted:
        rec = record[i]
        print(f"     r{rec['flagged_round']}  {rec['stem']}  "
              f"frac={rec['outlier_fraction']:.3f}")


if __name__ == "__main__":
    main()
