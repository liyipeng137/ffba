#!/usr/bin/env python3
"""
Apply a per-frame depth correction so feed-forward depth matches post-BA poses.

Why
---
After bundle adjustment the poses are updated but the dense FF depth is not, so
depth and pose disagree per frame. `diagnose_depth_pose_consistency.py` showed
the disagreement is per-frame (tight within a frame *spatially*, but frame
medians differ). This script corrects it from the refined sparse anchors.

Models (--mode)
---------------
For frame i, anchors are pairs (d = FF depth at a pixel, z = BA point depth in
that camera):
  scalar      : z ~ s*d            -> corrected = s_i * d
  affine      : z ~ a*d + b        -> corrected = a_i*d + b_i        (depth-space)
  inv_affine  : 1/z ~ a*(1/d) + b  -> corrected = d / (a_i + b_i*d)  (disparity)

`scalar` only removes a per-frame uniform scale. If the per-frame error varies
*with depth* (a real affine/shift bias of the depth model), `scalar` mis-scales
regions away from the anchor depths and can make TSDF layering worse there;
`affine` / `inv_affine` absorb that depth dependence. A printed
affine-vs-scalar residual comparison tells you whether the shift actually helps.

Where the scale comes from
--------------------------
Anchor-based -- it needs the refined `points3D`. Camera poses alone cannot give
a per-frame depth scale, so `--coarse` is only a global-scale cross-check /
last-resort fallback.

Robustness / guards
-------------------
- < --min_anchors usable anchors            -> fall back to the global scale.
- per-frame scalar s_i outside
  [global/clamp, global*clamp]              -> degenerate frame, fall back.
- affine/inv_affine on a frame whose anchors
  don't span enough depth (--min_depth_span)-> ill-posed shift, fall back to the
                                               per-frame scalar.
- corrected depth <= 0 (affine extrapolation)-> marked invalid (0), not a wrong
                                               surface.

Output
------
Mirrors `MERG3R/algos/utils.py::export_prediction_depth_maps`:
    depth_npy/<stem>.npy   float32 meters
    depth_u16/<stem>.png   uint16 = clip(depth * 1000)
    depth_vis/<stem>.png   TURBO colormap (2/98 percentile)
plus per_frame_scale.csv and apply_summary.json.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from diagnose_depth_pose_consistency import (  # noqa: E402
    camera_center,
    load_ff_depth,
    qvec2rotmat,
    read_colmap_model,
    umeyama_sim3,
)


# --------------------------------------------------------------------------- #
# Anchor collection + per-frame model fitting
# --------------------------------------------------------------------------- #

def collect_frame_anchors(image, cam, depth, points, depth_min, depth_max):
    """Return (d_ff, z_ba) arrays for this frame's usable sparse anchors."""
    R = qvec2rotmat(image["qvec"])
    t = image["tvec"]
    Hd, Wd = depth.shape[:2]
    sx = Wd / cam["width"]
    sy = Hd / cam["height"]

    ds, zs = [], []
    for xy, pid in zip(image["xys"], image["point3D_ids"], strict=False):
        pid = int(pid)
        if pid < 0:
            continue
        p = points.get(pid)
        if p is None:
            continue
        z_ba = float(R[2] @ p["xyz"] + t[2])
        if not (depth_min < z_ba < depth_max):
            continue
        u = int(round(float(xy[0]) * sx))
        v = int(round(float(xy[1]) * sy))
        if not (0 <= u < Wd and 0 <= v < Hd):
            continue
        d_ff = float(depth[v, u])
        if not (depth_min < d_ff < depth_max):
            continue
        ds.append(d_ff)
        zs.append(z_ba)
    return np.asarray(ds, dtype=np.float64), np.asarray(zs, dtype=np.float64)


def _ols_line(x, y):
    """OLS y ~ a*x + b. Falls back to through-origin scale if x has no spread."""
    xm, ym = x.mean(), y.mean()
    denom = float(np.sum((x - xm) ** 2))
    if denom < 1e-12:
        a = float(ym / xm) if abs(xm) > 1e-9 else 1.0
        return a, 0.0
    a = float(np.sum((x - xm) * (y - ym)) / denom)
    b = float(ym - a * xm)
    return a, b


def _robust_line(x, y, trim_sigma=3.0, iters=2):
    """OLS line with MAD-trim refit (rejects outlier anchors)."""
    a, b = _ols_line(x, y)
    for _ in range(iters):
        resid = y - (a * x + b)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        if mad < 1e-12:
            break
        keep = np.abs(resid - med) <= trim_sigma * 1.4826 * mad
        if keep.sum() < max(8, 0.5 * x.size):
            break
        a, b = _ols_line(x[keep], y[keep])
    return a, b


def _rel_resid_cv(d, z, pred):
    """Robust CV of the relative residual (pred - z)/z."""
    rel = (pred - z) / np.clip(z, 1e-6, None)
    return float(1.4826 * np.median(np.abs(rel - np.median(rel))))


def fit_frame_model(d, z, mode, ratio_clip, min_anchors, min_depth_span):
    """Fit the requested per-frame model.

    Returns (model, scalar_s, n_used, status_suffix, scalar_cv, model_cv)
    where model is ("scalar", s) | ("affine", a, b) | ("inv_affine", a, b),
    or model=None when there are too few anchors.
    """
    if d.size < min_anchors:
        return None, None, int(d.size), "", None, None
    # ratio-trim around the scalar median (consistent with the diagnostic)
    ratio = z / d
    med0 = np.median(ratio)
    keep = (ratio > med0 / ratio_clip) & (ratio < med0 * ratio_clip)
    d, z = d[keep], z[keep]
    if d.size < min_anchors:
        return None, None, int(d.size), "", None, None

    s = float(np.median(z / d))
    scalar_cv = _rel_resid_cv(d, z, s * d)
    if mode == "scalar":
        return ("scalar", s), s, int(d.size), "scalar", scalar_cv, scalar_cv

    # affine / inv_affine need the anchors to span a depth range, else a,b are
    # not separable -> fall back to the per-frame scalar.
    span = float((np.percentile(d, 95) - np.percentile(d, 5)) / max(np.median(d), 1e-6))
    if span < min_depth_span:
        return ("scalar", s), s, int(d.size), "affine_illposed_scalar", scalar_cv, scalar_cv

    if mode == "affine":
        a, b = _robust_line(d, z)
        model_cv = _rel_resid_cv(d, z, a * d + b)
        return ("affine", a, b), s, int(d.size), "affine", scalar_cv, model_cv
    if mode == "inv_affine":
        a, b = _robust_line(1.0 / d, 1.0 / z)   # 1/z ~ a*(1/d) + b
        denom = a + b * d
        # z = d / (a + b*d)  (NOT 1/(a+b*d)); matches apply_model.
        pred = np.where(denom > 1e-6, d / np.clip(denom, 1e-6, None), z)
        model_cv = _rel_resid_cv(d, z, pred)
        return ("inv_affine", a, b), s, int(d.size), "inv_affine", scalar_cv, model_cv
    raise ValueError(f"Unknown mode {mode!r}")


def apply_model(depth, model):
    """Apply a fitted model to a dense depth map, preserving invalid pixels."""
    valid = depth > 0
    kind = model[0]
    if kind == "scalar":
        out = depth * np.float32(model[1])
    elif kind == "affine":
        a, b = model[1], model[2]
        out = (a * depth + b).astype(np.float32)
    elif kind == "inv_affine":
        a, b = model[1], model[2]
        denom = a + b * depth
        out = np.where(denom > 1e-6, depth / denom, 0.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown model kind {kind!r}")
    # Keep originally-valid, strictly-positive pixels only (drops affine
    # extrapolation that goes <=0 rather than placing a wrong surface).
    out = np.where(valid & np.isfinite(out) & (out > 0), out, 0.0)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Depth output (mirrors export_prediction_depth_maps / _save_depth_frame_pngs)
# --------------------------------------------------------------------------- #

def save_depth_three_formats(depth, stem, out_dir):
    u16_dir = out_dir / "depth_u16"
    vis_dir = out_dir / "depth_vis"
    npy_dir = out_dir / "depth_npy"
    for d in (u16_dir, vis_dir, npy_dir):
        d.mkdir(parents=True, exist_ok=True)

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32, copy=False
    )
    depth_u16 = np.clip(depth * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    base_name = stem + ".png"

    cv2.imwrite(str(u16_dir / base_name), depth_u16)
    np.save(str(npy_dir / (stem + ".npy")), depth)

    valid_mask = np.isfinite(depth) & (depth > 0)
    if np.any(valid_mask):
        d = depth[valid_mask]
        d_min = np.percentile(d, 2.0)
        d_max = np.percentile(d, 98.0)
        if d_max <= d_min:
            d_max = d_min + 1e-6
        depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
        depth_vis_u8 = (depth_norm * 255.0).astype(np.uint8)
        depth_vis_u8[~valid_mask] = 0
        depth_color = cv2.applyColorMap(depth_vis_u8, cv2.COLORMAP_TURBO)
    else:
        h, w = depth.shape[:2]
        depth_color = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(vis_dir / base_name), depth_color)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--refined", required=True,
                    help="Post-BA COLMAP model dir (cameras/images/points3D)")
    ap.add_argument("--depth", required=True, help="Original FF depth dir")
    ap.add_argument("--output", required=True, help="Output dir for corrected depth")
    ap.add_argument("--coarse", default="",
                    help="Pre-BA COLMAP model dir (optional): global-scale cross-check")
    ap.add_argument("--mode", choices=["scalar", "affine", "inv_affine"],
                    default="scalar")
    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=50.0)
    ap.add_argument("--min_anchors", type=int, default=8)
    ap.add_argument("--ratio_clip", type=float, default=4.0)
    ap.add_argument("--clamp", type=float, default=1.5,
                    help="Frames with scalar s_i outside [global/clamp, global*clamp] "
                         "fall back to global scale")
    ap.add_argument("--min_depth_span", type=float, default=0.30,
                    help="affine/inv_affine need (d95-d5)/median(d) above this, "
                         "else the shift is ill-posed and the frame uses scalar")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[APPLY] Reading refined model: {args.refined}")
    cameras, images, points = read_colmap_model(args.refined)
    if not points:
        raise RuntimeError("Refined model has no points3D.")
    print(f"[APPLY] images={len(images)}, points3D={len(points)}, "
          f"cameras={len(cameras)}, mode={args.mode}")

    # ---- Pass 1: collect anchors, fit per-frame model ---- #
    fitted = {}   # stem -> dict(model, scalar_s, n, status, scalar_cv, model_cv)
    missing_depth = []
    for im in images.values():
        cam = cameras[im["camera_id"]]
        stem = Path(im["name"]).stem
        depth = load_ff_depth(args.depth, stem)
        if depth is None:
            missing_depth.append(stem)
            fitted[stem] = None
            continue
        d, z = collect_frame_anchors(
            im, cam, depth, points, args.depth_min, args.depth_max
        )
        model, s, n, status, scv, mcv = fit_frame_model(
            d, z, args.mode, args.ratio_clip, args.min_anchors, args.min_depth_span
        )
        fitted[stem] = dict(model=model, scalar_s=s, n=n, status=status,
                            scalar_cv=scv, model_cv=mcv)
    if missing_depth:
        print(f"[APPLY] WARNING: {len(missing_depth)} frames have no depth "
              f"(e.g. {missing_depth[:3]})")

    reliable_s = [f["scalar_s"] for f in fitted.values()
                  if f is not None and f["scalar_s"] is not None]
    global_scale = float(np.median(reliable_s)) if reliable_s else None

    # ---- Sim3 coarse->refined (cross-check + fallback) ---- #
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
            print(f"[APPLY] Sim3 cross-check skipped: {e}")

    if global_scale is None:
        global_scale = sim3_scale if sim3_scale is not None else 1.0
        print(f"[APPLY] WARNING: no reliable anchors; global scale={global_scale:.5f}")
    print(f"[APPLY] global scale (median per-frame) = {global_scale:.5f}"
          + (f"  | Sim3 coarse->refined = {sim3_scale:.5f}"
             if sim3_scale is not None else ""))

    lo, hi = global_scale / args.clamp, global_scale * args.clamp
    global_model = ("scalar", global_scale)

    # ---- Pass 2: decide final model, apply, write ---- #
    rows = []
    counts = {"per_frame": 0, "affine_illposed_scalar": 0, "fallback_few_anchors": 0,
              "fallback_clamp": 0, "no_depth": 0}
    scalar_cvs, model_cvs, improved = [], [], 0
    for im in images.values():
        stem = Path(im["name"]).stem
        f = fitted[stem]
        depth = load_ff_depth(args.depth, stem)
        if depth is None or f is None:
            counts["no_depth"] += 1
            rows.append(dict(stem=stem, n_anchors=0, status="no_depth",
                             scalar_s="", model="", scalar_cv="", model_cv=""))
            continue

        if f["model"] is None:
            model, status = global_model, "fallback_few_anchors"
            counts["fallback_few_anchors"] += 1
        elif not (lo <= f["scalar_s"] <= hi):
            model, status = global_model, "fallback_clamp"
            counts["fallback_clamp"] += 1
        else:
            model, status = f["model"], f["status"]
            # status in {"scalar", "affine", "inv_affine", "affine_illposed_scalar"}
            if status == "affine_illposed_scalar":
                counts["affine_illposed_scalar"] += 1
            else:
                counts["per_frame"] += 1
            if status in ("affine", "inv_affine"):
                scalar_cvs.append(f["scalar_cv"])
                model_cvs.append(f["model_cv"])
                if f["scalar_cv"] and f["model_cv"] < 0.8 * f["scalar_cv"]:
                    improved += 1

        save_depth_three_formats(apply_model(depth, model), stem, out)
        rows.append(dict(
            stem=stem, n_anchors=f["n"], status=status,
            scalar_s=("" if f["scalar_s"] is None else round(f["scalar_s"], 6)),
            model="|".join(str(round(float(x), 6)) for x in model[1:]),
            scalar_cv=("" if f["scalar_cv"] is None else round(f["scalar_cv"], 5)),
            model_cv=("" if f["model_cv"] is None else round(f["model_cv"], 5)),
        ))

    rows.sort(key=lambda r: r["stem"])
    with open(out / "per_frame_scale.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["stem", "n_anchors", "status",
                                           "scalar_s", "model", "scalar_cv", "model_cv"])
        w.writeheader()
        w.writerows(rows)

    resid_cmp = None
    if scalar_cvs:
        resid_cmp = {
            "frames_with_affine": len(scalar_cvs),
            "median_scalar_rel_cv": float(np.median(scalar_cvs)),
            "median_model_rel_cv": float(np.median(model_cvs)),
            "frac_frames_affine_better_20pct": float(improved / len(scalar_cvs)),
        }

    summary = {
        "inputs": {"refined": args.refined, "depth": args.depth,
                   "coarse": args.coarse or None, "output": str(out)},
        "mode": args.mode,
        "global_scale_median_per_frame": global_scale,
        "sim3_scale_coarse_to_refined": sim3_scale,
        "clamp_factor": args.clamp, "clamp_range": [lo, hi],
        "min_depth_span": args.min_depth_span,
        "counts": counts,
        "affine_vs_scalar_residual": resid_cmp,
        "fallback_clamp_frames": [r["stem"] for r in rows if r["status"] == "fallback_clamp"],
    }
    with open(out / "apply_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, default=str)

    print("\n================ APPLY PER-FRAME CORRECTION ================")
    print(f" mode                     : {args.mode}")
    print(f" global scale             : {global_scale:.5f}"
          + (f"  | Sim3={sim3_scale:.5f}" if sim3_scale is not None else ""))
    print(f" per-frame applied        : {counts['per_frame']}")
    print(f"   affine ill-posed->scalar: {counts['affine_illposed_scalar']}")
    print(f"   fallback few anchors   : {counts['fallback_few_anchors']}")
    print(f"   fallback clamp         : {counts['fallback_clamp']}  "
          f"{summary['fallback_clamp_frames'][:10]}")
    print(f"   no depth               : {counts['no_depth']}")
    if resid_cmp:
        print(f" affine vs scalar (rel CV): scalar={resid_cmp['median_scalar_rel_cv']:.4f} "
              f"-> model={resid_cmp['median_model_rel_cv']:.4f}  "
              f"(affine>20% better in {100*resid_cmp['frac_frames_affine_better_20pct']:.0f}% of frames)")
    print(f" output                   : {out}/depth_npy|depth_u16|depth_vis")
    print("===========================================================")


if __name__ == "__main__":
    main()
