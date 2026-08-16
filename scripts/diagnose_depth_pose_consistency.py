#!/usr/bin/env python3
"""
Diagnose feed-forward depth vs. post-BA pose consistency.

Background
----------
The feed-forward model emits (pose, depth) that are jointly scale-consistent,
so TSDF fusion with the *pre-BA* pose is clean. After bundle adjustment the
poses are updated but the dense depth is not, so depth and pose may disagree
in scale and TSDF fusion produces layered / doubled surfaces ("错层").

This script quantifies that disagreement *before* committing to any fix, and
tells you whether the disagreement is:

  - a single global scale          -> one scalar on all depth fixes it
  - a per-frame scale (drift)      -> one scalar per frame
  - a smooth in-frame field         -> low-frequency scale field per frame
  - high-frequency / structured     -> model-based refinement (e.g. lingbot)

How it works
------------
1. Load the post-BA reconstruction (`--refined`). Its `points3D` are the
   sparse, scale-correct anchors. For every observation of every 3D point:
     z_ba   = depth of the BA point in the observing camera (correct scale)
     d_ff   = feed-forward dense depth sampled at the same pixel (FF scale)
     r      = z_ba / d_ff   (the per-pixel scale correction FF depth needs)
2. Aggregate `r` per frame (median, spread) and fit a smooth in-frame plane
   r ~ a + b*u + c*v to detect a low-frequency field.
3. Optionally compare against the pre-BA poses (`--coarse`) via a Sim3
   (Umeyama) fit on camera centers. The residual *after* removing the best
   global similarity is the non-rigid deformation BA introduced -- that
   residual, not the global part, is what breaks depth consistency.
4. Independently of scale, unproject each observing frame's FF depth into the
   world with the refined pose and measure how far apart those world points
   land (the geometric "错层" magnitude in meters).

Inputs
------
--refined : COLMAP model dir (cameras/images/points3D, .bin or .txt), BA-后
--depth   : the `pred_depth` dir produced by the pipeline (uses depth_npy/)
--coarse  : (optional) COLMAP model dir, BA-前, for the Sim3 deformation report
--output  : where to write the JSON / CSV / optional PNG plots

The script reads COLMAP files directly (no pycolmap dependency).
"""

import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# COLMAP model reading (.bin and .txt), self-contained
# --------------------------------------------------------------------------- #

# model_id -> (model_name, num_params)
CAMERA_MODEL_IDS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_cameras_bin(path):
    cameras = {}
    with open(path, "rb") as fid:
        (num,) = _read_next_bytes(fid, 8, "Q")
        for _ in range(num):
            cam_id, model_id, width, height = _read_next_bytes(fid, 24, "iiQQ")
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = dict(
                model=model_name, width=int(width), height=int(height),
                params=np.array(params, dtype=np.float64),
            )
    return cameras


def read_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            e = line.split()
            cameras[int(e[0])] = dict(
                model=e[1], width=int(e[2]), height=int(e[3]),
                params=np.array(list(map(float, e[4:])), dtype=np.float64),
            )
    return cameras


def read_images_bin(path):
    images = {}
    with open(path, "rb") as fid:
        (num,) = _read_next_bytes(fid, 8, "Q")
        for _ in range(num):
            img_id, qw, qx, qy, qz, tx, ty, tz, cam_id = _read_next_bytes(
                fid, 64, "idddddddi"
            )
            name = b""
            while True:
                c = fid.read(1)
                if c == b"\x00":
                    break
                name += c
            (num_pts,) = _read_next_bytes(fid, 8, "Q")
            # x(double) y(double) point3D_id(int64) per observation
            buf = fid.read(24 * num_pts)
            arr = np.frombuffer(buf, dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("pid", "<i8")]))
            images[img_id] = dict(
                name=name.decode("utf-8"),
                qvec=np.array([qw, qx, qy, qz], dtype=np.float64),
                tvec=np.array([tx, ty, tz], dtype=np.float64),
                camera_id=cam_id,
                xys=np.stack([arr["x"], arr["y"]], axis=1) if num_pts else np.zeros((0, 2)),
                point3D_ids=arr["pid"].copy() if num_pts else np.zeros((0,), dtype=np.int64),
            )
    return images


def read_images_txt(path):
    images = {}
    with open(path) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        e = line.split()
        img_id = int(e[0])
        qvec = np.array(list(map(float, e[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, e[5:8])), dtype=np.float64)
        cam_id = int(e[8])
        name = e[9] if len(e) == 10 else "_".join(e[9:])
        pts_line = lines[i + 1].split()
        if len(pts_line) >= 3:
            vals = np.array(list(map(float, pts_line)), dtype=np.float64).reshape(-1, 3)
            xys = vals[:, :2]
            pids = vals[:, 2].astype(np.int64)
        else:
            xys = np.zeros((0, 2))
            pids = np.zeros((0,), dtype=np.int64)
        images[img_id] = dict(
            name=name, qvec=qvec, tvec=tvec, camera_id=cam_id,
            xys=xys, point3D_ids=pids,
        )
        i += 2
    return images


def read_points3d_bin(path):
    points = {}
    with open(path, "rb") as fid:
        (num,) = _read_next_bytes(fid, 8, "Q")
        for _ in range(num):
            pid, x, y, z, r, g, b, err = _read_next_bytes(fid, 43, "QdddBBBd")
            (track_len,) = _read_next_bytes(fid, 8, "Q")
            buf = fid.read(8 * track_len)  # (image_id int32, point2D_idx uint32)
            track = np.frombuffer(buf, dtype=np.dtype([("img", "<i4"), ("p2d", "<u4")]))
            points[pid] = dict(
                xyz=np.array([x, y, z], dtype=np.float64),
                error=float(err),
                track_image_ids=track["img"].copy(),
                track_p2d_idx=track["p2d"].copy(),
            )
    return points


def read_points3d_txt(path):
    points = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            e = line.split()
            pid = int(e[0])
            xyz = np.array(list(map(float, e[1:4])), dtype=np.float64)
            err = float(e[7])
            track = np.array(list(map(int, e[8:])), dtype=np.int64).reshape(-1, 2)
            points[pid] = dict(
                xyz=xyz, error=err,
                track_image_ids=track[:, 0].astype(np.int32),
                track_p2d_idx=track[:, 1].astype(np.uint32),
            )
    return points


def read_colmap_model(model_dir):
    model_dir = Path(model_dir)
    if (model_dir / "cameras.bin").exists():
        cameras = read_cameras_bin(model_dir / "cameras.bin")
        images = read_images_bin(model_dir / "images.bin")
        p3d_path = model_dir / "points3D.bin"
        points = read_points3d_bin(p3d_path) if p3d_path.exists() else {}
    elif (model_dir / "cameras.txt").exists():
        cameras = read_cameras_txt(model_dir / "cameras.txt")
        images = read_images_txt(model_dir / "images.txt")
        p3d_path = model_dir / "points3D.txt"
        points = read_points3d_txt(p3d_path) if p3d_path.exists() else {}
    else:
        raise FileNotFoundError(f"No COLMAP cameras.bin/.txt found in {model_dir}")
    return cameras, images, points


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #

def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def intrinsics_matrix(cam):
    p = cam["params"]
    if cam["model"] == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif cam["model"] in ("PINHOLE", "OPENCV", "FULL_OPENCV"):
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif cam["model"] in ("SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "FOV"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    elif cam["model"] in ("RADIAL", "RADIAL_FISHEYE"):
        fx = fy = p[0]; cx, cy = p[1], p[2]
    else:
        raise ValueError(f"Unsupported camera model: {cam['model']}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return K


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


# --------------------------------------------------------------------------- #
# Depth loading
# --------------------------------------------------------------------------- #

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


def robust_stats(values):
    v = np.asarray(values, dtype=np.float64)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    cv = float(1.4826 * mad / med) if med > 1e-9 else float("nan")
    return med, mad, cv


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refined", required=True, help="Post-BA COLMAP model dir (with points3D)")
    ap.add_argument("--depth", required=True, help="pred_depth dir (uses depth_npy/)")
    ap.add_argument("--coarse", default="", help="Pre-BA COLMAP model dir (optional, for Sim3 deformation report)")
    ap.add_argument("--output", default="depth_pose_diag", help="Output dir for json/csv/plots")
    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=50.0)
    ap.add_argument("--min_anchors", type=int, default=8, help="Min anchors per frame for per-frame stats")
    ap.add_argument("--ratio_clip", type=float, default=4.0, help="Reject anchors with r outside [1/c, c]*frame_median during stats")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[DIAG] Reading refined model: {args.refined}")
    cameras, images, points = read_colmap_model(args.refined)
    if not points:
        raise RuntimeError("Refined model has no points3D -- cannot anchor depth scale.")
    print(f"[DIAG] images={len(images)}, points3D={len(points)}, cameras={len(cameras)}")

    # Pre-cache per-image: K, R, t, center, depth, scale (depth grid vs camera res)
    img_cache = {}
    missing_depth = []
    for img_id, im in images.items():
        cam = cameras[im["camera_id"]]
        stem = Path(im["name"]).stem
        print(f"[DIAG] Loading depth for {stem}")
        depth = load_ff_depth(args.depth, stem)
        if depth is None:
            missing_depth.append(stem)
            continue
        Hd, Wd = depth.shape[:2]
        sx = Wd / cam["width"]
        sy = Hd / cam["height"]
        img_cache[img_id] = dict(
            stem=stem,
            K=intrinsics_matrix(cam),
            R=qvec2rotmat(im["qvec"]),
            t=im["tvec"],
            center=camera_center(im["qvec"], im["tvec"]),
            depth=depth, Hd=Hd, Wd=Wd, sx=sx, sy=sy,
        )
    if missing_depth:
        print(f"[DIAG] WARNING: {len(missing_depth)} frames have no depth (e.g. {missing_depth[:3]})")

    # Map (image_id, point2D_idx) -> observation pixel via image.xys
    # Walk every 3D point's track; accumulate per-frame ratios and per-point world spread.
    per_frame = {iid: {"r": [], "u": [], "v": []} for iid in img_cache}
    point_spread = []          # geometric 错层 magnitude per point (meters)
    point_spread_rel = []      # normalized by that point's mean depth
    n_obs_used = 0

    for pid, p in points.items():
        Xw = p["xyz"]
        world_pts = []
        depths_here = []
        for img_id, p2d_idx in zip(p["track_image_ids"].tolist(), p["track_p2d_idx"].tolist()):
            c = img_cache.get(int(img_id))
            if c is None:
                continue
            xy = images[int(img_id)]["xys"][int(p2d_idx)]
            x_h, y_h = float(xy[0]), float(xy[1])
            # z of the BA point in this camera (correct scale)
            z_ba = float(c["R"][2] @ Xw + c["t"][2])
            if not (args.depth_min < z_ba < args.depth_max):
                continue
            # sample FF depth at mapped low-res pixel
            u = int(round(x_h * c["sx"]))
            v = int(round(y_h * c["sy"]))
            if not (0 <= u < c["Wd"] and 0 <= v < c["Hd"]):
                continue
            d_ff = float(c["depth"][v, u])
            if not (args.depth_min < d_ff < args.depth_max):
                continue
            r = z_ba / d_ff
            per_frame[img_id]["r"].append(r)
            per_frame[img_id]["u"].append(x_h / cameras[images[img_id]["camera_id"]]["width"] - 0.5)
            per_frame[img_id]["v"].append(y_h / cameras[images[img_id]["camera_id"]]["height"] - 0.5)
            n_obs_used += 1

            # FF-depth world unprojection with refined pose (for 错层 spread)
            ray = np.linalg.inv(c["K"]) @ np.array([x_h, y_h, 1.0])
            X_cam = d_ff * ray  # assumes z-depth (Open3D TSDF convention)
            X_world_ff = c["R"].T @ (X_cam - c["t"])
            world_pts.append(X_world_ff)
            depths_here.append(d_ff)

        if len(world_pts) >= 2:
            wp = np.stack(world_pts, 0)
            spread = float(np.linalg.norm(wp - wp.mean(0), axis=1).mean())
            point_spread.append(spread)
            point_spread_rel.append(spread / max(np.mean(depths_here), 1e-6))

    print(f"[DIAG] anchors used={n_obs_used}, points with >=2 obs spread measured={len(point_spread)}")

    # ---------------- per-frame aggregation ---------------- #
    frame_rows = []
    frame_medians = []
    frame_cvs = []
    plane_r2s = []
    plane_grad_rels = []
    for img_id, d in per_frame.items():
        r = np.asarray(d["r"], dtype=np.float64)
        if r.size < args.min_anchors:
            continue
        med0 = np.median(r)
        keep = (r > med0 / args.ratio_clip) & (r < med0 * args.ratio_clip)
        r = r[keep]
        u = np.asarray(d["u"])[keep]
        v = np.asarray(d["v"])[keep]
        if r.size < args.min_anchors:
            continue
        med, mad, cv = robust_stats(r)
        # smooth in-frame plane fit r ~ a + b*u + c*v
        A = np.stack([np.ones_like(u), u, v], axis=1)
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((r - pred) ** 2))
        ss_tot = float(np.sum((r - r.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
        grad_rel = float(np.hypot(coef[1], coef[2]) / med) if med > 1e-9 else float("nan")
        resid = r - pred
        resid_cv = float(1.4826 * np.median(np.abs(resid - np.median(resid))) / med) if med > 1e-9 else float("nan")

        frame_rows.append(dict(
            stem=img_cache[img_id]["stem"], image_id=img_id, n_anchors=int(r.size),
            median_r=med, mad_r=mad, cv_r=cv, plane_R2=r2,
            plane_grad_rel=grad_rel, resid_cv_after_plane=resid_cv,
        ))
        frame_medians.append(med)
        frame_cvs.append(cv)
        plane_r2s.append(r2)
        plane_grad_rels.append(grad_rel)

    frame_rows.sort(key=lambda x: x["stem"])
    frame_medians = np.asarray(frame_medians)
    frame_cvs = np.asarray(frame_cvs)
    plane_r2s = np.asarray(plane_r2s)

    # ---------------- Sim3 deformation (coarse vs refined) ---------------- #
    sim3_report = None
    if args.coarse:
        try:
            _, c_images, _ = read_colmap_model(args.coarse)
            name_to_center_ref = {Path(images[i]["name"]).stem: img_cache[i]["center"]
                                  for i in img_cache}
            name_to_center_coarse = {Path(im["name"]).stem: camera_center(im["qvec"], im["tvec"])
                                     for im in c_images.values()}
            common = sorted(set(name_to_center_ref) & set(name_to_center_coarse))
            if len(common) >= 3:
                src = np.stack([name_to_center_coarse[n] for n in common])  # coarse
                dst = np.stack([name_to_center_ref[n] for n in common])     # refined
                s, R, t, resid = umeyama_sim3(src, dst)
                scene_scale = float(np.linalg.norm(dst - dst.mean(0), axis=1).mean())
                sim3_report = dict(
                    num_common_frames=len(common),
                    global_scale_coarse_to_refined=s,
                    residual_after_sim3_m={
                        "mean": float(resid.mean()), "median": float(np.median(resid)),
                        "p90": float(np.percentile(resid, 90)), "max": float(resid.max()),
                    },
                    residual_relative_to_scene={
                        "mean": float(resid.mean() / scene_scale) if scene_scale > 0 else None,
                        "p90": float(np.percentile(resid, 90) / scene_scale) if scene_scale > 0 else None,
                    },
                    scene_scale_m=scene_scale,
                    worst_frames=[common[k] for k in np.argsort(resid)[::-1][:10].tolist()],
                )
                print(f"[DIAG] Sim3 coarse->refined: scale={s:.4f}, "
                      f"residual median={np.median(resid):.4g}m, "
                      f"p90={np.percentile(resid,90):.4g}m (scene~{scene_scale:.3g}m)")
        except Exception as e:  # noqa: BLE001
            print(f"[DIAG] Sim3 report skipped: {e}")

    # ---------------- global summary + recommendation ---------------- #
    def frac(arr, thr, ge=False):
        a = np.asarray(arr)
        if a.size == 0:
            return None
        return float(np.mean(a >= thr) if ge else np.mean(a < thr))

    global_median_r = float(np.median(frame_medians)) if frame_medians.size else None
    frame_median_cv = (float(1.4826 * np.median(np.abs(frame_medians - np.median(frame_medians)))
                             / np.median(frame_medians))
                       if frame_medians.size else None)
    med_frame_cv = float(np.median(frame_cvs)) if frame_cvs.size else None
    med_plane_r2 = float(np.median(plane_r2s)) if plane_r2s.size else None

    # Decision heuristic (tunable)
    recommendation = "insufficient_data"
    reason = ""
    if frame_medians.size:
        if frame_median_cv is not None and frame_median_cv < 0.02 and med_frame_cv < 0.05:
            recommendation = "global_scalar"
            reason = "Per-frame median ratios agree (CV<2%) and in-frame scatter is small -> one global scale on all depth."
        elif med_frame_cv is not None and med_frame_cv < 0.06 and (med_plane_r2 or 0) < 0.4:
            recommendation = "per_frame_scalar"
            reason = "Each frame's ratio is internally tight but frame medians differ -> one scalar (or affine) per frame."
        elif (med_plane_r2 or 0) >= 0.4:
            recommendation = "per_frame_smooth_field"
            reason = "A smooth in-frame plane explains much of the ratio variation -> low-frequency scale field per frame."
        else:
            recommendation = "model_based_refinement"
            reason = "Ratio variation is high and not explained by a smooth field -> structural depth error; consider lingbot-depth refinement."

    summary = dict(
        inputs=dict(refined=args.refined, depth=args.depth, coarse=args.coarse or None),
        counts=dict(
            images=len(images), points3D=len(points), anchors_used=n_obs_used,
            frames_with_stats=len(frame_rows),
            frames_missing_depth=len(missing_depth),
            points_spread_measured=len(point_spread),
        ),
        ratio_global=dict(
            global_median_r=global_median_r,
            frame_median_ratio_cv=frame_median_cv,
            median_in_frame_cv=med_frame_cv,
            median_plane_R2=med_plane_r2,
            frame_median_r_min=float(frame_medians.min()) if frame_medians.size else None,
            frame_median_r_max=float(frame_medians.max()) if frame_medians.size else None,
        ),
        layering_world_spread=dict(
            note="Mean per-point spread of FF-depth world unprojections across observing frames (refined pose).",
            median_m=float(np.median(point_spread)) if point_spread else None,
            p90_m=float(np.percentile(point_spread, 90)) if point_spread else None,
            median_relative_to_depth=float(np.median(point_spread_rel)) if point_spread_rel else None,
            p90_relative_to_depth=float(np.percentile(point_spread_rel, 90)) if point_spread_rel else None,
            frac_points_rel_gt_1pct=frac(point_spread_rel, 0.01, ge=True),
            frac_points_rel_gt_2pct=frac(point_spread_rel, 0.02, ge=True),
        ),
        sim3_deformation=sim3_report,
        recommendation=recommendation,
        recommendation_reason=reason,
    )

    with open(out / "diag_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out / "per_frame.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "stem", "image_id", "n_anchors", "median_r", "mad_r", "cv_r",
            "plane_R2", "plane_grad_rel", "resid_cv_after_plane",
        ])
        w.writeheader()
        for row in frame_rows:
            w.writerow(row)

    # ---------------- optional plots ---------------- #
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            if frame_medians.size:
                axes[0, 0].hist(frame_medians, bins=40, color="steelblue")
                axes[0, 0].axvline(global_median_r, color="red", ls="--",
                                   label=f"global median={global_median_r:.4f}")
                axes[0, 0].set_title("Per-frame median scale ratio (z_ba / d_ff)")
                axes[0, 0].set_xlabel("ratio"); axes[0, 0].legend()

                axes[0, 1].hist(frame_cvs, bins=40, color="darkorange")
                axes[0, 1].set_title("Per-frame in-frame ratio CV (tightness)")
                axes[0, 1].set_xlabel("CV (robust)")

                xs = np.arange(len(frame_rows))
                axes[1, 0].plot(xs, [r["median_r"] for r in frame_rows], ".-", ms=3)
                axes[1, 0].set_title("Frame median ratio vs frame index (drift)")
                axes[1, 0].set_xlabel("frame (sorted by name)"); axes[1, 0].set_ylabel("median ratio")

                axes[1, 1].hist(plane_r2s, bins=40, color="seagreen")
                axes[1, 1].set_title("Per-frame in-frame plane fit R^2 (field-ness)")
                axes[1, 1].set_xlabel("R^2")
            fig.tight_layout()
            fig.savefig(out / "diag_plots.png", dpi=120)
            plt.close(fig)
            print(f"[DIAG] plots -> {out / 'diag_plots.png'}")
        except Exception as e:  # noqa: BLE001
            print(f"[DIAG] plotting skipped: {e}")

    # ---------------- console summary ---------------- #
    print("\n================ DIAGNOSIS ================")
    print(f" frames with stats        : {len(frame_rows)}")
    print(f" global median ratio      : {global_median_r}")
    print(f" frame-median ratio CV    : {frame_median_cv}   (small => global scale enough)")
    print(f" median in-frame CV       : {med_frame_cv}      (small => per-frame scalar enough)")
    print(f" median in-frame plane R^2: {med_plane_r2}      (large => smooth field needed)")
    if point_spread_rel:
        print(f" 错层 spread (rel depth)  : median={np.median(point_spread_rel):.4f}, "
              f"p90={np.percentile(point_spread_rel,90):.4f}")
    if sim3_report:
        print(f" Sim3 global scale        : {sim3_report['global_scale_coarse_to_refined']:.4f}")
        print(f" deformation (rel scene)  : median="
              f"{sim3_report['residual_relative_to_scene']['mean']}, "
              f"p90={sim3_report['residual_relative_to_scene']['p90']}")
    print(f"\n RECOMMENDATION           : {recommendation}")
    print(f"   {reason}")
    print(f"\n outputs: {out/'diag_summary.json'}, {out/'per_frame.csv'}")
    print("===========================================")


if __name__ == "__main__":
    main()
