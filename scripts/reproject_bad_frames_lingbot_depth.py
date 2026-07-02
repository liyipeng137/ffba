#!/usr/bin/env python3
"""
Reproject trusted TSDF geometry to selected frames, then run LingBot-Depth.

Intended flow:

  1. Run correct_and_flag_depth.py to get:
       - depth_scaled/
       - layering_frames.txt
  2. Run tsdf_colmap.py on depth_scaled/ with --exclude_frames layering_frames.txt
     to get a trusted mesh and/or sampled point cloud.
  3. This script projects that trusted geometry back to the selected frames and
     uses the projected depth as LingBot-Depth input. If --frames is omitted,
     all refined COLMAP images are selected.

Inputs
------
--mesh            TSDF mesh .ply from tsdf_colmap.py
--pcd             sampled/full point cloud .ply from tsdf_colmap.py
--frames          optional layering_frames.txt, one frame stem per line
--image_dir       original RGB image directory
--refined_colmap  post-BA COLMAP model directory
--source          pcd | mesh

Outputs
-------
<output>/projected_depth/depth_npy|depth_u16|depth_vis/
<output>/lingbot_depth/depth_npy|depth_u16|depth_vis/  (+ depth_png from helper)
<output>/projection_summary.json

Notes
-----
- pcd mode uses z-buffer projection and is the safer first pass for sparse
  depth completion.
- mesh mode uses Open3D RaycastingScene and can produce dense raycast depth.
  Use --sparsify_stride > 1 if you want to feed sparse anchors instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_MERG3R_ROOT = _REPO_ROOT / "MERG3R"
for _p in (_SCRIPT_DIR, _MERG3R_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from correct_depth_scale import save_depth_three_formats  # noqa: E402
from diagnose_depth_pose_consistency import (  # noqa: E402
    intrinsics_matrix,
    qvec2rotmat,
    read_cameras_bin,
    read_cameras_txt,
    read_images_bin,
    read_images_txt,
)


def camera_k(cam: dict) -> np.ndarray:
    return intrinsics_matrix(cam).astype(np.float32)


def image_to_frame(image_id: int, image: dict) -> dict:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec2rotmat(image["qvec"])
    w2c[:3, 3] = image["tvec"]
    return {
        "image_id": int(image_id),
        "name": image["name"],
        "camera_id": image["camera_id"],
        "w2c": w2c,
    }


def read_colmap_cameras_images(model_dir: Path) -> tuple[dict, dict, str]:
    if (model_dir / "cameras.bin").exists() and (model_dir / "images.bin").exists():
        return (
            read_cameras_bin(model_dir / "cameras.bin"),
            read_images_bin(model_dir / "images.bin"),
            "bin",
        )
    if (model_dir / "cameras.txt").exists() and (model_dir / "images.txt").exists():
        return (
            read_cameras_txt(model_dir / "cameras.txt"),
            read_images_txt(model_dir / "images.txt"),
            "txt",
        )
    raise FileNotFoundError(f"No COLMAP cameras/images found in {model_dir}")


def read_frame_stems(path: Path) -> list[str]:
    stems = []
    for raw in path.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        stems.append(Path(raw).stem)
    return stems


def build_image_index(image_dir: Path) -> dict[str, Path]:
    image_paths = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_paths.extend(image_dir.rglob(pattern))
    index = {}
    for p in sorted(image_paths):
        index.setdefault(p.name, p)
        index.setdefault(p.stem, p)
    return index


def resolve_image_path(
    image_dir: Path, image_index: dict[str, Path], frame_name: str
) -> Path | None:
    direct = image_dir / frame_name
    if direct.exists():
        return direct
    by_name = image_index.get(Path(frame_name).name)
    if by_name is not None:
        return by_name
    return image_index.get(Path(frame_name).stem)


def load_rgb_for_frame(
    image_dir: Path,
    image_index: dict[str, Path],
    frame_name: str,
    width: int,
    height: int,
) -> np.ndarray | None:
    image_path = resolve_image_path(image_dir, image_index, frame_name)
    if image_path is None:
        return None
    image = np.asarray(Image.open(image_path).convert("RGB"))
    if image.shape[1] != width or image.shape[0] != height:
        image = np.asarray(
            Image.fromarray(image).resize((width, height), Image.LANCZOS)
        )
    return image


def sparsify_depth(depth: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return depth
    out = np.zeros_like(depth, dtype=np.float32)
    out[::stride, ::stride] = depth[::stride, ::stride]
    return out


def zbuffer_project_points(
    points_world: np.ndarray,
    w2c: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
    depth_min: float,
    depth_max: float,
    chunk_size: int,
    splat_radius: int,
) -> np.ndarray:
    zbuf = np.full(height * width, np.inf, dtype=np.float32)
    r = w2c[:3, :3].astype(np.float64)
    t = w2c[:3, 3].astype(np.float64)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    offsets = [(0, 0)]
    if splat_radius > 0:
        offsets = [
            (du, dv)
            for dv in range(-splat_radius, splat_radius + 1)
            for du in range(-splat_radius, splat_radius + 1)
            if du * du + dv * dv <= splat_radius * splat_radius
        ]

    for start in range(0, len(points_world), chunk_size):
        pts = points_world[start : start + chunk_size].astype(np.float64, copy=False)
        cam = pts @ r.T + t
        z = cam[:, 2]
        valid = (z > depth_min) & (z < depth_max) & np.isfinite(z)
        if not np.any(valid):
            continue
        cam = cam[valid]
        z = z[valid].astype(np.float32, copy=False)
        u = np.rint(fx * (cam[:, 0] / cam[:, 2]) + cx).astype(np.int64)
        v = np.rint(fy * (cam[:, 1] / cam[:, 2]) + cy).astype(np.int64)

        for du, dv in offsets:
            uu = u + du
            vv = v + dv
            in_img = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            if not np.any(in_img):
                continue
            linear = vv[in_img] * width + uu[in_img]
            np.minimum.at(zbuf, linear, z[in_img])

    depth = zbuf.reshape(height, width)
    depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    return depth


def load_point_cloud_points(path: Path) -> np.ndarray:
    import open3d as o3d

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        raise RuntimeError(f"Point cloud has no points: {path}")
    return points


def build_raycast_scene(mesh_path: Path):
    import open3d as o3d

    mesh = o3d.t.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return o3d, scene


def raycast_mesh_depth(
    o3d,
    scene,
    w2c: np.ndarray,
    k: np.ndarray,
    width: int,
    height: int,
    depth_min: float,
    depth_max: float,
    max_normal_angle_deg: float,
    nthreads: int,
) -> np.ndarray:
    k_t = o3d.core.Tensor(k.astype(np.float32))
    w2c_t = o3d.core.Tensor(w2c.astype(np.float32))
    rays = scene.create_rays_pinhole(
        intrinsic_matrix=k_t,
        extrinsic_matrix=w2c_t,
        width_px=int(width),
        height_px=int(height),
    )
    ans = scene.cast_rays(rays, nthreads=nthreads)
    t_hit = ans["t_hit"].numpy()
    rays_np = rays.numpy()
    origin = rays_np[..., :3]
    direction = rays_np[..., 3:]

    valid = np.isfinite(t_hit)
    t_safe = np.where(valid, t_hit, 0.0).astype(np.float32)
    hit_world = origin + direction * t_safe[..., None]
    hit_h = np.concatenate(
        [hit_world, np.ones((*hit_world.shape[:2], 1), dtype=np.float32)],
        axis=-1,
    )
    hit_cam = hit_h @ w2c.astype(np.float32).T
    depth = hit_cam[..., 2].astype(np.float32)
    valid &= np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)

    if max_normal_angle_deg > 0.0 and "primitive_normals" in ans:
        normals = ans["primitive_normals"].numpy()
        dir_norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        view_dir = direction / np.clip(dir_norm, 1e-8, None)
        normal_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = normals / np.clip(normal_norm, 1e-8, None)
        cos_abs = np.abs(np.sum(normals * (-view_dir), axis=-1))
        valid &= cos_abs >= np.cos(np.deg2rad(max_normal_angle_deg))

    return np.where(valid, depth, 0.0).astype(np.float32)


def summarize_depth(depth: np.ndarray) -> dict:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return {
            "valid_pixels": 0,
            "valid_ratio": 0.0,
            "min": None,
            "median": None,
            "max": None,
        }
    vals = depth[valid]
    return {
        "valid_pixels": int(vals.size),
        "valid_ratio": float(vals.size / depth.size),
        "min": float(np.min(vals)),
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
    }


def run_lingbot(
    images: list[np.ndarray],
    image_names: list[str],
    intrinsics: list[np.ndarray],
    projected_depth_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    use_fp16: bool,
    enable_depth_mask: bool,
) -> dict:
    from algos.lingbot_depth_refine import run_lingbot_depth_refinement

    stats = run_lingbot_depth_refinement(
        images=images,
        image_names=image_names,
        depth_image_names=image_names,
        depth_npy_dir=projected_depth_dir / "depth_npy",
        output_dir=output_dir,
        intrinsic=np.stack(intrinsics, axis=0),
        model_name=model_name,
        device=device,
        use_fp16=use_fp16,
        enable_depth_mask=enable_depth_mask,
    )

    # The LingBot helper writes depth_npy/depth_vis/depth_png. Mirror the
    # MERG3R export format as well: depth_npy/depth_u16/depth_vis.
    npy_dir = output_dir / "depth_npy"
    for npy_path in sorted(npy_dir.glob("*.npy")):
        depth = np.load(npy_path).astype(np.float32)
        save_depth_three_formats(depth, npy_path.stem, output_dir)
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mesh", default="", help="TSDF mesh .ply from tsdf_colmap.py")
    ap.add_argument("--pcd", default="", help="TSDF sampled/full point cloud .ply")
    ap.add_argument("--source", choices=["pcd", "mesh"], default="pcd")
    ap.add_argument(
        "--frames",
        default="",
        help="Optional frame-stem list, e.g. layering_frames.txt. "
        "If omitted, all refined COLMAP images are processed.",
    )
    ap.add_argument("--image_dir", required=True, help="Original RGB image directory")
    ap.add_argument("--refined_colmap", required=True, help="Post-BA COLMAP model dir")
    ap.add_argument("--output", required=True, help="Output directory")

    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=50.0)
    ap.add_argument("--point_chunk_size", type=int, default=1_000_000)
    ap.add_argument(
        "--splat_radius",
        type=int,
        default=0,
        help="PCD projection splat radius in pixels",
    )
    ap.add_argument(
        "--sparsify_stride",
        type=int,
        default=1,
        help="Keep one projected depth every N pixels before LingBot",
    )
    ap.add_argument(
        "--mesh_max_normal_angle",
        type=float,
        default=0.0,
        help="Mesh raycast grazing-angle filter; 0 disables it",
    )
    ap.add_argument("--raycast_threads", type=int, default=0)

    ap.add_argument("--skip_lingbot", action="store_true")
    ap.add_argument(
        "--lingbot_model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5"
    )
    ap.add_argument("--lingbot_device", default="cuda")
    ap.add_argument("--no_lingbot_fp16", action="store_true")
    ap.add_argument("--disable_depth_mask", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "pcd" and not args.pcd:
        raise ValueError("--pcd is required when --source pcd")
    if args.source == "mesh" and not args.mesh:
        raise ValueError("--mesh is required when --source mesh")

    out = Path(args.output)
    projected_dir = out / "projected_depth"
    lingbot_dir = out / "lingbot_depth"
    out.mkdir(parents=True, exist_ok=True)
    projected_dir.mkdir(parents=True, exist_ok=True)

    cameras, images, colmap_fmt = read_colmap_cameras_images(Path(args.refined_colmap))
    frames = [image_to_frame(iid, image) for iid, image in sorted(images.items())]
    missing = []
    if args.frames:
        frame_by_stem = {Path(f["name"]).stem: f for f in frames}
        requested_stems = read_frame_stems(Path(args.frames))
        selected = []
        for stem in requested_stems:
            frame = frame_by_stem.get(stem)
            if frame is None:
                missing.append(stem)
            else:
                selected.append(frame)
        frame_selection = "file"
        requested_count = len(requested_stems)
    else:
        selected = frames
        frame_selection = "all"
        requested_count = len(frames)

    if not selected:
        raise RuntimeError("No frames selected from refined COLMAP images.")
    if missing:
        print(
            f"[REPROJECT] Warning: {len(missing)} frame stems not found in COLMAP: {missing[:10]}"
        )
    print(
        f"[REPROJECT] COLMAP={colmap_fmt}, source={args.source}, "
        f"frame_selection={frame_selection}, frames={len(selected)}/{requested_count}"
    )

    if args.source == "pcd":
        points_world = load_point_cloud_points(Path(args.pcd))
        print(
            f"[REPROJECT] Loaded point cloud: {args.pcd} ({len(points_world):,} points)"
        )
        raycast_ctx = None
    else:
        points_world = None
        raycast_ctx = build_raycast_scene(Path(args.mesh))
        print(f"[REPROJECT] Loaded raycast mesh: {args.mesh}")

    image_dir = Path(args.image_dir)
    image_index = build_image_index(image_dir)
    lingbot_images = []
    lingbot_names = []
    lingbot_intrinsics = []
    frame_stats = []

    for frame in selected:
        stem = Path(frame["name"]).stem
        cam = cameras[frame["camera_id"]]
        width, height = int(cam["width"]), int(cam["height"])
        k = camera_k(cam)
        w2c = frame["w2c"].astype(np.float64)

        if args.source == "pcd":
            depth = zbuffer_project_points(
                points_world=points_world,
                w2c=w2c,
                k=k,
                width=width,
                height=height,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                chunk_size=args.point_chunk_size,
                splat_radius=max(args.splat_radius, 0),
            )
        else:
            o3d, scene = raycast_ctx
            depth = raycast_mesh_depth(
                o3d=o3d,
                scene=scene,
                w2c=w2c,
                k=k,
                width=width,
                height=height,
                depth_min=args.depth_min,
                depth_max=args.depth_max,
                max_normal_angle_deg=args.mesh_max_normal_angle,
                nthreads=args.raycast_threads,
            )

        depth = sparsify_depth(depth, max(args.sparsify_stride, 1))
        save_depth_three_formats(depth, stem, projected_dir)
        stats = summarize_depth(depth)
        stats.update({"stem": stem, "image_name": frame["name"]})
        frame_stats.append(stats)
        print(
            f"[REPROJECT] {stem}: valid={stats['valid_pixels']} "
            f"({stats['valid_ratio'] * 100:.2f}%), median={stats['median']}"
        )

        if not args.skip_lingbot:
            image = load_rgb_for_frame(
                image_dir, image_index, frame["name"], width, height
            )
            if image is None:
                print(
                    f"[REPROJECT] LingBot skip {stem}: missing RGB image {frame['name']}"
                )
                continue
            lingbot_images.append(image)
            lingbot_names.append(Path(frame["name"]).name)
            lingbot_intrinsics.append(k)

    lingbot_stats = None
    if args.skip_lingbot:
        print("[REPROJECT] LingBot skipped by --skip_lingbot")
    elif not lingbot_images:
        print("[REPROJECT] LingBot skipped: no readable bad-frame RGB images")
    else:
        print(
            f"[REPROJECT] Running LingBot-Depth on {len(lingbot_images)} frames -> {lingbot_dir}"
        )
        lingbot_stats = run_lingbot(
            images=lingbot_images,
            image_names=lingbot_names,
            intrinsics=lingbot_intrinsics,
            projected_depth_dir=projected_dir,
            output_dir=lingbot_dir,
            model_name=args.lingbot_model,
            device=args.lingbot_device,
            use_fp16=not args.no_lingbot_fp16,
            enable_depth_mask=not args.disable_depth_mask,
        )

    summary = {
        "inputs": {
            "mesh": args.mesh or None,
            "pcd": args.pcd or None,
            "source": args.source,
            "frames": args.frames or None,
            "frame_selection": frame_selection,
            "image_dir": args.image_dir,
            "refined_colmap": args.refined_colmap,
        },
        "params": {
            "depth_min": args.depth_min,
            "depth_max": args.depth_max,
            "splat_radius": args.splat_radius,
            "sparsify_stride": args.sparsify_stride,
            "mesh_max_normal_angle": args.mesh_max_normal_angle,
        },
        "outputs": {
            "projected_depth": str(projected_dir),
            "lingbot_depth": None if lingbot_stats is None else str(lingbot_dir),
        },
        "missing_colmap_frame_stems": missing,
        "frame_stats": frame_stats,
        "lingbot": lingbot_stats,
    }
    with open(out / "projection_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, default=str)

    print("[REPROJECT] DONE")
    print(f"  projected depth : {projected_dir}")
    if lingbot_stats is not None:
        print(f"  lingbot depth   : {lingbot_dir}")
    print(f"  summary         : {out / 'projection_summary.json'}")


if __name__ == "__main__":
    main()
