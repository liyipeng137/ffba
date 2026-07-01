#!/usr/bin/env python3
"""
Build a hybrid delivery PLY from COLMAP sparse points and TSDF mesh samples.

The script keeps all original COLMAP points, then adds depth-derived mesh
samples only where they plausibly fill low-texture sparse holes:

  1. Sample candidate points from the final TSDF mesh.
  2. Drop candidates too close to existing COLMAP points with scipy cKDTree.
  3. Project each candidate to BA-refined views and require depth consistency
     with the final corrected depth in at least --min_support_views views.
  4. Build image-space sparse-hole tile maps from COLMAP feature observations.
     A candidate is added if at least one of its supporting views lands in a
     sparse-hole tile.

Output colors are intentionally source-only for visualization:
  - original COLMAP points: blue
  - added depth points: red
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from diagnose_depth_pose_consistency import (  # noqa: E402
    intrinsics_matrix,
    qvec2rotmat,
    read_cameras_bin,
    read_cameras_txt,
    read_images_bin,
    read_images_txt,
)


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


def image_w2c(image: dict) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec2rotmat(image["qvec"])
    w2c[:3, 3] = image["tvec"]
    return w2c


def import_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d is required for PLY/mesh IO.") from exc
    return o3d


def import_ckdtree():
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("scipy is required for cKDTree-based 3D dedup.") from exc
    return cKDTree


def load_point_cloud_points(path: Path) -> np.ndarray:
    o3d = import_open3d()
    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float64)
    if points.size == 0:
        raise RuntimeError(f"Point cloud has no points: {path}")
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.size == 0:
        raise RuntimeError(f"Point cloud has no finite points: {path}")
    return points


def sample_mesh_points(
    path: Path, num_points: int, method: str, poisson_init_factor: int
) -> np.ndarray:
    o3d = import_open3d()
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise RuntimeError(f"Mesh has no triangles: {path}")
    if method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    elif method == "poisson":
        pcd = mesh.sample_points_poisson_disk(
            number_of_points=num_points,
            init_factor=poisson_init_factor,
        )
    elif method == "vertices":
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
    else:
        raise ValueError(f"Unknown sample method: {method}")
    points = np.asarray(pcd.points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.size == 0:
        raise RuntimeError(f"Mesh sampling produced no finite points: {path}")
    return points


def resolve_depth_path(depth_dir: Path, stem: str, image_name: str) -> Path | None:
    name = Path(image_name).name
    candidates = [
        depth_dir / "depth_npy" / f"{stem}.npy",
        depth_dir / "depth_u16" / f"{stem}.png",
        depth_dir / "depth_png" / f"{stem}.png",
        depth_dir / f"{stem}.npy",
        depth_dir / f"{stem}.png",
        depth_dir / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_depth_for_frame(
    depth_dir: Path,
    stem: str,
    image_name: str,
    width: int,
    height: int,
    depth_scale: float,
) -> np.ndarray | None:
    path = resolve_depth_path(depth_dir, stem, image_name)
    if path is None:
        return None
    if path.suffix.lower() == ".npy":
        depth = np.load(path).astype(np.float32)
    else:
        depth = np.asarray(Image.open(path), dtype=np.float32) / depth_scale
    if depth.shape != (height, width):
        depth = np.asarray(
            Image.fromarray(depth).resize((width, height), Image.NEAREST),
            dtype=np.float32,
        )
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return depth.astype(np.float32, copy=False)


def build_sparse_hole_maps(
    cameras: dict,
    images: dict,
    tile_size: int,
    tile_min_sparse_points: int,
) -> tuple[dict[int, np.ndarray], dict]:
    hole_maps = {}
    stats = {}
    for image_id, image in images.items():
        cam = cameras[image["camera_id"]]
        width, height = int(cam["width"]), int(cam["height"])
        cols = (width + tile_size - 1) // tile_size
        rows = (height + tile_size - 1) // tile_size
        counts = np.zeros((rows, cols), dtype=np.int32)

        xys = np.asarray(image["xys"], dtype=np.float64)
        pids = np.asarray(image["point3D_ids"])
        valid = pids >= 0
        if xys.size > 0 and np.any(valid):
            xy = xys[valid]
            in_img = (
                (xy[:, 0] >= 0)
                & (xy[:, 0] < width)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < height)
            )
            xy = xy[in_img]
            if xy.size > 0:
                tx = np.clip((xy[:, 0] // tile_size).astype(np.int64), 0, cols - 1)
                ty = np.clip((xy[:, 1] // tile_size).astype(np.int64), 0, rows - 1)
                np.add.at(counts, (ty, tx), 1)

        hole = counts < tile_min_sparse_points
        hole_maps[int(image_id)] = hole
        stats[int(image_id)] = {
            "stem": Path(image["name"]).stem,
            "num_tiles": int(hole.size),
            "hole_tiles": int(np.count_nonzero(hole)),
            "hole_ratio": float(np.count_nonzero(hole) / hole.size),
            "observations": int(np.count_nonzero(pids >= 0)),
        }
    return hole_maps, stats


def dedup_against_colmap(
    candidates: np.ndarray,
    colmap_points: np.ndarray,
    radius: float,
    workers: int,
) -> tuple[np.ndarray, dict]:
    cKDTree = import_ckdtree()
    tree = cKDTree(colmap_points)
    dist, _ = tree.query(candidates, k=1, workers=workers)
    keep = dist > radius
    return candidates[keep], {
        "input_candidates": int(len(candidates)),
        "kept_after_dedup": int(np.count_nonzero(keep)),
        "removed_near_colmap": int(len(candidates) - np.count_nonzero(keep)),
        "dedup_radius": float(radius),
    }


def project_and_filter_candidates(
    candidates: np.ndarray,
    cameras: dict,
    images: dict,
    hole_maps: dict[int, np.ndarray],
    depth_dir: Path,
    *,
    depth_scale: float,
    depth_min: float,
    depth_max: float,
    depth_consistency_rel: float,
    min_support_views: int,
    tile_size: int,
    candidate_chunk_size: int,
) -> tuple[np.ndarray, dict]:
    support_counts = np.zeros(len(candidates), dtype=np.uint16)
    hole_support = np.zeros(len(candidates), dtype=bool)
    frames_used = 0
    frames_missing_depth = 0
    frame_stats = []

    for image_id, image in images.items():
        stem = Path(image["name"]).stem
        cam = cameras[image["camera_id"]]
        width, height = int(cam["width"]), int(cam["height"])
        depth = load_depth_for_frame(
            depth_dir=depth_dir,
            stem=stem,
            image_name=image["name"],
            width=width,
            height=height,
            depth_scale=depth_scale,
        )
        if depth is None:
            frames_missing_depth += 1
            continue

        k = intrinsics_matrix(cam)
        w2c = image_w2c(image)
        r = w2c[:3, :3]
        t = w2c[:3, 3]
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        hole_map = hole_maps[int(image_id)]
        frame_support = 0
        frame_hole_support = 0

        for start in range(0, len(candidates), candidate_chunk_size):
            end = min(start + candidate_chunk_size, len(candidates))
            pts = candidates[start:end]
            cam_pts = pts @ r.T + t
            z = cam_pts[:, 2]
            valid = (z > depth_min) & (z < depth_max) & np.isfinite(z)
            if not np.any(valid):
                continue

            local_indices = np.arange(start, end, dtype=np.int64)[valid]
            cam_v = cam_pts[valid]
            z_v = z[valid]
            u = np.rint(fx * (cam_v[:, 0] / z_v) + cx).astype(np.int64)
            v = np.rint(fy * (cam_v[:, 1] / z_v) + cy).astype(np.int64)
            in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            if not np.any(in_img):
                continue

            local_indices = local_indices[in_img]
            z_i = z_v[in_img]
            u_i = u[in_img]
            v_i = v[in_img]
            d_i = depth[v_i, u_i]
            valid_depth = (d_i > depth_min) & (d_i < depth_max) & np.isfinite(d_i)
            if not np.any(valid_depth):
                continue

            local_indices = local_indices[valid_depth]
            z_i = z_i[valid_depth]
            u_i = u_i[valid_depth]
            v_i = v_i[valid_depth]
            d_i = d_i[valid_depth]
            rel = np.abs(z_i - d_i) / np.maximum(z_i, 1e-6)
            support = rel <= depth_consistency_rel
            if not np.any(support):
                continue

            support_indices = local_indices[support]
            support_counts[support_indices] += 1
            frame_support += int(support_indices.size)

            tx = np.clip(
                (u_i[support] // tile_size).astype(np.int64), 0, hole_map.shape[1] - 1
            )
            ty = np.clip(
                (v_i[support] // tile_size).astype(np.int64), 0, hole_map.shape[0] - 1
            )
            hit_hole = hole_map[ty, tx]
            if np.any(hit_hole):
                hole_indices = support_indices[hit_hole]
                hole_support[hole_indices] = True
                frame_hole_support += int(hole_indices.size)

        frames_used += 1
        frame_stats.append(
            {
                "stem": stem,
                "support_hits": int(frame_support),
                "hole_support_hits": int(frame_hole_support),
            }
        )
        if frames_used == 1 or frames_used % 25 == 0:
            print(
                f"[HYBRID] validated frames={frames_used}, "
                f"current={stem}, support_hits={frame_support:,}, "
                f"hole_hits={frame_hole_support:,}"
            )

    selected_mask = (support_counts >= min_support_views) & hole_support
    selected = candidates[selected_mask]
    support_hist = np.bincount(
        support_counts.astype(np.int64), minlength=min_support_views + 2
    )
    stats = {
        "input_candidates": int(len(candidates)),
        "selected": int(len(selected)),
        "frames_used": int(frames_used),
        "frames_missing_depth": int(frames_missing_depth),
        "min_support_views": int(min_support_views),
        "depth_consistency_rel": float(depth_consistency_rel),
        "candidates_with_any_support": int(np.count_nonzero(support_counts > 0)),
        "candidates_with_hole_support": int(np.count_nonzero(hole_support)),
        "support_count_histogram": {
            str(i): int(v) for i, v in enumerate(support_hist.tolist()) if int(v) > 0
        },
        "frame_stats": frame_stats,
    }
    return selected, stats


def maybe_limit_points(
    points: np.ndarray, max_points: int, rng: np.random.Generator
) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx]


def write_hybrid_ply(
    colmap_points: np.ndarray, added_points: np.ndarray, output: Path
) -> None:
    o3d = import_open3d()
    points = np.concatenate([colmap_points, added_points], axis=0)
    colors = np.zeros((len(points), 3), dtype=np.float64)
    colors[: len(colmap_points)] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    colors[len(colmap_points) :] = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output), pcd)


def write_added_points_ply(added_points: np.ndarray, output: Path) -> None:
    if len(added_points) == 0:
        return
    o3d = import_open3d()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(added_points)
    pcd.colors = o3d.utility.Vector3dVector(
        np.repeat(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float64), len(added_points), axis=0
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output), pcd)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--points3d_ply", required=True, help="COLMAP sparse points PLY")
    ap.add_argument("--mesh", required=True, help="Final TSDF mesh PLY")
    ap.add_argument(
        "--depth_dir", required=True, help="Final corrected depth dir or depth_u16 dir"
    )
    ap.add_argument(
        "--refined_colmap", required=True, help="BA-refined COLMAP model dir"
    )
    ap.add_argument("--output", required=True, help="Output hybrid PLY")

    ap.add_argument("--num_candidates", type=int, default=1_000_000)
    ap.add_argument(
        "--sample_method", choices=["uniform", "poisson", "vertices"], default="uniform"
    )
    ap.add_argument("--poisson_init_factor", type=int, default=5)
    ap.add_argument("--voxel_size", type=float, default=0.01)
    ap.add_argument(
        "--dedup_radius", type=float, default=0.0, help="0 means 2 * voxel_size"
    )
    ap.add_argument("--min_support_views", type=int, default=2)
    ap.add_argument("--depth_consistency_rel", type=float, default=0.03)
    ap.add_argument("--depth_min", type=float, default=0.05)
    ap.add_argument("--depth_max", type=float, default=5.0)
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--tile_size", type=int, default=64)
    ap.add_argument("--tile_min_sparse_points", type=int, default=2)
    ap.add_argument("--candidate_chunk_size", type=int, default=200_000)
    ap.add_argument("--kdtree_workers", type=int, default=-1)
    ap.add_argument(
        "--max_added_points",
        type=int,
        default=0,
        help="Optional random cap after filtering; 0 disables",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--added_points_output", default="", help="Optional red-only added-points PLY"
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    cameras, images, colmap_fmt = read_colmap_cameras_images(Path(args.refined_colmap))
    print(
        f"[HYBRID] COLMAP={colmap_fmt}, cameras={len(cameras)}, images={len(images)}, "
        f"tile={args.tile_size}, tile_min={args.tile_min_sparse_points}"
    )

    colmap_points = load_point_cloud_points(Path(args.points3d_ply))
    print(f"[HYBRID] Loaded COLMAP points: {len(colmap_points):,}")

    candidates = sample_mesh_points(
        Path(args.mesh),
        num_points=args.num_candidates,
        method=args.sample_method,
        poisson_init_factor=args.poisson_init_factor,
    )
    print(
        f"[HYBRID] Sampled mesh candidates: {len(candidates):,} "
        f"(method={args.sample_method})"
    )

    dedup_radius = args.dedup_radius if args.dedup_radius > 0 else 2.0 * args.voxel_size
    candidates, dedup_stats = dedup_against_colmap(
        candidates,
        colmap_points,
        radius=dedup_radius,
        workers=args.kdtree_workers,
    )
    print(
        f"[HYBRID] Dedup: kept={len(candidates):,}, "
        f"removed={dedup_stats['removed_near_colmap']:,}, radius={dedup_radius:g}"
    )

    hole_maps, hole_stats = build_sparse_hole_maps(
        cameras,
        images,
        tile_size=args.tile_size,
        tile_min_sparse_points=args.tile_min_sparse_points,
    )
    mean_hole_ratio = float(np.mean([s["hole_ratio"] for s in hole_stats.values()]))
    print(f"[HYBRID] Sparse-hole maps built: mean_hole_ratio={mean_hole_ratio:.3f}")

    added_points, filter_stats = project_and_filter_candidates(
        candidates,
        cameras,
        images,
        hole_maps,
        Path(args.depth_dir),
        depth_scale=args.depth_scale,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        depth_consistency_rel=args.depth_consistency_rel,
        min_support_views=args.min_support_views,
        tile_size=args.tile_size,
        candidate_chunk_size=args.candidate_chunk_size,
    )
    before_cap = len(added_points)
    added_points = maybe_limit_points(added_points, args.max_added_points, rng)
    if len(added_points) != before_cap:
        print(f"[HYBRID] Capped added points: {before_cap:,} -> {len(added_points):,}")

    output = Path(args.output)
    write_hybrid_ply(colmap_points, added_points, output)
    print(
        f"[HYBRID] Wrote hybrid PLY: {output} "
        f"(colmap={len(colmap_points):,}, added={len(added_points):,})"
    )

    added_output = Path(args.added_points_output) if args.added_points_output else None
    if added_output is not None:
        write_added_points_ply(added_points, added_output)
        print(f"[HYBRID] Wrote added-points PLY: {added_output}")

    summary = {
        "inputs": {
            "points3d_ply": args.points3d_ply,
            "mesh": args.mesh,
            "depth_dir": args.depth_dir,
            "refined_colmap": args.refined_colmap,
        },
        "params": {
            "num_candidates": args.num_candidates,
            "sample_method": args.sample_method,
            "voxel_size": args.voxel_size,
            "dedup_radius": dedup_radius,
            "min_support_views": args.min_support_views,
            "depth_consistency_rel": args.depth_consistency_rel,
            "depth_min": args.depth_min,
            "depth_max": args.depth_max,
            "tile_size": args.tile_size,
            "tile_min_sparse_points": args.tile_min_sparse_points,
            "max_added_points": args.max_added_points,
        },
        "counts": {
            "colmap_points": int(len(colmap_points)),
            "added_points_before_cap": int(before_cap),
            "added_points": int(len(added_points)),
            "hybrid_points": int(len(colmap_points) + len(added_points)),
        },
        "dedup": dedup_stats,
        "filter": filter_stats,
        "hole_maps": {
            "mean_hole_ratio": mean_hole_ratio,
            "per_frame": hole_stats,
        },
        "outputs": {
            "hybrid_ply": str(output),
            "added_points_ply": None if added_output is None else str(added_output),
        },
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2, default=str)
    print(f"[HYBRID] Summary: {summary_path}")


if __name__ == "__main__":
    main()
