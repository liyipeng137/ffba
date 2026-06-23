#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qw * qz,
                2 * qz * qx + 2 * qw * qy,
            ],
            [
                2 * qx * qy + 2 * qw * qz,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qw * qx,
            ],
            [
                2 * qz * qx - 2 * qw * qy,
                2 * qy * qz + 2 * qw * qx,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )


def load_initial_transforms(transforms_json):
    with open(transforms_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    poses = {}
    metadata = {}
    for frame in data.get("frames", []):
        file_path = frame.get("file_path")
        if not file_path:
            continue
        image_name = Path(file_path).name
        c2w_opengl = np.asarray(frame["transform_matrix"], dtype=np.float64)
        c2w_opencv = c2w_opengl.copy()
        c2w_opencv[:3, 1:3] *= -1.0
        poses[image_name] = c2w_opencv
        metadata[image_name] = frame

    if not poses:
        raise ValueError(f"No frames found in {transforms_json}")
    return poses, metadata


def _is_colmap_image_header(parts):
    if len(parts) < 10:
        return False
    try:
        int(parts[0])
        [float(v) for v in parts[1:8]]
        int(parts[8])
    except ValueError:
        return False
    return True


def load_colmap_images_txt(images_txt):
    poses = {}
    with open(images_txt, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    idx = 0
    while idx < len(lines):
        parts = lines[idx].split()
        if not _is_colmap_image_header(parts):
            idx += 1
            continue

        qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        image_name = Path(parts[9]).name

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = qvec_to_rotmat(qvec)
        w2c[:3, 3] = tvec
        poses[image_name] = np.linalg.inv(w2c)
        idx += 2

    if not poses:
        raise ValueError(f"No COLMAP image poses found in {images_txt}")
    return poses


def resolve_bae_images_path(path):
    path = Path(path)
    if path.is_file():
        return path
    candidates = [
        path / "images_optimized.txt",
        path / "images.txt",
        path / "sparse" / "0" / "images_optimized.txt",
        path / "sparse" / "0" / "images.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find COLMAP images txt under {path}")


def estimate_umeyama(source, target, with_scale=True):
    if source.shape != target.shape:
        raise ValueError(f"source and target shape mismatch: {source.shape} vs {target.shape}")
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {source.shape}")
    if source.shape[0] < 3:
        raise ValueError("Need at least 3 common poses for global alignment")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered.T @ source_centered) / source.shape[0]
    U, singular_values, Vt = np.linalg.svd(covariance)

    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1.0

    rotation = U @ correction @ Vt
    if with_scale:
        source_var = np.mean(np.sum(source_centered * source_centered, axis=1))
        if source_var < 1e-12:
            raise ValueError("Source camera centers have near-zero variance")
        scale = float(np.sum(singular_values * np.diag(correction)) / source_var)
    else:
        scale = 1.0
    translation = target_mean - scale * (rotation @ source_mean)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return transform, scale, rotation, translation


def transform_points(points, transform):
    original_shape = points.shape
    flat = points.reshape(-1, 3).astype(np.float64)
    hom = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
    transformed = (transform @ hom.T).T
    transformed = transformed[:, :3] / np.clip(transformed[:, 3:], 1e-12, None)
    return transformed.reshape(original_shape).astype(np.float32)


def load_dense_metadata(dense_dir):
    dense_dir = Path(dense_dir)
    metadata_path = dense_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            frames = json.load(f).get("frames", [])
        return frames

    frames = []
    for dense_path in sorted(dense_dir.glob("*.npz")):
        with np.load(dense_path, allow_pickle=True) as data:
            image_name = str(data["image_name"].item()) if "image_name" in data else dense_path.stem
        frames.append({"dense_file": dense_path.name, "image_name": Path(image_name).name})
    if not frames:
        raise FileNotFoundError(f"No dense .npz files found in {dense_dir}")
    return frames


def write_ascii_ply(path, points, colors=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if colors is None:
        colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
    colors = np.asarray(colors)
    if colors.dtype != np.uint8:
        if colors.max(initial=0) <= 1.0:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Apply BAE optimized pose deltas to VGGT-SLAM framewise dense points."
    )
    parser.add_argument("--bae-images", required=True, help="BAE images_optimized.txt or directory containing it.")
    parser.add_argument("--initial-transforms", required=True, help="VGGT-SLAM exported transforms.json before BAE.")
    parser.add_argument("--dense-dir", required=True, help="VGGT-SLAM exported dense frame .npz directory.")
    parser.add_argument("--out-ply", required=True, help="Merged corrected dense point cloud PLY.")
    parser.add_argument("--out-dir", default=None, help="Optional directory for corrected per-frame .npz and diagnostics.")
    parser.add_argument("--alignment", choices=["sim3", "se3"], default="sim3")
    parser.add_argument("--mask-key", default="mask")
    parser.add_argument("--min-common-frames", type=int, default=3)
    args = parser.parse_args()

    global np
    import numpy as np

    initial_poses, _ = load_initial_transforms(args.initial_transforms)
    bae_images_path = resolve_bae_images_path(args.bae_images)
    optimized_poses_raw = load_colmap_images_txt(bae_images_path)

    common_names = sorted(set(initial_poses) & set(optimized_poses_raw))
    if len(common_names) < args.min_common_frames:
        raise ValueError(
            f"Only {len(common_names)} common frames between initial transforms and BAE poses; "
            f"need at least {args.min_common_frames}."
        )

    initial_centers = np.stack([initial_poses[name][:3, 3] for name in common_names], axis=0)
    optimized_centers = np.stack([optimized_poses_raw[name][:3, 3] for name in common_names], axis=0)
    gauge_transform, gauge_scale, gauge_rotation, gauge_translation = estimate_umeyama(
        optimized_centers,
        initial_centers,
        with_scale=args.alignment == "sim3",
    )

    optimized_poses = {
        name: gauge_transform @ pose for name, pose in optimized_poses_raw.items() if name in initial_poses
    }
    pose_deltas = {
        name: optimized_poses[name] @ np.linalg.inv(initial_poses[name]) for name in optimized_poses
    }

    aligned_centers = np.stack([optimized_poses[name][:3, 3] for name in common_names], axis=0)
    center_errors = np.linalg.norm(aligned_centers - initial_centers, axis=1)

    dense_dir = Path(args.dense_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        corrected_dir = out_dir / "corrected_frames"
        corrected_dir.mkdir(parents=True, exist_ok=True)
    else:
        corrected_dir = None

    merged_points = []
    merged_colors = []
    applied_frames = []
    skipped_frames = []

    for frame in load_dense_metadata(dense_dir):
        image_name = Path(frame["image_name"]).name
        dense_file = dense_dir / frame["dense_file"]
        if image_name not in pose_deltas:
            skipped_frames.append({"image_name": image_name, "reason": "missing_pose_delta"})
            continue

        with np.load(dense_file, allow_pickle=True) as data:
            pointcloud = data["pointcloud"]
            mask = data[args.mask_key].astype(bool)
            colors = data["colors"] if "colors" in data else None

        corrected = transform_points(pointcloud, pose_deltas[image_name])
        valid = mask & np.isfinite(corrected).all(axis=-1)
        frame_points = corrected[valid].reshape(-1, 3)
        if colors is None:
            colors = np.full(pointcloud.shape, 255, dtype=np.uint8)
        frame_colors = colors[valid].reshape(-1, 3)

        merged_points.append(frame_points)
        merged_colors.append(frame_colors)
        applied_frames.append(
            {
                "image_name": image_name,
                "dense_file": frame["dense_file"],
                "num_points": int(frame_points.shape[0]),
            }
        )

        if corrected_dir is not None:
            np.savez_compressed(
                corrected_dir / frame["dense_file"],
                pointcloud=corrected.astype(np.float32),
                mask=mask,
                colors=colors,
                delta=pose_deltas[image_name].astype(np.float32),
                c2w_before=initial_poses[image_name].astype(np.float32),
                c2w_after=optimized_poses[image_name].astype(np.float32),
                image_name=np.array(image_name),
            )

    if not merged_points:
        raise ValueError("No dense frames were corrected; check image name matching.")

    merged_points = np.concatenate(merged_points, axis=0)
    merged_colors = np.concatenate(merged_colors, axis=0)
    write_ascii_ply(args.out_ply, merged_points, merged_colors)

    diagnostics = {
        "bae_images": str(bae_images_path),
        "initial_transforms": str(Path(args.initial_transforms)),
        "dense_dir": str(dense_dir),
        "out_ply": str(Path(args.out_ply)),
        "alignment": args.alignment,
        "num_initial_poses": len(initial_poses),
        "num_optimized_poses": len(optimized_poses_raw),
        "num_common_poses": len(common_names),
        "gauge_alignment": {
            "scale": float(gauge_scale),
            "rotation_det": float(np.linalg.det(gauge_rotation)),
            "translation_norm": float(np.linalg.norm(gauge_translation)),
            "center_error_mean": float(center_errors.mean()),
            "center_error_median": float(np.median(center_errors)),
            "center_error_p95": float(np.percentile(center_errors, 95)),
        },
        "applied_frames": applied_frames,
        "skipped_frames": skipped_frames,
    }

    if out_dir is not None:
        with open(out_dir / "pose_deltas.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    name: {
                        "delta": pose_deltas[name].tolist(),
                        "c2w_before": initial_poses[name].tolist(),
                        "c2w_after": optimized_poses[name].tolist(),
                    }
                    for name in sorted(pose_deltas)
                },
                f,
                indent=4,
            )
        with open(out_dir / "diagnostics.json", "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=4)

    print(json.dumps(diagnostics["gauge_alignment"], indent=2))
    print(
        f"Corrected {len(applied_frames)} dense frames, skipped {len(skipped_frames)}, "
        f"wrote {merged_points.shape[0]} points to {args.out_ply}"
    )


if __name__ == "__main__":
    main()
