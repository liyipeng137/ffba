import dataclasses
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from hloc import extract_features, match_features, triangulation

from .colmap_io import (
    create_cameras_and_points_bin,
    create_images_from_pose_dict,
    export_points3d_to_ply,
    read_images_binary,
)
from .nerfstudio import NerfStudioDataset


@dataclasses.dataclass(frozen=True)
class PoseInitResult:
    sparse_dir: Path
    images_dir: Path
    ply_path: Path


def _validate_single_camera_dimensions(dimensions: list[tuple[int, int]]) -> tuple[int, int]:
    unique_dimensions = sorted(set(dimensions))
    if len(unique_dimensions) != 1:
        raise ValueError(
            "camera_id=1 requires all cached images to have the same width/height; "
            f"got {unique_dimensions}"
        )
    return unique_dimensions[0]


def _average_intrinsics(intrinsics: list[np.ndarray], dimensions: list[tuple[int, int]]) -> dict[str, float]:
    width, height = _validate_single_camera_dimensions(dimensions)
    values = np.stack(intrinsics, axis=0)
    return {
        "width": width,
        "height": height,
        "fx": float(np.mean(values[:, 0])),
        "fy": float(np.mean(values[:, 1])),
        "cx": float(np.mean(values[:, 2])),
        "cy": float(np.mean(values[:, 3])),
    }


def cache_dataset(dataset: NerfStudioDataset, workspace_dir: Path) -> None:
    images_dir = workspace_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images_data = []
    pose_dict = {}
    intrinsics = []
    dimensions = []

    print("Collecting dataset for HLOC pose initialization...")
    for image_id, camera in enumerate(tqdm(dataset, desc="Collecting data")):
        image_name = str(image_id).zfill(8)
        image_np = camera.image.numpy() * 255
        images_data.append((image_name, image_np))
        pose_dict[image_name] = camera.extrinsics.inverse().numpy()

        K = camera.intrinsics.numpy()
        intrinsics.append(np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64))
        dimensions.append((camera.image_width, camera.image_height))

    print("Saving cached images...")
    for image_name, image_np in tqdm(images_data, desc="Saving images"):
        Image.fromarray(np.uint8(image_np)).save(images_dir / f"{image_name}.jpg", quality=95)

    create_cameras_and_points_bin(workspace_dir, _average_intrinsics(intrinsics, dimensions))
    create_images_from_pose_dict(workspace_dir, pose_dict)


def pairs_from_poses(
    images,
    overlap: int = 5,
    loop_Rt_thresh=(30.0, 2.0),
    near_Rt_min_thresh=(1.0, 0.05),
    max_loops_per_image: int = 5,
) -> list[tuple[str, str]]:
    ordered = sorted(images.items(), key=lambda item: item[0])
    names = [image.name for _, image in ordered]
    R_w2c = np.stack([image.qvec2rotmat() for _, image in ordered], 0).astype(np.float32)
    t_w2c = np.stack([image.tvec for _, image in ordered], 0).astype(np.float32)

    R_c2w = R_w2c.transpose(0, 2, 1)
    t_c2w = -(R_c2w @ t_w2c[:, :, None])[:, :, 0]

    n_images = len(names)
    if n_images == 0:
        return []

    R_loop_max, t_loop_max = loop_Rt_thresh
    R_near_min, t_near_min = near_Rt_min_thresh

    dt = t_c2w @ t_c2w.T
    dt *= -2
    sq = np.einsum("ij,ij->i", t_c2w, t_c2w)
    dt += sq[:, None]
    dt += sq[None]
    np.clip(dt, 0, None, out=dt)
    np.sqrt(dt, out=dt)

    trace = np.einsum("nji,mji->nm", R_c2w, R_c2w, optimize=True)
    dR = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    dR = np.rad2deg(np.abs(np.arccos(dR)))

    pairs = []
    added = set()
    for i in range(n_images - 1):
        for j in range(i + 1, min(i + overlap + 1, n_images)):
            if dR[i, j] < R_near_min and dt[i, j] < t_near_min:
                continue
            if (i, j) not in added:
                pairs.append((names[i], names[j]))
                added.add((i, j))

    for i in range(n_images):
        start = i + overlap + 1
        if start >= n_images:
            continue
        cand_idx = np.arange(start, n_images)
        valid = (dR[i, cand_idx] < R_loop_max) & (dt[i, cand_idx] < t_loop_max)
        valid &= ~((dR[i, cand_idx] < R_near_min) & (dt[i, cand_idx] < t_near_min))
        if not np.any(valid):
            continue
        valid_candidates = cand_idx[valid]
        order = np.lexsort((dR[i, valid_candidates], dt[i, valid_candidates]))
        for j in valid_candidates[order][:max_loops_per_image]:
            if (i, j) not in added:
                pairs.append((names[i], names[j]))
                added.add((i, j))

    return pairs


def write_pairs_from_poses(model_dir: Path, output: Path, overlap: int = 5) -> None:
    images = read_images_binary(model_dir / "images.bin")
    pairs = pairs_from_poses(images, overlap=overlap)
    if not pairs:
        raise ValueError("No image pairs were generated from the provided poses")
    with open(output, "w") as f:
        f.write("\n".join(" ".join(pair) for pair in pairs))


def run_hloc(workspace_dir: Path, overlap: int = 5) -> None:
    images_dir = workspace_dir / "images"
    model_dir = workspace_dir / "model"
    sfm_pairs = workspace_dir / "pairs-sfm.txt"

    write_pairs_from_poses(model_dir, sfm_pairs, overlap=overlap)

    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superpoint+lightglue"]
    features = extract_features.main(
        feature_conf,
        images_dir,
        feature_path=model_dir / "features.h5",
        as_half=False,
    )
    sfm_matches = match_features.main(
        matcher_conf,
        sfm_pairs,
        features=model_dir / "features.h5",
        matches=model_dir / "matches.h5",
    )

    sparse_reconstruction_folder = workspace_dir / "sparse" / "0"
    sparse_reconstruction_folder.mkdir(parents=True, exist_ok=True)
    triangulation.main(
        sparse_reconstruction_folder,
        model_dir,
        images_dir,
        sfm_pairs,
        features,
        sfm_matches,
        skip_geometric_verification=True,
    )
    shutil.rmtree(model_dir)


def run_poseinit(
    source_path: str | Path,
    output_dir: str | Path,
    overwrite: bool = True,
    resolution: float = 1.0,
    overlap: int = 5,
) -> PoseInitResult:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    sparse_dir = output_dir / "sparse" / "0"
    images_dir = output_dir / "images"
    ply_path = sparse_dir / "points3D.ply"

    if overwrite and (output_dir / "model").exists():
        shutil.rmtree(output_dir / "model")
    if overwrite and sparse_dir.exists():
        shutil.rmtree(output_dir / "sparse")
    if overwrite and images_dir.exists():
        shutil.rmtree(images_dir)

    if overwrite or not sparse_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset = NerfStudioDataset(source_path)
        dataset.all_cameras = [camera.downsample_scale(resolution) for camera in dataset.all_cameras]
        cache_dataset(dataset, output_dir)
        run_hloc(output_dir, overlap=overlap)

    points3d_bin = sparse_dir / "points3D.bin"
    if not points3d_bin.exists():
        raise FileNotFoundError(f"HLOC did not produce {points3d_bin}")
    export_points3d_to_ply(points3d_bin, ply_path)

    return PoseInitResult(sparse_dir=sparse_dir, images_dir=images_dir, ply_path=ply_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=True)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--overlap", type=int, default=5)
    args = parser.parse_args()

    result = run_poseinit(
        source_path=args.source_path,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        resolution=args.resolution,
        overlap=args.overlap,
    )
    print(f"sparse_dir={result.sparse_dir}")
    print(f"images_dir={result.images_dir}")
    print(f"ply_path={result.ply_path}")


if __name__ == "__main__":
    main()
