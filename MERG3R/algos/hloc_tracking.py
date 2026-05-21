import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from algos.tracking_metrics import print_tracking_metrics
from algos.utils import get_sim_matrix


_REPO_ROOT = Path(__file__).resolve().parents[2]
_HLOC_ROOT = _REPO_ROOT / "Hierarchical-Localization"
_LOMA_ROOT = _REPO_ROOT / "LoMa" / "src"


def _ensure_hloc_on_path():
    if not _HLOC_ROOT.is_dir():
        raise FileNotFoundError(f"HLoc directory not found: {_HLOC_ROOT}")
    for path in (_REPO_ROOT, _HLOC_ROOT, _LOMA_ROOT):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _graph_pairs_from_similarity(images, k: int):
    sim_matrix = get_sim_matrix(images)
    num_images = images.shape[0]
    if num_images < 2:
        return []

    effective_k = min(k, num_images - 1)
    pairs = []
    seen_pairs = set()
    for i in range(num_images):
        sim_row = sim_matrix[i].clone()
        if num_images - i - 1 >= effective_k:
            indices = torch.arange(0, num_images, device=sim_row.device)
            sim_row[indices <= i] = -1
        else:
            sim_row[i] = -1

        top_k_neighbors = torch.topk(sim_row, effective_k)
        for n in top_k_neighbors.indices:
            j = int(n.item())
            if i == j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            pairs.append((i, j))
    return pairs


def _normalize_extrinsics(extrinsic):
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected extrinsic shape (N, 3, 4) or (N, 4, 4), got {extrinsic.shape}")
    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic[:, :3, :4]
    return extrinsic


def _mean_pinhole_intrinsics(intrinsic):
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.ndim != 3 or intrinsic.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsic shape (N, 3, 3), got {intrinsic.shape}")
    return {
        "fx": float(np.mean(intrinsic[:, 0, 0])),
        "fy": float(np.mean(intrinsic[:, 1, 1])),
        "cx": float(np.mean(intrinsic[:, 0, 2])),
        "cy": float(np.mean(intrinsic[:, 1, 2])),
    }


def _write_images(images, images_dir: Path):
    images_dir.mkdir(parents=True, exist_ok=True)
    image_names = []
    images_cpu = images.detach().cpu().float().clamp(0, 1)
    for idx, image in enumerate(images_cpu):
        name = f"frame_{idx:06d}.png"
        image_np = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        PILImage.fromarray(image_np).save(images_dir / name)
        image_names.append(name)
    return image_names


def _write_reference_model(model_dir: Path, image_names, image_hw, extrinsic, intrinsic):
    _ensure_hloc_on_path()
    from third_party.poseinit_hloc.colmap_io import (
        ColmapCamera,
        Image,
        rotmat2qvec,
        write_cameras_binary,
        write_images_binary,
        write_points3D_binary,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    height, width = image_hw
    intr = _mean_pinhole_intrinsics(intrinsic)
    cameras = {
        1: ColmapCamera(
            id=1,
            model="PINHOLE",
            width=int(width),
            height=int(height),
            params=np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"]], dtype=np.float64),
        )
    }

    extrinsic = _normalize_extrinsics(extrinsic)
    colmap_images = {}
    for idx, name in enumerate(image_names, start=1):
        R = extrinsic[idx - 1, :3, :3]
        t = extrinsic[idx - 1, :3, 3]
        colmap_images[idx] = Image(
            id=idx,
            qvec=rotmat2qvec(R),
            tvec=t,
            camera_id=1,
            name=name,
            xys=[],
            point3D_ids=[],
        )

    write_cameras_binary(cameras, model_dir / "cameras.bin")
    write_images_binary(colmap_images, model_dir / "images.bin")
    write_points3D_binary({}, model_dir / "points3D.bin")


def _write_pairs(pairs, image_names, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(f"{image_names[i]} {image_names[j]}" for i, j in pairs))


def _run_hloc_triangulation(
    workspace_dir: Path,
    images_dir: Path,
    pairs_path: Path,
    feature_conf_name: str,
    matcher_conf_name: str,
    skip_geometric_verification: bool,
):
    _ensure_hloc_on_path()
    from hloc import extract_features, match_features, triangulation

    model_dir = workspace_dir / "model"
    sparse_dir = workspace_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    feature_conf = extract_features.confs[feature_conf_name]
    matcher_conf = match_features.confs[matcher_conf_name]
    features = extract_features.main(
        feature_conf,
        images_dir,
        feature_path=model_dir / "features.h5",
        as_half=False,
        overwrite=True,
    )
    matches = match_features.main(
        matcher_conf,
        pairs_path,
        features=features,
        matches=model_dir / "matches.h5",
        overwrite=True,
    )
    triangulation.main(
        sparse_dir,
        model_dir,
        images_dir,
        pairs_path,
        features,
        matches,
        skip_geometric_verification=skip_geometric_verification,
    )
    return sparse_dir


def _read_hloc_tracks(sparse_dir: Path, num_images: int):
    _ensure_hloc_on_path()
    from third_party.poseinit_hloc.colmap_io import read_images_binary, read_points3D_binary

    images_by_id = read_images_binary(sparse_dir / "images.bin")
    points3d_by_id = read_points3D_binary(sparse_dir / "points3D.bin")
    image_id_to_frame = {
        image_id: int(Path(image.name).stem.split("_")[-1])
        for image_id, image in images_by_id.items()
    }

    final_track = [[] for _ in range(num_images)]
    points_id = [[] for _ in range(num_images)]
    points_3d = []
    points_conf = []

    for point in points3d_by_id.values():
        curr_point_id = len(points_3d)
        observations = []
        seen_frames = set()
        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
            frame_idx = image_id_to_frame.get(int(image_id))
            if frame_idx is None or frame_idx < 0 or frame_idx >= num_images:
                continue
            if frame_idx in seen_frames:
                continue
            image = images_by_id[int(image_id)]
            point2d_idx = int(point2d_idx)
            if point2d_idx < 0 or point2d_idx >= len(image.xys):
                continue
            xy = np.asarray(image.xys[point2d_idx], dtype=np.float32)
            if not np.isfinite(xy).all():
                continue
            seen_frames.add(frame_idx)
            observations.append((frame_idx, xy))

        if len(observations) < 2 or not np.isfinite(point.xyz).all():
            continue

        for frame_idx, xy in observations:
            final_track[frame_idx].append(xy)
            points_id[frame_idx].append(curr_point_id)

        track_len = len(observations)
        reproj_error = float(point.error) if np.isfinite(point.error) else 1e6
        conf = min(track_len, 10) / 10.0 / (1.0 + max(reproj_error, 0.0))
        points_3d.append(np.asarray(point.xyz, dtype=np.float32))
        points_conf.append(conf)

    final_track = [np.stack(track).astype(np.float32) if track else np.array([]) for track in final_track]
    points_id = [np.stack(idx).astype(np.int64) if idx else np.array([]) for idx in points_id]

    if not points_3d:
        return (
            final_track,
            points_id,
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    return (
        final_track,
        points_id,
        np.stack(points_3d).astype(np.float32),
        np.asarray(points_conf, dtype=np.float32),
    )


@torch.no_grad()
def graph_extract_matches_hloc(
    images,
    extrinsic,
    intrinsic,
    k=5,
    workspace_dir=None,
    feature_conf_name="loma_aachen",
    matcher_conf_name="loma",
    skip_geometric_verification=False,
    overwrite=True,
):
    """
    HLoc graph tracking for MERG3R.

    Uses MERG3R's DINO similarity graph for pair selection, shared PINHOLE
    intrinsics for a fixed-pose COLMAP reference model, HLoc for feature
    extraction/matching, and pycolmap triangulation to produce sparse tracks.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected images shape (N, C, H, W), got {tuple(images.shape)}")
    if images.shape[0] < 2:
        raise ValueError("HLoc tracking requires at least two images")

    pairs = _graph_pairs_from_similarity(images, k=k)
    if not pairs:
        raise ValueError("No image pairs were generated for HLoc tracking")

    owns_workspace = workspace_dir is None
    tmp_ctx = tempfile.TemporaryDirectory(prefix="merg3r_hloc_tracking_") if owns_workspace else None
    workspace = Path(tmp_ctx.name if owns_workspace else workspace_dir)
    try:
        if overwrite and workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        images_dir = workspace / "images"
        model_dir = workspace / "model"
        pairs_path = workspace / "pairs-sfm.txt"

        image_names = _write_images(images, images_dir)
        _write_reference_model(model_dir, image_names, images.shape[-2:], extrinsic, intrinsic)
        _write_pairs(pairs, image_names, pairs_path)

        sparse_dir = _run_hloc_triangulation(
            workspace,
            images_dir,
            pairs_path,
            feature_conf_name=feature_conf_name,
            matcher_conf_name=matcher_conf_name,
            skip_geometric_verification=skip_geometric_verification,
        )
        result = _read_hloc_tracks(sparse_dir, num_images=images.shape[0])
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    print("End HLoc tracking.")
    print("Num of HLoc graph pairs: ", len(pairs))
    print("Num of HLoc tracks: ", result[2].shape[0])
    print_tracking_metrics(
        "HLocGraph",
        result[0],
        result[1],
        result[2],
        extrinsic,
        intrinsic,
        points_conf=result[3],
        extra_stats={
            "pair_count": len(pairs),
            "feature_conf": feature_conf_name,
            "matcher_conf": matcher_conf_name,
            "skip_geometric_verification": skip_geometric_verification,
            "workspace_dir": workspace,
        },
    )
    return result
