from pathlib import Path

from hloc import (
    extract_features,
    match_features,
    reconstruction,
    triangulation
)
import numpy as np
from typing import List, Dict, Tuple
import json
import os
import glob
import natsort

def normalize_path(p: str) -> str:
    return p.replace("\\", "/")

def parse_w2c_from_transform_entry(frame: Dict) -> np.ndarray:
    if "transform_matrix" not in frame:
        raise ValueError("transform_matrix is missing in transforms frame")
    c2w_gl = np.asarray(frame["transform_matrix"], dtype=np.float32)
    if c2w_gl.shape != (4, 4):
        raise ValueError(f"transform_matrix must be 4x4, got {c2w_gl.shape}")
    # OpenGL c2w -> OpenCV c2w
    c2w_cv = np.array(c2w_gl, copy=True)
    c2w_cv[:3, 1:3] *= -1
    # OpenCV c2w -> OpenCV w2c
    w2c = np.linalg.inv(c2w_cv).astype(np.float32)
    return w2c

def canonical_rel_path(p: str) -> str:
    p = normalize_path(p).strip()
    while p.startswith("./"):
        p = p[2:]
    return p

def parse_intrinsic_from_transform_entry(frame: Dict, root: Dict) -> np.ndarray:
    fl_x = frame.get("fl_x", root.get("fl_x"))
    fl_y = frame.get("fl_y", root.get("fl_y"))
    cx = frame.get("cx", root.get("cx"))
    cy = frame.get("cy", root.get("cy"))
    if any(x is None for x in [fl_x, fl_y, cx, cy]):
        raise ValueError("fl_x/fl_y/cx/cy is missing in transforms.json")
    K = np.eye(3, dtype=np.float32)
    K[0, 0], K[1, 1] = float(fl_x), float(fl_y)
    K[0, 2], K[1, 2] = float(cx), float(cy)
    return K

def build_transforms_lookup(frames: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    exact_lookup = dict()
    base_lookup = dict()
    base_duplicated = set()
    for frame in frames:
        if "file_path" not in frame:
            continue
        path = canonical_rel_path(frame["file_path"])
        exact_lookup[path] = frame
        base = os.path.basename(path)
        if base in base_lookup:
            base_duplicated.add(base)
        else:
            base_lookup[base] = frame
    for base in base_duplicated:
        base_lookup.pop(base, None)
    return exact_lookup, base_lookup

def build_depth_lookup(depth_root: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    depth_paths = natsort.natsorted(glob.glob(os.path.join(depth_root, "**", "*.png"), recursive=True))
    if len(depth_paths) == 0:
        raise FileNotFoundError(f"No depth png files found under {depth_root}")

    rel_lookup = dict()
    stem_lookup = dict()
    stem_duplicated = set()
    for dpath in depth_paths:
        rel = normalize_path(os.path.relpath(dpath, depth_root))
        rel_no_ext = os.path.splitext(rel)[0]
        rel_lookup[rel_no_ext] = dpath

        stem = os.path.splitext(os.path.basename(dpath))[0]
        if stem in stem_lookup:
            stem_duplicated.add(stem)
        else:
            stem_lookup[stem] = dpath

    for stem in stem_duplicated:
        stem_lookup.pop(stem, None)
    return rel_lookup, stem_lookup

def build_payload_from_transforms(
    rgb_paths: List[str],
    data_root: str,
    depth_root: str,
    transforms_json_path: str,
):
    with open(transforms_json_path, "r") as f:
        transforms_data = json.load(f)
    frames = transforms_data.get("frames", [])
    if len(frames) == 0:
        raise ValueError(f"No frames found in {transforms_json_path}")

    frame_exact, frame_base = build_transforms_lookup(frames)
    depth_rel_lookup, depth_stem_lookup = build_depth_lookup(depth_root)

    payload = []
    missing_prior = []
    missing_depth = []
    used_frames = set()

    for idx, rgb_path in enumerate(rgb_paths):
        rel_rgb = normalize_path(os.path.relpath(rgb_path, data_root))
        stem = os.path.splitext(os.path.basename(rel_rgb))[0]
        rel_no_ext = os.path.splitext(rel_rgb)[0]

        frame = None
        if rel_rgb in frame_exact:
            frame = frame_exact[rel_rgb]
        elif os.path.basename(rel_rgb) in frame_base:
            frame = frame_base[os.path.basename(rel_rgb)]

        if frame is None:
            missing_prior.append(rel_rgb)
            continue

        try:
            K = parse_intrinsic_from_transform_entry(frame, transforms_data)
            w2c = parse_w2c_from_transform_entry(frame)
        except Exception as e:
            missing_prior.append(f"{rel_rgb} ({str(e)})")
            continue

        depth_path = None
        if rel_no_ext in depth_rel_lookup:
            depth_path = depth_rel_lookup[rel_no_ext]
        elif stem in depth_stem_lookup:
            depth_path = depth_stem_lookup[stem]
        if depth_path is None:
            missing_depth.append(rel_rgb)
            continue

        payload.append(
            {
                "idx": idx,
                "rgb_path": rgb_path,
                "rgb_rel": rel_rgb,
                "file_path": canonical_rel_path(frame.get("file_path", rel_rgb)),
                "depth_path": depth_path,
                "K": K,
                "w2c": w2c,
            }
        )
        used_frames.add(id(frame))

    unused_prior = []
    for frame in frames:
        if id(frame) not in used_frames:
            fp = frame.get("file_path", "<missing file_path>")
            unused_prior.append(canonical_rel_path(fp) if isinstance(fp, str) else str(fp))

    report = {
        "n_rgb": len(rgb_paths),
        "n_payload": len(payload),
        "missing_prior": missing_prior,
        "missing_depth": missing_depth,
        "unused_prior": unused_prior,
    }
    return payload, report

def build_pairs_from_prior_poses(
    payload: List[Dict],
    overlap: int = 5,
    loop_Rt_thresh: Tuple[float, float] = (30.0, 2.0),
    near_Rt_min_thresh: Tuple[float, float] = (1.0, 0.05),
    max_loops_per_image: int = 5,
):
    payload_sorted = sorted(payload, key=lambda x: x["idx"])
    if len(payload_sorted) == 0:
        return []

    poses_w2c = np.stack([x["w2c"] for x in payload_sorted], axis=0).astype(np.float32)
    indices = [int(x["idx"]) for x in payload_sorted]

    R_w2c = poses_w2c[:, :3, :3]
    t_w2c = poses_w2c[:, :3, 3]
    R_c2w = np.transpose(R_w2c, (0, 2, 1))
    t_c2w = -(R_c2w @ t_w2c[:, :, None])[:, :, 0]

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
    nfrm = len(indices)

    for i in range(nfrm - 1):
        for j in range(i + 1, min(i + overlap + 1, nfrm)):
            if dR[i, j] < R_near_min and dt[i, j] < t_near_min:
                continue
            key = (indices[i], indices[j])
            if key not in added:
                pairs.append(key)
                added.add(key)

    for i in range(nfrm):
        start = i + overlap + 1
        if start >= nfrm:
            continue
        cand_idx = np.arange(start, nfrm)
        valid = (dR[i, cand_idx] < R_loop_max) & (dt[i, cand_idx] < t_loop_max)
        not_too_near = ~((dR[i, cand_idx] < R_near_min) & (dt[i, cand_idx] < t_near_min))
        valid &= not_too_near
        if not np.any(valid):
            continue

        vc = cand_idx[valid]
        order = np.lexsort((dR[i, vc], dt[i, vc]))
        vc = vc[order][:max_loops_per_image]

        for j in vc:
            key = (indices[i], indices[j])
            if key not in added:
                pairs.append(key)
                added.add(key)

    if len(pairs) == 0:
        raise RuntimeError("No valid pose-based pairs were generated from prior poses.")
    return pairs

def write_selected_pairs(dst_perscene: str, pairs: List[Tuple[int, int]]) -> str:
    pairs_path = os.path.join(dst_perscene, "selected_pairs.txt")
    with open(pairs_path, "w") as f:
        for idx1, idx2 in pairs:
            f.write(f"{idx1} {idx2}\n")
    return pairs_path

def create_images_from_pose_dict():
    """
    Write R T to colmap images.bin
    """
    pass


transforms_json_path = ""
images = Path("")
outputs = Path("outputs/sfm/")
sfm_pairs = outputs / "pairs-netvlad.txt"
sfm_dir = outputs / "sfm_superpoint+superglue"


payload, report = build_payload_from_transforms(
            rgb_paths=images,
            data_root="",
            depth_root="",
            transforms_json_path=transforms_json_path,
        )
selected_pairs = build_pairs_from_prior_poses(payload=payload)
pairs_path = write_selected_pairs(dst_perscene="", pairs=selected_pairs)


retrieval_conf = extract_features.confs["netvlad"]
feature_conf = extract_features.confs["superpoint_aachen"]
matcher_conf = match_features.confs["loma"]

# write pose to colmap images.bin
create_images_from_pose_dict()


# pair generation from retrieval
retrieval_path = extract_features.main(retrieval_conf, images, outputs)

# Feature extraction
feature_path = extract_features.main(feature_conf, images, outputs)
match_path = match_features.main(
    matcher_conf, sfm_pairs, feature_conf["output"], outputs
)

# SfM reconstruction
# model = reconstruction.main(sfm_dir, images, sfm_pairs, feature_path, match_path)


# triangulation ?

triangulation.main(outputs, Path(f'{outputs}/model'), images,
                            sfm_pairs, feature_path, match_path, skip_geometric_verification=True)