import json
import math
import struct
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np

from vggt_slam.pi3_solver import rotation_angle_degrees


ColmapImage = namedtuple("ColmapImage", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
ColmapPoint3D = namedtuple("ColmapPoint3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_colmap_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = _read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = props[8]
            image_name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]
            num_points2d = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            elems = _read_next_bytes(fid, num_bytes=24 * num_points2d, format_char_sequence="ddq" * num_points2d)
            xys = np.column_stack([tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))])
            point3d_ids = np.array(tuple(map(int, elems[2::3])), dtype=np.int64)
            images[image_id] = ColmapImage(image_id, qvec, tvec, camera_id, image_name, xys, point3d_ids)
    return images


def read_colmap_points3d_binary(path):
    points3d = {}
    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3d_id = props[0]
            xyz = np.array(props[1:4], dtype=np.float64)
            rgb = np.array(props[4:7], dtype=np.uint8)
            error = float(props[7])
            track_length = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = _read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])), dtype=np.int64)
            point2d_idxs = np.array(tuple(map(int, track_elems[1::2])), dtype=np.int64)
            points3d[point3d_id] = ColmapPoint3D(point3d_id, xyz, rgb, error, image_ids, point2d_idxs)
    return points3d


def load_lingbot_poses(transforms_json_path, image_names):
    with open(transforms_json_path, "r", encoding="utf-8") as f:
        transforms_data = json.load(f)

    pose_by_name = {}
    for index, frame in enumerate(transforms_data.get("frames", [])):
        file_path = frame.get("file_path")
        transform_matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float32)
        if file_path is None or transform_matrix.shape != (4, 4):
            continue
        c2w_opencv = transform_matrix.copy()
        c2w_opencv[:3, 1:3] *= -1.0
        basename = Path(file_path).name
        if basename in pose_by_name:
            raise ValueError(f"Duplicate basename in LingBot transforms.json: {basename}")
        pose_by_name[basename] = {
            "transforms_index": index,
            "c2w": c2w_opencv,
        }

    records = []
    missing = []
    seen = set()
    for frame_id, image_name in enumerate(image_names):
        basename = Path(image_name).name
        if basename in seen:
            raise ValueError(f"Duplicate input image basename: {basename}")
        seen.add(basename)
        pose = pose_by_name.get(basename)
        if pose is None:
            missing.append(basename)
            continue
        records.append(
            {
                "frame_id": frame_id,
                "image_path": str(image_name),
                "image_name": basename,
                "c2w": pose["c2w"],
                "transforms_index": pose["transforms_index"],
            }
        )

    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} images are missing from LingBot transforms.json: {preview}")
    return records


def _pose_distance(c2w_a, c2w_b, translation_scale):
    translation = np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3])
    rotation = rotation_angle_degrees(c2w_a[:3, :3].T @ c2w_b[:3, :3])
    return float(translation / translation_scale + rotation / 30.0)


def compute_pose_distance_matrix(poses):
    num_frames = len(poses)
    centers = np.stack([pose[:3, 3] for pose in poses], axis=0)
    if num_frames > 1:
        step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        translation_scale = float(np.median(step[step > 1e-6])) if np.any(step > 1e-6) else 1.0
    else:
        translation_scale = 1.0
    translation_scale = max(translation_scale, 1e-3)

    dists = np.zeros((num_frames, num_frames), dtype=np.float32)
    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            dist = _pose_distance(poses[i], poses[j], translation_scale)
            dists[i, j] = dist
            dists[j, i] = dist
    return dists, translation_scale


def load_hloc_observation_stats(sparse_dir, image_names, features_h5=None):
    sparse_dir = Path(sparse_dir)
    images_path = sparse_dir / "images.bin"
    points_path = sparse_dir / "points3D.bin"
    if not images_path.exists():
        raise FileNotFoundError(f"HLoc sparse images.bin not found: {images_path}")
    if not points_path.exists():
        raise FileNotFoundError(f"HLoc sparse points3D.bin not found: {points_path}")

    colmap_images = read_colmap_images_binary(images_path)
    colmap_points = read_colmap_points3d_binary(points_path)
    basename_to_image = {}
    for image in colmap_images.values():
        basename = Path(image.name).name
        if basename in basename_to_image:
            raise ValueError(f"Duplicate image basename in COLMAP images.bin: {basename}")
        basename_to_image[basename] = image

    id_to_basename = {image.id: Path(image.name).name for image in colmap_images.values()}
    connected_neighbors = defaultdict(set)
    covisible_observations = defaultdict(int)
    for point in colmap_points.values():
        basenames = [id_to_basename[image_id] for image_id in point.image_ids if image_id in id_to_basename]
        for basename in basenames:
            covisible_observations[basename] += max(0, len(basenames) - 1)
            for other in basenames:
                if other != basename:
                    connected_neighbors[basename].add(other)

    feature_counts = {}
    if features_h5 is not None:
        feature_path = Path(features_h5)
        if feature_path.exists():
            try:
                import h5py

                with h5py.File(str(feature_path), "r") as fd:
                    for key in fd.keys():
                        basename = Path(key).name
                        if "keypoints" in fd[key]:
                            feature_counts[basename] = int(fd[key]["keypoints"].shape[0])
            except ImportError:
                print("h5py is not available; skipping feature_count stats.")

    stats = []
    missing_sparse = []
    for frame_id, image_name in enumerate(image_names):
        basename = Path(image_name).name
        colmap_image = basename_to_image.get(basename)
        if colmap_image is None:
            missing_sparse.append(basename)
            observation_count = 0
            total_points2d = 0
        else:
            valid = colmap_image.point3D_ids != -1
            observation_count = int(valid.sum())
            total_points2d = int(colmap_image.point3D_ids.shape[0])

        stats.append(
            {
                "frame_id": frame_id,
                "image_name": basename,
                "image_path": str(image_name),
                "observation_count": observation_count,
                "total_points2d": total_points2d,
                "feature_count": int(feature_counts.get(basename, 0)),
                "connected_neighbor_count": int(len(connected_neighbors.get(basename, set()))),
                "covisible_observation_count": int(covisible_observations.get(basename, 0)),
            }
        )

    if missing_sparse:
        print(
            "HLoc sparse model is missing some input images; treating them as zero-observation frames:",
            missing_sparse[:10],
        )
    return stats


def choose_keyframes(
    pose_records,
    observation_stats,
    keyframe_ratio=0.2,
    min_observation_quantile=0.25,
    max_temporal_gap=40,
):
    num_frames = len(pose_records)
    if num_frames == 0:
        raise ValueError("No frames available for keyframe selection.")

    poses = [record["c2w"] for record in pose_records]
    pose_dists, translation_scale = compute_pose_distance_matrix(poses)
    observations = np.array([stat["observation_count"] for stat in observation_stats], dtype=np.float32)
    connected = np.array([stat["connected_neighbor_count"] for stat in observation_stats], dtype=np.float32)

    positive_obs = observations[observations > 0]
    min_observations = float(np.quantile(positive_obs, min_observation_quantile)) if positive_obs.size else 0.0
    eligible = (observations >= min_observations) & (connected > 0)
    if not eligible.any():
        eligible[:] = True

    obs_norm = observations / max(float(observations.max()), 1.0)
    conn_norm = connected / max(float(connected.max()), 1.0)
    quality = 0.75 * obs_norm + 0.25 * conn_norm

    target_count = int(math.ceil(num_frames * keyframe_ratio))
    target_count = max(2 if num_frames > 1 else 1, min(num_frames, target_count))
    selected = {0}
    if num_frames > 1:
        selected.add(num_frames - 1)

    while len(selected) < target_count:
        selected_list = sorted(selected)
        novelty = pose_dists[:, selected_list].min(axis=1)
        novelty_norm = novelty / max(float(novelty.max()), 1e-6)
        score = novelty_norm * (0.35 + 0.65 * quality)
        score[list(selected)] = -1.0
        score[~eligible] *= 0.25
        next_id = int(np.argmax(score))
        if score[next_id] < 0:
            break
        selected.add(next_id)

    if max_temporal_gap and max_temporal_gap > 0:
        changed = True
        while changed:
            changed = False
            sorted_selected = sorted(selected)
            for left, right in zip(sorted_selected[:-1], sorted_selected[1:]):
                if right - left <= max_temporal_gap:
                    continue
                segment = list(range(left + 1, right))
                if not segment:
                    continue
                midpoint = (left + right) * 0.5
                segment_scores = []
                for idx in segment:
                    temporal_center = 1.0 - abs(idx - midpoint) / max((right - left) * 0.5, 1.0)
                    eligibility_bonus = 1.0 if eligible[idx] else 0.25
                    segment_scores.append((quality[idx] * eligibility_bonus + 0.25 * temporal_center, idx))
                selected.add(max(segment_scores)[1])
                changed = True
                break

    keyframes = []
    for frame_id in sorted(selected):
        reasons = []
        if frame_id == 0:
            reasons.append("first_frame")
        if frame_id == num_frames - 1:
            reasons.append("last_frame")
        if observations[frame_id] >= min_observations:
            reasons.append("hloc_observations")
        reasons.append("pose_coverage")
        keyframes.append(
            {
                "frame_id": int(frame_id),
                "image_name": pose_records[frame_id]["image_name"],
                "image_path": pose_records[frame_id]["image_path"],
                "observation_count": int(observations[frame_id]),
                "connected_neighbor_count": int(connected[frame_id]),
                "quality_score": float(quality[frame_id]),
                "reasons": reasons,
            }
        )

    return {
        "keyframes": keyframes,
        "parameters": {
            "keyframe_ratio": float(keyframe_ratio),
            "min_observation_quantile": float(min_observation_quantile),
            "min_observations": float(min_observations),
            "max_temporal_gap": int(max_temporal_gap),
            "translation_scale": float(translation_scale),
        },
    }


def _nearest_keyframes_by_pose(frame_ids, keyframe_ids, pose_dists, limit, exclude=None):
    exclude = set(exclude or [])
    if not keyframe_ids or limit <= 0:
        return []
    center = int(round(float(np.mean(frame_ids)))) if frame_ids else keyframe_ids[0]
    ranked = []
    for keyframe_id in keyframe_ids:
        if keyframe_id in exclude:
            continue
        ranked.append((float(pose_dists[center, keyframe_id]), keyframe_id))
    ranked.sort()
    return [idx for _, idx in ranked[:limit]]


def _select_rolling_shared_anchors(previous_anchor_ids, target_frame_ids, pose_dists, limit):
    if not previous_anchor_ids or limit <= 0:
        return []
    target_start = min(target_frame_ids)
    target_center = int(round(float(np.mean(target_frame_ids))))
    ranked = []
    for keyframe_id in previous_anchor_ids:
        temporal_gap = max(0, target_start - keyframe_id)
        pose_gap = float(pose_dists[target_center, keyframe_id])
        # Prefer anchors close to the new batch boundary; pose distance breaks ties.
        ranked.append((temporal_gap, pose_gap, -keyframe_id, keyframe_id))
    ranked.sort()
    return [item[-1] for item in ranked[:limit]]


def build_local_batches(
    image_names,
    pose_records,
    keyframe_plan,
    submap_size=32,
    anchor_count=4,
    min_shared_anchors=2,
):
    num_frames = len(image_names)
    keyframe_ids = sorted(int(kf["frame_id"]) for kf in keyframe_plan["keyframes"])
    poses = [record["c2w"] for record in pose_records]
    pose_dists, _ = compute_pose_distance_matrix(poses)

    anchor_count = max(0, min(anchor_count, max(submap_size - 1, 0)))
    target_count = max(1, submap_size - anchor_count)

    batches = []
    cursor = 0
    previous_anchor_ids = []
    batch_id = 0
    while cursor < num_frames:
        target_frame_ids = list(range(cursor, min(num_frames, cursor + target_count)))
        target_set = set(target_frame_ids)
        target_keyframes = [idx for idx in target_frame_ids if idx in keyframe_ids]

        anchors = _select_rolling_shared_anchors(
            previous_anchor_ids,
            target_frame_ids,
            pose_dists,
            limit=min_shared_anchors,
        )

        progressive_keyframes = [idx for idx in keyframe_ids if idx <= target_frame_ids[-1]]
        preferred = target_keyframes + _nearest_keyframes_by_pose(
            target_frame_ids,
            progressive_keyframes,
            pose_dists,
            limit=anchor_count,
            exclude=anchors,
        )
        for idx in preferred:
            if idx not in anchors:
                anchors.append(idx)
            if len(anchors) >= anchor_count:
                break

        frame_ids = sorted(set(target_frame_ids + anchors))
        if len(frame_ids) > submap_size:
            removable = [idx for idx in frame_ids if idx in anchors and idx not in target_set]
            while len(frame_ids) > submap_size and removable:
                frame_ids.remove(removable.pop())
                anchors = [idx for idx in anchors if idx in frame_ids]
            if len(frame_ids) > submap_size:
                frame_ids = frame_ids[:submap_size]
                anchors = [idx for idx in anchors if idx in frame_ids]
                target_frame_ids = [idx for idx in target_frame_ids if idx in frame_ids]

        anchor_keyframes = [idx for idx in anchors if idx in frame_ids]
        batch = {
            "batch_id": batch_id,
            "frame_ids": frame_ids,
            "image_names": [Path(image_names[idx]).name for idx in frame_ids],
            "image_paths": [str(image_names[idx]) for idx in frame_ids],
            "anchor_keyframes": anchor_keyframes,
            "target_keyframes": [idx for idx in target_frame_ids if idx in keyframe_ids],
            "non_keyframes": [idx for idx in target_frame_ids if idx not in keyframe_ids],
            "target_frame_ids": target_frame_ids,
            "shared_with_previous": sorted(set(frame_ids).intersection(previous_anchor_ids)) if batches else [],
        }
        batches.append(batch)

        previous_anchor_ids = anchor_keyframes if anchor_keyframes else [idx for idx in frame_ids if idx in keyframe_ids]
        cursor += target_count
        batch_id += 1

    return {
        "batches": batches,
        "parameters": {
            "submap_size": int(submap_size),
            "anchor_count": int(anchor_count),
            "min_shared_anchors": int(min_shared_anchors),
            "target_count": int(target_count),
        },
    }


def generate_offline_plan(
    image_names,
    lingbot_transforms_json,
    hloc_sparse_dir,
    output_dir,
    hloc_features_h5=None,
    keyframe_ratio=0.2,
    min_observation_quantile=0.25,
    max_temporal_gap=40,
    submap_size=32,
    anchor_count=4,
    min_shared_anchors=2,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_records = load_lingbot_poses(lingbot_transforms_json, image_names)
    observation_stats = load_hloc_observation_stats(hloc_sparse_dir, image_names, features_h5=hloc_features_h5)
    keyframe_plan = choose_keyframes(
        pose_records,
        observation_stats,
        keyframe_ratio=keyframe_ratio,
        min_observation_quantile=min_observation_quantile,
        max_temporal_gap=max_temporal_gap,
    )
    batch_plan = build_local_batches(
        image_names,
        pose_records,
        keyframe_plan,
        submap_size=submap_size,
        anchor_count=anchor_count,
        min_shared_anchors=min_shared_anchors,
    )

    image_index = [
        {
            "frame_id": record["frame_id"],
            "image_name": record["image_name"],
            "image_path": record["image_path"],
            "lingbot_transforms_index": record["transforms_index"],
        }
        for record in pose_records
    ]

    outputs = {
        "image_index": output_dir / "image_index.json",
        "observation_stats": output_dir / "image_observation_stats.json",
        "keyframes": output_dir / "keyframes.json",
        "local_batches": output_dir / "local_batches.json",
    }
    with open(outputs["image_index"], "w", encoding="utf-8") as f:
        json.dump({"images": image_index}, f, indent=2)
    with open(outputs["observation_stats"], "w", encoding="utf-8") as f:
        json.dump({"frames": observation_stats}, f, indent=2)
    with open(outputs["keyframes"], "w", encoding="utf-8") as f:
        json.dump(keyframe_plan, f, indent=2)
    with open(outputs["local_batches"], "w", encoding="utf-8") as f:
        json.dump(batch_plan, f, indent=2)

    return outputs


def load_local_batch_plan(plan_json, image_names):
    with open(plan_json, "r", encoding="utf-8") as f:
        plan = json.load(f)
    if "batches" not in plan:
        raise ValueError(f"Submap plan {plan_json} does not contain a 'batches' field.")

    basename_to_path = {}
    for index, image_name in enumerate(image_names):
        basename = Path(image_name).name
        if basename in basename_to_path:
            raise ValueError(f"Duplicate input image basename: {basename}")
        basename_to_path[basename] = (index, image_name)

    batches = []
    for batch in plan["batches"]:
        if "frame_ids" in batch:
            frame_ids = [int(i) for i in batch["frame_ids"]]
            paths = [image_names[i] for i in frame_ids]
        elif "image_names" in batch:
            frame_ids = []
            paths = []
            for name in batch["image_names"]:
                basename = Path(name).name
                if basename not in basename_to_path:
                    raise ValueError(f"Plan image {basename} is not present in --image_folder.")
                frame_id, path = basename_to_path[basename]
                frame_ids.append(frame_id)
                paths.append(path)
        else:
            raise ValueError(f"Batch {batch.get('batch_id')} needs frame_ids or image_names.")

        metadata = dict(batch)
        metadata["frame_ids"] = frame_ids
        metadata["image_paths"] = [str(path) for path in paths]
        metadata["image_names"] = [Path(path).name for path in paths]
        batches.append((paths, metadata))
    return batches
