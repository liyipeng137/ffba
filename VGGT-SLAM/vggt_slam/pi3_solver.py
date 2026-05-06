import sys
import time
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from termcolor import colored
from torchvision.transforms.functional import to_tensor

# from vggt.utils.geometry import closed_form_inverse_se3

from vggt_slam.scale_solver import estimate_scale_pairwise
from vggt_slam.solver import Solver
from vggt_slam.submap import Submap


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def _ensure_pi3_import_path():
    pi3_root = Path(__file__).resolve().parents[1] / "Pi3"
    if not pi3_root.exists():
        raise FileNotFoundError(f"Pi3 directory not found at {pi3_root}")

    pi3_root_str = str(pi3_root)
    if pi3_root_str not in sys.path:
        sys.path.insert(0, pi3_root_str)


def load_pi3x_model(device, ckpt_path=None):
    _ensure_pi3_import_path()
    from pi3.models.pi3x import Pi3X

    if ckpt_path:
        model = Pi3X().to(device).eval()
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            weights = load_file(ckpt_path)
        else:
            weights = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(weights, strict=False)
        return model

    return Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()


def estimate_similarity_transform(source_points, target_points):
    """
    Estimate a Sim(3) transform T such that:
        target ~= T * source

    Returns:
        transform_4x4: (4, 4)
        scale: float
        rotation: (3, 3)
        translation: (3,)
    """
    if source_points.shape != target_points.shape:
        raise ValueError(
            "Source and target point sets must have the same shape. "
            f"Got {source_points.shape} vs {target_points.shape}."
        )
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 point clouds, got {source_points.shape}.")
    if source_points.shape[0] < 3:
        raise ValueError("Need at least 3 points to estimate a similarity transform.")

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)

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

    source_var = np.mean(np.sum(source_centered ** 2, axis=1))
    if source_var < 1e-12:
        raise ValueError("Source point cloud variance is too small for similarity estimation.")

    scale = np.sum(singular_values * np.diag(correction)) / source_var
    translation = target_mean - scale * (rotation @ source_mean)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation

    return transform.astype(np.float32), float(scale), rotation.astype(np.float32), translation.astype(np.float32)


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float32)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    transformed = (transform @ points_h.T).T
    return transformed[:, :3] / np.clip(transformed[:, 3:], 1e-8, None)


def rotation_angle_degrees(rotation):
    trace_value = np.trace(rotation)
    cos_theta = np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


class Pi3Solver(Solver):
    """Solver variant that consumes Pi3X local point maps instead of VGGT depth."""

    def __init__(
        self,
        init_conf_threshold: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        lc_thres: float = 0.80,
        vis_voxel_size: float = None,
        lingbot_transforms_json: str = None,
        loop_window_radius: int = 2,
        min_loop_frame_gap: int = 32,
        loop_translation_thresh: float = 1.0,
        loop_rotation_thresh_deg: float = 30.0,
        min_shared_anchors: int = 2,
    ):
        super().__init__(
            init_conf_threshold=init_conf_threshold,
            lc_thres=lc_thres,
            vis_voxel_size=vis_voxel_size,
        )
        self.shared_intrinsic = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.loop_window_radius = int(loop_window_radius)
        self.min_loop_frame_gap = int(min_loop_frame_gap)
        self.loop_translation_thresh = float(loop_translation_thresh)
        self.loop_rotation_thresh_deg = float(loop_rotation_thresh_deg)
        self.min_shared_anchors = int(min_shared_anchors)
        self.lingbot_pose_db = self._load_lingbot_prior_poses(lingbot_transforms_json)

    def _load_original_images(self, image_names):
        images = []
        image_size = None
        for image_name in image_names:
            image = Image.open(image_name).convert("RGB")
            width, height = image.size

            if image_size is None:
                image_size = (height, width)
            elif image_size != (height, width):
                raise ValueError(
                    "Pi3Solver expects all images in a submap to have the same original size. "
                    f"Got {height}x{width} for {image_name}, expected {image_size[0]}x{image_size[1]}."
                )

            if height % 14 != 0 or width % 14 != 0:
                raise ValueError(
                    "Pi3X requires original image height and width to be multiples of 14. "
                    f"Got {height}x{width} for {image_name}."
                )

            images.append(to_tensor(image))

        return torch.stack(images, dim=0)

    def _intrinsics_for_frames(self, num_frames):
        return np.repeat(self.shared_intrinsic[None, ...], num_frames, axis=0)

    def _load_lingbot_prior_poses(self, transforms_json_path):
        if not transforms_json_path:
            return {}

        transforms_path = Path(transforms_json_path)
        if not transforms_path.exists():
            raise FileNotFoundError(f"LingBot transforms.json not found at {transforms_path}")

        with open(transforms_path, "r", encoding="utf-8") as f:
            transforms_data = json.load(f)

        frames = transforms_data.get("frames", [])
        pose_db = {}
        for index, frame in enumerate(frames):
            file_path = frame.get("file_path")
            transform_matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float32)
            if file_path is None or transform_matrix.shape != (4, 4):
                continue

            basename = Path(file_path).name
            c2w_opencv = transform_matrix.copy()
            c2w_opencv[:3, 1:3] *= -1.0
            pose_db[basename] = {
                "global_index": index,
                "c2w": c2w_opencv,
                "w2c": np.linalg.inv(c2w_opencv).astype(np.float32),
            }

        print(
            "Loaded LingBot prior poses:",
            {
                "path": str(transforms_path),
                "num_frames": len(pose_db),
            },
        )
        return pose_db

    def _rotation_delta_degrees_from_c2w(self, c2w_a, c2w_b):
        relative_rotation = c2w_a[:3, :3].T @ c2w_b[:3, :3]
        return rotation_angle_degrees(relative_rotation)

    def _get_window_bounds(self, center_index, num_frames):
        start_index = max(0, center_index - self.loop_window_radius)
        end_index = min(num_frames, center_index + self.loop_window_radius + 1)
        return start_index, end_index

    def _run_pi3_inference_on_images(self, image_names, model):
        device = next(model.parameters()).device
        images = self._load_original_images(image_names).to(device)
        intrinsics_np = self._intrinsics_for_frames(images.shape[0])
        intrinsics = torch.from_numpy(intrinsics_np).float()[None].to(device)

        if device.type == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
            autocast_context = torch.amp.autocast("cuda", dtype=dtype)
        else:
            autocast_context = nullcontext()

        with torch.no_grad():
            with self.vggt_timer:
                with autocast_context:
                    predictions = model(
                        imgs=images[None],
                        intrinsics=intrinsics,
                        mask_add_ray=torch.ones((1, images.shape[0]), dtype=torch.bool, device=device),
                        mask_add_depth=torch.zeros((1, images.shape[0]), dtype=torch.bool, device=device),
                        mask_add_pose=torch.zeros((1, images.shape[0]), dtype=torch.bool, device=device),
                    )

        return {
            "images": images.detach().float().cpu().numpy(),
            "local_points": predictions["local_points"][0].detach().float().cpu().numpy(),
            "world_points_pi3": predictions["points"][0].detach().float().cpu().numpy(),
            "point_conf": torch.sigmoid(predictions["conf"][0, ..., 0]).detach().float().cpu().numpy(),
            "camera_poses": predictions["camera_poses"][0].detach().float().cpu().numpy(),
            "intrinsic": intrinsics_np,
        }

    def _find_loop_candidate(self, image_names):
        if not self.lingbot_pose_db:
            return None
        if self.map.get_largest_key(ignore_loop_closure_submaps=True) is None:
            return None

        best_candidate = None
        best_score = None

        for current_index, image_name in enumerate(image_names):
            current_basename = Path(image_name).name
            current_prior = self.lingbot_pose_db.get(current_basename)
            if current_prior is None:
                print(colored(f"Current prior {current_basename} not found in LingBot pose database, skipping...", "red", attrs=["bold"]))
                continue

            current_center = current_prior["c2w"][:3, 3]
            current_global_index = current_prior["global_index"]

            for submap in self.map.ordered_submaps_by_key():
                if submap.get_lc_status():
                    continue

                for prior_index, prior_image_name in enumerate(submap.img_names):
                    prior_basename = Path(prior_image_name).name
                    prior_prior = self.lingbot_pose_db.get(prior_basename)
                    if prior_prior is None:
                        continue

                    frame_gap = abs(current_global_index - prior_prior["global_index"])
                    if frame_gap < self.min_loop_frame_gap:
                        continue

                    prior_center = prior_prior["c2w"][:3, 3]
                    translation_distance = float(np.linalg.norm(current_center - prior_center))
                    if translation_distance > self.loop_translation_thresh:
                        continue

                    rotation_delta = self._rotation_delta_degrees_from_c2w(current_prior["c2w"], prior_prior["c2w"])
                    if rotation_delta > self.loop_rotation_thresh_deg:
                        continue

                    candidate_score = (translation_distance, rotation_delta, frame_gap)
                    if best_score is None or candidate_score < best_score:
                        best_score = candidate_score
                        best_candidate = {
                            "query_submap_id": int(self.current_working_submap.get_id()),
                            "query_submap_frame": int(current_index),
                            "detected_submap_id": int(submap.get_id()),
                            "detected_submap_frame": int(prior_index),
                            "current_global_index": int(current_global_index),
                            "detected_global_index": int(prior_prior["global_index"]),
                            "translation_distance": translation_distance,
                            "rotation_deg": float(rotation_delta),
                            "query_image_name": image_name,
                            "detected_image_name": prior_image_name,
                            "query_prior_c2w": current_prior["c2w"],
                            "detected_prior_c2w": prior_prior["c2w"],
                        }

        if best_candidate is not None:
            print(
                colored("Pi3 loop candidate", "cyan"),
                {
                    "query_frame": best_candidate["query_submap_frame"],
                    "detected_submap_id": best_candidate["detected_submap_id"],
                    "detected_frame": best_candidate["detected_submap_frame"],
                    "frame_gap": abs(
                        best_candidate["current_global_index"] - best_candidate["detected_global_index"]
                    ),
                    "translation_distance": best_candidate["translation_distance"],
                    "rotation_deg": best_candidate["rotation_deg"],
                    "query_image": Path(best_candidate["query_image_name"]).name,
                    "detected_image": Path(best_candidate["detected_image_name"]).name,

                },
            )
        else:
            print("Pi3 loop candidate: none")

        return best_candidate

    def _verify_loop_candidate(self, loop_candidate, current_image_names, model):
        prior_submap = self.map.get_submap(loop_candidate["detected_submap_id"])
        prior_start, prior_end = self._get_window_bounds(
            loop_candidate["detected_submap_frame"], len(prior_submap.img_names)
        )
        current_start, current_end = self._get_window_bounds(
            loop_candidate["query_submap_frame"], len(current_image_names)
        )

        prior_window = prior_submap.img_names[prior_start:prior_end]
        current_window = current_image_names[current_start:current_end]
        loop_window_image_names = list(prior_window) + list(current_window)
        prior_center_index = loop_candidate["detected_submap_frame"] - prior_start
        current_center_index = len(prior_window) + (loop_candidate["query_submap_frame"] - current_start)

        t1 = time.time()
        loop_predictions = self._run_pi3_inference_on_images(loop_window_image_names, model)
        print(f"Pi3 loop window inference took {time.time() - t1:.2f} seconds")

        loop_c2w = loop_predictions["camera_poses"]
        loop_w2c = closed_form_inverse_se3(loop_c2w)
        loop_relative_h = (loop_w2c[prior_center_index] @ loop_c2w[current_center_index]).astype(np.float32)

        loop_conf = loop_predictions["point_conf"]
        current_center_conf = float(loop_conf[current_center_index].mean())
        prior_center_conf = float(loop_conf[prior_center_index].mean())

        prior_relative_h = (
            np.linalg.inv(loop_candidate["detected_prior_c2w"]) @ loop_candidate["query_prior_c2w"]
        ).astype(np.float32)
        prior_rotation = rotation_angle_degrees(prior_relative_h[:3, :3])
        prior_translation = float(np.linalg.norm(prior_relative_h[:3, 3]))

        pi3_rotation = rotation_angle_degrees(loop_relative_h[:3, :3])
        pi3_translation = float(np.linalg.norm(loop_relative_h[:3, 3]))

        print(
            colored("Pi3 loop verification", "green"),
            {
                "query_frame": loop_candidate["query_submap_frame"],
                "detected_submap_id": loop_candidate["detected_submap_id"],
                "detected_frame": loop_candidate["detected_submap_frame"],
                "prior_window": [Path(p).name for p in prior_window],
                "current_window": [Path(p).name for p in current_window],
                "prior_center_conf": prior_center_conf,
                "current_center_conf": current_center_conf,
                "pi3_rotation_deg": pi3_rotation,
                "pi3_translation_norm": pi3_translation,
                "lingbot_rotation_deg": prior_rotation,
                "lingbot_translation_norm": prior_translation,
            },
        )

        verified_loop = {
            "query_submap_id": int(loop_candidate["query_submap_id"]),
            "query_submap_frame": int(loop_candidate["query_submap_frame"]),
            "detected_submap_id": int(loop_candidate["detected_submap_id"]),
            "detected_submap_frame": int(loop_candidate["detected_submap_frame"]),
            "relative_h": loop_relative_h,
            "prior_window_paths": list(prior_window),
            "current_window_paths": list(current_window),
        }
        return verified_loop

    def _compute_filtered_confidence(self, local_points, point_conf):
        _ensure_pi3_import_path()
        from pi3.utils.geometry import depth_edge

        local_depth = torch.from_numpy(local_points[..., 2]).float()
        non_edge_mask = (~depth_edge(local_depth, rtol=0.03)).cpu().numpy()
        filtered_conf = point_conf.copy()
        filtered_conf[~non_edge_mask] = 0.0

        print(
            "Pi3 confidence filtering:",
            {
                "total_pixels": int(point_conf.size),
                "non_edge_pixels": int(non_edge_mask.sum()),
                "edge_pixels": int((~non_edge_mask).sum()),
                "edge_ratio": float((~non_edge_mask).mean()),
            },
        )
        return filtered_conf.astype(np.float32), non_edge_mask

    def _infer_overlap_count(self, current_submap, prior_submap):
        current_img_names = [str(Path(p).name) for p in getattr(current_submap, "img_names", [])]
        prior_img_names = [str(Path(p).name) for p in getattr(prior_submap, "img_names", [])]

        if current_img_names and prior_img_names:
            max_overlap = min(len(current_img_names), len(prior_img_names))
            overlap_count = 0
            for count in range(max_overlap, 0, -1):
                current_prefix = current_img_names[:count]
                prior_suffix = prior_img_names[-count:]
                if current_prefix == prior_suffix:
                    overlap_count = count
                    break

            print(
                "Pi3 overlap detection by image names:",
                {
                    "current_prefix": current_img_names[: min(3, len(current_img_names))],
                    "prior_suffix": prior_img_names[-min(3, len(prior_img_names)) :],
                    "overlap_count": overlap_count,
                },
            )
            if overlap_count > 0:
                return overlap_count

        current_frame_ids = current_submap.get_frame_ids()
        prior_frame_ids = prior_submap.get_frame_ids()
        max_overlap = min(len(current_frame_ids), len(prior_frame_ids))

        overlap_count = 0
        for count in range(max_overlap, 0, -1):
            current_prefix = current_frame_ids[:count]
            prior_suffix = prior_frame_ids[-count:]
            if current_prefix == prior_suffix:
                overlap_count = count
                break

        print(
            "Pi3 overlap detection fallback by frame ids:",
            {
                "current_prefix": current_frame_ids[: min(3, len(current_frame_ids))],
                "prior_suffix": prior_frame_ids[-min(3, len(prior_frame_ids)) :],
                "overlap_count": overlap_count,
            },
        )
        return overlap_count

    def _prefix_overlap_pairs(self, current_submap, prior_submap, overlap_count):
        prior_last_index = prior_submap.get_last_non_loop_frame_index()
        prior_overlap_start = prior_last_index - overlap_count + 1
        return [
            {
                "current_index": offset,
                "prior_index": prior_overlap_start + offset,
                "global_frame_id": int(current_submap.get_global_frame_ids()[offset]),
                "source": "prefix_suffix_overlap",
            }
            for offset in range(overlap_count)
        ]

    def _shared_anchor_pairs(self, current_submap, prior_submap):
        shared_global_ids = current_submap.get_shared_global_frame_ids(prior_submap)
        pairs = []
        for global_frame_id in shared_global_ids:
            current_index = current_submap.get_local_index_for_global_frame_id(global_frame_id)
            prior_index = prior_submap.get_local_index_for_global_frame_id(global_frame_id)
            if current_index is None or prior_index is None:
                continue
            pairs.append(
                {
                    "current_index": int(current_index),
                    "prior_index": int(prior_index),
                    "global_frame_id": int(global_frame_id),
                    "source": "shared_anchor",
                }
            )
        return pairs

    def _find_alignment_prior_submap(self, current_submap, fallback_submap_id):
        if fallback_submap_id is None:
            return None, []

        best_submap = None
        best_pairs = []
        for candidate in self.map.ordered_submaps_by_key():
            if candidate.get_lc_status():
                continue
            if candidate.get_id() == current_submap.get_id():
                continue
            pairs = self._shared_anchor_pairs(current_submap, candidate)
            if len(pairs) > len(best_pairs):
                best_submap = candidate
                best_pairs = pairs

        if best_submap is not None and len(best_pairs) >= self.min_shared_anchors:
            print(
                "Pi3 shared-anchor prior selected:",
                {
                    "prior_submap_id": int(best_submap.get_id()),
                    "shared_count": int(len(best_pairs)),
                    "shared_global_frame_ids": [pair["global_frame_id"] for pair in best_pairs],
                },
            )
            return best_submap, best_pairs

        return self.map.get_submap(fallback_submap_id), []

    def _estimate_submap_alignment(self, current_submap, prior_submap, alignment_pairs):
        current_w2c = current_submap.get_all_poses()
        prior_w2c = prior_submap.get_all_poses()
        current_c2w = np.linalg.inv(current_w2c)
        prior_c2w = np.linalg.inv(prior_w2c)

        source_points_all = []
        target_points_all = []
        overlap_debug = []
        pose_diagnostics = []

        for pair in alignment_pairs:
            current_index = pair["current_index"]
            prior_index = pair["prior_index"]

            current_conf = current_submap.get_conf_masks_frame(current_index)
            prior_conf = prior_submap.get_conf_masks_frame(prior_index)
            good_mask = (prior_conf > prior_submap.get_conf_threshold()) & (
                current_conf > current_submap.get_conf_threshold()
            )
            good_mask = good_mask.reshape(-1)

            current_points_local = current_submap.get_frame_pointcloud(current_index).reshape(-1, 3)
            prior_points_local = prior_submap.get_frame_pointcloud(prior_index).reshape(-1, 3)

            finite_mask = np.isfinite(current_points_local).all(axis=1) & np.isfinite(prior_points_local).all(axis=1)
            norm_mask = (np.linalg.norm(current_points_local, axis=1) > 1e-8) & (
                np.linalg.norm(prior_points_local, axis=1) > 1e-8
            )
            good_mask = good_mask & finite_mask & norm_mask

            current_points_submap = transform_points(current_points_local[good_mask], current_c2w[current_index])
            prior_points_submap = transform_points(prior_points_local[good_mask], prior_c2w[prior_index])

            num_valid = current_points_submap.shape[0]
            overlap_debug.append(
                {
                    "current_index": current_index,
                    "prior_index": prior_index,
                    "global_frame_id": pair.get("global_frame_id"),
                    "source": pair.get("source"),
                    "valid_points": int(num_valid),
                }
            )

            if num_valid >= 3:
                source_points_all.append(current_points_submap)
                target_points_all.append(prior_points_submap)

                pose_transform = prior_c2w[prior_index] @ current_w2c[current_index]
                pose_rotation = pose_transform[:3, :3]
                pose_translation = pose_transform[:3, 3]
                pose_diagnostics.append(
                    {
                        "current_index": current_index,
                        "prior_index": prior_index,
                        "global_frame_id": pair.get("global_frame_id"),
                        "rotation_deg": rotation_angle_degrees(pose_rotation),
                        "translation_norm": float(np.linalg.norm(pose_translation)),
                    }
                )

        print("Pi3 overlap frame stats:", overlap_debug)
        if pose_diagnostics:
            print("Pi3 overlap pose diagnostics:", pose_diagnostics)

        if not source_points_all:
            raise ValueError("No overlap frame had at least 3 valid non-edge point correspondences.")

        source_points = np.concatenate(source_points_all, axis=0)
        target_points = np.concatenate(target_points_all, axis=0)

        try:
            transform, scale, rotation, translation = estimate_similarity_transform(source_points, target_points)
        except ValueError as exc:
            print(colored(f"Sim3 estimation failed ({exc}); falling back to scale-only in submap coordinates.", "red"))
            scale_factor_est_output = estimate_scale_pairwise(source_points, target_points)
            scale = float(scale_factor_est_output[0])
            rotation = np.eye(3, dtype=np.float32)
            translation = np.zeros(3, dtype=np.float32)
            transform = np.diag((scale, scale, scale, 1.0)).astype(np.float32)

        aligned_source = transform_points(source_points, transform)
        residuals = np.linalg.norm(aligned_source - target_points, axis=1)

        print(
            colored("Pi3 overlap Sim3", "green"),
            {
                "num_frames": len(alignment_pairs),
                "num_points": int(source_points.shape[0]),
                "scale": scale,
                "rotation_deg": rotation_angle_degrees(rotation),
                "translation_norm": float(np.linalg.norm(translation)),
                "residual_mean": float(residuals.mean()),
                "residual_median": float(np.median(residuals)),
                "residual_p95": float(np.percentile(residuals, 95)),
            },
        )

        return transform, alignment_pairs

    def run_predictions(self, image_names, model, max_loops, clip_model, clip_preprocess, batch_metadata=None):
        t1 = time.time()
        device = next(model.parameters()).device
        images = self._load_original_images(image_names).to(device)
        print(f"Loaded {len(image_names)} original images in {time.time() - t1:.2f} seconds")
        print(f"Original image tensor shape: {images.shape}")

        if self.map.get_largest_key() is None:
            new_pcd_num = 0
        else:
            new_pcd_num = self.map.get_largest_key() + self.map.get_latest_submap().get_last_non_loop_frame_index() + 1

        print(f"Creating new Pi3 submap with id {new_pcd_num}")
        new_submap = Submap(new_pcd_num)
        new_submap.add_all_frames(images.detach().cpu())
        new_submap.set_frame_ids(image_names)
        new_submap.set_batch_metadata(batch_metadata)
        new_submap.set_last_non_loop_frame_index(images.shape[0] - 1)
        new_submap.set_all_retrieval_vectors([])
        new_submap.set_img_names(image_names)
        self.current_working_submap = new_submap

        loop_candidates = []
        with self.loop_closure_timer:
            if max_loops > 0:
                loop_candidate = self._find_loop_candidate(image_names)
                if loop_candidate is not None:
                    verified_loop = self._verify_loop_candidate(loop_candidate, image_names, model)
                    loop_candidates.append(verified_loop)

        t1 = time.time()
        inference_outputs = self._run_pi3_inference_on_images(image_names, model)
        print(f"Pi3X model inference took {time.time() - t1:.2f} seconds")

        return {
            "images": inference_outputs["images"],
            "local_points": inference_outputs["local_points"],
            "world_points_pi3": inference_outputs["world_points_pi3"],
            "point_conf": inference_outputs["point_conf"],
            "camera_poses": inference_outputs["camera_poses"],
            "intrinsic": inference_outputs["intrinsic"],
            "detected_loops": loop_candidates[: max_loops if max_loops > 0 else 0],
            "batch_metadata": batch_metadata,
        }

    def add_edge(self, submap_id_curr, frame_id_curr, submap_id_prev=None, frame_id_prev=None, is_loop_closure=False):
        if is_loop_closure:
            print(colored("Pi3Solver loop-closure edge requested but loop closure is disabled in phase 1.", "yellow"))
            return

        current_submap = self.map.get_submap(submap_id_curr)
        current_w2c = current_submap.get_all_poses()
        current_c2w = np.linalg.inv(current_w2c)
        G_map_from_current_submap = np.linalg.inv(current_c2w[0])
        alignment_pairs = []
        prior_submap = None

        if submap_id_prev is not None:
            prior_submap, shared_pairs = self._find_alignment_prior_submap(current_submap, submap_id_prev)
            if shared_pairs:
                alignment_pairs = shared_pairs
            else:
                overlap_count = self._infer_overlap_count(current_submap, prior_submap)
                if overlap_count <= 0:
                    raise ValueError(
                        "Pi3Solver expected shared anchors or at least one prefix/suffix overlapping frame "
                        "between adjacent submaps."
                    )
                alignment_pairs = self._prefix_overlap_pairs(current_submap, prior_submap, overlap_count)

            H_prior_from_current_submap, alignment_pairs = self._estimate_submap_alignment(
                current_submap=current_submap,
                prior_submap=prior_submap,
                alignment_pairs=alignment_pairs,
            )

            anchor_pair = alignment_pairs[0]
            prior_anchor_index = anchor_pair["prior_index"]
            prior_anchor_node_id = prior_submap.get_id() + prior_anchor_index
            prior_anchor_homography = self.graph.get_homography(prior_anchor_node_id)
            prior_anchor_w2c = prior_submap.get_all_poses()[prior_anchor_index]
            G_map_from_prior_submap = prior_anchor_homography @ prior_anchor_w2c
            G_map_from_current_submap = G_map_from_prior_submap @ H_prior_from_current_submap

            print(
                "Pi3 overlap anchor:",
                {
                    "alignment_source": anchor_pair.get("source"),
                    "num_alignment_pairs": len(alignment_pairs),
                    "prior_submap_id": int(prior_submap.get_id()),
                    "prior_anchor_index": int(prior_anchor_index),
                    "current_anchor_index": int(anchor_pair["current_index"]),
                    "global_frame_id": anchor_pair.get("global_frame_id"),
                },
            )
        else:
            assert (submap_id_curr == 0 and frame_id_curr == 0), "First added node must be submap 0 frame 0"

        for index, pose in enumerate(current_w2c):
            current_node = G_map_from_current_submap @ current_c2w[index]
            self.graph.add_homography(submap_id_curr + index, current_node.astype(np.float32))

        if submap_id_prev is None:
            self.graph.add_prior_factor(submap_id_curr + frame_id_curr, self.graph.get_homography(submap_id_curr + frame_id_curr))

        for index, pose in enumerate(current_w2c):
            if index == 0:
                continue
            H_inner = current_w2c[index - 1] @ np.linalg.inv(pose)
            self.graph.add_between_factor(
                submap_id_curr + index - 1,
                submap_id_curr + index,
                H_inner,
                self.graph.inner_submap_noise,
            )

        if submap_id_prev is not None and alignment_pairs:
            prior_w2c = prior_submap.get_all_poses()
            prior_anchor_pair = alignment_pairs[0]
            G_map_from_prior_submap = self.graph.get_homography(
                prior_submap.get_id() + prior_anchor_pair["prior_index"]
            ) @ prior_w2c[prior_anchor_pair["prior_index"]]
            H_prior_from_current_submap = np.linalg.inv(G_map_from_prior_submap) @ G_map_from_current_submap

            extra_constraints = []
            for pair in alignment_pairs:
                prior_index = pair["prior_index"]
                current_index = pair["current_index"]
                prior_node_id = prior_submap.get_id() + prior_index
                current_node_id = submap_id_curr + current_index
                relative_h = (
                    prior_w2c[prior_index] @ H_prior_from_current_submap @ current_c2w[current_index]
                ).astype(np.float32)
                self.graph.add_between_factor(
                    prior_node_id,
                    current_node_id,
                    relative_h,
                    self.graph.intra_submap_noise,
                )
                extra_constraints.append(
                    {
                        "prior_node_id": int(prior_node_id),
                        "current_node_id": int(current_node_id),
                        "global_frame_id": pair.get("global_frame_id"),
                        "source": pair.get("source"),
                    }
                )

            print("Pi3 shared/overlap constraints:", extra_constraints)

    def add_points(self, pred_dict):
        images = pred_dict["images"]
        local_points = pred_dict["local_points"]
        conf = pred_dict["point_conf"]
        camera_poses = pred_dict["camera_poses"]
        intrinsics_cam = pred_dict["intrinsic"]
        filtered_conf, non_edge_mask = self._compute_filtered_confidence(local_points, conf)

        colors = (images.transpose(0, 2, 3, 1) * 255).astype(np.uint8)
        world_to_cam = closed_form_inverse_se3(camera_poses)

        num_frames = camera_poses.shape[0]
        K_4x4 = np.tile(np.eye(4), (num_frames, 1, 1))
        K_4x4[:, :3, :3] = intrinsics_cam

        submap_id_prev = self.map.get_largest_key(ignore_loop_closure_submaps=True)
        submap_id_curr = self.current_working_submap.get_id()
        frame_id_curr = 0
        frame_id_prev = None

        if submap_id_prev is not None:
            frame_id_prev = self.map.get_latest_submap(ignore_loop_closure_submaps=True).get_last_non_loop_frame_index()

        print(
            "Pi3 add_points summary:",
            {
                "submap_id": int(submap_id_curr),
                "num_frames": int(num_frames),
                "mean_conf_raw": float(conf.mean()),
                "mean_conf_filtered": float(filtered_conf.mean()),
                "non_edge_ratio": float(non_edge_mask.mean()),
            },
        )

        self.current_working_submap.add_all_poses(world_to_cam)
        self.current_working_submap.add_all_points(local_points, colors, filtered_conf, self.init_conf_threshold, K_4x4)
        self.current_working_submap.set_conf_masks(filtered_conf)
        self.map.add_submap(self.current_working_submap)

        self.add_edge(submap_id_curr, frame_id_curr, submap_id_prev, frame_id_prev, is_loop_closure=False)

        detected_loops = pred_dict.get("detected_loops", [])
        for loop in detected_loops:
            hist_node_id = loop["detected_submap_id"] + loop["detected_submap_frame"]
            curr_node_id = submap_id_curr + loop["query_submap_frame"]
            relative_h = np.asarray(loop["relative_h"], dtype=np.float32)
            self.graph.add_between_factor(
                hist_node_id,
                curr_node_id,
                relative_h,
                self.graph.loop_noise,
            )
            self.graph.increment_loop_closure()
            print(
                colored("Pi3 loop edge added", "yellow"),
                {
                    "hist_node_id": int(hist_node_id),
                    "curr_node_id": int(curr_node_id),
                    "query_submap_frame": int(loop["query_submap_frame"]),
                    "detected_submap_id": int(loop["detected_submap_id"]),
                    "detected_submap_frame": int(loop["detected_submap_frame"]),
                },
            )
