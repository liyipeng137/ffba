import os
import json
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from vggt_slam.slam_utils import decompose_camera, cosine_similarity

class GraphMap:
    def __init__(self):
        self.submaps = dict()
        self.rectifying_H_mats = []
        self.non_lc_submap_ids = []
    
    def get_num_submaps(self):
        return len(self.submaps)

    def add_submap(self, submap):
        submap_id = submap.get_id()
        self.submaps[submap_id] = submap
        if not submap.get_lc_status():
            self.non_lc_submap_ids.append(submap_id)
    
    def get_largest_key(self, ignore_loop_closure_submaps=False):
        """
        Get the largest key of the first node of any submap.
        Return: The largest key, or None if the dictionary is empty.
        """
        if len(self.submaps) == 0:
            return None
        if ignore_loop_closure_submaps:
            non_lc_keys = [key for key, submap in self.submaps.items() if not submap.get_lc_status()]
            return max(non_lc_keys)
        return max(self.submaps.keys())
    
    def get_submap(self, id):
        return self.submaps[id]

    def get_latest_submap(self, ignore_loop_closure_submaps=False):
        return self.get_submap(self.get_largest_key(ignore_loop_closure_submaps))

    def retrieve_best_semantic_frame(self, query_text_vector):
        overall_best_score = 0.0
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            submap = self.submaps[submap_key]
            if submap.get_lc_status():
                continue
            submap_embeddings = submap.get_all_semantic_vectors()
            scores = []
            for index, embedding in enumerate(submap_embeddings):
                score = cosine_similarity(embedding, query_text_vector)
                scores.append(score)
            
            best_score_id = np.argmax(scores)
            best_score = scores[best_score_id]

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_submap_id = submap_key
                overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index
    
    def retrieve_best_score_frame(self, query_vector, current_submap_id, ignore_last_submap=True):
        overall_best_score = 1000
        overall_best_submap_id = 0
        overall_best_frame_index = 0
        # search for best image to target image
        sorted_keys = sorted(self.submaps.keys())
        for index, submap_key in enumerate(sorted_keys):
            if submap_key == current_submap_id:
                continue

            if self.non_lc_submap_ids and ignore_last_submap and submap_key == self.non_lc_submap_ids[-1]:
                continue

            else:
                submap = self.submaps[submap_key]
                if submap.get_lc_status():
                    continue
                submap_embeddings = submap.get_all_retrieval_vectors()
                scores = []
                for index, embedding in enumerate(submap_embeddings):
                    score = torch.linalg.norm(embedding-query_vector)
                    # score = embedding @ query_vector.t()
                    scores.append(score.item())

                # for now assume we can only have at most one loop closure per submap
                
                best_score_id = np.argmin(scores)
                best_score = scores[best_score_id]

                if best_score < overall_best_score:
                    overall_best_score = best_score
                    overall_best_submap_id = submap_key
                    overall_best_frame_index = best_score_id

        return overall_best_score, overall_best_submap_id, overall_best_frame_index

    def get_frames_from_loops(self, loops):
        frames = []
        for detected_loop in loops:
            frames.append(self.submaps[detected_loop.detected_submap_id].get_frame_at_index(detected_loop.detected_submap_frame))
        return frames
    
    def get_submaps(self):
        return self.submaps.values()

    def ordered_submaps_by_key(self):
        for k in sorted(self.submaps):
            yield self.submaps[k]
    
    def get_all_homographies(self, graph):
        homographies = []
        for submap in self.ordered_submaps_by_key():
            for pose_num in range(len(submap.poses)):
                id = int(submap.get_id() + pose_num)
                homographies.append(graph.get_homography(id))
        return np.stack(homographies)

    def get_all_cam_matricies(self, graph, give_camera_mat):
        cam_mats = []
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
            poses = submap.get_all_poses_world(graph, give_camera_mat=give_camera_mat)
            cam_mats.append(poses)
        return np.vstack(cam_mats)

    def write_poses_to_file(self, file_name, graph, give_camera_mat=False, kitti_format=False):
        all_poses = self.get_all_cam_matricies(give_camera_mat=True, graph=graph)
        with open(file_name, "w") as f:

            if self.rectifying_H_mats:
                assert len(self.rectifying_H_mats) == len(all_poses), "Number of rectifying mats and number of poses do not match"
                print("Using rectifying homographies when writing poses to file.")
            count = 0
            for submap_index, submap in enumerate(self.ordered_submaps_by_key()):
                if submap.get_lc_status():
                    continue
                frame_ids = submap.get_frame_ids()
                if getattr(submap, "global_frame_ids", None) is not None:
                    frame_ids = submap.get_global_frame_ids()
                print(frame_ids)
                for frame_index, frame_id in enumerate(frame_ids):
                    pose = all_poses[count]
                    K, rotation_matrix, t, scale = decompose_camera(pose)
                    # print("Decomposed K:\n", K)
                    count += 1
                    x, y, z = t
                    if kitti_format:
                        pose_matrix = np.eye(4)
                        pose_matrix[:3, :3] = rotation_matrix
                        pose_matrix[:3, 3] = t
                        output = pose_matrix.flatten()[:-4]
                        output = np.array([float(frame_id), *output])
                    else:
                        quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w
                        output = np.array([float(frame_id), x, y, z, *quaternion])
                    f.write(" ".join(f"{v:.8f}" for v in output) + "\n")

    def _iter_frame_exports(self, graph, unique_images=True):
        seen_image_names = set()

        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue

            c2w_opencv = submap.get_all_poses_world(graph, give_camera_mat=False)
            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame(graph)
            if getattr(submap, "global_frame_ids", None) is not None:
                frame_ids = submap.get_global_frame_ids()

            for frame_index, (pointcloud, frame_id, conf_mask) in enumerate(
                zip(pointclouds, frame_ids, conf_masks)
            ):
                image_path = submap.img_names[frame_index] if frame_index < len(submap.img_names) else str(frame_id)
                image_name = Path(image_path).name
                if unique_images and image_name in seen_image_names:
                    continue
                seen_image_names.add(image_name)

                intrinsic = submap.proj_mats[frame_index][:3, :3]
                height, width = pointcloud.shape[:2]
                colors = submap.colors[frame_index] if submap.colors is not None else None

                yield {
                    "node_id": int(submap.get_id() + frame_index),
                    "submap_id": int(submap.get_id()),
                    "frame_index": int(frame_index),
                    "frame_id": float(frame_id),
                    "image_path": str(image_path),
                    "image_name": image_name,
                    "c2w_opencv": c2w_opencv[frame_index],
                    "intrinsic": intrinsic,
                    "height": int(height),
                    "width": int(width),
                    "pointcloud": pointcloud,
                    "mask": conf_mask,
                    "colors": colors,
                }

    def write_lingbot_transforms_json(
        self,
        file_name,
        graph,
        path_mode="basename",
        unique_images=True,
    ):
        records = list(self._iter_frame_exports(graph, unique_images=unique_images))
        if not records:
            raise ValueError("No frames available to export transforms.json")
        if path_mode not in ("basename", "absolute"):
            raise ValueError("path_mode must be 'basename' or 'absolute'")

        intrinsics = np.stack([record["intrinsic"] for record in records], axis=0)
        mean_k = intrinsics.mean(axis=0)

        frames = []
        for record in records:
            c2w_opengl = np.array(record["c2w_opencv"], copy=True)
            c2w_opengl[:3, 1:3] *= -1.0
            intrinsic = record["intrinsic"]
            file_path = record["image_name"] if path_mode == "basename" else str(Path(record["image_path"]).resolve())
            frames.append(
                {
                    "file_path": file_path,
                    "transform_matrix": c2w_opengl.tolist(),
                    "w": record["width"],
                    "h": record["height"],
                    "fl_x": float(intrinsic[0, 0]),
                    "fl_y": float(intrinsic[1, 1]),
                    "cx": float(intrinsic[0, 2]),
                    "cy": float(intrinsic[1, 2]),
                    "image_name": record["image_name"],
                    "source_image_path": str(record["image_path"]),
                    "node_id": record["node_id"],
                    "submap_id": record["submap_id"],
                    "frame_index": record["frame_index"],
                    "frame_id": record["frame_id"],
                }
            )

        data = {
            "camera_model": "OpenGL",
            "fl_x": float(mean_k[0, 0]),
            "fl_y": float(mean_k[1, 1]),
            "cx": float(mean_k[0, 2]),
            "cy": float(mean_k[1, 2]),
            "frames": frames,
        }

        os.makedirs(os.path.dirname(file_name) or ".", exist_ok=True)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def save_framewise_dense_points(self, graph, output_dir, unique_images=True):
        os.makedirs(output_dir, exist_ok=True)
        metadata = []

        for export_index, record in enumerate(self._iter_frame_exports(graph, unique_images=unique_images)):
            stem = Path(record["image_name"]).stem
            dense_file = f"{export_index:06d}_{stem}.npz"
            dense_path = os.path.join(output_dir, dense_file)

            save_data = {
                "pointcloud": record["pointcloud"].astype(np.float32),
                "mask": record["mask"],
                "c2w_opencv": record["c2w_opencv"].astype(np.float32),
                "intrinsic": record["intrinsic"].astype(np.float32),
                "node_id": np.array(record["node_id"], dtype=np.int64),
                "submap_id": np.array(record["submap_id"], dtype=np.int64),
                "frame_index": np.array(record["frame_index"], dtype=np.int64),
                "frame_id": np.array(record["frame_id"], dtype=np.float64),
                "image_name": np.array(record["image_name"]),
                "image_path": np.array(record["image_path"]),
            }
            if record["colors"] is not None:
                save_data["colors"] = record["colors"]
            np.savez_compressed(dense_path, **save_data)

            metadata.append(
                {
                    "dense_file": dense_file,
                    "image_name": record["image_name"],
                    "image_path": record["image_path"],
                    "node_id": record["node_id"],
                    "submap_id": record["submap_id"],
                    "frame_index": record["frame_index"],
                    "frame_id": record["frame_id"],
                    "width": record["width"],
                    "height": record["height"],
                }
            )

        with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"frames": metadata}, f, indent=4)

    def save_framewise_pointclouds(self, graph, file_name):
        os.makedirs(file_name, exist_ok=True)
        count = 0
        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue
                count += len(submap.poses)
            pointclouds, frame_ids, conf_masks = submap.get_points_list_in_world_frame(graph)
            for frame_id, pointcloud, conf_masks in zip(frame_ids, pointclouds, conf_masks):
                # save pcd as numpy array
                np.savez(f"{file_name}/{frame_id}.npz", pointcloud=pointcloud, mask=conf_masks)
        assert count == len(self.rectifying_H_mats), "Number of rectifying mats and number of point maps do not match"
                

    def write_points_to_file(self, graph, file_name):
        pcd_all = []
        colors_all = []
        for submap in self.ordered_submaps_by_key():
            pcd = submap.get_points_in_world_frame(graph)
            pcd = pcd.reshape(-1, 3)
            pcd_all.append(pcd)
            colors_all.append(submap.get_points_colors())
        pcd_all = np.concatenate(pcd_all, axis=0)
        colors_all = np.concatenate(colors_all, axis=0)
        if colors_all.max() > 1.0:
            colors_all = colors_all / 255.0
        pcd_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_all))
        pcd_all.colors = o3d.utility.Vector3dVector(colors_all)
        o3d.io.write_point_cloud(file_name, pcd_all)

    def write_submaps_to_dir(self, graph, output_dir, coordinate_mode="world"):
        os.makedirs(output_dir, exist_ok=True)
        exported_files = []

        for submap in self.ordered_submaps_by_key():
            if submap.get_lc_status():
                continue

            if coordinate_mode == "world":
                pcd = submap.get_points_in_world_frame(graph)
            elif coordinate_mode == "local":
                pcd = submap.get_points_in_local_frame()
            else:
                raise ValueError(f"Unknown coordinate_mode '{coordinate_mode}'. Expected 'world' or 'local'.")

            colors = submap.get_points_colors()
            if pcd is None or len(pcd) == 0:
                continue

            if colors.max() > 1.0:
                colors = colors / 255.0

            pcd_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.reshape(-1, 3)))
            pcd_o3d.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))

            file_name = os.path.join(output_dir, f"submap_{int(submap.get_id()):06d}_{coordinate_mode}.ply")
            o3d.io.write_point_cloud(file_name, pcd_o3d)
            exported_files.append(file_name)

        return exported_files
