"""
COLMAP loader adapted from Gaussian Splatting style parsing and converted
to the same dictionary schema as `read_bal_data`.

Supported inputs:
- Text model: cameras.txt / images.txt / points3D.txt
- Binary model: cameras.bin / images.bin / points3D.bin

Current BA scope:
- Single shared PINHOLE camera model
"""

from __future__ import annotations

import collections
import os
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

DTYPE = torch.float64

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in CAMERA_MODELS}


class Image(BaseImage):
    pass


def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def _resolve_colmap_files(
    cameras_file: Optional[str],
    images_file: Optional[str],
    points3d_file: Optional[str],
    input_dir: Optional[str],
) -> Tuple[str, str, str, str]:
    if input_dir is not None:
        bin_files = (
            os.path.join(input_dir, "cameras.bin"),
            os.path.join(input_dir, "images.bin"),
            os.path.join(input_dir, "points3D.bin"),
        )
        txt_files = (
            os.path.join(input_dir, "cameras.txt"),
            os.path.join(input_dir, "images.txt"),
            os.path.join(input_dir, "points3D.txt"),
        )
        has_bin = all(os.path.exists(p) for p in bin_files)
        has_txt = all(os.path.exists(p) for p in txt_files)
        if has_bin:
            return (*bin_files, "bin")
        if has_txt:
            return (*txt_files, "txt")
        raise FileNotFoundError(
            "Could not find a complete COLMAP model in input_dir. "
            "Expected either cameras/images/points3D in .bin or .txt format."
        )

    if not all([cameras_file, images_file, points3d_file]):
        raise ValueError("Provide either input_dir or all of cameras_file/images_file/points3d_file")

    ext = os.path.splitext(cameras_file)[1].lower()
    if ext not in (".txt", ".bin"):
        raise ValueError("Unsupported COLMAP file extension; expected .txt or .bin")

    same_ext = all(os.path.splitext(p)[1].lower() == ext for p in (images_file, points3d_file))
    if not same_ext:
        raise ValueError("cameras/images/points3D must use the same extension (.txt or .bin)")

    return cameras_file, images_file, points3d_file, ext[1:]


def read_intrinsics_text(path: str) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            if model != "PINHOLE":
                raise ValueError(f"Only PINHOLE camera model is supported, got {model}")
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))
            if params.shape[0] != 4:
                raise ValueError(f"PINHOLE expects 4 params (fx, fy, cx, cy), got {params.shape[0]}")
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=params,
            )
    return cameras


def read_extrinsics_text(path: str) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue

            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]

            points_line = fid.readline()
            points_elems = points_line.split() if points_line else []
            if len(points_elems) == 0:
                xys = np.empty((0, 2), dtype=np.float64)
                point3d_ids = np.empty((0,), dtype=np.int64)
            else:
                xys = np.column_stack(
                    [
                        tuple(map(float, points_elems[0::3])),
                        tuple(map(float, points_elems[1::3])),
                    ]
                )
                point3d_ids = np.array(tuple(map(int, points_elems[2::3])))

            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images


def read_points3d_text(
    path: str,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, List[List[Tuple[int, int]]]]:
    point_ids: List[int] = []
    xyzs: List[Tuple[float, float, float]] = []
    rgbs: List[Tuple[int, int, int]] = []
    errors: List[float] = []
    tracks: List[List[Tuple[int, int]]] = []

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) == 0 or line[0] == "#":
                continue
            elems = line.split()
            if len(elems) < 8:
                continue

            pid = int(elems[0])
            xyz = tuple(map(float, elems[1:4]))
            rgb = tuple(map(int, elems[4:7]))
            err = float(elems[7])
            track: List[Tuple[int, int]] = []
            for i in range(8, len(elems), 2):
                if i + 1 >= len(elems):
                    break
                track.append((int(elems[i]), int(elems[i + 1])))

            point_ids.append(pid)
            xyzs.append(xyz)
            rgbs.append(rgb)
            errors.append(err)
            tracks.append(track)

    return (
        point_ids,
        np.array(xyzs, dtype=np.float64) if xyzs else np.empty((0, 3), dtype=np.float64),
        np.array(rgbs, dtype=np.int64) if rgbs else np.empty((0, 3), dtype=np.int64),
        np.array(errors, dtype=np.float64) if errors else np.empty((0,), dtype=np.float64),
        tracks,
    )


def read_intrinsics_binary(path: str) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = _read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = _read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )

            if model_name != "PINHOLE":
                raise ValueError(f"Only PINHOLE camera model is supported, got {model_name}")

            params_np = np.array(params)
            if params_np.shape[0] != 4:
                raise ValueError(f"PINHOLE expects 4 params (fx, fy, cx, cy), got {params_np.shape[0]}")

            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=params_np,
            )
    return cameras


def read_extrinsics_binary(path: str) -> Dict[int, Image]:
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = _read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]
            num_points2d = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = _read_next_bytes(
                fid,
                num_bytes=24 * num_points2d,
                format_char_sequence="ddq" * num_points2d,
            )
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id_s[0::3])),
                    tuple(map(float, x_y_id_s[1::3])),
                ]
            )
            point3d_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images


def read_points3d_binary(
    path: str,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, List[List[Tuple[int, int]]]]:
    point_ids: List[int] = []
    xyzs: List[Tuple[float, float, float]] = []
    rgbs: List[Tuple[int, int, int]] = []
    errors: List[float] = []
    tracks: List[List[Tuple[int, int]]] = []

    with open(path, "rb") as fid:
        num_points = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = _read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point_id = int(props[0])
            xyz = (float(props[1]), float(props[2]), float(props[3]))
            rgb = (int(props[4]), int(props[5]), int(props[6]))
            error = float(props[7])

            track_length = _read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = _read_next_bytes(
                fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length
            )
            track = []
            for i in range(0, len(track_elems), 2):
                track.append((int(track_elems[i]), int(track_elems[i + 1])))

            point_ids.append(point_id)
            xyzs.append(xyz)
            rgbs.append(rgb)
            errors.append(error)
            tracks.append(track)

    return (
        point_ids,
        np.array(xyzs, dtype=np.float64) if xyzs else np.empty((0, 3), dtype=np.float64),
        np.array(rgbs, dtype=np.int64) if rgbs else np.empty((0, 3), dtype=np.int64),
        np.array(errors, dtype=np.float64) if errors else np.empty((0,), dtype=np.float64),
        tracks,
    )


def _build_bae_data(
    cameras: Dict[int, Camera],
    images: Dict[int, Image],
    point_ids: List[int],
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    points_error: np.ndarray,
    point_tracks: List[List[Tuple[int, int]]],
) -> dict:
    if len(cameras) != 1:
        raise ValueError(f"Expected exactly one PINHOLE camera, got {len(cameras)}")
    shared_cam_id, shared_cam = next(iter(cameras.items()))

    pid_to_idx = {pid: idx for idx, pid in enumerate(point_ids)}

    camera_params: List[List[float]] = []
    points_2d: List[Tuple[float, float]] = []
    cam_indices: List[int] = []
    pt_indices: List[int] = []
    image_records: List[dict] = []

    for image_id in sorted(images.keys()):
        image = images[image_id]
        if image.camera_id != shared_cam_id:
            raise ValueError("Multiple camera intrinsics detected; only one PINHOLE camera is supported")

        all_obs = []
        valid_obs = 0
        cam_idx = len(camera_params)
        for (xy, pid) in zip(image.xys, image.point3D_ids):
            x = float(xy[0])
            y = float(xy[1])
            pid = int(pid)
            all_obs.append((x, y, pid))
            if pid >= 0 and pid in pid_to_idx:
                points_2d.append((x, y))
                cam_indices.append(cam_idx)
                pt_indices.append(pid_to_idx[pid])
                valid_obs += 1

        if valid_obs == 0:
            continue

        qvec = image.qvec  # [qw, qx, qy, qz]
        tvec = image.tvec
        camera_params.append(
            [
                float(tvec[0]),
                float(tvec[1]),
                float(tvec[2]),
                float(qvec[1]),
                float(qvec[2]),
                float(qvec[3]),
                float(qvec[0]),
            ]
        )
        image_records.append(
            {
                "image_id": image.id,
                "name": image.name,
                "camera_id": image.camera_id,
                "width": shared_cam.width,
                "height": shared_cam.height,
                "points_2d": all_obs,
            }
        )

    shared_intrinsics = torch.tensor(
        [
            float(shared_cam.params[0]),
            float(shared_cam.params[1]),
            float(shared_cam.params[2]),
            float(shared_cam.params[3]),
        ],
        dtype=DTYPE,
    )

    return {
        "problem_name": "colmap_import",
        "camera_params": torch.tensor(camera_params, dtype=DTYPE),
        "points_3d": torch.tensor(points_xyz, dtype=DTYPE),
        "points_2d": torch.tensor(points_2d, dtype=DTYPE),
        "camera_index_of_observations": torch.tensor(cam_indices, dtype=torch.int64),
        "point_index_of_observations": torch.tensor(pt_indices, dtype=torch.int64),
        "intrinsics": shared_intrinsics,
        "metadata": {
            "camera_models": cameras,
            "shared_cam_id": shared_cam_id,
            "point_ids": point_ids,
            "point_colors": points_rgb.tolist(),
            "point_errors": points_error.tolist(),
            "point_tracks": point_tracks,
            "images": image_records,
            "intrinsics": shared_intrinsics,
        },
    }


def read_colmap_data(
    cameras_file: Optional[str] = None,
    images_file: Optional[str] = None,
    points3d_file: Optional[str] = None,
    input_dir: Optional[str] = None,
) -> dict:
    """
    Read COLMAP model and convert to BAE input format.

    Usage:
    - read_colmap_data(input_dir="colmap_data")
    - read_colmap_data("cameras.txt", "images.txt", "points3D.txt")
    - read_colmap_data("cameras.bin", "images.bin", "points3D.bin")

    If input_dir is provided, loader auto-detects format:
    - Prefer .bin when full .bin set exists
    - Fallback to .txt when full .txt set exists
    """
    cameras_path, images_path, points3d_path, fmt = _resolve_colmap_files(
        cameras_file, images_file, points3d_file, input_dir
    )

    if fmt == "bin":
        cameras = read_intrinsics_binary(cameras_path)
        images = read_extrinsics_binary(images_path)
        point_ids, points_xyz, points_rgb, points_error, point_tracks = read_points3d_binary(points3d_path)
    else:
        cameras = read_intrinsics_text(cameras_path)
        images = read_extrinsics_text(images_path)
        point_ids, points_xyz, points_rgb, points_error, point_tracks = read_points3d_text(points3d_path)

    return _build_bae_data(
        cameras,
        images,
        point_ids,
        points_xyz,
        points_rgb,
        points_error,
        point_tracks,
    )


def save_colmap_result(
    images_out: str,
    points3d_out: str,
    data: dict,
    camera_params: torch.Tensor,
    points_3d: torch.Tensor,
) -> None:
    """Save optimized camera poses and points in COLMAP text format."""
    meta = data.get("metadata", {})
    images_meta = meta.get("images", [])
    point_ids = meta.get("point_ids", [])
    point_colors = meta.get("point_colors", [])
    point_errors = meta.get("point_errors", [])
    point_tracks = meta.get("point_tracks", [])

    if camera_params.shape[0] != len(images_meta):
        raise ValueError("camera_params rows do not match metadata images")
    if points_3d.shape[0] != len(point_ids):
        raise ValueError("points_3d rows do not match metadata points")

    with open(images_out, "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for idx, rec in enumerate(images_meta):
            pose = camera_params[idx]
            tx, ty, tz = pose[:3].tolist()
            qx, qy, qz, qw = pose[3:7].tolist()
            f.write(
                f"{rec['image_id']} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {rec['camera_id']} {rec['name']}\n"
            )
            obs = rec.get("points_2d", [])
            if len(obs) == 0:
                f.write("\n")
            else:
                f.write(" ".join(f"{x} {y} {pid}" for x, y, pid in obs) + "\n")

    with open(points3d_out, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for idx, (pid, pt) in enumerate(zip(point_ids, points_3d.tolist())):
            x, y, z = pt
            if idx < len(point_colors):
                r, g, b = point_colors[idx]
            else:
                r, g, b = 0, 0, 0
            error = float(point_errors[idx]) if idx < len(point_errors) else 0.0
            track = point_tracks[idx] if idx < len(point_tracks) else []
            track_str = " ".join(f"{img_id} {pt_idx}" for img_id, pt_idx in track)
            base = f"{pid} {x} {y} {z} {r} {g} {b} {error}"
            f.write(base + (f" {track_str}" if track_str else "") + "\n")


def save_colmap_cameras(cameras_out: str, data: dict, intrinsics: torch.Tensor) -> None:
    """Save shared PINHOLE intrinsics to cameras.txt."""
    meta = data.get("metadata", {})
    cameras = meta.get("camera_models", {})
    shared_cam_id = meta.get("shared_cam_id", None)

    if intrinsics is None:
        intrinsics = meta.get("intrinsics", None)
    if intrinsics is None:
        raise ValueError("intrinsics not provided")
    if intrinsics.dim() > 1:
        intrinsics = intrinsics.squeeze(0)
    if intrinsics.shape[0] != 4:
        raise ValueError("intrinsics must contain 4 values: fx, fy, cx, cy")

    if shared_cam_id not in cameras:
        raise ValueError("shared camera metadata is missing")
    cam = cameras[shared_cam_id]

    fx, fy, cx, cy = intrinsics.tolist()
    with open(cameras_out, "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"{cam.id} PINHOLE {cam.width} {cam.height} {fx} {fy} {cx} {cy}\n")


__all__ = ["read_colmap_data", "save_colmap_result", "save_colmap_cameras"]
