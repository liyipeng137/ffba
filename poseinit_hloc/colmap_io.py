# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# This file keeps the small COLMAP binary subset needed by poseinit_hloc.

import collections
import struct
from pathlib import Path

import numpy as np

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
ColmapCamera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

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
CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def write_next_bytes(fid, data, format_char_sequence: str, endian_character: str = "<") -> None:
    if isinstance(data, (list, tuple)):
        packed = struct.pack(endian_character + format_char_sequence, *data)
    else:
        packed = struct.pack(endian_character + format_char_sequence, data)
    fid.write(packed)


def read_images_binary(path_to_model_file: str | Path) -> dict[int, Image]:
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(image_id, qvec, tvec, camera_id, image_name, xys, point3D_ids)
    return images


def read_points3D_binary(path_to_model_file: str | Path) -> dict[int, Point3D]:
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = props[0]
            xyz = np.array(props[1:4])
            rgb = np.array(props[4:7])
            error = np.array(props[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(point3D_id, xyz, rgb, error, image_ids, point2D_idxs)
    return points3D


def write_images_binary(images: dict[int, Image], path_to_model_file: str | Path) -> None:
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, point3D_id], "ddq")


def write_cameras_binary(cameras: dict[int, ColmapCamera], path_to_model_file: str | Path) -> None:
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            write_next_bytes(fid, [cam.id, model_id, cam.width, cam.height], "iiQQ")
            for param in cam.params:
                write_next_bytes(fid, float(param), "d")


def write_points3D_binary(points3D: dict[int, Point3D], path_to_model_file: str | Path) -> None:
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, point in points3D.items():
            write_next_bytes(fid, point.id, "Q")
            write_next_bytes(fid, point.xyz.tolist(), "ddd")
            write_next_bytes(fid, point.rgb.tolist(), "BBB")
            write_next_bytes(fid, point.error, "d")
            write_next_bytes(fid, point.image_ids.shape[0], "Q")
            for image_id, point2D_id in zip(point.image_ids, point.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def create_cameras_and_points_bin(target: str | Path, intrinsics: dict) -> None:
    model_dir = Path(target) / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    write_points3D_binary({}, model_dir / "points3D.bin")
    cameras = {
        1: ColmapCamera(
            id=1,
            model="PINHOLE",
            width=int(intrinsics["width"]),
            height=int(intrinsics["height"]),
            params=np.array([intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]]),
        )
    }
    write_cameras_binary(cameras, model_dir / "cameras.bin")


def create_images_from_pose_dict(ws_dir: str | Path, pose_dict: dict[str, np.ndarray]) -> None:
    images = {}
    for image_name, camera_to_world in pose_dict.items():
        world_to_camera = np.linalg.inv(camera_to_world)
        r = world_to_camera[:3, :3]
        tvec = world_to_camera[:3, 3]
        qvec = rotmat2qvec(r)
        image_id = int(image_name)
        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=1,
            name=f"{image_name}.jpg",
            xys=[],
            point3D_ids=[],
        )
    write_images_binary(images, Path(ws_dir) / "model" / "images.bin")


def export_points3d_to_ply(points3d_path: str | Path, ply_path: str | Path) -> None:
    from plyfile import PlyData, PlyElement

    points3d = read_points3D_binary(points3d_path)
    if not points3d:
        raise ValueError(f"No points found in {points3d_path}")
    xyz = np.array([points3d[k].xyz for k in points3d])
    rgb = np.array([points3d[k].rgb for k in points3d])
    normals = np.zeros_like(xyz)
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate((xyz, normals, rgb), axis=1)))
    Path(ply_path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(ply_path)
