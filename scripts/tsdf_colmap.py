
#!/usr/bin/env python3
"""
TSDF Fusion from COLMAP model (text or binary)

Input:
  - colmap/         : cameras.{txt,bin} + images.{txt,bin} (auto-detected, prefers .bin)
  - depth/*.png     : uint16 PNG, depth = pixel_value / depth_scale (meters)
  - image/*.jpg     : RGB color images

Output:
  - mesh.ply
  - sampled point cloud ply (per-frame random sampling from TSDF volume)
"""

import argparse
import collections
import struct
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from tqdm import tqdm

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
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
CAMERA_MODEL_IDS = {m.model_id: m for m in CAMERA_MODELS}


# --------------------------------------------------------------------------- #
# COLMAP I/O
# --------------------------------------------------------------------------- #

def _read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def _camera_params_from_model(model: str, params) -> tuple:
    if model == "PINHOLE":
        fx, fy, cx, cy = map(float, params[:4])
    elif model == "SIMPLE_PINHOLE":
        fx = fy = float(params[0])
        cx, cy = float(params[1]), float(params[2])
    else:
        raise ValueError(f"Unsupported camera model: {model}")
    return fx, fy, cx, cy


def resolve_colmap_model(colmap_dir: Path) -> tuple:
    """Return (cameras_path, images_path, fmt) where fmt is 'bin' or 'txt'. Prefers .bin."""
    bin_files = (colmap_dir / "cameras.bin", colmap_dir / "images.bin")
    txt_files = (colmap_dir / "cameras.txt", colmap_dir / "images.txt")
    if all(p.exists() for p in bin_files):
        return bin_files[0], bin_files[1], "bin"
    if all(p.exists() for p in txt_files):
        return txt_files[0], txt_files[1], "txt"
    raise FileNotFoundError(
        f"Could not find cameras/images in {colmap_dir}. "
        "Expected either cameras.bin+images.bin or cameras.txt+images.txt."
    )


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion [qw, qx, qy, qz] -> 3x3 rotation matrix."""
    return np.array([
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
    ])


def _image_to_frame(name: str, camera_id: int, qvec: np.ndarray, tvec: np.ndarray) -> dict:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec2rotmat(qvec)
    w2c[:3, 3] = tvec
    return dict(name=name, camera_id=camera_id, w2c=w2c)


def parse_colmap_cameras(cameras_path: Path) -> dict:
    """Parse cameras.txt -> {camera_id: {w,h,fx,fy,cx,cy}}."""
    cameras = {}
    with open(cameras_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            els = line.split()
            camera_id = int(els[0])
            model = els[1]
            w, h = int(els[2]), int(els[3])
            fx, fy, cx, cy = _camera_params_from_model(model, els[4:])
            cameras[camera_id] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    if not cameras:
        raise ValueError(f"No cameras found in {cameras_path}")
    return cameras


def parse_colmap_cameras_binary(cameras_path: Path) -> dict:
    """Parse cameras.bin -> {camera_id: {w,h,fx,fy,cx,cy}}."""
    cameras = {}
    with open(cameras_path, "rb") as fid:
        num_cameras = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, w, h = _read_next_bytes(fid, 24, "iiQQ")
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = _read_next_bytes(fid, 8 * num_params, "d" * num_params)
            fx, fy, cx, cy = _camera_params_from_model(model_name, params)
            cameras[camera_id] = dict(w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy)
    if not cameras:
        raise ValueError(f"No cameras found in {cameras_path}")
    return cameras


def parse_colmap_images(images_path: Path) -> list:
    """
    Parse images.txt. Poses are stored as OpenCV w2c: X_cam = R @ X_world + t.
    Returns list of {name, camera_id, w2c}.
    """
    frames = []
    with open(images_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            qvec = np.array(list(map(float, elems[1:5])))
            tvec = np.array(list(map(float, elems[5:8])))
            camera_id = int(elems[8])
            name = elems[9] if len(elems) == 10 else "_".join(elems[9:])
            f.readline()  # skip POINTS2D line
            frames.append(_image_to_frame(name, camera_id, qvec, tvec))
    if not frames:
        raise ValueError(f"No images found in {images_path}")
    return frames


def parse_colmap_images_binary(images_path: Path) -> list:
    """Parse images.bin -> list of {name, camera_id, w2c}."""
    frames = []
    with open(images_path, "rb") as fid:
        num_reg_images = _read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = _read_next_bytes(fid, 64, "idddddddi")
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = ""
            current_char = _read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                name += current_char.decode("utf-8")
                current_char = _read_next_bytes(fid, 1, "c")[0]
            num_points2d = _read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2d)  # skip POINTS2D
            frames.append(_image_to_frame(name, camera_id, qvec, tvec))
    if not frames:
        raise ValueError(f"No images found in {images_path}")
    return frames


def load_colmap_model(colmap_dir: Path) -> tuple:
    """Load cameras and frames from COLMAP dir (auto-detect .bin / .txt)."""
    cameras_path, images_path, fmt = resolve_colmap_model(colmap_dir)
    if fmt == "bin":
        cameras = parse_colmap_cameras_binary(cameras_path)
        frames = parse_colmap_images_binary(images_path)
    else:
        cameras = parse_colmap_cameras(cameras_path)
        frames = parse_colmap_images(images_path)
    return cameras, frames, fmt


# --------------------------------------------------------------------------- #
# TSDF wrapper
# --------------------------------------------------------------------------- #

class TSDFVolume:
    def __init__(self, voxel_size: float, sdf_trunc: float):
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    def integrate(
        self,
        color: np.ndarray,      # H×W×3 uint8
        depth: np.ndarray,      # H×W float32, meters (0 = invalid)
        intrinsic: o3d.camera.PinholeCameraIntrinsic,
        extrinsic: np.ndarray,  # 4×4 OpenCV w2c
        depth_trunc: float = 10.0,
    ):
        color_o3d = o3d.geometry.Image(np.ascontiguousarray(color.astype(np.uint8)))
        depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        self.volume.integrate(rgbd, intrinsic, extrinsic)

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = self.volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_pcd(self) -> o3d.geometry.PointCloud:
        return self.volume.extract_point_cloud()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="TSDF fusion from COLMAP model (.txt or .bin)")
    parser.add_argument("--colmap", required=True, help="COLMAP directory (cameras/images in .txt or .bin)")
    parser.add_argument("--depth", required=True, help="Path to depth directory")
    parser.add_argument("--image", required=True, help="Path to image directory")
    parser.add_argument("--output",    default="mesh.ply",   help="Output mesh path")
    parser.add_argument("--pcd_output", default="",          help="Output full point cloud (optional)")
    parser.add_argument("--sampled_pcd_output", default="",
                        help="Output sampled point cloud (default: <output_stem>_points.ply)")
    parser.add_argument("--num_points", type=int, default=200_000,
                        help="Total points to sample across frames (0=disable)")
    parser.add_argument("--voxel_size", type=float, default=0.01,  help="Voxel size in meters")
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Divide depth pixel value by this to get meters (default 1000)")
    parser.add_argument("--depth_min",  type=float, default=0.1,   help="Min depth in meters")
    parser.add_argument("--depth_max",  type=float, default=5.0,   help="Max depth in meters")
    parser.add_argument("--max_frames", type=int,   default=0,     help="Process only first N frames (0=all)")
    parser.add_argument("--exclude_frames", default="",
                        help="Frames to skip during integration: a path to a text file "
                             "(one stem per line) or a comma-separated list of stems. "
                             "Used to validate layering-frame candidates.")
    args = parser.parse_args()

    colmap_dir = Path(args.colmap)
    cameras, frames, colmap_fmt = load_colmap_model(colmap_dir)
    print(f"Loaded COLMAP model ({colmap_fmt}): {len(cameras)} cameras, {len(frames)} frames")
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    exclude = set()
    if args.exclude_frames:
        p = Path(args.exclude_frames)
        raw = p.read_text().split() if p.exists() else args.exclude_frames.split(",")
        # Accept stems with or without extension.
        exclude = {Path(s.strip()).stem for s in raw if s.strip()}
        print(f"Excluding {len(exclude)} frames from integration: "
              f"{sorted(exclude)[:10]}{' ...' if len(exclude) > 10 else ''}")

    tsdf = TSDFVolume(args.voxel_size, sdf_trunc=args.voxel_size * 3)

    integrated = 0
    skipped = 0
    sample_points = args.num_points > 0
    samples_per_frame = max((args.num_points + len(frames) - 1) // len(frames), 1) if sample_points else 0
    points_list = []
    colors_list = []

    for frame in tqdm(frames, desc="Integrating"):
        color_name = Path(frame["name"]).name
        color_path = Path(args.image) / color_name
        stem = Path(color_name).stem
        if stem in exclude:
            skipped += 1
            continue
        depth_path = Path(args.depth) / f"{stem}.png"
        if not depth_path.exists():
            depth_path = Path(args.depth) / color_name

        if not color_path.exists() or not depth_path.exists():
            skipped += 1
            print(f"Skipping frame {color_name} because it does not exist")
            continue

        cam = cameras[frame["camera_id"]]
        w, h = cam["w"], cam["h"]
        fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        extrinsic = frame["w2c"]

        color = np.array(Image.open(color_path).convert("RGB"))
        if color.shape[0] != h or color.shape[1] != w:
            color = np.array(Image.fromarray(color).resize((w, h), Image.LANCZOS))

        if str(depth_path).endswith(".npy"):
            depth = np.load(depth_path).astype(np.float32)
        else:
            depth_img = np.array(Image.open(depth_path), dtype=np.float32)
            depth = depth_img / args.depth_scale

        if depth.shape[0] != h or depth.shape[1] != w:
            depth = np.array(
                Image.fromarray(depth).resize((w, h), Image.NEAREST), dtype=np.float32
            )

        valid = (depth > args.depth_min) & (depth < args.depth_max)
        depth_filtered = np.where(valid, depth, 0.0).astype(np.float32)

        if valid.sum() < 1000:
            skipped += 1
            continue

        tsdf.integrate(color, depth_filtered, intrinsic, extrinsic, depth_trunc=args.depth_max)
        integrated += 1

        if sample_points:
            pcd = tsdf.extract_pcd()
            if len(pcd.points) > 0:
                pick_num = min(samples_per_frame, len(pcd.points))
                idx = np.random.choice(len(pcd.points), size=pick_num, replace=False)
                points_list.append(np.asarray(pcd.points)[idx])
                colors_list.append(np.asarray(pcd.colors)[idx])

    print(f"Integrated {integrated} frames, skipped {skipped}")

    mesh = tsdf.extract_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print(f"Mesh saved: {output_path}  ({len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces)")

    if sample_points and points_list:
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(np.concatenate(points_list, axis=0))
        sampled_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(colors_list, axis=0))
        sampled_path = Path(args.sampled_pcd_output) if args.sampled_pcd_output else output_path.with_name(f"{output_path.stem}_points.ply")
        sampled_path.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(sampled_path), sampled_pcd)
        print(f"Sampled point cloud saved: {sampled_path}  ({len(sampled_pcd.points):,} pts)")

    if args.pcd_output:
        pcd = tsdf.extract_pcd()
        o3d.io.write_point_cloud(args.pcd_output, pcd)
        print(f"Point cloud saved: {args.pcd_output}  ({len(pcd.points):,} pts)")


if __name__ == "__main__":
    main()
