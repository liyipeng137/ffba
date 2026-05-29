
#!/usr/bin/env python3
"""
TSDF Fusion from COLMAP text model

Input:
  - colmap/         : cameras.txt (PINHOLE intrinsics) + images.txt (OpenCV w2c poses)
  - depth/*.png     : uint16 PNG, depth = pixel_value / depth_scale (meters)
  - image/*.jpg     : RGB color images

Output:
  - mesh.ply
"""

import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# COLMAP I/O
# --------------------------------------------------------------------------- #

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
            if model == "PINHOLE":
                fx, fy, cx, cy = map(float, els[4:8])
            elif model == "SIMPLE_PINHOLE":
                fx = fy = float(els[4])
                cx, cy = float(els[5]), float(els[6])
            else:
                raise ValueError(f"Unsupported camera model: {model}")
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

            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = qvec2rotmat(qvec)
            w2c[:3, 3] = tvec
            frames.append(dict(name=name, camera_id=camera_id, w2c=w2c))
    if not frames:
        raise ValueError(f"No images found in {images_path}")
    return frames


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
    parser = argparse.ArgumentParser(description="TSDF fusion from COLMAP text model")
    parser.add_argument("--colmap", required=True, help="COLMAP directory (cameras.txt + images.txt)")
    parser.add_argument("--depth", required=True, help="Path to depth directory")
    parser.add_argument("--image", required=True, help="Path to image directory")
    parser.add_argument("--output",    default="mesh.ply",   help="Output mesh path")
    parser.add_argument("--pcd_output", default="",          help="Output point cloud (optional)")
    parser.add_argument("--voxel_size", type=float, default=0.01,  help="Voxel size in meters")
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Divide depth pixel value by this to get meters (default 1000)")
    parser.add_argument("--depth_min",  type=float, default=0.1,   help="Min depth in meters")
    parser.add_argument("--depth_max",  type=float, default=5.0,   help="Max depth in meters")
    parser.add_argument("--max_frames", type=int,   default=0,     help="Process only first N frames (0=all)")
    args = parser.parse_args()

    colmap_dir = Path(args.colmap)
    cameras = parse_colmap_cameras(colmap_dir / "cameras.txt")
    frames = parse_colmap_images(colmap_dir / "images.txt")
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    tsdf = TSDFVolume(args.voxel_size, sdf_trunc=args.voxel_size * 3)

    integrated = 0
    skipped = 0

    for frame in tqdm(frames, desc="Integrating"):
        color_name = Path(frame["name"]).name
        color_path = Path(args.image) / color_name
        stem = Path(color_name).stem
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

    print(f"Integrated {integrated} frames, skipped {skipped}")

    mesh = tsdf.extract_mesh()
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    print(f"Mesh saved: {output_path}  ({len(mesh.vertices):,} verts, {len(mesh.triangles):,} faces)")

    if args.pcd_output:
        pcd = tsdf.extract_pcd()
        o3d.io.write_point_cloud(args.pcd_output, pcd)
        print(f"Point cloud saved: {args.pcd_output}  ({len(pcd.points):,} pts)")


if __name__ == "__main__":
    main()
