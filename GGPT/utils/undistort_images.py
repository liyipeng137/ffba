"""
Utility to read COLMAP cameras.txt and undistort images to linear cameras with center principle points.

Usage:
  # From command line:
  python utils/undistort_images.py /path/to/data --cameras cameras.txt --output /path/to/undistorted

  # From Python:
  from utils.undistort_images import read_camera_intrinsics, undistort_image

  cameras = read_camera_intrinsics("cameras.txt")
  K, dist, w, h = cameras[1]  # camera_id 1
  img_undist, K_new = undistort_image("image.png", K, dist, "output.png")
"""

import argparse
import os
import numpy as np
import cv2
from pathlib import Path

# Add project root for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.colmap_loader import read_intrinsics_text


def read_camera_intrinsics(cameras_path):
    """
    Read cameras.txt and return dict: camera_id -> (K, dist_coeffs, width, height).
    K is 3x3 camera matrix, dist_coeffs is (k1,k2,p1,p2,k3,k4,k5,k6) for FULL_OPENCV.
    """
    cameras = read_intrinsics_text(cameras_path)
    out = {}
    for cam_id, camera in cameras.items():
        if camera.model != "FULL_OPENCV":
            raise ValueError(f"Only FULL_OPENCV supported, got {camera.model}")
        K, dist = full_opencv_to_opencv(camera)
        out[cam_id] = (K, dist, camera.width, camera.height)
    return out


def full_opencv_to_opencv(camera):
    """
    Extract OpenCV camera matrix and distortion coefs from FULL_OPENCV params.
    FULL_OPENCV params: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
    """
    params = camera.params
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    dist_coeffs = np.array(params[4:12])  # k1, k2, p1, p2, k3, k4, k5, k6

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return K, dist_coeffs


def get_optimal_new_camera_matrix(K, dist_coeffs, width, height, alpha=0.0):
    """
    Get new camera matrix for undistortion with center principle point.
    alpha=0: crops to valid pixels only
    alpha=1: keeps all pixels (may include black regions)
    alpha=0.5: balanced
    """
    try:
        K_new, _ = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, (width, height), alpha,
            newImgSize=(width, height),
            centerPrincipalPoint=True
        )
    except TypeError:
        # Older OpenCV: centerPrincipalPoint not available.
        # Use optimal focal lengths but manually center the principal point.
        K_opt, _ = cv2.getOptimalNewCameraMatrix(
            K, dist_coeffs, (width, height), alpha,
            newImgSize=(width, height)
        )
        K_new = np.array([
            [K_opt[0, 0], 0, (width - 1) / 2.0],
            [0, K_opt[1, 1], (height - 1) / 2.0],
            [0, 0, 1]
        ], dtype=np.float64)
    return K_new


def get_undistort_maps(K, dist_coeffs, width, height, alpha=0.0):
    """
    Get remap maps and new camera matrix for undistortion.
    Returns (map1, map2, K_new) where K_new is the linear pinhole intrinsics with center pp.
    """
    K_new = get_optimal_new_camera_matrix(K, dist_coeffs, width, height, alpha)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, K_new, (width, height), cv2.CV_32FC1
    )
    return map1, map2, K_new


def undistort_image(image_path, K, dist_coeffs, output_path=None, alpha=0.0):
    """
    Undistort a single image and optionally save it.
    Returns undistorted image and new camera matrix.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]

    K_new = get_optimal_new_camera_matrix(K, dist_coeffs, w, h, alpha)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, dist_coeffs, None, K_new, (w, h), cv2.CV_32FC1
    )
    img_undist = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, img_undist)

    return img_undist, K_new


def process_folder(folder, cameras_path, output_folder=None, alpha=0.0,
                   camera_id_to_subfolder=None):
    """
    Process all images in a folder structure, undistorting using cameras.txt intrinsics.

    Args:
        folder: Root folder containing images/ with Camera1/, Camera2/, Camera3/ subdirs
        cameras_path: Path to cameras.txt
        output_folder: Where to save undistorted images (default: folder/undistorted/)
        alpha: Undistortion alpha (0=crop, 1=keep all, 0.5=balanced)
        camera_id_to_subfolder: Dict mapping camera_id (1,2,3) to subfolder name (e.g. Camera1)
    """
    cameras = read_intrinsics_text(cameras_path)
    if not cameras:
        raise ValueError(f"No cameras found in {cameras_path}")

    output_folder = output_folder or os.path.join(folder, "undistorted")
    camera_id_to_subfolder = camera_id_to_subfolder or {
        1: "Camera1", 2: "Camera2", 3: "Camera3"
    }

    new_intrinsics = {}
    images_dir = os.path.join(folder, "images")

    for cam_id, camera in cameras.items():
        if camera.model != "FULL_OPENCV":
            raise ValueError(f"Only FULL_OPENCV supported, got {camera.model}")

        K, dist_coeffs = full_opencv_to_opencv(camera)
        subfolder = camera_id_to_subfolder.get(cam_id, f"Camera{cam_id}")
        input_dir = os.path.join(images_dir, subfolder)
        output_dir = os.path.join(output_folder, "images", subfolder)

        if not os.path.isdir(input_dir):
            print(f"Skipping camera {cam_id}: {input_dir} not found")
            continue

        os.makedirs(output_dir, exist_ok=True)
        files = sorted([f for f in os.listdir(input_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])

        w, h = camera.width, camera.height
        K_new = get_optimal_new_camera_matrix(K, dist_coeffs, w, h, alpha)
        new_intrinsics[cam_id] = K_new

        for fname in files:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)
            try:
                undistort_image(in_path, K, dist_coeffs, out_path, alpha)
            except Exception as e:
                print(f"Error processing {in_path}: {e}")

        print(f"Camera {cam_id}: undistorted {len(files)} images from {subfolder}")

    # Write new cameras.txt with linear (PINHOLE) intrinsics
    out_cameras_path = os.path.join(output_folder, "cameras.txt")
    with open(out_cameras_path, "w") as f:
        f.write("# Undistorted cameras (linear pinhole, center principle point)\n")
        for cam_id, K_new in new_intrinsics.items():
            fx, fy = K_new[0, 0], K_new[1, 1]
            cx, cy = K_new[0, 2], K_new[1, 2]
            w, h = cameras[cam_id].width, cameras[cam_id].height
            # PINHOLE: fx, fy, cx, cy
            f.write(f"{cam_id} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")
    print(f"Wrote {out_cameras_path}")
    return new_intrinsics


def main():
    parser = argparse.ArgumentParser(description="Undistort images using COLMAP cameras.txt")
    parser.add_argument("folder", help="Data folder containing images/Camera1/, etc.")
    parser.add_argument("--cameras", "-c", default="cameras.txt",
                        help="Path to cameras.txt (default: folder/cameras.txt)")
    parser.add_argument("--output", "-o", help="Output folder (default: folder/undistorted/)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Undistortion alpha: 0=crop invalid, 1=keep all, 0.5=balanced (default: 0)")
    args = parser.parse_args()

    cameras_path = args.cameras if os.path.isabs(args.cameras) else os.path.join(args.folder, args.cameras)
    if not os.path.exists(cameras_path):
        parser.error(f"cameras.txt not found: {cameras_path}")

    process_folder(args.folder, cameras_path, args.output, args.alpha)


if __name__ == "__main__":
    main()
