import copy
import gc
import os
import glob
from pyexpat import model
import shutil
import tempfile
import time
import struct

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
import requests
from PIL import Image
from scipy.spatial.transform import Rotation

from algos.feedforward_paths import ensure_feedforward_on_path

ensure_feedforward_on_path()

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from pi3.models.pi3 import Pi3
from pi3x_model.models.pi3x import Pi3X
from pi3x_model.utils.transforms_utils import recover_intrinsics_from_output


def extrinsic_to_colmap_format(extrinsics):
    """Convert extrinsic matrices to COLMAP format (quaternion + translation)."""
    num_cameras = extrinsics.shape[0]
    quaternions = []
    translations = []

    for i in range(num_cameras):
        # VGGT's extrinsic is camera-to-world (R|t) format
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]

        # Convert rotation matrix to quaternion
        # COLMAP quaternion format is [qw, qx, qy, qz]
        rot = Rotation.from_matrix(R)
        quat = rot.as_quat()  # scipy returns [x, y, z, w]
        quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]

        quaternions.append(quat)
        translations.append(t)

    return np.array(quaternions), np.array(translations)


def _download_file_from_url(url, filename):
    """Downloads a file from a URL, handling redirects."""
    try:
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()

        if response.status_code == 302:
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            response = requests.get(url, stream=True)
            response.raise_for_status()

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False


def _run_skyseg(onnx_session, input_size, image):
    """Runs sky segmentation inference using ONNX model."""
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def _segment_sky(image_path, onnx_session, mask_filename=None):
    """Segments sky from an image using an ONNX model."""
    image = cv2.imread(image_path)

    result_map = _run_skyseg(onnx_session, [320, 320], image)
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255

    if mask_filename is not None:
        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask)

    return output_mask


def _hash_point(point, scale=100):
    """Create a hash for a 3D point by quantizing coordinates."""
    quantized = tuple(np.round(point * scale).astype(int))
    return hash(quantized)


def filter_and_prepare_points(
    predictions,
    conf_threshold,
    mask_sky=False,
    mask_black_bg=False,
    mask_white_bg=False,
    stride=1,
    prediction_mode="Depthmap and Camera Branch",
):
    """
    Filter points based on confidence and prepare for COLMAP format.
    Implementation matches the conventions in the original VGGT code.
    """
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get(
                "world_points_conf", np.ones_like(pred_world_points[..., 0])
            )
        else:
            print(
                "Warning: world_points not found in predictions, falling back to depth-based points"
            )
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get(
                "depth_conf", np.ones_like(pred_world_points[..., 0])
            )
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get(
            "depth_conf", np.ones_like(pred_world_points[..., 0])
        )

    if "colmap_images" in predictions.keys():
        colors_rgb = predictions["colmap_images"]
    else:
        colors_rgb = predictions["images"]

    S, H, W = pred_world_points.shape[:3]
    if colors_rgb.shape[:3] != (S, H, W):
        print(f"Reshaping colors_rgb from {colors_rgb.shape} to match {(S, H, W, 3)}")
        reshaped_colors = np.zeros((S, H, W, 3), dtype=np.float32)
        for i in range(S):
            if i < len(colors_rgb):
                reshaped_colors[i] = cv2.resize(colors_rgb[i], (W, H))
        colors_rgb = reshaped_colors

    colors_rgb = (colors_rgb * 255).astype(np.uint8)

    if mask_sky:
        print("Applying sky segmentation mask")
        try:
            import onnxruntime

            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Created temporary directory for sky segmentation: {temp_dir}")
                temp_images_dir = os.path.join(temp_dir, "images")
                sky_masks_dir = os.path.join(temp_dir, "sky_masks")
                os.makedirs(temp_images_dir, exist_ok=True)
                os.makedirs(sky_masks_dir, exist_ok=True)

                image_list = []
                for i, img in enumerate(colors_rgb):
                    img_path = os.path.join(temp_images_dir, f"image_{i:04d}.png")
                    image_list.append(img_path)
                    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                skyseg_path = os.path.join(temp_dir, "skyseg.onnx")
                if not os.path.exists("skyseg.onnx"):
                    print("Downloading skyseg.onnx...")
                    download_success = _download_file_from_url(
                        "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                        skyseg_path,
                    )
                    if not download_success:
                        print("Failed to download skyseg model, skipping sky filtering")
                        mask_sky = False
                else:
                    shutil.copy("skyseg.onnx", skyseg_path)

                if mask_sky:
                    skyseg_session = onnxruntime.InferenceSession(skyseg_path)
                    sky_mask_list = []

                    for img_path in image_list:
                        mask_path = os.path.join(
                            sky_masks_dir, os.path.basename(img_path)
                        )
                        sky_mask = _segment_sky(img_path, skyseg_session, mask_path)

                        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                            sky_mask = cv2.resize(sky_mask, (W, H))

                        sky_mask_list.append(sky_mask)

                    sky_mask_array = np.array(sky_mask_list)
                    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
                    pred_world_points_conf = pred_world_points_conf * sky_mask_binary
                    print(f"Applied sky mask, shape: {sky_mask_binary.shape}")

        except (ImportError, Exception) as e:
            print(f"Error in sky segmentation: {e}")
            mask_sky = False

    vertices_3d = pred_world_points.reshape(-1, 3)
    conf = pred_world_points_conf.reshape(-1)
    colors_rgb_flat = colors_rgb.reshape(-1, 3)

    if len(conf) != len(colors_rgb_flat):
        print(
            f"WARNING: Shape mismatch between confidence ({len(conf)}) and colors ({len(colors_rgb_flat)})"
        )
        min_size = min(len(conf), len(colors_rgb_flat))
        conf = conf[:min_size]
        vertices_3d = vertices_3d[:min_size]
        colors_rgb_flat = colors_rgb_flat[:min_size]

    if conf_threshold == 0.0:
        conf_thres_value = 0.0
    else:
        conf_thres_value = np.percentile(conf, conf_threshold)

    print(
        f"Using confidence threshold: {conf_threshold}% (value: {conf_thres_value:.4f})"
    )
    conf_mask = (conf >= conf_thres_value) & (conf > 1e-5)

    if mask_black_bg:
        print("Filtering black background")
        black_bg_mask = colors_rgb_flat.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        print("Filtering white background")
        white_bg_mask = ~(
            (colors_rgb_flat[:, 0] > 240)
            & (colors_rgb_flat[:, 1] > 240)
            & (colors_rgb_flat[:, 2] > 240)
        )
        conf_mask = conf_mask & white_bg_mask

    filtered_vertices = vertices_3d[conf_mask]
    filtered_colors = colors_rgb_flat[conf_mask]

    if len(filtered_vertices) == 0:
        print("Warning: No points remaining after filtering. Using default point.")
        filtered_vertices = np.array([[0, 0, 0]])
        filtered_colors = np.array([[200, 200, 200]])

    print(f"Filtered to {len(filtered_vertices)} points")

    points3D = []
    point_indices = {}
    image_points2D = [[] for _ in range(len(pred_world_points))]

    print(f"Preparing points for COLMAP format with stride {stride}...")

    total_points = 0
    for img_idx in range(S):
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                flat_idx = img_idx * H * W + y * W + x

                if flat_idx >= len(conf):
                    continue

                if conf[flat_idx] < conf_thres_value or conf[flat_idx] <= 1e-5:
                    continue

                if mask_black_bg and colors_rgb_flat[flat_idx].sum() < 16:
                    continue

                if mask_white_bg and all(colors_rgb_flat[flat_idx] > 240):
                    continue

                point3D = vertices_3d[flat_idx]
                rgb = colors_rgb_flat[flat_idx]

                if not np.all(np.isfinite(point3D)):
                    continue

                point_hash = _hash_point(point3D, scale=100)

                if point_hash not in point_indices:
                    point_idx = len(points3D)
                    point_indices[point_hash] = point_idx

                    point_entry = {
                        "id": point_idx,
                        "xyz": point3D,
                        "rgb": rgb,
                        "error": 1.0,
                        "track": [(img_idx, len(image_points2D[img_idx]))],
                    }
                    points3D.append(point_entry)
                    total_points += 1
                else:
                    point_idx = point_indices[point_hash]
                    points3D[point_idx]["track"].append(
                        (img_idx, len(image_points2D[img_idx]))
                    )

                image_points2D[img_idx].append(
                    (x, y, point_indices[point_hash])
                )

    print(
        f"Prepared {len(points3D)} 3D points with {sum(len(pts) for pts in image_points2D)} observations for COLMAP"
    )
    return points3D, image_points2D


def write_colmap_cameras_txt(file_path, intrinsics, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.txt format."""
    with open(file_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(intrinsics)}\n")

        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1
            model = "PINHOLE"

            fx = intrinsic[0, 0]
            fy = intrinsic[1, 1]
            cx = intrinsic[0, 2]
            cy = intrinsic[1, 2]

            f.write(
                f"{camera_id} {model} {image_width} {image_height} {fx} {fy} {cx} {cy}\n"
            )


def write_colmap_images_txt(
    file_path, quaternions, translations, image_points2D, image_names, shared_camera=False
):
    """Write camera poses and keypoints to COLMAP images.txt format."""
    with open(file_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        num_points = sum(len(points) for points in image_points2D)
        avg_points = num_points / len(image_points2D) if image_points2D else 0
        f.write(
            f"# Number of images: {len(quaternions)}, mean observations per image: {avg_points:.1f}\n"
        )

        for i in range(len(quaternions)):
            image_id = i + 1
            camera_id = 1 if shared_camera else i + 1

            qw, qx, qy, qz = quaternions[i]
            tx, ty, tz = translations[i]

            f.write(
                f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {os.path.basename(image_names[i])}\n"
            )

            points_line = " ".join(
                [
                    f"{x} {y} {point3d_id+1}"
                    for x, y, point3d_id in image_points2D[i]
                ]
            )
            f.write(f"{points_line}\n")


def write_colmap_cameras_bin(file_path, intrinsics, image_width, image_height):
    """Write camera intrinsics to COLMAP cameras.bin format."""
    with open(file_path, 'wb') as fid:
        # Write number of cameras (uint64)
        fid.write(struct.pack('<Q', len(intrinsics)))
        
        for i, intrinsic in enumerate(intrinsics):
            camera_id = i + 1
            model_id = 1 
            
            fx = float(intrinsic[0, 0])
            fy = float(intrinsic[1, 1])
            cx = float(intrinsic[0, 2])
            cy = float(intrinsic[1, 2])
            
            # Camera ID (uint32)
            fid.write(struct.pack('<I', camera_id))
            # Model ID (uint32)
            fid.write(struct.pack('<I', model_id))
            # Width (uint64)
            fid.write(struct.pack('<Q', image_width))
            # Height (uint64)
            fid.write(struct.pack('<Q', image_height))
            
            # Parameters (double)
            fid.write(struct.pack('<dddd', fx, fy, cx, cy))

def write_colmap_images_bin(file_path, quaternions, translations, image_points2D, image_names, shared_camera=False):
    """Write camera poses and keypoints to COLMAP images.bin format."""
    with open(file_path, 'wb') as fid:
        # Write number of images (uint64)
        fid.write(struct.pack('<Q', len(quaternions)))
        
        for i in range(len(quaternions)):
            image_id = i + 1
            camera_id = 1 if shared_camera else i + 1
            
            qw, qx, qy, qz = quaternions[i].astype(float)
            tx, ty, tz = translations[i].astype(float)
            
            image_name = os.path.basename(image_names[i]).encode()
            points = image_points2D[i]
            
            # Image ID (uint32)
            fid.write(struct.pack('<I', image_id))
            # Quaternion (double): qw, qx, qy, qz
            fid.write(struct.pack('<dddd', qw, qx, qy, qz))
            # Translation (double): tx, ty, tz
            fid.write(struct.pack('<ddd', tx, ty, tz))
            # Camera ID (uint32)
            fid.write(struct.pack('<I', camera_id))
            # Image name
            fid.write(struct.pack('<I', len(image_name)))
            fid.write(image_name)
            
            # Write number of 2D points (uint64)
            fid.write(struct.pack('<Q', len(points)))
            
            # Write 2D points: x, y, point3D_id
            for x, y, point3d_id in points:
                fid.write(struct.pack('<dd', float(x), float(y)))
                fid.write(struct.pack('<Q', point3d_id + 1))

def write_colmap_points3D_bin(file_path, points3D):
    """Write 3D points and tracks to COLMAP points3D.bin format."""
    with open(file_path, 'wb') as fid:
        # Write number of points (uint64)
        fid.write(struct.pack('<Q', len(points3D)))
        
        for point in points3D:
            point_id = point["id"] + 1
            x, y, z = point["xyz"].astype(float)
            r, g, b = point["rgb"].astype(np.uint8)
            error = float(point["error"])
            track = point["track"]
            
            # Point ID (uint64)
            fid.write(struct.pack('<Q', point_id))
            # Position (double): x, y, z
            fid.write(struct.pack('<ddd', x, y, z))
            # Color (uint8): r, g, b
            fid.write(struct.pack('<BBB', int(r), int(g), int(b)))
            # Error (double)
            fid.write(struct.pack('<d', error))
            
            # Track: list of (image_id, point2D_idx)
            fid.write(struct.pack('<Q', len(track)))
            for img_id, point2d_idx in track:
                fid.write(struct.pack('<II', img_id + 1, point2d_idx))


def write_colmap_points3D_txt(file_path, points3D):
    """Write 3D points and tracks to COLMAP points3D.txt format."""
    with open(file_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )

        avg_track_length = (
            sum(len(point["track"]) for point in points3D) / len(points3D)
            if points3D
            else 0
        )
        f.write(
            f"# Number of points: {len(points3D)}, mean track length: {avg_track_length:.4f}\n"
        )

        for point in points3D:
            point_id = point["id"] + 1
            x, y, z = point["xyz"]
            r, g, b = point["rgb"]
            error = point["error"]

            track = " ".join(
                f"{img_id+1} {point2d_idx}" for img_id, point2d_idx in point["track"]
            )

            f.write(
                f"{point_id} {x} {y} {z} {int(r)} {int(g)} {int(b)} {error} {track}\n"
            )



def load_model(model_name="vggt", device=None):
    """Load and initialize a supported geometric foundation model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if model_name == 'vggt':
        model = VGGT.from_pretrained("facebook/VGGT-1B")
    elif model_name == 'pi3':
        model = Pi3.from_pretrained("yyfz233/Pi3")
    elif model_name == 'pi3x':
        model = Pi3X.from_pretrained("yyfz233/Pi3X")
        # else:
        #     model = Pi3X()
        #     if pi3x_ckpt.endswith(".safetensors"):
        #         from safetensors.torch import load_file
        #         state_dict = load_file(pi3x_ckpt)
        #     else:
        #         state_dict = torch.load(pi3x_ckpt, map_location="cpu", weights_only=False)
        #         if isinstance(state_dict, dict) and "model" in state_dict:
        #             state_dict = state_dict["model"]
        #     model.load_state_dict(state_dict, strict=False)
    else:
        raise NotImplementedError("Other model backbones are not implemented!")

    
    model.eval()
    model = model.to(device)
    return model, device



def run_inference_step_by_step(model, batches, size_hw, device, need_features=False, pi3x_intrinsics_method="lstsq"):
    """
    Output:
     - extrinsic: (N, 3, 4)
     - intrinsic: (N, 3, 3)
     - depth: (N, H, W, 1)
     - depth_conf: (N, H, W)
    """
    # Construct a customized prediction
    predictions = []
    start = time.time()

    if isinstance(model, VGGT):
        for i, images in enumerate(batches):
            prediction = dict()

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    images = images[None].to(device)  # add batch dimension
                    aggregated_tokens_list, ps_idx, patch_tokens = model.aggregator(images)
            
                    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                    # Predict depth maps
                    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
                    # No need to predict point maps here
                    # Predict feature maps for tracking
                    if need_features:
                        feature_maps = model.track_head.feature_extractor(aggregated_tokens_list, images, ps_idx)

            prediction['depth'] = depth_map.to(dtype=torch.float32, device='cpu')
            prediction['depth_conf'] = depth_conf.to(dtype=torch.float32, device='cpu')
            prediction['pose_enc'] = pose_enc.to(dtype=torch.float32, device='cpu')

            extri, intri = pose_encoding_to_extri_intri(prediction["pose_enc"], size_hw)
            prediction['extrinsic'] = extri.squeeze(0)
            prediction['intrinsic'] = intri.squeeze(0)
            prediction['world_points'] = unproject_depth_map_to_point_map(prediction['depth'].squeeze(0), extri.squeeze(0), intri.squeeze(0))

            if need_features:
                prediction['features'] = feature_maps.to(dtype=torch.float32, device='cpu')
                
            for key in patch_tokens.keys():
                if isinstance(patch_tokens[key], torch.Tensor):
                    val = patch_tokens[key].to(dtype=torch.float32, device='cpu')
                    patch_tokens[key] = val
            # prediction['dino_features'] = patch_tokens.to("cpu")

            del aggregated_tokens_list, ps_idx, patch_tokens, pose_enc, depth_map, depth_conf, images

            predictions.append(prediction)
            torch.cuda.empty_cache()
            gc.collect()
        
    elif isinstance(model, Pi3):
        for i, images in enumerate(batches):
            prediction = dict()
            with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        images = images[None].to(device)  # add batch dimension
                        res = model(images)
            
            prediction['extrinsic'] = remove_homogeneous_row(torch.linalg.inv(res['camera_poses'].squeeze(0))).to(dtype=torch.float32, device='cpu')
            prediction['world_points'] = res['points'].to(dtype=torch.float32, device='cpu').squeeze(0)
            prediction['depth_conf'] = res['conf'].to(dtype=torch.float32, device='cpu').squeeze(-1)
            prediction['depth_conf'] = torch.sigmoid(prediction['depth_conf'])
            prediction['local_points'] = res['local_points'].to(dtype=torch.float32, device='cpu').squeeze(0)
            # prediction['dino_features'] = res['dino_features']

            intrinsic, depth = estimate_intrinsics_and_depth(res['local_points'].squeeze(0))
            prediction['intrinsic'] = intrinsic.to(dtype=torch.float32, device='cpu').squeeze(0)
            prediction['depth'] = compute_depth(prediction['world_points'], prediction['extrinsic']).to(dtype=torch.float32, device='cpu').unsqueeze(0).unsqueeze(-1)

            predictions.append(prediction)

            del res, images
            
            gc.collect()
            torch.cuda.empty_cache()

    elif isinstance(model, Pi3X):
        if pi3x_intrinsics_method not in ("lstsq", "moge"):
            raise ValueError(f"Unsupported pi3x_intrinsics_method: {pi3x_intrinsics_method}")

        for i, images in enumerate(batches):
            prediction = dict()
            if images.shape[-2] % 14 != 0 or images.shape[-1] % 14 != 0:
                raise ValueError(
                    "Pi3X requires image height and width to be multiples of 14. "
                    f"Got {tuple(images.shape[-2:])} for subset {i}."
                )

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    images = images[None].to(device)  # add batch dimension
                    res = model(imgs=images)

            c2w = res['camera_poses'].squeeze(0)
            prediction['extrinsic'] = remove_homogeneous_row(torch.linalg.inv(c2w)).to(dtype=torch.float32, device='cpu')
            prediction['world_points'] = res['points'].to(dtype=torch.float32, device='cpu').squeeze(0)
            prediction['depth_conf'] = res['conf'].to(dtype=torch.float32, device='cpu').squeeze(0).squeeze(-1)
            prediction['depth_conf'] = torch.sigmoid(prediction['depth_conf'])
            prediction['local_points'] = res['local_points'].to(dtype=torch.float32, device='cpu').squeeze(0)

            if pi3x_intrinsics_method == "moge":
                intrinsic = recover_intrinsics_from_output(res)
                prediction['intrinsic'] = torch.from_numpy(intrinsic).to(dtype=torch.float32, device='cpu')
                depth = res['local_points'].squeeze(0)[..., 2]
            else:
                intrinsic, depth = estimate_intrinsics_and_depth(res['local_points'].squeeze(0))
                prediction['intrinsic'] = intrinsic.to(dtype=torch.float32, device='cpu')

            prediction['depth'] = depth.to(dtype=torch.float32, device='cpu').unsqueeze(0).unsqueeze(-1)

            predictions.append(prediction)

            del res, images, c2w
            
            gc.collect()
            torch.cuda.empty_cache()
            

    end = time.time()

    print(f"[INFERENCE] Time used: {end - start}s. ")
    
    return predictions


def compute_depth(points, extrin):
    """
    points: (N, H, W, 3) in world coordinates
    extrin: (N, 3, 4) camera extrinsics (world -> camera)
    intrin: (N, 3, 3) camera intrinsics (unused for depth directly)
    
    Returns:
        depth: (N, H, W)
    """
    N, H, W, _ = points.shape

    # Convert to homogeneous coords: (N, H, W, 4)
    ones = torch.ones((N, H, W, 1), dtype=points.dtype, device=points.device)
    homog_points = torch.cat([points, ones], dim=-1)

    # Transform world -> camera: (N, H, W, 3)
    # First expand extrin to match dimensions
    cam_points = torch.einsum('nij,nhwj->nhwi', extrin, homog_points)

    # Depth is the z-coordinate in camera space
    depth = cam_points[..., 2]

    return depth


def _world_points_from_local_frame(local_points_frame, extrinsic_frame):
    extrinsic_frame = np.asarray(extrinsic_frame, dtype=np.float32)
    if extrinsic_frame.shape == (4, 4):
        extrinsic_frame = extrinsic_frame[:3, :4]
    elif extrinsic_frame.shape != (3, 4):
        raise ValueError(f"Expected extrinsic frame shape (3, 4) or (4, 4), got {extrinsic_frame.shape}")

    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :4] = extrinsic_frame
    c2w = np.linalg.inv(w2c)
    return local_points_frame.astype(np.float32) @ c2w[:3, :3].T + c2w[:3, 3]


def _image_frame_to_colors(image_frame, target_hw):
    if isinstance(image_frame, torch.Tensor):
        image_frame = image_frame.detach().cpu().numpy()
    if image_frame.shape[0] == 3:
        image_frame = np.transpose(image_frame, (1, 2, 0))
    height, width = target_hw
    return (cv2.resize(image_frame, (width, height)) * 255).clip(0, 255).astype(np.uint8)


def _write_binary_ply_header(file_obj, num_vertices):
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_vertices}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    file_obj.write(header.encode("ascii"))


def _write_binary_ply_vertices(file_obj, points, colors):
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("red", np.uint8),
            ("green", np.uint8),
            ("blue", np.uint8),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    file_obj.write(vertices.tobytes())


def collect_dense_world_points(
    local_points,
    extrinsic,
    images,
    conf,
    conf_threshold=50.0,
    stride=1,
    max_points=2_000_000,
    include_colors=True,
):
    """
    Collect a sampled dense local point map into one world-space point set.

    The returned points are intended to be shared by dense PLY export and
    camera-depth projection so both outputs use the same sampled global cloud.
    """
    if isinstance(local_points, torch.Tensor):
        local_points = local_points.detach().cpu().numpy()
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.detach().cpu().numpy()
    if isinstance(conf, torch.Tensor):
        conf = conf.detach().cpu().numpy()

    local_points = np.asarray(local_points, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    conf = np.asarray(conf)
    if conf.ndim == 4 and conf.shape[-1] == 1:
        conf = conf[..., 0]

    num_frames, height, width, _ = local_points.shape
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if max_points is not None and max_points <= 0:
        raise ValueError(f"max_points must be positive or None, got {max_points}")
    if conf.shape[:3] != (num_frames, height, width):
        raise ValueError(
            "Confidence shape must match local_points spatial shape. "
            f"Got conf={conf.shape}, local_points={local_points.shape}."
        )
    if len(extrinsic) != num_frames:
        raise ValueError(f"Expected {num_frames} extrinsics, got {len(extrinsic)}")

    threshold_value = 0.0 if conf_threshold == 0.0 else np.percentile(conf, conf_threshold)
    valid_counts = []
    for i in range(num_frames):
        conf_i = conf[i, ::stride, ::stride]
        local_i = local_points[i, ::stride, ::stride]
        valid = (conf_i >= threshold_value) & (conf_i > 1e-5) & np.isfinite(local_i).all(axis=-1)
        valid_counts.append(int(np.count_nonzero(valid)))

    total_valid = int(np.sum(valid_counts))
    stats = {
        "num_frames": int(num_frames),
        "height": int(height),
        "width": int(width),
        "stride": int(stride),
        "conf_threshold": float(conf_threshold),
        "conf_threshold_value": float(threshold_value),
        "num_valid_points_before_cap": int(total_valid),
        "max_points": None if max_points is None else int(max_points),
    }
    if total_valid == 0:
        stats["num_points"] = 0
        empty_colors = np.empty((0, 3), dtype=np.uint8) if include_colors else None
        return np.empty((0, 3), dtype=np.float32), empty_colors, stats

    frame_quotas = np.asarray(valid_counts, dtype=np.int64)
    if max_points is not None and total_valid > max_points:
        max_points = int(max_points)
        valid_counts_np = np.asarray(valid_counts, dtype=np.float64)
        ideal_quotas = valid_counts_np * (max_points / float(total_valid))
        frame_quotas = np.floor(ideal_quotas).astype(np.int64)
        remainder = max_points - int(frame_quotas.sum())
        if remainder > 0:
            order = np.argsort(-(ideal_quotas - frame_quotas))
            frame_quotas[order[:remainder]] += 1

    total_points = int(frame_quotas.sum())
    world_points = np.empty((total_points, 3), dtype=np.float32)
    colors = np.empty((total_points, 3), dtype=np.uint8) if include_colors else None
    offset = 0
    for i in range(num_frames):
        quota = int(frame_quotas[i])
        if quota <= 0:
            continue

        local_i = local_points[i, ::stride, ::stride]
        conf_i = conf[i, ::stride, ::stride]
        valid = (conf_i >= threshold_value) & (conf_i > 1e-5) & np.isfinite(local_i).all(axis=-1)
        if not np.any(valid):
            continue

        valid_indices = np.flatnonzero(valid.reshape(-1))
        if quota < len(valid_indices):
            sample_positions = np.linspace(0, len(valid_indices) - 1, quota, dtype=np.int64)
            valid_indices = valid_indices[sample_positions]

        local_flat = local_i.reshape(-1, 3)
        world_i = _world_points_from_local_frame(local_flat[valid_indices], extrinsic[i])
        world_points[offset : offset + len(world_i)] = world_i

        if include_colors:
            color_i = _image_frame_to_colors(images[i], (height, width))[::stride, ::stride]
            color_flat = color_i.reshape(-1, 3)
            colors[offset : offset + len(world_i)] = color_flat[valid_indices]

        offset += len(world_i)

    if offset != total_points:
        world_points = world_points[:offset]
        if include_colors:
            colors = colors[:offset]
        total_points = int(offset)

    stats["num_points"] = int(total_points)
    return world_points, colors, stats


def export_dense_world_points_ply(output_path, world_points, colors, stats=None, write_chunk_size=1_000_000):
    world_points = np.asarray(world_points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(f"Expected world_points shape (P, 3), got {world_points.shape}")
    if colors.shape != (world_points.shape[0], 3):
        raise ValueError(f"Expected colors shape ({world_points.shape[0]}, 3), got {colors.shape}")
    if write_chunk_size <= 0:
        raise ValueError(f"write_chunk_size must be positive, got {write_chunk_size}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as file_obj:
        _write_binary_ply_header(file_obj, len(world_points))
        for start in range(0, len(world_points), write_chunk_size):
            end = min(start + write_chunk_size, len(world_points))
            _write_binary_ply_vertices(file_obj, world_points[start:end], colors[start:end])

    if stats is None:
        print(f"[OUTPUT WRITING] Exported {len(world_points)} dense model points to {output_path}")
    elif "conf_threshold" in stats:
        print(
            f"[OUTPUT WRITING] Exported {len(world_points)} dense model points to {output_path} "
            f"(valid={stats['num_valid_points_before_cap']}, max_points={stats['max_points']}, "
            f"conf percentile={stats['conf_threshold']}, value={stats['conf_threshold_value']:.4f}, "
            f"stride={stats['stride']})"
        )
    else:
        print(
            f"[OUTPUT WRITING] Exported {len(world_points)} dense model points to {output_path} "
            f"(valid={stats['num_valid_points_before_cap']}, max_points={stats['max_points']}, "
            f"stride={stats['stride']})"
        )


def _depth_npy_path_for_image(depth_npy_dir, image_name, frame_idx):
    stem = os.path.splitext(os.path.basename(str(image_name)))[0]
    path = os.path.join(depth_npy_dir, f"{stem}.npy")
    if os.path.exists(path):
        return path, stem

    fallback_stem = f"frame_{frame_idx:04d}"
    fallback = os.path.join(depth_npy_dir, f"{fallback_stem}.npy")
    if os.path.exists(fallback):
        return fallback, fallback_stem
    return path, stem


def _local_points_from_depth_frame(depth_frame, intrinsic, stride):
    height, width = depth_frame.shape
    y_coords = np.arange(0, height, stride, dtype=np.float32)
    x_coords = np.arange(0, width, stride, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(x_coords, y_coords)
    depth = depth_frame[::stride, ::stride].astype(np.float32, copy=False)

    fx = float(intrinsic[0, 0])
    fy = float(intrinsic[1, 1])
    cx = float(intrinsic[0, 2])
    cy = float(intrinsic[1, 2])
    if abs(fx) < 1e-8 or abs(fy) < 1e-8:
        raise ValueError(f"Invalid focal lengths fx={fx}, fy={fy}")

    x = (u_grid - cx) / fx * depth
    y = (v_grid - cy) / fy * depth
    return np.stack([x, y, depth], axis=-1)


def collect_depth_npy_world_points(
    depth_npy_dir,
    image_names,
    images,
    extrinsic,
    intrinsic,
    stride=1,
    max_points=2_000_000,
    min_depth=1e-6,
    max_depth=None,
    valid_mask_depth_npy_dir=None,
    valid_mask_min_depth=1e-6,
):
    """
    Back-project float32 depth .npy maps into a sampled world-space point set.
    """
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.detach().cpu().numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.detach().cpu().numpy()

    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    num_frames = len(extrinsic)
    if intrinsic.ndim == 2:
        intrinsic = np.repeat(intrinsic[None], num_frames, axis=0)
    if intrinsic.shape != (num_frames, 3, 3):
        raise ValueError(f"Expected intrinsic shape (3, 3) or ({num_frames}, 3, 3), got {intrinsic.shape}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if max_points is not None and max_points <= 0:
        raise ValueError(f"max_points must be positive or None, got {max_points}")

    frame_items = []
    valid_counts = []
    missing = 0
    for frame_idx in range(num_frames):
        image_name = image_names[frame_idx] if frame_idx < len(image_names) else f"frame_{frame_idx:04d}.png"
        depth_path, stem = _depth_npy_path_for_image(depth_npy_dir, image_name, frame_idx)
        if not os.path.exists(depth_path):
            frame_items.append((None, stem))
            valid_counts.append(0)
            missing += 1
            continue

        depth_frame = np.load(depth_path).astype(np.float32)
        depth_frame = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
        depth_s = depth_frame[::stride, ::stride]
        valid = np.isfinite(depth_s) & (depth_s > min_depth)
        if max_depth is not None:
            valid &= depth_s <= max_depth
        if valid_mask_depth_npy_dir is not None:
            mask_path, _ = _depth_npy_path_for_image(valid_mask_depth_npy_dir, image_name, frame_idx)
            if os.path.exists(mask_path):
                mask_depth = np.load(mask_path).astype(np.float32)
                mask_depth = np.nan_to_num(mask_depth, nan=0.0, posinf=0.0, neginf=0.0)
                if mask_depth.shape != depth_frame.shape:
                    mask_depth = cv2.resize(mask_depth, (depth_frame.shape[1], depth_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_depth_s = mask_depth[::stride, ::stride]
                valid &= np.isfinite(mask_depth_s) & (mask_depth_s > valid_mask_min_depth)
            else:
                valid &= False
        frame_items.append((depth_path, stem))
        valid_counts.append(int(np.count_nonzero(valid)))

    total_valid = int(np.sum(valid_counts))
    stats = {
        "num_frames": int(num_frames),
        "num_missing_depths": int(missing),
        "stride": int(stride),
        "min_depth": float(min_depth),
        "max_depth": None if max_depth is None else float(max_depth),
        "valid_mask_depth_npy_dir": None if valid_mask_depth_npy_dir is None else str(valid_mask_depth_npy_dir),
        "valid_mask_min_depth": float(valid_mask_min_depth),
        "num_valid_points_before_cap": int(total_valid),
        "max_points": None if max_points is None else int(max_points),
    }
    if total_valid == 0:
        stats["num_points"] = 0
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), stats

    frame_quotas = np.asarray(valid_counts, dtype=np.int64)
    if max_points is not None and total_valid > max_points:
        max_points = int(max_points)
        valid_counts_np = np.asarray(valid_counts, dtype=np.float64)
        ideal_quotas = valid_counts_np * (max_points / float(total_valid))
        frame_quotas = np.floor(ideal_quotas).astype(np.int64)
        remainder = max_points - int(frame_quotas.sum())
        if remainder > 0:
            order = np.argsort(-(ideal_quotas - frame_quotas))
            frame_quotas[order[:remainder]] += 1

    total_points = int(frame_quotas.sum())
    world_points = np.empty((total_points, 3), dtype=np.float32)
    colors = np.empty((total_points, 3), dtype=np.uint8)
    offset = 0
    for frame_idx, ((depth_path, _stem), quota) in enumerate(zip(frame_items, frame_quotas)):
        quota = int(quota)
        if depth_path is None or quota <= 0:
            continue

        depth_frame = np.load(depth_path).astype(np.float32)
        depth_frame = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
        local_i = _local_points_from_depth_frame(depth_frame, intrinsic[frame_idx], stride)
        depth_s = local_i[..., 2]
        valid = np.isfinite(depth_s) & (depth_s > min_depth) & np.isfinite(local_i).all(axis=-1)
        if max_depth is not None:
            valid &= depth_s <= max_depth
        if valid_mask_depth_npy_dir is not None:
            image_name = image_names[frame_idx] if frame_idx < len(image_names) else f"frame_{frame_idx:04d}.png"
            mask_path, _ = _depth_npy_path_for_image(valid_mask_depth_npy_dir, image_name, frame_idx)
            if os.path.exists(mask_path):
                mask_depth = np.load(mask_path).astype(np.float32)
                mask_depth = np.nan_to_num(mask_depth, nan=0.0, posinf=0.0, neginf=0.0)
                if mask_depth.shape != depth_frame.shape:
                    mask_depth = cv2.resize(mask_depth, (depth_frame.shape[1], depth_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_depth_s = mask_depth[::stride, ::stride]
                valid &= np.isfinite(mask_depth_s) & (mask_depth_s > valid_mask_min_depth)
            else:
                valid &= False
        if not np.any(valid):
            continue

        valid_indices = np.flatnonzero(valid.reshape(-1))
        if quota < len(valid_indices):
            sample_positions = np.linspace(0, len(valid_indices) - 1, quota, dtype=np.int64)
            valid_indices = valid_indices[sample_positions]

        local_flat = local_i.reshape(-1, 3)
        world_i = _world_points_from_local_frame(local_flat[valid_indices], extrinsic[frame_idx])
        color_i = _image_frame_to_colors(images[frame_idx], depth_frame.shape)[::stride, ::stride]
        color_flat = color_i.reshape(-1, 3)
        world_points[offset : offset + len(world_i)] = world_i
        colors[offset : offset + len(world_i)] = color_flat[valid_indices]
        offset += len(world_i)

    if offset != total_points:
        world_points = world_points[:offset]
        colors = colors[:offset]
        total_points = int(offset)

    stats["num_points"] = int(total_points)
    return world_points, colors, stats


def export_depth_npy_world_points_ply(
    output_path,
    depth_npy_dir,
    image_names,
    images,
    extrinsic,
    intrinsic,
    stride=1,
    max_points=2_000_000,
    min_depth=1e-6,
    max_depth=None,
    valid_mask_depth_npy_dir=None,
    valid_mask_min_depth=1e-6,
):
    world_points, colors, stats = collect_depth_npy_world_points(
        depth_npy_dir,
        image_names,
        images,
        extrinsic,
        intrinsic,
        stride=stride,
        max_points=max_points,
        min_depth=min_depth,
        max_depth=max_depth,
        valid_mask_depth_npy_dir=valid_mask_depth_npy_dir,
        valid_mask_min_depth=valid_mask_min_depth,
    )
    if len(world_points) == 0:
        print(f"[OUTPUT WRITING] No depth points passed filtering for {output_path}")
        return stats
    export_dense_world_points_ply(output_path, world_points, colors, stats=stats)
    return stats


def export_dense_local_point_map_ply(
    output_path,
    local_points,
    extrinsic,
    images,
    conf,
    conf_threshold=50.0,
    stride=1,
    max_points=2_000_000,
):
    """
    Export a dense model point cloud to PLY using local point maps and final poses.
    """
    world_points, colors, stats = collect_dense_world_points(
        local_points,
        extrinsic,
        images,
        conf,
        conf_threshold=conf_threshold,
        stride=stride,
        max_points=max_points,
        include_colors=True,
    )
    if len(world_points) == 0:
        print(f"[OUTPUT WRITING] No dense points passed filtering for {output_path}")
        return
    export_dense_world_points_ply(output_path, world_points, colors, stats=stats)


def project_world_points_to_depth(world_points, extrinsic, intrinsics, image_size, chunk_size=1_000_000):
    """
    Project world-space points into each camera and build depth maps via z-buffer.

    Args:
        world_points: (P, 3) merged dense world points.
        extrinsic:    (N, 3, 4) or (N, 4, 4) world-to-camera matrices.
        intrinsics:   (N, 3, 3) camera intrinsics.
        image_size:   (H, W).
    Returns:
        depth_maps: (N, H, W), metric camera-space z depth.
    """
    world_points = np.asarray(world_points, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    H, W = image_size
    N = extrinsic.shape[0]
    depth_maps = np.zeros((N, H, W), dtype=np.float32)

    if world_points.size == 0:
        return depth_maps
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(f"Expected world_points shape (P, 3), got {world_points.shape}")
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected extrinsic shape (N, 3, 4) or (N, 4, 4), got {extrinsic.shape}")
    if intrinsics.shape != (N, 3, 3):
        raise ValueError(f"Expected intrinsics shape ({N}, 3, 3), got {intrinsics.shape}")
    if chunk_size == -1:
        chunk_size = world_points.shape[0]
    elif chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, or -1 for no chunking, got {chunk_size}")

    for i in range(N):
        w2c = extrinsic[i, :3, :4].astype(np.float32, copy=False)
        R = w2c[:, :3]
        t = w2c[:, 3]
        K = intrinsics[i].astype(np.float32, copy=False)
        depth_flat = np.full(H * W, np.inf, dtype=np.float32)

        for start in range(0, world_points.shape[0], chunk_size):
            points = world_points[start : start + chunk_size]
            cam_points = points @ R.T + t
            z = cam_points[:, 2]
            valid = np.isfinite(z) & (z > 1e-6)
            if not np.any(valid):
                continue

            cam_points = cam_points[valid]
            z = z[valid].astype(np.float32, copy=False)
            u = K[0, 0] * (cam_points[:, 0] / z) + K[0, 2]
            v = K[1, 1] * (cam_points[:, 1] / z) + K[1, 2]

            valid_uv = np.isfinite(u) & np.isfinite(v)
            if not np.any(valid_uv):
                continue

            u = np.rint(u[valid_uv]).astype(np.int32)
            v = np.rint(v[valid_uv]).astype(np.int32)
            z = z[valid_uv]

            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not np.any(in_bounds):
                continue

            idx = v[in_bounds] * W + u[in_bounds]
            np.minimum.at(depth_flat, idx, z[in_bounds])

        depth = depth_flat.reshape(H, W)
        depth[~np.isfinite(depth)] = 0.0
        depth_maps[i] = depth

    return depth_maps


@torch.no_grad()
def project_world_points_to_depth_torch(
    world_points,
    extrinsic,
    intrinsics,
    image_size,
    point_chunk_size=1_000_000,
    camera_batch_size=4,
    device="cuda",
):
    """
    CUDA implementation of world-point z-buffer projection using torch scatter_reduce.

    The function batches both cameras and points to keep a 24GB GPU within a
    predictable memory envelope. Depth is returned on CPU as float32.
    """
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA depth projection requested but torch.cuda.is_available() is False")

    world_points = np.asarray(world_points, dtype=np.float32)
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    H, W = image_size
    N = extrinsic.shape[0]
    depth_maps = np.zeros((N, H, W), dtype=np.float32)

    if world_points.size == 0:
        return depth_maps
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        raise ValueError(f"Expected world_points shape (P, 3), got {world_points.shape}")
    if extrinsic.ndim != 3 or extrinsic.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"Expected extrinsic shape (N, 3, 4) or (N, 4, 4), got {extrinsic.shape}")
    if intrinsics.shape != (N, 3, 3):
        raise ValueError(f"Expected intrinsics shape ({N}, 3, 3), got {intrinsics.shape}")
    if point_chunk_size == -1:
        point_chunk_size = world_points.shape[0]
    elif point_chunk_size <= 0:
        raise ValueError(f"point_chunk_size must be positive, or -1 for no chunking, got {point_chunk_size}")
    if camera_batch_size <= 0:
        raise ValueError(f"camera_batch_size must be positive, got {camera_batch_size}")

    device = torch.device(device)
    extrinsic_t = torch.as_tensor(extrinsic[:, :3, :4], dtype=torch.float32, device=device)
    intrinsics_t = torch.as_tensor(intrinsics, dtype=torch.float32, device=device)
    image_pixels = H * W

    for camera_start in range(0, N, camera_batch_size):
        camera_end = min(camera_start + camera_batch_size, N)
        batch_size = camera_end - camera_start
        R = extrinsic_t[camera_start:camera_end, :, :3]
        t = extrinsic_t[camera_start:camera_end, :, 3]
        K = intrinsics_t[camera_start:camera_end]
        depth_flat = torch.full(
            (batch_size * image_pixels,),
            torch.inf,
            dtype=torch.float32,
            device=device,
        )
        camera_offsets = torch.arange(batch_size, device=device, dtype=torch.int64).view(batch_size, 1) * image_pixels

        for point_start in range(0, world_points.shape[0], point_chunk_size):
            point_end = min(point_start + point_chunk_size, world_points.shape[0])
            points = torch.as_tensor(world_points[point_start:point_end], dtype=torch.float32, device=device)

            cam_points = torch.einsum("pj,bij->bpi", points, R) + t[:, None, :]
            z = cam_points[..., 2]
            valid = torch.isfinite(z) & (z > 1e-6)
            if not torch.any(valid):
                continue

            safe_z = torch.where(valid, z, torch.ones_like(z))
            u = K[:, None, 0, 0] * (cam_points[..., 0] / safe_z) + K[:, None, 0, 2]
            v = K[:, None, 1, 1] * (cam_points[..., 1] / safe_z) + K[:, None, 1, 2]
            u_round = torch.round(u).to(torch.int64)
            v_round = torch.round(v).to(torch.int64)

            in_bounds = (
                valid
                & torch.isfinite(u)
                & torch.isfinite(v)
                & (u_round >= 0)
                & (u_round < W)
                & (v_round >= 0)
                & (v_round < H)
            )
            if not torch.any(in_bounds):
                continue

            idx = camera_offsets + v_round * W + u_round
            depth_flat.scatter_reduce_(
                0,
                idx[in_bounds],
                z[in_bounds],
                reduce="amin",
                include_self=True,
            )

        depth_batch = depth_flat.view(batch_size, H, W)
        depth_batch = torch.where(torch.isfinite(depth_batch), depth_batch, torch.zeros_like(depth_batch))
        depth_maps[camera_start:camera_end] = depth_batch.cpu().numpy()

    return depth_maps


def save_depth_pngs(depth_np, image_names, output_dir):
    """
    Save depth maps in three formats:
      1) uint16 millimeter PNGs under depth_u16
      2) uint8 pseudo-color PNGs under depth_vis
      3) float32 NPY files under depth_npy
    """
    depth_u16_dir = os.path.join(output_dir, "depth_u16")
    depth_vis_dir = os.path.join(output_dir, "depth_vis")
    depth_npy_dir = os.path.join(output_dir, "depth_npy")
    os.makedirs(depth_u16_dir, exist_ok=True)
    os.makedirs(depth_vis_dir, exist_ok=True)
    os.makedirs(depth_npy_dir, exist_ok=True)

    for idx in range(depth_np.shape[0]):
        depth_frame = np.nan_to_num(depth_np[idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        depth_u16 = np.clip(depth_frame * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)

        if idx < len(image_names):
            stem = os.path.splitext(os.path.basename(str(image_names[idx])))[0]
        else:
            stem = f"frame_{idx:04d}"
        base_name = stem + ".png"

        cv2.imwrite(os.path.join(depth_u16_dir, base_name), depth_u16)
        np.save(os.path.join(depth_npy_dir, stem + ".npy"), depth_frame.astype(np.float32, copy=False))

        valid_mask = np.isfinite(depth_frame) & (depth_frame > 0)
        if np.any(valid_mask):
            d = depth_frame[valid_mask]
            d_min = np.percentile(d, 2.0)
            d_max = np.percentile(d, 98.0)
            if d_max <= d_min:
                d_max = d_min + 1e-6

            depth_norm = (depth_frame - d_min) / (d_max - d_min)
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
            depth_vis_u8 = (depth_norm * 255.0).astype(np.uint8)
            depth_vis_u8[~valid_mask] = 0
            depth_color = cv2.applyColorMap(depth_vis_u8, cv2.COLORMAP_TURBO)
        else:
            h, w = depth_frame.shape[:2]
            depth_color = np.zeros((h, w, 3), dtype=np.uint8)

        cv2.imwrite(os.path.join(depth_vis_dir, base_name), depth_color)


def export_dense_projected_depth_maps(
    output_dir,
    local_points,
    extrinsic,
    intrinsic,
    image_names,
    conf,
    conf_threshold=50.0,
    stride=1,
    max_points=20_000_000,
    chunk_size=1_000_000,
    world_points=None,
    image_size=None,
    backend="cuda",
    camera_batch_size=4,
):
    """
    Merge dense local point maps with final poses, project the merged cloud to every camera,
    and save per-view depth maps.
    """
    if isinstance(extrinsic, torch.Tensor):
        extrinsic = extrinsic.detach().cpu().numpy()
    if isinstance(intrinsic, torch.Tensor):
        intrinsic = intrinsic.detach().cpu().numpy()

    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    intrinsic = np.asarray(intrinsic, dtype=np.float32)

    if world_points is None:
        if local_points is None or conf is None:
            raise ValueError("local_points and conf are required when world_points is not provided")
        world_points, _, collect_stats = collect_dense_world_points(
            local_points,
            extrinsic,
            images=None,
            conf=conf,
            conf_threshold=conf_threshold,
            stride=stride,
            max_points=max_points,
            include_colors=False,
        )
        if image_size is None:
            image_size = (collect_stats["height"], collect_stats["width"])
    else:
        world_points = np.asarray(world_points, dtype=np.float32)

    if image_size is None:
        if local_points is None:
            raise ValueError("image_size is required when projecting precomputed world_points without local_points")
        local_shape = local_points.shape if not isinstance(local_points, torch.Tensor) else tuple(local_points.shape)
        image_size = (int(local_shape[1]), int(local_shape[2]))

    num_frames = len(extrinsic)
    if len(intrinsic) != num_frames:
        raise ValueError(f"Expected {num_frames} intrinsics, got {len(intrinsic)}")

    if len(world_points) == 0:
        print(f"[DEPTH EXPORT] No dense points passed filtering for {output_dir}")
        stats = {
            "num_depth_frames": int(num_frames),
            "num_merged_points": 0,
            "num_valid_points_before_cap": 0,
            "num_nonzero_depth_pixels": 0,
            "backend": backend,
            "output_dir": output_dir,
        }
        return stats

    backend = backend.lower()
    if backend == "cuda":
        depth_np = project_world_points_to_depth_torch(
            world_points=world_points,
            extrinsic=extrinsic,
            intrinsics=intrinsic,
            image_size=image_size,
            point_chunk_size=chunk_size,
            camera_batch_size=camera_batch_size,
            device="cuda",
        )
    elif backend == "cpu":
        depth_np = project_world_points_to_depth(
            world_points=world_points,
            extrinsic=extrinsic,
            intrinsics=intrinsic,
            image_size=image_size,
            chunk_size=chunk_size,
        )
    else:
        raise ValueError(f"Unsupported dense depth backend: {backend}")
    save_depth_pngs(depth_np=depth_np, image_names=image_names, output_dir=output_dir)

    nonzero_pixels = int(np.count_nonzero(depth_np > 0))
    print(
        f"[DEPTH EXPORT] Saved projected dense depth maps to {output_dir} "
        f"(backend={backend}, frames={num_frames}, points={len(world_points)}, "
        f"nonzero_pixels={nonzero_pixels})"
    )
    return {
        "num_depth_frames": int(num_frames),
        "num_merged_points": int(len(world_points)),
        "num_valid_points_before_cap": int(len(world_points)),
        "num_nonzero_depth_pixels": nonzero_pixels,
        "backend": backend,
        "output_dir": output_dir,
    }


@torch.no_grad()
def estimate_intrinsics_and_depth(points: torch.Tensor):
    """
    Estimate intrinsics K (fx, fy, cx, cy, zero skew) from points in camera coords,
    and also return per-pixel depths.

    Args:
        points: (B, H, W, 3)  3D points expressed in the CAMERA frame
                              (i = row = y, j = col = x).
                              Last dimension = (X, Y, Z).

    Returns:
        K:      (B, 3, 3) intrinsics matrix for each camera
        depth:  (B, H, W) depth = Z coordinate (positive forward)
    """
    B, H, W, _ = points.shape
    device = points.device
    # Pixel grids: u = x (cols), v = y (rows)
    u_grid = torch.arange(W, device=device, dtype=torch.float).view(1, 1, W).expand(B, H, W)
    v_grid = torch.arange(H, device=device, dtype=torch.float).view(1, H, 1).expand(B, H, W)

    X = points[..., 0]
    Y = points[..., 1]
    Z = points[..., 2]   # depth map

    # Valid mask
    valid = torch.isfinite(Z) & (Z > 1e-6)

    # Prepare outputs
    K = torch.zeros((B, 3, 3), dtype=points.dtype, device=device)
    K[:, 2, 2] = 1.0

    for b in range(B):
        m = valid[b]
        if m.sum().item() < 4:
            K[b] = torch.full((3, 3), float("nan"), dtype=points.dtype, device=device)
            K[b, 2, 2] = 1.0
            continue

        # Build linear systems: u = fx * (X/Z) + cx ; v = fy * (Y/Z) + cy
        a_u = (X[b][m] / Z[b][m]).unsqueeze(1)
        a_v = (Y[b][m] / Z[b][m]).unsqueeze(1)
        A_u = torch.cat([a_u, torch.ones_like(a_u)], dim=1)
        A_v = torch.cat([a_v, torch.ones_like(a_v)], dim=1)

        u = u_grid[b][m].unsqueeze(1)
        v = v_grid[b][m].unsqueeze(1)

        sol_u = torch.linalg.lstsq(A_u, u).solution.squeeze(1)
        sol_v = torch.linalg.lstsq(A_v, v).solution.squeeze(1)

        fx, cx = sol_u[0], sol_u[1]
        fy, cy = sol_v[0], sol_v[1]

        K[b, 0, 0] = fx
        K[b, 1, 1] = fy
        K[b, 0, 2] = cx
        K[b, 1, 2] = cy

    return K, Z   # Z is the depth map



def convert_to_homogeneous_matrix(input: torch.Tensor):
    if len(input.shape) == 2:
        # No batch dimension
        return torch.cat([input, torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(input.device)])

    elif len(input.shape) == 3:
        # With batch dimension
        bs = input.shape[0]
        homo_part = torch.stack([torch.tensor([0, 0, 0, 1]).reshape(1, 4) for _ in range(bs)]).to(input.device)
        return torch.cat([input, homo_part], dim=1)

    else:
        raise ValueError("Input shape incorrect for homogeneous matrix. ")


def remove_homogeneous_row(matrix: torch.Tensor) -> torch.Tensor:
    if len(matrix.shape) == 2:
        # No batch dimension, shape should be (4, 4)
        if matrix.shape != (4, 4):
            raise ValueError("Expected shape (4, 4) for single matrix.")
        return matrix[:3]

    elif len(matrix.shape) == 3:
        # With batch dimension, shape should be (B, 4, 4)
        if matrix.shape[1:] != (4, 4):
            raise ValueError("Expected shape (B, 4, 4) for batched matrix.")
        return matrix[:, :3]

    else:
        raise ValueError("Invalid input shape.")
    

def output_to_colmap(predictions, img_names, output_dir, image_points2D, points3D, idx=None, format="txt", shared_camera=False):
    name = "colmap" if idx is None else f"colmap_{idx}"
    quaternions, translations = extrinsic_to_colmap_format(predictions['extrinsic'])
    
    height, width = predictions["depth"].shape[1:3]
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    intrinsics = predictions["intrinsic"]
    if shared_camera:
        intrinsics = np.mean(intrinsics, axis=0, keepdims=True)
    if format == "txt":
        write_colmap_cameras_txt(
        os.path.join(output_dir, name, "cameras.txt"), 
        intrinsics, width, height)
        write_colmap_images_txt(
            os.path.join(output_dir, name, "images.txt"), 
            quaternions, translations, image_points2D, img_names, shared_camera=shared_camera)
        write_colmap_points3D_txt(
            os.path.join(output_dir, name, "points3D.txt"), 
            points3D)
    elif format == "bin":
        write_colmap_cameras_bin(
            os.path.join(output_dir, name, "cameras.bin"), 
            intrinsics, width, height)
        write_colmap_images_bin(
            os.path.join(output_dir, name, "images.bin"), 
            quaternions, translations, image_points2D, img_names, shared_camera=shared_camera)
        write_colmap_points3D_bin(
            os.path.join(output_dir, name, "points3D.bin"), 
            points3D)


def write_recon_to_colmap(output_dir, predictions, images, names, stride=100, conf_threshold=50.0, format="txt", shared_camera=False):
    size_hw = images.shape[-2:]
    if "extrinsic" not in predictions.keys():
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], size_hw)
        predictions["extrinsic"] = extrinsic
    if "intrinsic" not in predictions.keys():
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], size_hw)
        predictions["intrinsic"] = intrinsic
    
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy() 
            if predictions[key].shape[0] == 1:
                predictions[key] = predictions[key].squeeze(0) # remove batch dimension
    
    predictions["original_images"] = images
    print("[OUTPUT WRITING] Computing 3D points from depth maps...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    S, H, W = world_points.shape[:3]
    normalized_images = np.zeros((S, H, W, 3), dtype=np.float32)
    for j, img in enumerate(images):
        img = img.permute(1, 2, 0)
        resized_img = cv2.resize(img.cpu().numpy(), (W, H))
        # normalized_images[j] = resized_img / 255.0
        normalized_images[j] = resized_img
    
    predictions["colmap_images"] = normalized_images

    print(f"[OUTPUT WRITING] Filtering points with confidence threshold {conf_threshold}% and stride {stride}...")
    points3D, image_points2D = filter_and_prepare_points(
        predictions, 
        conf_threshold, 
        mask_sky=False, 
        mask_black_bg=False,
        mask_white_bg=False,
        stride=stride,
        prediction_mode="Depthmap and Camera Branch"
    )
    
    # Export 3D point clouds
    points_3d = []
    points_rgb = []
    for point in points3D:
        point_id = point["id"] + 1  
        x, y, z = point["xyz"]
        r, g, b = point["rgb"]

        points_3d.append((x, y, z))
        points_rgb.append((r, g, b, 1))
    trimesh.PointCloud(points_3d, points_rgb).export(os.path.join(output_dir, "points.ply"))
    output_to_colmap(predictions, names, output_dir, image_points2D, points3D, format=format, shared_camera=shared_camera)


def restore_predictions_order(predictions):
    """
    Sort the prediction based on image_ids to restore the order of all the attributes
    to the original order.
    """
    image_ids = predictions['image_ids']

    for key in predictions.keys():
        if isinstance(predictions[key], np.ndarray):
            rearrangement = np.argsort(image_ids)
            if predictions[key].shape[0] == rearrangement.shape[0]:
                predictions[key] = predictions[key][rearrangement]


def extract_frames_from_video(video_path, subsample=1, num_images=-1, output_dir=None):
    """
    Extract frames from a video file and save them as images.
    
    Args:
        video_path: Path to the video file
        subsample: Extract every Nth frame (default: 1, extract all frames)
        num_images: Maximum number of frames to extract (-1 for all frames)
        output_dir: Directory to save frames. If None, creates a temporary directory.
    
    Returns:
        tuple: (list of frame file paths, output directory path)
    """
    print(f"[UTILS] Processing video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="video_frames_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    frame_idx = 0
    saved_frame_idx = 0
    image_names = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply subsampling
        if frame_idx % subsample == 0:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Save frame as image
            frame_path = os.path.join(output_dir, f"frame_{saved_frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            image_names.append(frame_path)
            saved_frame_idx += 1
            
            # Apply num_images limit if specified
            if num_images != -1 and len(image_names) >= num_images:
                break
        
        frame_idx += 1
    
    cap.release()
    
    if len(image_names) == 0:
        if output_dir is None or output_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(output_dir, ignore_errors=True)
        raise ValueError(f"No frames extracted from video: {video_path}")
    
    print(f"[UTILS] Extracted {len(image_names)} frames from video")
    return image_names, output_dir


def process_images(image_dir, subsample, device, num_images, multi_dirs=False):
    """Process images with VGGT and return predictions. Also supports video files."""
    
    # Check if input is a video file
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV', '.MKV', '.WEBM')
    is_video = os.path.isfile(image_dir) and image_dir.lower().endswith(video_extensions)
    
    if is_video:
        # Extract frames from video
        image_names, _ = extract_frames_from_video(image_dir, subsample, num_images)
    else:
        # Original image directory processing
        if multi_dirs:
            image_names = glob.glob(os.path.join(image_dir, "*", "**"), recursive=True)
        else:
            image_names = glob.glob(os.path.join(image_dir, "*"))
        image_names = sorted([f for f in image_names if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if num_images != -1:
            image_names = image_names[:num_images]
        image_names = image_names[::subsample] # subsampling code from mast3r-sfm
        print(f"[UTILS] Found {len(image_names)} images")
        
        if len(image_names) == 0:
            raise ValueError(f"No images found in {image_dir}")

    original_images = []
    for img_path in image_names:
        img = Image.open(img_path).convert('RGB')
        original_images.append(np.array(img))
    
    images = load_and_preprocess_images(image_names, mode="raw").to(device)

    print(f"[UTILS] Preprocessed images shape: {images.shape}")
    
    return images, image_names
    

@torch.no_grad()
def mnn_one_to_many(A, B, tau=0.6):
    """
    A: (P, D)  normalized patch tokens (image i)
    B: (Bsz, P, D) normalized patch tokens (candidate images)
    tau: cosine threshold

    Returns:
        scores: (Bsz,) MNN score for each candidate in B
    """
    # Ensure correct dims
    if B.dim() == 2:
        B = B.unsqueeze(0)  # (1, P, D)

    # (Bsz, P, P): S[b, i, j] = <B[b,i], A[j]>
    # This matches each patch in B to all patches in A.
    S = torch.matmul(B, A.t())  # (Bsz, P, P)

    # For each patch i in B: best j in A
    j_best = S.argmax(dim=2)  # (Bsz, P)

    # For each patch j in A: best i in B
    i_best = S.argmax(dim=1)  # (Bsz, P)

    P = A.shape[0]
    i_idx = torch.arange(P, device=A.device).view(1, P).expand(B.shape[0], P)  # (Bsz, P)

    # mutual: i_best[b, j_best[b,i]] == i
    i_back = torch.gather(i_best, dim=1, index=j_best)  # (Bsz, P)
    mutual = (i_back == i_idx)

    # confidence: S[b, i, j_best[b,i]] > tau
    sim_ij = torch.gather(S, dim=2, index=j_best.unsqueeze(2)).squeeze(2)  # (Bsz, P)
    confident = (sim_ij > tau)

    good = mutual & confident
    return good.float().mean(dim=1)  # (Bsz,)      # (Bsz,)


@torch.no_grad()
def mnn_from_dino_candidates(
    X, sim_dino, K=30, tau=0.6, batch_cand=8, use_fp16=True, symmetric=True
):
    """
    X: (M, P, D) patch tokens
    sim_dino: (M, M) DINO similarity matrix (larger = more similar)
    K: number of candidates per image to verify with MNN
    tau: patch-level cosine threshold for confident MNN matches
    batch_cand: compute MNN for this many candidates at once
    symmetric: mirror scores to make S_mnn symmetric

    Returns:
        S_mnn: (M, M) float32 matrix with only candidate entries filled (others 0)
        cand_idx: (M, K) candidate indices used per row
    """
    device = X.device
    M, P, D = X.shape

    # Normalize patch tokens for cosine
    X = F.normalize(X, p=2, dim=-1)

    # Make sure diagonal doesn't get selected
    sim = sim_dino.clone()
    sim.fill_diagonal_(-1e9)

    # Candidate indices (top-K per row)
    cand_vals, cand_idx = torch.topk(sim, k=K, dim=1, largest=True, sorted=True)  # (M,K)

    S_mnn = torch.zeros((M, M), device=device, dtype=torch.float32)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if (use_fp16 and device.type == "cuda")
        else torch.cpu.amp.autocast(enabled=False)
    )

    for i in range(M):
        A = X[i]  # (P,D)
        js = cand_idx[i]  # (K,)

        # optional: if symmetric, only compute j>i to halve work
        if symmetric:
            js = js[js > i]
            if js.numel() == 0:
                continue

        # process candidates in batches
        for t in range(0, js.numel(), batch_cand):
            j_batch = js[t:t+batch_cand]
            B = X[j_batch]  # (Bsz,P,D)

            with autocast_ctx:
                scores = mnn_one_to_many(A, B, tau=tau)  # (Bsz,)

            S_mnn[i, j_batch] = scores.float()

    if symmetric:
        S_mnn = torch.maximum(S_mnn, S_mnn.t())  # keep larger of two directions
        S_mnn.fill_diagonal_(0.0)

    return S_mnn, cand_idx


def get_sim_matrix(
    images,
    model_name="dinov3",
    device="cuda",
    subset_size=100,
    feature_size=768,
    alpha=0.3,
    return_feats=False
):
        """Extract normalized feature embeddings for all images.

        Loads a DINOv2/DINOv3 model, normalizes images using
        ImageNet statistics, and obtains per-frame patch embeddings.
        Similarity is computed as a blend of:
        - Global mean-patch cosine similarity (diagonal zeroed).
        - MNN (mutual nearest-neighbor) patch consistency.

        """
        _RESNET_MEAN = [0.485, 0.456, 0.406]
        _RESNET_STD = [0.229, 0.224, 0.225]
        use_hf_dinov3 = model_name == "dinov3"
        if use_hf_dinov3:
            # repo_dir = os.environ.get(
            #     "MERG3R_DINOV3_REPO_DIR",
            #     os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "dinov3"),
            # )
            # weights_url = os.environ.get("MERG3R_DINOV3_WEIGHTS_URL", None)
            # if not os.path.isdir(repo_dir):
            #     raise ValueError(
            #         f"DINOv3 repo directory not found: {repo_dir}. "
            #         "Use model_name='dinov2' for the default public model."
            #     )
            # model = torch.hub.load(
            #     repo_dir, "dinov3_vitb16", source="local", weights=weights_url
            # )

            from transformers import AutoModel
            model_id = os.environ.get(
                "MERG3R_DINOV3_MODEL_ID",
                "facebook/dinov3-vitb16-pretrain-lvd1689m",
            )
            model = AutoModel.from_pretrained(
                model_id,
            )
        else:
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")

        model.eval()
        model = model.to(device)
        
        resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
        resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
        images_resnet_norm = (images - resnet_mean) / resnet_std
        
        # Need to split up into subsets because of feature extractor GPU limitations
        num_subsets = (len(images) + subset_size - 1) // subset_size
        frame_feat = torch.empty(size=(0, feature_size), device=device)
        frame_chunks = []
        with torch.no_grad():
            for i in range(num_subsets):
                image_subset = images_resnet_norm[i * subset_size : (i+1) * subset_size]
                if image_subset.shape[0] == 0:
                    continue

                if use_hf_dinov3:
                    outputs = model(image_subset)
                    num_register_tokens = getattr(model.config, "num_register_tokens", 0)
                    frame_feat_subset = outputs.last_hidden_state[:, 1 + num_register_tokens :, :]
                else:
                    frame_feat_subset = model(image_subset, is_training=True)
                    # frame_feat_subset = frame_feat_subset["x_norm_clstoken"]

                    # --- Mean pooling patches
                    frame_feat_subset = frame_feat_subset["x_norm_patchtokens"]
                # frame_feat_subset = torch.mean(frame_feat_subset, dim=1)

                frame_chunks.append(frame_feat_subset)
                del image_subset, frame_feat_subset
        frame_feat = torch.cat(frame_chunks, dim=0)
        frame_feat_norm = torch.nn.functional.normalize(frame_feat, p=2, dim=-1)
        frame_feat_norm_mean = torch.mean(frame_feat_norm, dim=1)
        
        del model, resnet_mean, resnet_std, images_resnet_norm
        sim_matrix = frame_feat_norm_mean @ frame_feat_norm_mean.t()
        sim_matrix = sim_matrix.fill_diagonal_(0)

        mnn_sim_matrix, cand_idx = mnn_from_dino_candidates(frame_feat_norm, sim_matrix)
        sim_matrix = alpha * sim_matrix + (1 - alpha) * mnn_sim_matrix

        gc.collect()
        torch.cuda.empty_cache()
        if return_feats:
            return sim_matrix, frame_feat_norm_mean
        return sim_matrix



def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }
