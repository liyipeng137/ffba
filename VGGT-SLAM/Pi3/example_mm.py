import torch
import argparse
import numpy as np
import os
import cv2
from pi3.utils.basic import load_multimodal_data, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3x import Pi3X



# depth_frame = np.nan_to_num(depth_np[idx], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
# depth_u16 = np.clip(depth_frame * 1000.0, 0, np.iinfo(np.uint16).max).astype(np.uint16)
# stem = os.path.splitext(os.path.basename(str(image_names[idx])))[0]
# base_name = stem + ".png"
# cv2.imwrite(os.path.join(depth_u16_dir, base_name), depth_u16)


def generate_sampled_image_names(data_path, num_frames, interval):
    if os.path.isdir(data_path):
        filenames = sorted(
            [x for x in os.listdir(data_path) if x.lower().endswith((".png", ".jpg", ".jpeg", ".heic"))]
        )
        return filenames[::interval][:num_frames]
    return [f"frame_{i:04d}.png" for i in range(num_frames)]


def project_world_points_to_depth(world_points, camera_poses, intrinsics, image_size):
    """
    Reproject world-space points (from exported ply) into each camera and build depth maps via z-buffer.
    Args:
        world_points: (P, 3) world points
        camera_poses: (N, 4, 4) OpenCV cam2world
        intrinsics:   (N, 3, 3) per-frame intrinsics
        image_size:   (H, W)
    Returns:
        depth_maps: (N, H, W), metric depth in meters
    """
    H, W = image_size
    N = camera_poses.shape[0]
    depth_maps = np.zeros((N, H, W), dtype=np.float32)

    if world_points.size == 0:
        return depth_maps

    world_points = np.asarray(world_points, dtype=np.float32)
    world_points_h = np.concatenate(
        [world_points, np.ones((world_points.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    for i in range(N):
        K = intrinsics[i].astype(np.float32, copy=False)
        T_w2c = np.linalg.inv(camera_poses[i]).astype(np.float32)
        cam_points = (T_w2c @ world_points_h.T).T[:, :3]

        z = cam_points[:, 2]
        valid = np.isfinite(z) & (z > 1e-6)
        if not np.any(valid):
            continue

        cam_points = cam_points[valid]
        x = cam_points[:, 0]
        y = cam_points[:, 1]
        z = cam_points[:, 2]

        u = K[0, 0] * (x / z) + K[0, 2]
        v = K[1, 1] * (y / z) + K[1, 2]

        valid_uv = np.isfinite(u) & np.isfinite(v)
        if not np.any(valid_uv):
            continue

        u = np.rint(u[valid_uv]).astype(np.int32)
        v = np.rint(v[valid_uv]).astype(np.int32)
        z = z[valid_uv].astype(np.float32, copy=False)

        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_bounds):
            continue

        u = u[in_bounds]
        v = v[in_bounds]
        z = z[in_bounds]

        depth_flat = np.full(H * W, np.inf, dtype=np.float32)
        idx = v * W + u
        np.minimum.at(depth_flat, idx, z)

        depth = depth_flat.reshape(H, W)
        depth[~np.isfinite(depth)] = 0.0
        depth_maps[i] = depth

    return depth_maps


def save_depth_pngs(depth_np, image_names, output_dir):
    """
    Save depth as:
      1) uint16 millimeter PNG for downstream usage
      2) uint8 pseudo-color PNG for quick visual inspection
      3) float32 NPY per-frame depth for downstream numeric processing
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


def build_shared_intrinsics(num_frames, fx, fy, cx, cy):
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.repeat(K[None], num_frames, axis=0)


if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    
    # parser.add_argument("--conditions_path", type=str, default='examples/room/condition.npz',
    parser.add_argument("--conditions_path", type=str, default=None,
                        help="Optional path to a .npz file containing 'poses', 'depths', 'intrinsics'.")
    parser.add_argument("--fx", type=float, default=None,
                        help="Manual shared focal length fx in pixels. If set with fy/cx/cy, overrides intrinsics from conditions.")
    parser.add_argument("--fy", type=float, default=None,
                        help="Manual shared focal length fy in pixels. If set with fx/cx/cy, overrides intrinsics from conditions.")
    parser.add_argument("--cx", type=float, default=None,
                        help="Manual shared principal point cx in pixels. If set with fx/fy/cy, overrides intrinsics from conditions.")
    parser.add_argument("--cy", type=float, default=None,
                        help="Manual shared principal point cy in pixels. If set with fx/fy/cx, overrides intrinsics from conditions.")

    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--save_depth_dir", type=str, default=None,
                        help="Optional directory to save projected per-view depth maps (uint16 + visualization png).")
    parser.add_argument("--save_transforms", type=str, default=None,
                        help="Path to save transforms.json with camera poses and intrinsics. Default: None (not saved)")
    parser.add_argument("--use_moge_intrinsics", action='store_true',
                        help="Use MoGe method to recover intrinsics from local_points. Default: False (use input or default)")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
                        
    args = parser.parse_args()
    manual_intrinsics_values = [args.fx, args.fy, args.cx, args.cy]
    has_manual_intrinsics = any(v is not None for v in manual_intrinsics_values)
    if has_manual_intrinsics and not all(v is not None for v in manual_intrinsics_values):
        raise ValueError("If using manual intrinsics, please provide all of --fx --fy --cx --cy.")

    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3X().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight, strict=False)
    else:
        model = Pi3X.from_pretrained("yyfz233/Pi3X").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data

    # Load optional conditions from .npz
    poses = None
    depths = None
    intrinsics = None

    if args.conditions_path is not None and os.path.exists(args.conditions_path):
        print(f"Loading conditions from {args.conditions_path}...")
        data_npz = np.load(args.conditions_path, allow_pickle=True)

        poses = data_npz['poses']             # Expected (N, 4, 4) OpenCV camera-to-world
        depths = data_npz['depths']           # Expected (N, H, W)
        intrinsics = data_npz['intrinsics']   # Expected (N, 3, 3)

    conditions = dict(
        intrinsics=intrinsics,
        poses=poses,
        depths=depths
    )

    # Load images (Required)
    imgs, conditions = load_multimodal_data(args.data_path, conditions, interval=args.interval, device=device) 
    N = imgs.shape[1]

    # Apply user-provided shared intrinsics (highest priority)
    if has_manual_intrinsics:
        intrinsics_np = build_shared_intrinsics(N, args.fx, args.fy, args.cx, args.cy)
        conditions['intrinsics'] = torch.from_numpy(intrinsics_np).float()[None].to(device)
        print(
            "Using manual shared intrinsics for all frames: "
            f"fx={args.fx:.3f}, fy={args.fy:.3f}, cx={args.cx:.3f}, cy={args.cy:.3f}"
        )

    # Keep a single source of truth for intrinsics used by downstream exports.
    intrinsics_used_np = None
    if conditions.get('intrinsics') is not None:
        intrinsics_used_np = conditions['intrinsics'][0].detach().cpu().numpy()

    """
    Args:
        imgs (torch.Tensor): Input RGB images valued in [0, 1].
            Shape: (B, N, 3, H, W).
        intrinsics (torch.Tensor, optional): Camera intrinsic matrices.
            Shape: (B, N, 3, 3).
            Values are in pixel coordinates (not normalized).
        rays (torch.Tensor, optional): Pre-computed ray directions (unit vectors).
            Shape: (B, N, H, W, 3).
            Can replace `intrinsics` as a geometric condition.
        poses (torch.Tensor, optional): Camera-to-World matrices.
            Shape: (B, N, 4, 4).
            Coordinate system: OpenCV convention (Right-Down-Forward).
        depths (torch.Tensor, optional): Ground truth or prior depth maps.
            Shape: (B, N, H, W).
            Invalid values (e.g., sky or missing data) should be set to 0.
        mask_add_depth (torch.Tensor, optional): Mask for depth condition.
            Shape: (B, N, N).
        mask_add_ray (torch.Tensor, optional): Mask for ray/intrinsic condition.
            Shape: (B, N, N).
        mask_add_pose (torch.Tensor, optional): Mask for pose condition.
            Shape: (B, N, N).
            Note: Requires at least two frames to be True to establish a meaningful
            coordinate system (absolute pose for a single frame provides no relative constraint).
    """

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(
                imgs=imgs, 
                **conditions
            )

    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}")
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        
    write_ply(res['points'][0][masks].cpu(), imgs[0].permute(0, 2, 3, 1)[masks], args.save_path)
    print("Done.")
    
    # 6. Save depth maps from exported ply points + camera poses (optional)
    if args.save_depth_dir:
        print(f"Saving depth maps to: {args.save_depth_dir}")
        from pi3.utils.transforms_utils import recover_intrinsics_from_output

        camera_poses = res['camera_poses'][0].detach().cpu().numpy()  # (N, 4, 4), OpenCV cam2world
        if intrinsics_used_np is None:
            intrinsics_np = recover_intrinsics_from_output(res, imgs)  # (N, 3, 3)
            print("No input intrinsics found. Recovered intrinsics from local_points for depth export.")
        else:
            intrinsics_np = intrinsics_used_np
            print("Using provided intrinsics for depth export.")
        world_points = res['points'][0][masks].detach().cpu().numpy()  # same points exported to ply
        H, W = imgs.shape[-2:]

        depth_np = project_world_points_to_depth(
            world_points=world_points,
            camera_poses=camera_poses,
            intrinsics=intrinsics_np,
            image_size=(H, W),
        )

        image_names = generate_sampled_image_names(args.data_path, camera_poses.shape[0], args.interval)
        save_depth_pngs(depth_np=depth_np, image_names=image_names, output_dir=args.save_depth_dir)
        print("Depth maps saved.")

    # 7. Save transforms.json (optional)
    if args.save_transforms:
        print("\n" + "="*60)
        print("保存相机位姿和内参到 transforms.json...")
        print("="*60)
        
        from pi3.utils.transforms_utils import save_transforms_json, generate_image_paths
        from PIL import Image
        
        # 提取位姿 (OpenCV camera-to-world)
        camera_poses = res['camera_poses'][0].cpu().numpy()  # (N, 4, 4)
        N = camera_poses.shape[0]
        H, W = imgs.shape[-2:]
        
        # 获取内参
        intrinsics_np = intrinsics_used_np
        if intrinsics_np is not None:
            print("使用输入/手动设置的内参")

        # 生成图像路径
        image_paths = generate_image_paths(args.data_path, N, save_dir='images')
        
        # 保存 resize 后的图片（先保存图片，再生成正确的路径）
        output_dir = os.path.dirname(args.save_transforms) or '.'
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        print(f"保存 resize 后的图片到: {images_dir}")
        imgs_np = imgs[0].cpu().numpy()  # (N, 3, H, W)
        
        # 保存图片并更新路径
        updated_image_paths = []
        for i in range(N):
            # 提取文件名（从 image_paths 中获取，如 "./images/frame_0000.png"）
            img_filename = os.path.basename(image_paths[i])
            
            # 对于非标准扩展名（如 .heic），统一保存为 .png
            name_without_ext, ext = os.path.splitext(img_filename)
            if ext.lower() in ['.heic', '.jpeg']:
                img_filename = name_without_ext + '.png'
            elif ext.lower() == '.jpg':
                img_filename = name_without_ext + '.jpg'  # 保持 jpg
            else:
                img_filename = name_without_ext + '.png'  # 默认 png
            
            save_path = os.path.join(images_dir, img_filename)
            
            # 转换为 PIL Image (0-1范围转为0-255)
            img_array = imgs_np[i].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
            img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_array)
            
            # 保存
            img_pil.save(save_path)
            
            # 更新路径
            updated_image_paths.append(f"./images/{img_filename}")
        
        print(f"已保存 {N} 张图片到 {images_dir}")
        
        # 保存 transforms.json（使用更新后的路径）
        save_transforms_json(
            output_path=args.save_transforms,
            camera_poses=camera_poses,
            intrinsics=intrinsics_np,
            image_paths=updated_image_paths,
            image_size=(H, W),
            res=res if args.use_moge_intrinsics else None,
            imgs=imgs if args.use_moge_intrinsics else None,
            use_moge_recovery=args.use_moge_intrinsics
        )
        print("="*60 + "\n")
