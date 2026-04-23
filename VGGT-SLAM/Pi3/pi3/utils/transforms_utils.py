"""
Utility functions for saving camera poses and intrinsics to transforms.json format
Compatible with Nerfstudio and other NeRF frameworks
"""

import os
import json
import numpy as np
import torch
from pathlib import Path


def save_transforms_json(
    output_path,
    camera_poses,      # (N, 4, 4) OpenCV cam2world
    intrinsics,        # (N, 3, 3) or None
    image_paths,       # List[str], length N
    image_size,        # (H, W) tuple
    res=None,          # Optional: model output for focal recovery
    imgs=None,         # Optional: for focal recovery
    use_moge_recovery=True,  # Whether to use MoGe's recover_focal_shift
):
    """
    Save camera poses and intrinsics to transforms.json (Nerfstudio format)
    
    Args:
        output_path: Path to save transforms.json
        camera_poses: (N, 4, 4) camera-to-world matrices in OpenCV convention
        intrinsics: (N, 3, 3) intrinsic matrices, or None
        image_paths: List of image file paths (relative)
        image_size: (H, W) tuple
        res: Optional model output dict containing 'local_points', 'conf'
        imgs: Optional input images for recovering intrinsics
        use_moge_recovery: Whether to use MoGe method to recover intrinsics
    """
    N = camera_poses.shape[0]
    H, W = image_size
    
    # Get or recover intrinsics
    if intrinsics is None:
        if use_moge_recovery and res is not None and imgs is not None:
            print("从 local_points 恢复相机内参...")
            intrinsics = recover_intrinsics_from_output(res, imgs)
            print(f"恢复的内参 (第1帧): fx={intrinsics[0,0,0]:.2f}, fy={intrinsics[0,1,1]:.2f}")
        else:
            print("使用默认内参 (焦距 = max(H, W))...")
            f = max(H, W)
            intrinsics = np.array([[
                [f, 0, W/2],
                [0, f, H/2],
                [0, 0, 1]
            ]] * N, dtype=np.float32)
    
    # Convert OpenCV c2w (RDF: +x right, +y down, +z forward)
    # to OpenGL/Blender c2w (RUB: +x right, +y up, -z forward).
    camera_poses = np.array(camera_poses, copy=True)
    camera_poses[:, :3, 1:3] *= -1

    # Build transforms.json structure
    transforms_data = {
        "camera_model": "OpenGL",
    }
    
    # Check if all frames have the same intrinsics (shared camera mode)
    use_shared_camera = False
    if N > 1 and np.allclose(intrinsics[0], intrinsics[1:], rtol=1e-3):
        use_shared_camera = True
        K = intrinsics[0]
        transforms_data.update({
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "w": int(W),
            "h": int(H),
        })
    
    # Build frames
    frames = []
    for i in range(N):
        frame = {
            "file_path": image_paths[i],
            "transform_matrix": camera_poses[i].tolist()
        }
        
        # Add per-frame camera parameters if not using shared camera
        if not use_shared_camera:
            K = intrinsics[i]
            frame.update({
                "w": int(W),
                "h": int(H),
                "fl_x": float(K[0, 0]),
                "fl_y": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
            })
        
        frames.append(frame)
    
    transforms_data["frames"] = frames
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(transforms_data, f, indent=4)
    
    print(f"已保存 transforms.json ({N} 帧) 到: {output_path}")

def recover_intrinsics_from_output(res, imgs):
    from pi3.utils.geometry_torch import recover_focal_shift
    from pi3.utils.geometry import depth_edge
    
    points = res["local_points"]  # (B, N, H, W, 3)
    masks = torch.sigmoid(res["conf"][..., 0]) > 0.1  # (B, N, H, W)
    non_edge = ~depth_edge(points[..., 2], rtol=0.03, mask=masks)
    masks = torch.logical_and(masks, non_edge)
    
    B, N, H, W = points.shape[:4]
    assert B == 1, "仅支持 batch size = 1"
    
    try:
        # 批量处理所有帧
        focal_norm, shift = recover_focal_shift(
            points[0],  # (N, H, W, 3)
            masks[0],   # (N, H, W)
            downsample_size=(64, 64)
        )
        # focal_norm: (N,), shift: (N,)
        
    except Exception as e:
        print(f"批量焦距恢复失败，使用默认值。错误: {e}")
        focal_norm = torch.ones(N)
    
    # 转换为内参矩阵
    intrinsics_list = []
    aspect_ratio = W / H
    
    for i in range(N):
        f = focal_norm[i].item()
        
        # 正确的转换公式：先计算归一化焦距，再乘以图像尺寸
        # focal_norm 是相对于归一化视平面的焦距
        factor = (1 + aspect_ratio**2)**0.5
        fx_norm = f / 2 * factor / aspect_ratio  # 归一化焦距（相对于宽度）
        fy_norm = f / 2 * factor                  # 归一化焦距（相对于高度）
        
        # 转换为像素焦距
        fx = fx_norm * W  # 乘以宽度，不是对角线！
        fy = fy_norm * H  # 乘以高度，不是对角线！
        # 注意：对于方形像素，fx 和 fy 应该接近相等
        
        K = np.array([
            [fx, 0, W/2],
            [0, fy, H/2],
            [0, 0, 1]
        ], dtype=np.float32)
        intrinsics_list.append(K)
    
    return np.stack(intrinsics_list, axis=0)

# def recover_intrinsics_from_output(res, imgs):
#     """
#     使用 MoGe 方法从模型输出恢复相机内参
    
#     Args:
#         res: Model output dict with 'local_points', 'conf'
#         imgs: Input images (B, N, 3, H, W)
    
#     Returns:
#         intrinsics: (N, 3, 3) numpy array
#     """
#     try:
#         from pi3.utils.geometry_torch import recover_focal_shift
#     except ImportError:
#         raise ImportError("无法导入 moge_utils，使用默认内参")
#         print("警告: 无法导入 moge_utils，使用默认内参")
#         # B, N, _, H, W = imgs.shape
#         # f = max(H, W)
#         # return np.array([[
#         #     [f, 0, W/2],
#         #     [0, f, H/2],
#         #     [0, 0, 1]
#         # ]] * N, dtype=np.float32)
    
#     points = res["local_points"]  # (B, N, H, W, 3)
#     masks = torch.sigmoid(res["conf"][..., 0]) > 0.1  # (B, N, H, W)
    
#     B, N, H, W = points.shape[:4]
#     assert B == 1, "仅支持 batch size = 1"
    
#     intrinsics_list = []
    
#     for i in range(N):
#         points_i = points[0, i]  # (H, W, 3)
#         masks_i = masks[0, i]    # (H, W)
        
#         # 恢复归一化焦距 (相对于半对角线)
#         try:
#             focal_norm, shift = recover_focal_shift(
#                 points_i.unsqueeze(0), 
#                 masks_i.unsqueeze(0), 
#                 downsample_size=(64, 64)
#             )
#             focal_norm = focal_norm.item()
#         except Exception as e:
#             print(f"警告: 第 {i} 帧焦距恢复失败，使用默认值。错误: {e}")
#             focal_norm = 1.0
        
#         # 转换归一化焦距为像素焦距
#         aspect_ratio = W / H
        
#         # 正确的转换公式
#         factor = (1 + aspect_ratio**2)**0.5
#         fx_norm = focal_norm / 2 * factor / aspect_ratio
#         fy_norm = focal_norm / 2 * factor
#         fx = fx_norm * W  # 乘以宽度
#         fy = fy_norm * H  # 乘以高度
        
#         # 构建内参矩阵
#         K = np.array([
#             [fx, 0, W/2],
#             [0, fy, H/2],
#             [0, 0, 1]
#         ], dtype=np.float32)
        
#         intrinsics_list.append(K)
    
#     return np.stack(intrinsics_list, axis=0)


def generate_image_paths(data_path, num_frames, save_dir='images'):
    """
    根据数据路径生成图像文件路径列表
    
    Args:
        data_path: 原始数据路径 (目录或视频文件)
        num_frames: 帧数
        save_dir: 保存图像的相对目录名
    
    Returns:
        image_paths: List[str] 相对路径列表
    """
    image_paths = []
    
    if os.path.isdir(data_path):
        # 从目录读取
        filenames = sorted([x for x in os.listdir(data_path) 
                          if x.lower().endswith((".png", ".jpg", ".jpeg", ".heic"))])
        for i in range(min(num_frames, len(filenames))):
            image_paths.append(f"./{save_dir}/{filenames[i]}")
    else:
        # 视频输入，生成帧文件名
        for i in range(num_frames):
            image_paths.append(f"./{save_dir}/frame_{i:04d}.png")
    
    return image_paths
