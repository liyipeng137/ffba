import torch
import numpy as np 
import cv2 
import math


def crop_image_depth_and_intrinsic_by_pp(image, depth_map, intrinsic, target_shape, strict, intr_convention='opencv'):
    original_h, original_w = image.shape[:2]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    target_h, target_w = target_shape[0], target_shape[1]
    if strict == False: # Prevent OutOfBounds
        if intr_convention == "opencv":
            target_h = min(target_h, math.floor(cy+0.5) * 2, math.floor(original_h-0.5 - cy) * 2)
            target_w = min(target_w, math.floor(cx+0.5) * 2, math.floor(original_w-0.5 - cx) * 2)
        else:
            target_h = min(target_h, math.floor(cy) * 2, math.floor(original_h - cy) * 2) #Take floor, be conservative.
            target_w = min(target_w, math.floor(cx) * 2, math.floor(original_w - cx) * 2)

    if original_w < target_w:
        print(f"Target width {target_w} is larger than original width {original_w}. Zero-padding will be applied")
    if original_h < target_h:
        print(f"Target height {target_h} is larger than original height {original_h}. Zero-padding will be applied")
    
    tx = cx - (target_w - 1) / 2
    ty = cy - (target_h - 1) / 2 

    M = np.array([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (target_w, target_h), flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    if depth_map is not None:
        depth_map = cv2.warpAffine(depth_map, M, (target_w, target_h), flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE,borderValue=0)
    
    intrinsic = np.copy(intrinsic)
    if intr_convention == "opencv":
        intrinsic[0, 2] = (target_w - 1) / 2
        intrinsic[1, 2] = (target_h - 1) / 2
    elif intr_convention == "colmap":
        intrinsic[0, 2] = target_w / 2
        intrinsic[1, 2] = target_h / 2
    
    return image, depth_map, intrinsic
    

def resize_image_depth_and_intrinsic(
    image,
    depth_map,
    intrinsic,
    target_shape,
    safe_bound=4,
    intr_convention='opencv'
):  
    original_shape = np.array(image.shape[:2])
    resize_scales = (target_shape + safe_bound) / original_shape
    max_resize_scale = np.max(resize_scales)
    intrinsic = np.copy(intrinsic)

    input_resolution = original_shape[::-1]  # (H, W) -> (W, H)
    output_resolution = np.floor(input_resolution * max_resize_scale).astype(int)
    image = cv2.resize(image,tuple(output_resolution),interpolation=cv2.INTER_LANCZOS4 if max_resize_scale < 1 else cv2.INTER_CUBIC)
    if depth_map is not None:
        depth_map = cv2.resize(depth_map,output_resolution,interpolation=cv2.INTER_NEAREST_EXACT)
    actual_size = np.array(image.shape[:2])[::-1]  # (H, W) -> (W, H)
    actual_resize_scale = actual_size / input_resolution

    intrinsic[0,:] = intrinsic[0, :] * actual_resize_scale[0]
    intrinsic[1,:] = intrinsic[1, :] * actual_resize_scale[1]

    if intr_convention == "opencv":
        intrinsic[0, 2] = intrinsic[0, 2] + 0.5*(actual_resize_scale[0] - 1)
        intrinsic[1, 2] = intrinsic[1, 2] + 0.5*(actual_resize_scale[1] - 1)
        # If the input image is center-pp, then the output image is also center-pp
    else:
        pass 
    return image, depth_map, intrinsic

