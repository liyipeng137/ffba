'''
https://github.com/jasonyzhang/RayDiffusion/blob/main/ray_diffusion/eval/utils.py
'''
import numpy as np
import torch
from .metric_posediffuse  import rotation_angle, translation_angle, calculate_auc_np
from utils.geometry import closed_form_inverse_se3


def compute_optimal_alignment(A, B):
    """
    Compute the optimal scale s, rotation R, and translation t that minimizes:
    || A - (s * B @ R + T) || ^ 2

    Reference: Umeyama (TPAMI 91)

    Args
        A (torch.Tensor): (N, 3).
        B (torch.Tensor): (N, 3).

    Returns:
        s (float): scale.
        R (torch.Tensor): rotation matrix (3, 3).
        t (torch.Tensor): translation (3,).
    """
    A_bar = A.mean(0)
    B_bar = B.mean(0)
    # normally with R @ B, this would be A @ B.T
    H = (B - B_bar).T @ (A - A_bar)
    U, S, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    variance = torch.sum((B - B_bar) ** 2)
    scale = 1 / variance * torch.trace(torch.diag(S) @ S_prime)
    R = U @ S_prime @ Vh
    t = A_bar - scale * B_bar @ R


    A_hat = scale * B @ R + t
    return A_hat, scale, R, t


def compute_angular_error_batch(c2w_pred, c2w_gt):
    rotation1 = c2w_pred[:, :3, :3]
    rotation2 = c2w_gt[:, :3, :3]
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation2, rotation1.transpose(0, 2, 1))
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi


def compute_camera_center_error(c2w_pred, c2w_gt):
    cc_gt = c2w_gt[:, :3, 3]
    centroid = torch.mean(cc_gt, dim=0)
    diffs = cc_gt - centroid
    norms = torch.linalg.norm(diffs, dim=1)
    furthest_index = torch.argmax(norms).item()
    gt_scene_scale = norms[furthest_index].item()
    
    cc_pred = c2w_pred[:, :3, 3]

    A_hat, _, _, _ = compute_optimal_alignment(cc_gt, cc_pred)
    norm = torch.linalg.norm(cc_gt - A_hat, dim=1) / gt_scene_scale

    norms = np.ndarray.tolist(norm.detach().cpu().numpy())
    return norms, A_hat

def compute_extrinsic_error(c2w_pred, c2w_gt):
    #First create pairs of camera poses
    c2w_pred_ = []
    c2w_gt_ = []
    n_view = c2w_pred.shape[0]
    for src_view in range(n_view):
        for tgt_view in range(src_view+1, n_view):
            # Transform c2w to the src view's coordinate
            c2w_pred_.append(closed_form_inverse_se3(c2w_pred[src_view]) @ c2w_pred[tgt_view])
            c2w_gt_.append(closed_form_inverse_se3(c2w_gt[src_view]) @ c2w_gt[tgt_view])
    c2w_pred_ = np.stack(c2w_pred_)
    c2w_gt_ = np.stack(c2w_gt_)
    #Copmute the rotation_angle
    rel_angle_deg = rotation_angle(c2w_gt_[:, :3, :3], c2w_pred_[:, :3, :3])
    rel_tangle_deg = translation_angle(c2w_gt_[:, :3, 3], c2w_pred_[:, :3, 3])
    auc = {}
    for max_threshold in [1,3,5,10]:#[1,3,5,10,20,30]:
        auc[f'extrinsics/auc@{max_threshold:02d}(%)'] = round(calculate_auc_np(rel_angle_deg, rel_tangle_deg, max_threshold=max_threshold)*100,1)
    return auc




def compute_intrinsic_error(intrinsics_pred, intrinsics_gt, Hs, Ws):
    #Here we only evaluate the fov error
    fovx_pred = 2 * np.arctan(Ws / (2 * intrinsics_pred[:, 0, 0])) * 180 / np.pi
    fovy_pred = 2 * np.arctan(Hs / (2 * intrinsics_pred[:, 1, 1])) * 180 / np.pi
    fovx_gt = 2 * np.arctan(Ws / (2 * intrinsics_gt[:, 0, 0])) * 180 / np.pi
    fovy_gt = 2 * np.arctan(Hs / (2 * intrinsics_gt[:, 1, 1])) * 180 / np.pi
    fovx_error = np.abs(fovx_pred - fovx_gt)
    fovy_error = np.abs(fovy_pred - fovy_gt)
    return {
        'fovx_error_mean(deg)': round(float(np.mean(fovx_error)),2),
        'fovy_error_mean(deg)': round(float(np.mean(fovy_error)),2),
        'fov_error_mean(deg)': round(float(np.mean(fovx_error)+np.mean(fovy_error))/2,2)
    }


    