import torch
import numpy as np
# import pycolmap
# from pytorch3d.loss import chamfer_distance
import os
import cv2
from utils.io import save_images_as_grid, create_error_map, save_xyzrgb_to_ply
from utils.geometry import closed_form_inverse_K, closed_form_inverse_se3
from utils.metric_camera import compute_extrinsic_error, compute_intrinsic_error
import json

import torch.distributed as dist
class EvalLogger():
    def __init__(self,dirname):
        self.dirname = dirname
        self.dic = {}
    def write(self, metric_dict, prefix, dataset_key, seq_key):
        if dataset_key not in self.dic:
            self.dic[dataset_key] = {}
        if seq_key not in self.dic[dataset_key]:
            self.dic[dataset_key][seq_key] = {}
        for m,v in metric_dict.items():
            if isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel()==1):
                self.dic[dataset_key][seq_key][f'{prefix}/{m}'] = float("{:.3g}".format(v))
    def save(self, ddp_sync=False):
        os.makedirs(os.path.join(self.dirname,'per_rank'), exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        with open(os.path.join(self.dirname,'per_rank', f'rank{rank}.json'),'w') as f:
            json.dump(self.dic, f, indent=4)
        
        if ddp_sync:
            if dist.is_initialized():
                dist.barrier()
            if rank == 0:
                world_size = dist.get_world_size() if dist.is_initialized() else 1
                # Read and merge all other ranks' json files
                dic_gathered = {}
                for r in range(world_size):
                    rank_filename = os.path.join(self.dirname,'per_rank', f'rank{r}.json')
                    if not os.path.exists(rank_filename):
                        print(f"Warning: rank file {rank_filename} does not exist.")
                        continue
                    with open(rank_filename,'r') as f:
                        dic_r = json.load(f)
                    for dataset_key in dic_r:
                        if dataset_key not in dic_gathered:
                            dic_gathered[dataset_key] = {}
                        for seq_key in dic_r[dataset_key]:
                            if seq_key in dic_gathered[dataset_key]:
                                print(f"Warning: duplicate seq_key {seq_key} in dataset {dataset_key} from rank {r}")
                            dic_gathered[dataset_key][seq_key] = dic_r[dataset_key][seq_key]
                with open(os.path.join(self.dirname, f'all_ranks.json'),'w') as f:
                    json.dump(dic_gathered, f, indent=4)
                # Compute per dataset average
                dataset2average = {}
                for dataset_key in dic_gathered:
                    average = {}
                    count = {}
                    for seq_key in dic_gathered[dataset_key]:
                        for m,v in dic_gathered[dataset_key][seq_key].items():
                            if m not in average:
                                average[m] = 0.0
                                count[m] = 0
                            average[m] += v
                            count[m] += 1
                    for m in average:
                        average[m] = float("{:.3g}".format(average[m]/count[m]))
                    dataset2average[dataset_key] = average
                with open(os.path.join(self.dirname, f'average.json'),'w') as f:
                    json.dump(dataset2average, f, indent=4)
                return dataset2average
            else:
                return None
        else:
            return None


def umeyama_alignment(B, A, mask=None, max_error=0.03):
    import pycolmap
    if mask is None:
        mask = torch.ones_like(A[:,0], dtype=torch.bool)
    src = A[mask].cpu().numpy()
    tgt = B[mask].cpu().numpy()

    while True:
        estimation_options = pycolmap.RANSACOptions(max_error=max_error, min_inlier_ratio=0.8) #In pycolmap, the maxerror in align-via-points is 0.005 and the min_inlier_ratio=0.9
        #print('Running robust alignment via points with max_error:', max_error)
        sim3d = pycolmap.estimate_sim3d_robust(src, tgt, estimation_options=estimation_options)
        if sim3d is None:
            if max_error > 0.3:
                print(f"Aligning points failed, max_error is too large: {max_error}, give up using robust alignment")
                sim3d = pycolmap.estimate_sim3d(src, tgt) # Use non-robust alignment
                break
            else:
                max_error *= 3
                print(f"Aligning points failed, increasing max_error to {max_error}")
        else:
            break
    #Do some reshaping
    Ashape0 = A.shape
    Bshape0 = B.shape
    if len(A.shape)!=2:
        A = A.reshape(-1,3)
    if len(B.shape)!=2:
        B = B.reshape(-1,3)
    if sim3d is None:
        A_aligned = A
        sim3d_mat = torch.eye(4, device=A.device).float()[:3,:]
    else:
        A_homo = torch.cat([A, torch.ones_like(A[:,-1:])], axis=1)
        sim3d_mat = torch.from_numpy(sim3d['tgt_from_src'].matrix()).to(A_homo.device).float()
        A_aligned = torch.einsum('ij,nj->ni', sim3d_mat, A_homo)
    A_aligned = A_aligned.reshape(Ashape0)
    return A_aligned, sim3d_mat

def eval_multiview_depths(gt_depths, gt_masks, pred_depths, pred_to_gt_scale=None):
    if pred_to_gt_scale is None:
        raise NotImplementedError #TODO estimate the scale
    pred_depths = pred_depths*pred_to_gt_scale
    output_dict = {}
    rel = (gt_depths[gt_masks] - pred_depths[gt_masks]).abs() / (gt_depths[gt_masks] + 1e-6)
    rel = rel.mean().item()
    output_dict['rel(%)'] = round(rel*100,2)
    r1, r2 = gt_depths[gt_masks]/pred_depths[gt_masks], pred_depths[gt_masks]/gt_depths[gt_masks]
    rmax = torch.where(r1>r2, r1, r2)
    for threshold in [1.01,1.03,1.10,1.25]:
        output_dict[f'tau@{threshold:.2f}'] = round((rmax<threshold).float().mean().item()*100,1)
    return output_dict

def rmse(gt_points, pred_points, unit=('cm', 100), eval_mask=None):
    if eval_mask is None:
        eval_mask = torch.ones_like(gt_points[..., 0], dtype=torch.bool)
    output_dict = {}
    rmses = torch.sqrt(torch.sum((gt_points[eval_mask] - pred_points[eval_mask]) ** 2, axis=-1))*unit[1]
    output_dict['rmses'] = rmses
    output_dict[f'points/rmse_mean({unit[0]})'] = round(rmses.mean().item(),2)
    output_dict[f'points/rmse_median({unit[0]})'] = round(rmses.median().item(),2)
    threshold = torch.arange(1,11).to(rmses.device) #1 to 10
    recalls = (rmses[:,None]<threshold[None,:]).float().mean(0) # (Npts, Nthr)-> (Nthr,)
    aucs = torch.cumsum(recalls, axis=0)/threshold #(Nthr,)

    for i in [1, 3, 5, 10]:
        output_dict[f'points/auc@{i:02d}{unit[0]}(%)'] = round(aucs[i-1].item()*100,2)
    return output_dict

def eval_points(gt_points, pred_points, eval_mask=None, align_mask=None, 
            umeyama=True, unit=('cm',100), umeyama_max_error=0.03,
            save_errormap=False, output_filename=None):
    output_dict = {}

    N, H, W, _ = gt_points.shape
    device = gt_points.device
    if eval_mask is None:
        eval_mask = torch.ones([N,H,W], dtype=torch.bool).to(device)
    if umeyama:
        if align_mask is None:
            align_mask = eval_mask
        pred_points_aligned, sim3d_mat = umeyama_alignment(B=gt_points, A=pred_points, mask=eval_mask, max_error=umeyama_max_error)
    else:
        pred_points_aligned = pred_points 
        sim3d_mat = torch.eye(4).to(device)[:3,:] #3,4
    rmses_dict = rmse(gt_points=gt_points, pred_points=pred_points_aligned, eval_mask=eval_mask, unit=unit)
    output_dict.update(rmses_dict)

    if save_errormap:
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        rmses = rmses_dict['rmses']
        errormap1D, errorbar = create_error_map(rmses, min_val=0, max_val=10, cmap='jet')
        cv2.imwrite(output_filename.replace('.ply', '_bar.png'), errorbar[..., [2,1,0]]) #RGB to BGR
        errormap2D = np.ones([N,H,W,3], dtype=np.uint8)*255
        errormap2D[eval_mask.cpu().numpy()] = errormap1D
        save_images_as_grid(errormap2D, output_filename.replace('.ply', '2D.png'), num_per_row=4)
        save_xyzrgb_to_ply(points=pred_points_aligned[eval_mask], rgb=errormap1D, filename=output_filename)
    return output_dict, pred_points_aligned, sim3d_mat


def eval_cameras(gt_extrinsics, pred_extrinsics, gt_intrinsics, pred_intrinsics):
    c2w_pred = closed_form_inverse_se3(pred_extrinsics) #(N,4,4)
    c2w_gt = closed_form_inverse_se3(gt_extrinsics) #(N,4,4)
    ex_metrics = compute_extrinsic_error(c2w_pred=c2w_pred.cpu().numpy(), c2w_gt=c2w_gt.cpu().numpy())
    Hs = (gt_intrinsics[0,1,2]+0.5)*2
    Ws = (gt_intrinsics[0,0,2]+0.5)*2 #linear camera (same cameras), opencv convention
    in_metrics = compute_intrinsic_error(intrinsics_pred=pred_intrinsics.cpu().numpy(), intrinsics_gt=gt_intrinsics.cpu().numpy(), Hs=Hs.item(), Ws=Ws.item())
    return {**ex_metrics, **in_metrics}






def eval_chamfer_distance(gt_points, pred_points, max_num_pts=100000, unit=('cm', 100)):
    """
    gt_points (N,3)
    pred_points (M,3)
    return {
    "gt2pred': {'rmse_mean': , 'rmse_median': , 'auc@01cm(%)': , 'auc@5': , 'auc@10': }: # Completeness
    "pred2gt": {'rmse_mean': , 'rmse_median': , 'auc@01cm(%)': , 'auc@5': , 'auc@10': }: # Accuracy
    }
    Subsamples both point clouds to max_num_pts using deterministic evenly-spaced indices.
    """
    device = gt_points.device
    gt_points = gt_points.reshape(-1, 3)
    pred_points = pred_points.reshape(-1, 3)
    n_gt, n_pred = gt_points.shape[0], pred_points.shape[0]

    def _deterministic_subsample(pts, n_pts, max_n):
        """Subsample to at most max_n points via evenly-spaced indices (no RNG)."""
        if n_pts <= max_n:
            return pts
        # Evenly spaced indices (deterministic)
        indices = torch.round(
            torch.linspace(0, n_pts - 1, max_n, device=device, dtype=torch.float32)
        ).long().clamp(0, n_pts - 1)
        return pts[indices]

    gt_sub = _deterministic_subsample(gt_points, n_gt, max_num_pts)
    pred_sub = _deterministic_subsample(pred_points, n_pred, max_num_pts)

    # pytorch3d expects (N, P, D); use batch size 1
    gt_batch = gt_sub.unsqueeze(0)   # (1, n_gt_sub, 3)
    pred_batch = pred_sub.unsqueeze(0)  # (1, n_pred_sub, 3)
    # point_reduction=None returns per-point (squared) distances; batch_reduction must be None
    loss_per_pt, _ = chamfer_distance(
        gt_batch, pred_batch,
        batch_reduction=None, point_reduction=None, norm=2, single_directional=False
    )
    # loss_per_pt is (cham_x, cham_y): squared L2 distances, (1, n_gt) and (1, n_pred)
    gt2pred_sq = loss_per_pt[0].squeeze(0)   # completeness: gt -> nearest pred
    pred2gt_sq = loss_per_pt[1].squeeze(0)   # accuracy: pred -> nearest gt
    gt2pred_dists = torch.sqrt(gt2pred_sq.clamp(min=0.0))
    pred2gt_dists = torch.sqrt(pred2gt_sq.clamp(min=0.0))

    def _metrics(dists, unit):
        dists_cm = dists * unit[1]
        mean_cm = dists_cm.mean().item()
        median_cm = dists_cm.median().item()
        threshold = torch.arange(1, 11, device=device, dtype=dists.dtype)
        recalls = (dists_cm.unsqueeze(1) < threshold.unsqueeze(0)).float().mean(dim=0)
        return recalls

    return_dict = {}
    completeness_recalls = _metrics(gt2pred_dists, unit)
    accuracy_recalls = _metrics(pred2gt_dists, unit)
    for interval in [1,3,5,10]:
        return_dict[f'chamfer/completeness@{interval}{unit[0]}(%)'] = round(completeness_recalls[interval-1].item()*100,2)
        return_dict[f'chamfer/accuracy@{interval}{unit[0]}(%)'] = round(accuracy_recalls[interval-1].item()*100,2)
    return return_dict