
import torch
import os
import cv2
import numpy as np

import torch
import os
import cv2
import numpy as np

def error_to_jet(error, max_error=8.0):
    """
    error: scalar (float or 0-D torch tensor), already clipped into [0, max_error]
    returns: (R, G, B) in [0, 255], ints
    """
    # normalize to [0,1]
    if hasattr(error, "item"):
        e = float(error.item())
    else:
        e = float(error)
    x = max(0.0, min(1.0, e / max_error))

    # jet-like mapping (RGB)
    def _jet_channel(x, shift):
        v = 1.5 - abs(4.0 * x - shift)
        return max(0.0, min(1.0, v))

    r = _jet_channel(x, 3.0)
    g = _jet_channel(x, 2.0)
    b = _jet_channel(x, 1.0)

    return (int(255 * r), int(255 * g), int(255 * b))

def add_margin_to_image(img, top=0, bottom=0, left=0, right=0, color=(255,255,255)):
    H, W, C = img.shape
    new_H = H + top + bottom
    new_W = W + left + right
    canvas = np.ones((new_H, new_W, C), dtype=img.dtype)
    canvas = canvas * np.array(color, dtype=img.dtype).reshape(1,1,3)
    canvas[top:top+H, left:left+W, :] = img
    return canvas

def vis_matches_in_multiview(images, matches, gt_matches, eval_mask, is_query_mask,vis_num=512, 
        filename=None, col_num=4):
    '''
    images: N,H,W,3
    matches: N,M,2
    gt_matches: N,M,2
    eval_mask: N,H,W bool tensor
    vis: N,M bool tensor
    filename: None
    '''
    N, H, W, C = images.shape
    N, M, _ = matches.shape
    images_show = images.clone()
    images_show = (images_show.cpu().numpy()*255).astype(np.uint8)
    for m in np.linspace(0, M-1, vis_num):
        m = int(m)
        for i in range(N):
            xys = matches[i,m]
            gt_xys = gt_matches[i,m]
            xys = xys.cpu().numpy().astype(np.int32)
            gt_xys = gt_xys.cpu().numpy().astype(np.int32)
            #Clip
            xys[0] = np.clip(xys[0], 0, W-1)
            xys[1] = np.clip(xys[1], 0, H-1)
            gt_xys[0] = np.clip(gt_xys[0], 0, W-1)
            gt_xys[1] = np.clip(gt_xys[1], 0, H-1)
            if eval_mask[i, m] or is_query_mask[i, m]:
                #The color depends on the error
                if not is_query_mask[i, m]:
                    error = np.linalg.norm(xys - gt_xys).clip(0,8)
                    color = error_to_jet(error, max_error=8)
                else: #gray (query)
                    color = (128, 128, 128)
                color = tuple([int(c) for c in color])
                cv2.circle(images_show[i], xys, radius=4, color=color, thickness=1) #big unfilled circle
                cv2.circle(images_show[i], gt_xys, radius=3, color=color, thickness=-1) #filled
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    row_num = int(np.ceil(len(images_show) / col_num))
    images_grid = np.ones((row_num*H, col_num*W, 3), dtype=np.uint8)*255
    for i in range(row_num):
        for j in range(col_num):
            if i*col_num+j < len(images_show):
                images_grid[i*H:(i+1)*H, j*W:(j+1)*W] = images_show[i*col_num+j]
    cv2.imwrite(filename, images_grid[:,:,[2,1,0]])
    return images_grid[:,:,[2,1,0]]

def vis_matches_in_pairs(
    view1, view2,
    xy_view1, xy_view2_pred, xy_view2_gt, error, 
    quantile=[0.1,0.9], vis_num=30,
    filename=None):
    """
    view1: H, W, 3 (float)
    view2: H, W, 3 (float32)
    xy_view1: N, 2 (in view0)
    xy_view2_pred: N, 2 (in view1)
    xy_view2_gt: N, 2 (in view1)
    error: N, 1
    quantile: [0.1,0.9]
    vis_num: 30
    filename: None
    """
    H, W = view1.shape[:2]
    images_show = torch.cat([view1, view2], dim=1).cpu().numpy()
    images_show = (images_show*255).astype(np.uint8)
    #sort tracks by error (ascending)
    sort_idx = torch.argsort(error)
    xy_view1 = xy_view1[sort_idx]
    xy_view2_pred = xy_view2_pred[sort_idx]
    xy_view2_gt = xy_view2_gt[sort_idx]
    error = error[sort_idx]
    #vis_num tracks
    # We visualize two panes side by side:
    # left: tracks from the lower-error quantile, right: from the higher-error quantile.
    def draw_matches(images_show, xy_view1, xy_view2_pred, xy_view2_gt, error, quantile, toptext=""):
        images_show = images_show.copy()
        colors = torch.rand(len(xy_view1), 3) 
        colors = (colors*255).cpu().numpy().astype(np.uint8)
        xy_view2_pred = xy_view2_pred.clone(); xy_view2_pred[:,0] += W
        xy_view2_gt = xy_view2_gt.clone(); xy_view2_gt[:,0] += W
        average_error = error.mean()
        for i in range(len(xy_view1)):
            # convert coordinates to Python ints for OpenCV
            xy0 = tuple(int(v) for v in xy_view1[i].tolist())
            xy1_pred = tuple(int(v) for v in xy_view2_pred[i].tolist())
            xy1_gt = tuple(int(v) for v in xy_view2_gt[i].tolist())
            color = tuple([int(c) for c in colors[i].tolist()])
            images_show = cv2.circle(images_show, xy0, radius=7, color=color, thickness=-1)
            images_show = cv2.circle(images_show, xy1_pred, radius=7, color=color, thickness=-1)
            images_show = cv2.circle(images_show, xy1_gt, radius=7, color=color, thickness=3)
            images_show = cv2.line(images_show, xy0, xy1_pred, color=color, thickness=1)
            images_show = cv2.line(images_show, xy1_pred, xy1_gt, color=color, thickness=3, lineType=cv2.LINE_AA)
        # Add text on top of the image showing the average error
        # (Pad white margin to the top)
        images_show = add_margin_to_image(images_show, top=50)
        cv2.putText(images_show, toptext, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return images_show
    # If no tracks, just return the concatenated image
    N = len(xy_view1)
    if N == 0:
        return images_show

    # Uniformly sample up to vis_num tracks from low-error and high-error ranges
    max_idx = N - 1
    # low-error range: [0, hi1]
    hi1 = int(N * quantile[0]) - 1
    hi1 = max(hi1, 0)
    # high-error range: [lo2, max_idx]
    lo2 = int(N * (1 - quantile[0]))
    lo2 = min(lo2, max_idx)

    steps1 = min(vis_num, hi1 - 0 + 1)
    steps2 = min(vis_num, max_idx - lo2 + 1)

    indices1 = torch.linspace(0, hi1, steps=steps1, dtype=torch.long)
    indices2 = torch.linspace(lo2, max_idx, steps=steps2, dtype=torch.long)
    images_show1 = draw_matches(images_show, xy_view1[indices1], xy_view2_pred[indices1], xy_view2_gt[indices1], error[indices1], quantile[0], toptext=f"Quantile {quantile[0]:.2f}: Avg. error: {error[indices1].mean():.2f}")
    images_show2 = draw_matches(images_show, xy_view1[indices2], xy_view2_pred[indices2], xy_view2_gt[indices2], error[indices2], quantile[1], toptext=f"Quantile {quantile[1]:.2f}: Avg. error: {error[indices2].mean():.2f}")
    images_show = np.concatenate([images_show1, images_show2], axis=1)
    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        cv2.imwrite(filename, images_show[:,:,[2,1,0]])
    return images_show

def vis_matches(images, matches, visibility, filename, vis_num_track=5, vis_mask=None):
    '''
    images: N,C,H,W (0,1) tensor
    matches: N,M,2 (x,y) tensor (opencv convention)
    visibility: N,M bool tensor
    '''
    N, C, H, W = images.shape
    images_show = (images.permute(2,0,3,1).reshape(H,N*W,C)*255).cpu().numpy().astype(np.uint8)
    # Randomly choose 5 tracks to visualize
    if vis_mask is  None:
        vis_num_track = min(vis_num_track, matches.shape[1])
        show_track_ids = torch.randperm(matches.shape[1])[:vis_num_track]
    else:
        show_track_ids = torch.nonzero(vis_mask).squeeze(1)
        if len(show_track_ids) == 0:
            print(f"No tracks to visualize in {filename}.")
            return
        if len(show_track_ids) > vis_num_track:
            show_track_ids = show_track_ids[torch.randperm(len(show_track_ids))[:vis_num_track]]
    colors = (torch.rand(vis_num_track,3)*255).cpu().numpy().astype(np.uint8)
    matches = matches.cpu().numpy().astype(np.int32)
    visibility = visibility.cpu().numpy().astype(np.bool_)
    show_track_ids = show_track_ids.cpu().numpy().astype(np.int32)
    for track_id, color in zip(show_track_ids, colors):
        xys, vis = matches[0,track_id], visibility[0,track_id]
        color = tuple([int(c) for c in color])
        images_show = cv2.circle(images_show, xys, radius=4, color=color, thickness=-1 if vis else 2)
        last_xys = xys
        for ii in range(1,N):
            xys, vis = matches[ii,track_id].copy(), visibility[ii,track_id]
            xys[0] = np.clip(xys[0], 0, W-1)
            xys[1] = np.clip(xys[1], 0, H-1)
            xys[0] += ii*W
            images_show = cv2.circle(images_show, xys, radius=4, color=color, thickness=-1 if vis else 2)
            images_show = cv2.line(images_show, last_xys, xys, color=color, thickness=2 )
            last_xys = xys
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, images_show[:,:,[2,1,0]])
    return  images_show[:,:,[2,1,0]]