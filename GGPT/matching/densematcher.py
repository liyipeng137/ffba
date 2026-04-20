import torch 
from PIL import Image
import torch.nn.functional as F
from tqdm import tqdm
import os


def homo(xy):
    """
    Convert 2D/3D points from (x, y) to (x, y, 1) homogeneous coordinates.
    """
    return torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)

def match_dense(model, query_points_hr, images_hr, hr_to_lr,  model_name='roma', output_min_resolution=1920):
    # ! query_points_hr is ignored in this memory-efficient version.
    # The output resolution of RoMA (upsample_res) = (lr_h, lr_w) We don't need query points anymore.
    device = images_hr.device
    N,C, hr_h, hr_w = images_hr.shape
    lr_h, lr_w = round(hr_to_lr[1,1].item()*hr_h), round(hr_to_lr[0,0].item()*hr_w)
    symmetric = True
    dense_matcher = model
    if model_name == 'roma':
        #dense_matcher.upsample_res = (hr_h, hr_w)
        assert lr_h*lr_w == query_points_hr[0].shape[0] # just a sanity check
        dense_matcher.upsample_res = (lr_h, lr_w)
        # Note that!! Jan 24: I found this might hurt the performance as the input to the refine decoder only has resolution (lr_h, lr_w)
        # TODO decoder-input's resolution (hr_h, hr_w), output's resolution (lr_h, lr_w)
        # Now Input resolution (lr_h, lr_w), output_resolution (lr_h/2, lr_w/2)
        output_min_resolution = min(hr_w, output_min_resolution)
        upsample_scales = []
        for s in [8,4,2,1]:
            upsample_scales.append(s)
            if hr_w//s>output_min_resolution:
                break
        dense_matcher.decoder.upsample_scales = [str(ss) for ss in upsample_scales]
        print(f"Roma upsample scales: {dense_matcher.decoder.upsample_scales}")


    images_hr_np = (images_hr.permute(0,2,3,1).cpu().numpy()*255).astype('uint8')
    #images_hr_np = (images_square.permute(0,2,3,1).cpu().numpy()*255).astype('uint8')  #(N,H,W,3) in [0,255]
    #Do it in pair-wise manner

    dense_matcher.eval()
    batchify = True
    batch_size = 4
    SCALE_FIRST = True
    assert SCALE_FIRST, 'numerical stability'
    # Batchify
    if batchify:
        #exhaustive_ids
        if symmetric:
            exhaustive_ids = [[i, j] for i in range(N) for j in range(i+1, N)]
        else:
            exhaustive_ids = [[i, j] for i in range(N) for j in range(N) if i != j]
        exhaustive_ids = torch.tensor(exhaustive_ids, device=device)
        #im_As = images_hr[exhaustive_ids[:,0]]
        #im_Bs = images_hr[exhaustive_ids[:,1]] (This is too memory consuming)!
        for b in tqdm(range(0, exhaustive_ids.shape[0], batch_size)):
            #im_A = im_As[b:b+batch_size]
            #im_B = im_Bs[b:b+batch_size]
            ii = exhaustive_ids[b:b+batch_size,0]
            jj = exhaustive_ids[b:b+batch_size,1]
            im_A = images_hr[ii]
            im_B = images_hr[jj]

            with torch.no_grad():
                if model_name == 'roma':
                    warp, certainty = dense_matcher.match(im_A, im_B, device=device, batched=True)
                    output_w = warp.shape[-2]//2 #B,H,W,2
                    warp_AB, warp_BA = warp[...,:output_w,2:], warp[...,output_w:,:2]
                    certainty_AB, certainty_BA = certainty[...,:output_w], certainty[...,output_w:]
                elif  'romav2' in model_name:
                    res = dense_matcher.match(im_A, im_B)
                    warp_AB, warp_BA = res['warp_AB'], res['warp_BA'] #B,H,W,2
                    certainty_AB, certainty_BA = res['overlap_AB'][...,0], res['overlap_BA'][...,0] #B,H,W #In RoMAv2, it
                    # In roma v2, the output resolution is different from h, w. We resize it here?
                    warp_AB = F.interpolate(warp_AB.permute(0,3,1,2), size=(lr_h, lr_w), mode='bilinear', align_corners=False).permute(0,2,3,1)
                    warp_BA = F.interpolate(warp_BA.permute(0,3,1,2), size=(lr_h, lr_w), mode='bilinear', align_corners=False).permute(0,2,3,1)
                    certainty_AB = F.interpolate(certainty_AB.unsqueeze(1), size=(lr_h, lr_w), mode='bilinear', align_corners=False).squeeze(1)
                    certainty_BA = F.interpolate(certainty_BA.unsqueeze(1), size=(lr_h, lr_w), mode='bilinear', align_corners=False).squeeze(1)
                elif  'ufm' in model_name:
                    im_AB = torch.cat([im_A, im_B], dim=0)
                    im_BA = torch.cat([im_B, im_A], dim=0)
                    result = dense_matcher.predict_correspondences_batched(source_image=im_AB, target_image=im_BA, data_norm_type='identity') #it also accept float32 BCHW tensor
                    flow_output = result.flow.flow_output #2*B,2,H,W
                    output_h, output_w = flow_output.shape[-2], flow_output.shape[-1]
                    flow_output = F.interpolate(flow_output, size=(lr_h, lr_w), mode='bilinear', align_corners=False)
                    flow_output[:,0] *= lr_w/output_w #dx
                    flow_output[:,1] *= lr_h/output_h #dy
                    xy0 = torch.meshgrid(torch.arange(lr_w, device=device), torch.arange(lr_h, device=device), indexing='xy')
                    xy0 = torch.stack(xy0, dim=-1).float() #H,W,2
                    warp = flow_output.permute(0,2,3,1)+xy0[None] #2*B,H,W,2 #[0,w-1] in open-cv coordinate
                    warp_AB, warp_BA = torch.split(warp, im_A.shape[0], dim=0)
                    certainty = result.covisibility.mask #2*B, H,W
                    certainty = F.interpolate(certainty.unsqueeze(1), size=(lr_h, lr_w), mode='bilinear', align_corners=False).squeeze(1)
                    certainty_AB, certainty_BA = torch.split(certainty, im_A.shape[0], dim=0)
                    # set those OOI as 0
                    certainty_AB *= (warp_AB[...,0]>=0)&(warp_AB[...,0]<lr_w)&(warp_AB[...,1]>=0)&(warp_AB[...,1]<lr_h)
                    certainty_BA *= (warp_BA[...,0]>=0)&(warp_BA[...,0]<lr_w)&(warp_BA[...,1]>=0)&(warp_BA[...,1]<lr_h)
            assert warp_AB.shape[-3]==lr_h and warp_AB.shape[-2]==lr_w, f"{warp_AB.shape}, {lr_h}, {lr_w}"
            if b==0:
                warp_all_denseoutput = torch.zeros([N,N,lr_h,lr_w,2], device=device)
                certainty_all_denseoutput = torch.ones([N,N,lr_h,lr_w], device=device)
                cycle_error_denseoutput = torch.zeros([N,N,lr_h,lr_w], device=device)
                if SCALE_FIRST:
                    meshgrid = torch.stack(torch.meshgrid(torch.arange(lr_w), torch.arange(lr_h), indexing='xy'), axis=2).to(device).float()#h,w,2
                else:
                    meshgrid = torch.stack(torch.meshgrid(torch.linspace(-1+1/lr_w,1-1/lr_w,lr_w, device=device), torch.linspace(-1+1/lr_h,1-1/lr_h,lr_h, device=device), indexing='xy'), axis=2)
                warp_all_denseoutput = meshgrid[None,None].repeat(N,N,1,1,1)
            if symmetric:
                #In roma, -1 is the corner of the pixel
                if 'roma' in model_name:
                    if SCALE_FIRST:
                        A_to_B = (warp_AB+1)/2
                        B_to_A = (warp_BA+1)/2 #[0,1]
                        A_to_B[...,0] = A_to_B[...,0]*lr_w-0.5
                        A_to_B[...,1] = A_to_B[...,1]*lr_h-0.5
                        B_to_A[...,0] = B_to_A[...,0]*lr_w-0.5
                        B_to_A[...,1] = B_to_A[...,1]*lr_h-0.5 #Convert[0,H] to [-0.5,H-0.5] (in open-cv coordinate)
                    else:
                        A_to_B = warp_AB
                        B_to_A = warp_BA
                else:
                    assert  SCALE_FIRST
                    A_to_B = warp_AB
                    B_to_A = warp_BA
                warp_all_denseoutput[ii,jj] = A_to_B
                certainty_all_denseoutput[ii,jj] = certainty_AB
                warp_all_denseoutput[jj,ii] = B_to_A
                certainty_all_denseoutput[jj,ii] = certainty_BA
                '''
                 Compute cycle error
                https://github.com/cvg/glue-factory/blob/2d17e3b3bd7d30f0c828d4c4d3eac4ecefbf283d/gluefactory/utils/image.py#L232
                '''
                def cycle_error(A_to_B, B_to_A):
                    #A_to_B, B_to_A: b,h,w,2
                    #A_to_B: opencv convention 
                    A_to_B_normalized = A_to_B.clone() #0->-1+1/w, w-1->1-1/w
                    A_to_B_normalized[...,0]= 2*A_to_B_normalized[...,0]/lr_w-1+1/lr_w
                    A_to_B_normalized[...,1] = 2*A_to_B_normalized[...,1]/lr_h-1+1/lr_h
                    A_to_B_to_A = F.grid_sample(
                        B_to_A.permute(0,3,1,2), #b,2,h,w
                        A_to_B_normalized, #b,h,w,2 already normalized
                        mode='bilinear', align_corners=False)#b,2,h,w
                    A_to_B_to_A = A_to_B_to_A.permute(0,2,3,1) #b,h,w,2
                    # meshgrid = torch.stack(torch.meshgrid(torch.linspace(-1+1/lr_w,1-1/lr_w,lr_w, device=device), 
                    #         torch.linspace(-1+1/lr_h,1-1/lr_h,lr_h, device=device), indexing='xy'), axis=2) #h,w,2
                    meshgrid = torch.stack(torch.meshgrid(torch.arange(0,lr_w, device=device),torch.arange(0,lr_h, device=device), indexing='xy'), axis=2)
                    error = torch.linalg.norm(A_to_B_to_A-meshgrid[None], dim=-1)
                    return error
                cycle_error_denseoutput[ii,jj] = cycle_error(A_to_B, B_to_A)
                cycle_error_denseoutput[jj,ii] = cycle_error(B_to_A, A_to_B)
    pred_tracks_lr = warp_all_denseoutput.view(N, N, -1, 2) #N,N,hr_h,hr_w,2
    pred_scores = certainty_all_denseoutput.view(N, N, -1)  #N,N,hr_h,hr_w
    pred_cycle_error = cycle_error_denseoutput.view(N, N, -1)  #
    return {
        'pred_matches_lr': pred_tracks_lr, 'pred_scores': pred_scores,
        'pred_cycle_error': pred_cycle_error
    }
