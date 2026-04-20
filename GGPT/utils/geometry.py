import torch
import numpy as np


def homo(x):
    ones = torch.ones_like(x[...,-1:])
    homo_x = torch.cat([x,ones],axis=-1)
    return homo_x

def project_point_map_to_depth_map_torch(point_map, extrinsics_cam, intrinsics_cam):
    if point_map.ndim == 3:
        point_map, extrinsics_cam, intrinsics_cam = point_map[None,...], extrinsics_cam[None,...], intrinsics_cam[None,...]
    S, H, W, _ = point_map.shape
    depth_map = torch.zeros((S, H, W, 1), device=point_map.device)
    proj = torch.einsum('nij,njk->nik', intrinsics_cam, extrinsics_cam[:,:3,:4])
    point_map_projected = torch.einsum('nij,nhwj->nhwi', proj, homo(point_map))
    depth_map = point_map_projected[...,-1] # (N,H,W)
    return depth_map

def unproject_depth_map_to_point_map_torch(depth_map, extrinsics_cam, intrinsics_cam, eps=1e-4):
    """
    Unproject a batch of depth maps to 3D world coordinates, and reset invalid points to zero.
    
    Args:
        depth_map (torch.tensor): Batch of depth maps of shape (S, H, W, 1) or (S, H, W)
        extrinsics_cam (torch.tensor): Batch of camera extrinsic matrices of shape (S, 3, 4)
        intrinsics_cam (torch.tensor): Batch of camera intrinsic matrices of shape (S, 3, 3)
    """
    S, H, W = depth_map.shape[:3]
    if depth_map.ndim == 3:
        depth_map = depth_map.unsqueeze(-1)

    point_mask = depth_map[...,0] > 1e-4 #S,H,W

    u, v = torch.meshgrid(
        torch.arange(W, device=depth_map.device),
        torch.arange(H, device=depth_map.device),
        indexing="xy",
    ) #For open cv convention, uv refers to the pixel center now
    uv = torch.stack((u, v), dim=-1).float().unsqueeze(0).repeat(S,1,1,1)  #(S,H,W,2)
    uv_homo = homo(uv) #S,H,W,3
    Kinv = closed_form_inverse_K(intrinsics_cam)  #(S,3,3)
    ray_d = torch.einsum("sij,shwj->shwi", Kinv, uv_homo) # (S,h,w,3)
    cam_pts = depth_map.repeat(1,1,1,3)*ray_d # (S,h,w,1)
    c2w = closed_form_inverse_se3(extrinsics_cam) #(S,3,4) or (S,4,4)
    world_pts = torch.einsum('sij,shwj->shwi', c2w, homo(cam_pts))[...,:3] #S,3
    world_pts[~point_mask] = 0
    return world_pts



def depth_to_world_coords_points_torch(
    depth_map: torch.tensor,
    extrinsic: torch.tensor,
    intrinsic: torch.tensor,
    eps=1e-8):
    if depth_map is None:
        return None, None, None

    # Valid depth mask
    point_mask = depth_map > eps

    # Convert depth map to camera coordinates
    cam_coords_points = depth_to_cam_coords_points_torch(depth_map, intrinsic)

    # The extrinsic is camera-from-world, so invert it to transform camera->world
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0] #It accepts torch.tensor
    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    world_coords_points = torch.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask

def depth_to_cam_coords_points_torch(
    depth_map: torch.tensor, intrinsic: torch.tensor
) -> torch.tensor:
    """
    Unprojects a depth map into camera coordinates, returning (H, W, 3).

    Args:
        depth_map (torch.tensor):
            Depth map of shape (H, W).
        intrinsic (torch.tensor):
            3x3 camera intrinsic matrix.
            Assumes zero skew and standard OpenCV layout:
            [ fx   0   cx ]
            [  0  fy   cy ]
            [  0   0    1 ]

    Returns:
        torch.tensor:
            An (H, W, 3) array, where each pixel is mapped to (x, y, z) in the camera frame.
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert (
        intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0
    ), "Intrinsic matrix must have zero skew"

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = torch.meshgrid(
        torch.arange(W, device=depth_map.device),
        torch.arange(H, device=depth_map.device),
        indexing="xy",
    ) #For open cv convention, uv refers to the pixel center now

    # Unproject to camera coordinates
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # Stack to form camera coordinates
    return torch.stack((x_cam, y_cam, z_cam), dim=-1).float()

def closed_form_inverse_K(K):
    ndim = K.ndim
    if K.ndim == 2:
        K = K[None,:,:]
    S, _, _ = K.shape
    assert K[...,0,1].sum()<1e-3 and K[...,1,0].sum()<1e-3, "K must be in linear camera model format"
    assert K[...,2,0].sum()<1e-3 and K[...,2,1].sum()<1e-3 and K[...,2,2].mean()<1+1e-3, "K must be in linear camera model format"
    fx = K[...,0,0]
    fy = K[...,1,1]
    cx = K[...,0,2]
    cy = K[...,1,2]
    K_inv = torch.zeros_like(K)
    K_inv[...,0,0] = 1.0/fx
    K_inv[...,1,1] = 1.0/fy
    K_inv[...,0,2] = -cx/fx
    K_inv[...,1,2] = -cy/fy
    K_inv[...,2,2] = 1.0
    if ndim ==2:
        K_inv = K_inv[0]
    return K_inv


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    REMOVE_BATCH_DIM = False
    if se3.ndim == 2:
        se3 = se3[None, :, :]  # Add batch dimension
        REMOVE_BATCH_DIM = True
    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    if REMOVE_BATCH_DIM:
        inverted_matrix = inverted_matrix[0]
    return inverted_matrix


def compute_epipolar_errors(w2c_0,w2c_s, K_0, K_s, matches):
    '''
    w2c_0: [3,4]
    w2c_s: [N,3,4]
    K_0: [3,3]
    K_s: [N,3,3]
    matches: [N, H, W, 2]  #The correspondence of (h,w) pixel in view 0 in view (n)
    '''
    N = w2c_s.shape[0]
    R1, t1 = w2c_0[:3,:3], w2c_0[:3,-1]
    R2s, t2s = w2c_s[:,:3,:3], w2c_s[:,:3,-1] #N,3,3 N,3
    Rs = torch.einsum('nij,jk->nik', R2s, R1.T) #N,3,3
    ts = t2s - torch.einsum('nij,j->ni', Rs, t1) #N,3
    degenerate_msk = torch.norm(ts, dim=1)<1e-1

    tsx = torch.zeros([N,3,3]).to(Rs.device)
    tsx[:,0,1] = -ts[:,2]
    tsx[:,0,2] = ts[:,1]
    tsx[:,1,0] = ts[:,2]
    tsx[:,1,2] = -ts[:,0]
    tsx[:,2,0] = -ts[:,1]
    tsx[:,2,1] = ts[:,0]
    Es = torch.einsum('nij,njk->nik', tsx, Rs) #N,3,3

    Fs_1 = torch.einsum('nij,jk->nik', Es, closed_form_inverse_K(K_0)) #N,3,3
    Fs = torch.einsum('nij,njk->nik', closed_form_inverse_K(K_s).permute(0,2,1), Fs_1) #N,3,3

    H, W = matches.shape[1], matches.shape[2]
    device = matches.device
    xy0 = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy') #w,h
    xy0 = torch.stack(xy0, axis=-1).float() #(h,w,2)
    line_0tos = torch.einsum('hwi,nij->nhwj',homo(xy0), Fs.permute(0,2,1)) #(N,H,W,3)
    line_sto0 = torch.einsum('nhwi,nij->nhwj',homo(matches),Fs)


    def perpendicular_distance(line, uvs):
        #line : (N,3)
        #uvs:   (N,2) -> (N,3)
        nume = torch.einsum('nk,nk->n', line, homo(uvs)).abs() #N,
        deno = torch.norm(line[:,:2], dim=1) #N,
        return nume/(deno+1e-10)
    
    dis_a = perpendicular_distance(line_0tos.reshape(-1,3), matches.reshape(-1,2)).reshape(N,H,W)
    xy0 = xy0[None,...].repeat(N,1,1,1)
    dis_b = perpendicular_distance(line_sto0.reshape(-1,3), xy0.reshape(-1,2)).reshape(N,H,W)
    #! The colinear/pure rotation case! we don't consider teh dis_a and dis_b can be quite large
    return dis_a, dis_b

def compute_infrustum(extrinsics, intrinsics, points=None, depths=None, downsample=1):
    """
    Compute the pairwise covisibility scores. The inputs can be either from GT or Pred
    extrinsics: (N, 4, 4) 
    intrinsics: (N, 3, 3)
    points: (N, H, W, 3)
    depths: (N, H, W)
    outputs: (Ntgt,Nsrc,h,w) # downsample
    """
    if points is None:
        assert depths is not None
        points = unproject_depth_map_to_point_map_torch(depths, extrinsics, intrinsics) #N,H,W,3 
    if depths is None:
        assert points is not None
        depths = project_point_map_to_depth_map_torch(points, extrinsics, intrinsics)
    N, H, W = depths.shape
    device = depths.device
    if downsample>1:
        points = points[:,::downsample,::downsample,:]
        depths = depths[:,::downsample,::downsample]
        h, w = depths.shape[1:3]
    else:
        h, w = H, W

    points_ = points[None,...].repeat(N,1,1,1,1).reshape(-1,h,w,3) #(Ntgt*Nsrc,h,w,3)
    extrinsics_ = extrinsics[:,None,...].repeat(1,N,1,1).reshape(-1,4,4) #(Ntgt*Nsrc,4,4)
    intrinsics_ = intrinsics[:,None,...].repeat(1,N,1,1).reshape(-1,3,3) #(Ntgt*Nsrc,3,3)
    proj_xyzs = project_point_map_to_depth_map_torch(points_, extrinsics_, intrinsics_) #(Ntgt*Nsrc,h,w)
    proj_xyzs = proj_xyzs.reshape(N,N,h,w,3)
    proj_depths = proj_xyzs[...,-1]
    proj_xys = proj_xyzs[...,:2]/(proj_depths[... ,None]+1e-10)  #(Ntgt,Nsrc,h,w,2)
    in_frustum = (proj_depths>1e-4) & (proj_xys[...,0]>=0) & (proj_xys[...,0]<W) & (proj_xys[...,1]>=0) & (proj_xys[...,1]<H)
    return in_frustum  #(Ntgt, Nsrc, h, w)



    



