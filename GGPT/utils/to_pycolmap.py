# import pycolmap 
import numpy as np

def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri

def batch_torch_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):
    import pycolmap
    """

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index



    NOTE Yutong: if points3d is not None, leave points3d/2d empty
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    extrinsics, intrinsics = extrinsics.cpu().numpy(), intrinsics.cpu().numpy()
    points3d, tracks, masks = points3d.cpu().numpy(), tracks.cpu().numpy(), masks.cpu().numpy()
    assert len(intrinsics) == N
    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()
    points2d_negconfs = [] #[(N_pts,)] # For query points in that frame, the conf is 

    if points3d is not None:
        assert len(points3d) == P
        inlier_num = masks.sum(0) #N_track
        valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
        valid_idx = np.nonzero(valid_mask)[0]
        # Only add 3D points that have sufficient 2D points
        for vidx in valid_idx:
            # Use RGB colors if provided, otherwise use zeros
            rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
            reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)
        num_points3D = len(valid_idx)
    else:
        valid_mask = None #We do not filter any tracks at this stage, only output reconstruction
    camera = None

    # frame idx
    for fidx in range(N):
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)
            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)
            if True or pycolmap.__version__ == '3.13.0': #To check the version
                rig = pycolmap.Rig(rig_id=fidx+1)
                rig.add_ref_sensor(camera.sensor_id)
                reconstruction.add_rig(rig)


        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans
        if True or pycolmap.__version__ == '3.13.0':
            image = pycolmap.Image(frame_id=fidx+1, image_id=fidx+1, camera_id=camera.camera_id, name=f"image_{fidx + 1}.png")
            frame = pycolmap.Frame(frame_id=fidx+1, rig_id=rig.rig_id, rig_from_world=cam_from_world)
            frame.add_data_id(image.data_id)
            
        else:
            image = pycolmap.Image(
                image_id=fidx + 1, name=f"image_{fidx + 1}.png", camera_id=camera.camera_id, cam_from_world=cam_from_world)




        if points3d is not None:
            points2D_list = []

            point2D_idx = 0
            # NOTE point3D_id start by 1f
            for point3D_id in range(1, num_points3D + 1):
                original_track_idx = valid_idx[point3D_id - 1]
                if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all():
                    if masks[fidx][original_track_idx]:
                        # It seems we don't need +0.5 for BA ??
                        point2D_xy = tracks[fidx][original_track_idx]
                        # Please note when adding the Point2D object
                        # It not only requires the 2D xy location, but also the id to 3D point
                        points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))
                        # add element
                        track = reconstruction.points3D[point3D_id].track
                        track.add_element(fidx + 1, point2D_idx)
                        point2D_idx += 1

            assert point2D_idx == len(points2D_list)
            if point2D_idx == 0:
                print(f"frame {fidx + 1} does not have any points.")
                #return None, valid_mask, points2d_negconfs

            #image.points2D = pycolmap.ListPoint2D(points2D_list) (colmap 3.9)
            image.points2D = pycolmap.Point2DList(points2D_list)  # colmap 3.12
        #image.registered = True #colmap 3.9 Comment this for colmap 3.12
        '''
        try:
            print(f"frame {fidx + 1} ", len(points2D_list))
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"frame {fidx + 1} is out of BA")
            #image.registered = False
        '''
        
        # add image
        if True or pycolmap.__version__ == '3.13.0': #To check the version
            reconstruction.add_frame(frame)
            reconstruction.add_image(image)
        else:
            reconstruction.add_image(image)

    return reconstruction
