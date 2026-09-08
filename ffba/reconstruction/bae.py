"""Real-track BAE controller with the existing final-round filtering schedule."""

from gluemap.utils.colmap import camera_from_intrinsics_matrix
import numpy as np
import pycolmap

from dataclasses import dataclass
import logging

from gluemap.math.reprojection_error import (
    ReprojectionErrorType,
    filter_reconstruction_by_reprojection_error,
)

logger = logging.getLogger(__name__)


@dataclass
class IterativeBAOptions:
    max_filter_iterations: int
    normalized_reproj_threshold: float
    min_track_length: int
    bae_device: str
    bae_max_iterations: int
    bae_optimize_intrinsics: bool
    bae_fix_gauge: str
    bae_robust_loss: str
    bae_huber_delta: float
    run_post_ba_filter: bool = True
    last_ba_summary: dict | None = None


def iterative_bundle_adjustment(
    reconstruction,
    virtual_reconstruction=None,
    negative_depth_observations=None,
    *,
    options,
):
    """Optimize once; on the final round filter at 3x, 2x, 1x thresholds."""
    if virtual_reconstruction is not None or negative_depth_observations:
        raise ValueError("The formal BAE controller accepts real tracks only")
    from gluemap.estimators.bae_solver import bundle_adjustment_bae

    reconstruction, _, summary = bundle_adjustment_bae(
        reconstruction,
        None,
        {},
        max_num_iterations=options.bae_max_iterations,
        device=options.bae_device,
        optimize_intrinsics=options.bae_optimize_intrinsics,
        real_only=True,
        fix_gauge=options.bae_fix_gauge,
        robust_loss=options.bae_robust_loss,
        huber_delta=options.bae_huber_delta,
    )
    options.last_ba_summary = summary
    if options.run_post_ba_filter:
        for iteration in range(options.max_filter_iterations):
            threshold = max(3 - iteration, 1) * options.normalized_reproj_threshold
            filter_reconstruction_by_reprojection_error(
                reconstruction,
                ReprojectionErrorType.NORMALIZED,
                threshold,
                options.min_track_length,
                negative_depth_observations=None,
                log_level=logging.DEBUG,
                log_prefix="real: ",
            )
    return reconstruction, None


def build_seed_reconstruction_for_ba(
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_intrinsics: list,
    intrinsics_mapping: dict[int, int],
    keypoints_per_image: dict[int, np.ndarray],
    image_sizes: list[tuple[int, int]] | None = None,
    images_list: list[str] | None = None,
    camera_model: str = "SIMPLE_PINHOLE",
) -> pycolmap.Reconstruction:
    """Build a registered pose/keypoint seed without any 3D points."""
    return build_reconstruction_for_ba(
        global_rotations,
        global_centers,
        global_intrinsics,
        intrinsics_mapping,
        points3D={},
        keypoints_per_image=keypoints_per_image,
        image_sizes=image_sizes,
        images_list=images_list,
        camera_model=camera_model,
    )


def build_reconstruction_for_ba(
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_intrinsics: list,
    intrinsics_mapping: dict[int, int],
    points3D: dict[int, pycolmap.Point3D],
    keypoints_per_image: dict[int, np.ndarray],
    image_sizes: list[tuple[int, int]] | None = None,
    images_list: list[str] | None = None,
    camera_model: str = "SIMPLE_PINHOLE",
) -> pycolmap.Reconstruction:
    """
    Build pycolmap.Reconstruction from separate data structures.

    All inputs use 0-indexed image / camera IDs (matching keypoints_per_image,
    global_rotations, etc.); the returned reconstruction uses 1-indexed image
    and camera IDs to match the COLMAP convention used by the database written
    in prepare_glomap_prior.

    Args:
        global_rotations: Dict[image_id, np.ndarray(3,3)] (0-indexed keys)
        global_centers: Dict[image_id, np.ndarray(3,)] (0-indexed keys)
        global_intrinsics: List of intrinsics tensors (0-indexed)
        intrinsics_mapping: Dict[image_id, camera_type_idx] (0-indexed both
            sides)
        points3D: Dict[point3D_id, pycolmap.Point3D] with 0-indexed image_ids in
            track elements
        keypoints_per_image: Dict[image_id, np.ndarray(N,2)] (0-indexed keys)
        image_sizes: List[(height, width)] indexed by 0-indexed image_id -
            optional
        images_list: List[str] of image filenames indexed by 0-indexed image_id
            - optional
        camera_model: Camera model string

    Returns:
        pycolmap.Reconstruction with 1-indexed image_id / camera_id.
    """
    reconstruction = pycolmap.Reconstruction()

    # Add cameras (camera_id is 1-indexed in the output reconstruction)
    for camera_id, intrinsics in enumerate(global_intrinsics):
        if intrinsics is None:
            continue

        # Find image size for this camera (use first image with this camera_id)
        width, height = None, None
        if image_sizes is not None:
            for img_id, cam_id in intrinsics_mapping.items():
                if cam_id == camera_id and img_id < len(image_sizes):
                    height, width = image_sizes[img_id]
                    break

        # Note that there is an extra dimension in intrinsics, so we take
        # intrinsics[0]
        camera = camera_from_intrinsics_matrix(
            intrinsics[0], camera_model, width, height, camera_id + 1
        )
        reconstruction.add_camera_with_trivial_rig(camera)

    # Add images (image_id is 1-indexed in the output reconstruction)
    for image_id in global_rotations:
        if image_id not in global_centers:
            continue
        if image_id not in intrinsics_mapping:
            continue

        R = global_rotations[image_id]
        center = global_centers[image_id]

        image = pycolmap.Image()
        image.image_id = image_id + 1
        image.camera_id = intrinsics_mapping[image_id] + 1

        # Add 2D points
        if image_id in keypoints_per_image:
            for xy in keypoints_per_image[image_id]:
                image.points2D.append(pycolmap.Point2D(xy))

        # Set image name from images_list if available, otherwise use image_id
        if images_list is not None and image_id < len(images_list):
            image.name = images_list[image_id]
        else:
            image.name = str(image_id)

        # Set pose: cam_from_world
        # t = -R @ center
        t = -R @ center
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    # Add 3D points. Track elements arrive with 0-indexed image_ids; rebuild
    # each track with image_id+1 so observations point to the 1-indexed images.
    for point3D in points3D.values():
        xyz = point3D.xyz.reshape(3, 1) if point3D.xyz.ndim == 1 else point3D.xyz
        new_track = pycolmap.Track()
        for elem in point3D.track.elements:
            new_track.add_element(elem.image_id + 1, elem.point2D_idx)
        reconstruction.add_point3D(xyz, new_track)

    return reconstruction
