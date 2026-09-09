"""geometry for the formal SIFT + prior + BAE pipeline."""

import numpy as np


def camera_centers_from_w2c(extrinsic):
    rotations = extrinsic[:, :3, :3]
    translations = extrinsic[:, :3, 3]
    return np.einsum(
        "nij,nj->ni",
        -np.transpose(rotations, (0, 2, 1)),
        translations,
    )


def camera_viewing_axes_from_w2c(extrinsic):
    rotations_c2w = np.transpose(np.asarray(extrinsic)[:, :3, :3], (0, 2, 1))
    return rotations_c2w[:, :, -1]


def to_homogeneous_w2c(extrinsic):
    extrinsic = np.asarray(extrinsic)
    if extrinsic.shape[-2:] == (4, 4):
        return extrinsic
    if extrinsic.shape[-2:] != (3, 4):
        raise ValueError(
            f"Expected w2c extrinsic shape (...,3,4), got {extrinsic.shape}"
        )
    bottom_shape = extrinsic.shape[:-2] + (1, 4)
    bottom = np.zeros(bottom_shape, dtype=extrinsic.dtype)
    bottom[..., 0, 3] = 1.0
    return np.concatenate([extrinsic, bottom], axis=-2)


def center_local_extrinsics(extrinsic, group):
    w2c = to_homogeneous_w2c(extrinsic[group])
    center_inv = np.linalg.inv(w2c[0])
    local = np.einsum("nij,jk->nik", w2c, center_inv)
    return local[:, :3, :4]


def global_pose_dicts_from_w2c(extrinsic):
    rotations = {}
    centers = {}
    for idx in range(extrinsic.shape[0]):
        rotations[idx] = np.asarray(extrinsic[idx, :3, :3], dtype=np.float64)
        centers[idx] = np.asarray(
            -rotations[idx].T @ extrinsic[idx, :3, 3], dtype=np.float64
        )
    return rotations, centers
