import numpy as np
import torch
from ffba.initialization.sim3 import weighted_align_point_maps
from utils.geometry import unproject_depth_map_to_point_map


def weighted_iterative_alignment(sequence, device="cuda"):
    transforms = {}  # Matrices transforming child frames into parent frames
    scales = {}
    for parent_id in sequence.edges:
        parent = sequence.edges[parent_id]
        for child_id in parent:
            edge = parent[child_id]
            transform = torch.zeros((3, 4)).to(device)

            parent_overlap_indices, child_overlap_indices = (
                edge["parent_overlap_indices"],
                edge["child_overlap_indices"],
            )
            prev_set = sequence.predictions[parent_id]
            curr_set = sequence.predictions[child_id]
            A_depths, A_intri, A_extri = (
                prev_set["depth"][parent_overlap_indices],
                prev_set["intrinsic"][parent_overlap_indices],
                prev_set["extrinsic"][parent_overlap_indices],
            )
            B_depths, B_intri, B_extri = (
                curr_set["depth"][child_overlap_indices],
                curr_set["intrinsic"][child_overlap_indices],
                curr_set["extrinsic"][child_overlap_indices],
            )
            A_depth_conf, B_depth_conf = (
                prev_set["depth_conf"][parent_overlap_indices, ..., None],
                curr_set["depth_conf"][child_overlap_indices, ..., None],
            )

            # Proper 0–1 normalization
            normalized_depth_confs_A = (A_depth_conf - np.min(A_depth_conf)) / (
                np.max(A_depth_conf) - np.min(A_depth_conf)
            )
            normalized_depth_confs_B = (B_depth_conf - np.min(B_depth_conf)) / (
                np.max(B_depth_conf) - np.min(B_depth_conf)
            )

            percentile = 70
            conf_threshold = min(
                np.percentile(normalized_depth_confs_A, percentile),
                np.percentile(normalized_depth_confs_B, percentile),
            )

            A = unproject_depth_map_to_point_map(
                A_depths, A_extri, A_intri
            )  # Target frame (y)
            B = unproject_depth_map_to_point_map(
                B_depths, B_extri, B_intri
            )  # Source frame (x)

            c, R, t = weighted_align_point_maps(
                A,
                normalized_depth_confs_A.squeeze(),
                B,
                normalized_depth_confs_B.squeeze(),
                conf_threshold=conf_threshold,
                align_method="numpy",
            )

            transform[:3, :3] = torch.from_numpy(R).float().to(device)
            transform[:3, 3] = torch.from_numpy(t).float().to(device)

            if parent_id not in transforms:
                transforms[parent_id] = {}
            if parent_id not in scales:
                scales[parent_id] = {}
            # Transform: child -> parent
            transforms[parent_id][child_id] = transform
            scales[parent_id][child_id] = c

    return sequence.transform_to_shared_frame(transforms, scales), transforms, scales


def align_extrinsics(sequence, ba=False, device="cuda", method="weighted_iterative"):
    if ba or method != "weighted_iterative":
        raise ValueError(
            "The formal pipeline uses weighted_iterative alignment without BA"
        )
    predictions, transforms, scales = weighted_iterative_alignment(
        sequence, device=device
    )
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            predictions[key] = value.cpu().numpy()
            if predictions[key].shape[0] == 1:
                predictions[key] = predictions[key].squeeze(0)
    return predictions, transforms, scales


def restore_predictions_order(predictions):
    """
    Sort the prediction based on image_ids to restore the order of all the attributes
    to the original order.
    """
    image_ids = predictions["image_ids"]

    for key in predictions.keys():
        if isinstance(predictions[key], np.ndarray):
            rearrangement = np.argsort(image_ids)
            if predictions[key].shape[0] == rearrangement.shape[0]:
                predictions[key] = predictions[key][rearrangement]
