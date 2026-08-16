import torch


def intrinsics_averaging(
    intrinsics_all: list[torch.Tensor],
    communities: list[list[int]],
    intrinsics_mapping: dict[int, int],
    camera_model: str = "PINHOLE",
) -> list[torch.Tensor | None]:
    """
    Median-average per-camera intrinsics matrices grouped by community.

    Args:
        intrinsics_all: One tensor per cluster of shape (B, N, 3, 3) holding
            per-image intrinsics matrices for the images in that cluster.
        communities: For each cluster, the list of image indices whose
            intrinsics live at the matching slot of ``intrinsics_all``.
        intrinsics_mapping: Maps image index to a camera bucket index. All
            images sharing a camera average their intrinsics together.
        camera_model: COLMAP-style camera model name. When it starts with
            ``"SIMPLE"``, fx and fy are forced equal before averaging.

    Returns:
        List indexed by camera bucket; each entry is the median intrinsics
        tensor for that bucket, or ``None`` if no observations existed.
    """
    max_idx = max(intrinsics_mapping.values())
    intrinsics = [[] for _ in range(max_idx + 1)]

    for idx_cluster, community in enumerate(communities):
        for i, idx in enumerate(community):
            intrinsics[intrinsics_mapping[idx]].append(
                intrinsics_all[idx_cluster][:, i]
            )

    # Now, we need to average the intrinsics for each type of camera
    for i in range(len(intrinsics)):
        if len(intrinsics[i]) == 0:
            intrinsics[i] = None
            continue

        intrinsics_curr = torch.stack(intrinsics[i])
        if camera_model.startswith("SIMPLE"):
            focals = (
                intrinsics_curr[:, :, 0, 0] + intrinsics_curr[:, :, 1, 1]
            ) / 2
            intrinsics_curr[:, :, 0, 0] = focals
            intrinsics_curr[:, :, 1, 1] = focals

        intrinsics[i] = torch.median(intrinsics_curr, dim=0).values

    return intrinsics
