from collections.abc import Iterable

import numpy as np
import torch


def rescale_intrinsics(
    global_intrinsics: list[torch.Tensor | None],
    intrinsics_scales: list[tuple[float, float, float, float]],
    inverse: bool = False,
) -> list[torch.Tensor | None]:
    """Rescale intrinsics in place between resized and original image space.

    Args:
        global_intrinsics: List of intrinsic matrices, each shape ``(1, 3, 3)``
            (or ``None`` for missing entries). Modified in place.
        intrinsics_scales: Per-image ``(x_scale, y_scale, x_offset, y_offset)``
            describing the resize/crop applied to the source image.
        inverse: If ``False``, undo the resize (rescaled -> original). If
            ``True``, apply the resize (original -> rescaled).

    Returns:
        The same ``global_intrinsics`` list, mutated in place.
    """
    for i in range(len(global_intrinsics)):
        if global_intrinsics[i] is None:
            continue
        x_scale, y_scale, x_offset, y_offset = intrinsics_scales[i]
        if not inverse:
            global_intrinsics[i][0, 0, 2] -= x_offset
            global_intrinsics[i][0, 1, 2] -= y_offset
            global_intrinsics[i][0, 0] /= x_scale
            global_intrinsics[i][0, 1] /= y_scale
        else:
            global_intrinsics[i][0, 0] *= x_scale
            global_intrinsics[i][0, 1] *= y_scale
            global_intrinsics[i][0, 0, 2] += x_offset
            global_intrinsics[i][0, 1, 2] += y_offset

    return global_intrinsics


def keep_inframes(
    predictions_dict: dict,
    image_shape_ori: dict[int, tuple[int, int]],
    indexes: Iterable[int] | None = None,
) -> None:
    """Zero-out visibility for tracks that fall outside the original image.

    Reads ``predictions_dict["indexes"]`` and
    ``predictions_dict["tracks"]`` and mutates
    ``predictions_dict["vis"]`` in place: any track whose pixel coordinate lies
    outside ``[0, W) x [0, H)`` for its source image gets its visibility flag
    multiplied by zero.

    Args:
        predictions_dict: Dict containing ``"indexes"``, ``"tracks"``, and
            ``"vis"``. ``"vis"`` is mutated in place.
        image_shape_ori: Mapping from image index to ``(H, W)``.
        indexes: Iterable of star indexes to process. ``None`` means all.
    """
    if indexes is None:
        indexes = range(len(predictions_dict["indexes"]))

    for idx in indexes:
        index = predictions_dict["indexes"][idx]
        for i, idx_inner in enumerate(index):
            is_valid_x = (predictions_dict["tracks"][idx][0, i, :, 0] >= 0) * (
                predictions_dict["tracks"][idx][0, i, :, 0]
                < image_shape_ori[idx_inner][1]
            )
            is_valid_y = (predictions_dict["tracks"][idx][0, i, :, 1] >= 0) * (
                predictions_dict["tracks"][idx][0, i, :, 1]
                < image_shape_ori[idx_inner][0]
            )

            predictions_dict["vis"][idx][0, i] = (
                predictions_dict["vis"][idx][0, i] * is_valid_x * is_valid_y
            )


# Change from original images to rescaled images
def standardize_query_points(
    query_points: torch.Tensor,
    image_change: tuple[float, float, float, float] | np.ndarray,
) -> torch.Tensor:
    """Map query points from original to rescaled image coordinates.

    Args:
        query_points: ``(K, 2)`` or ``(1, K, 2)``. Mutated in place.
        image_change: ``(x_scale, y_scale, x_shift, y_shift)`` as a 4-tuple
            or a length-4 ``np.ndarray``.

    Returns:
        The same ``query_points`` tensor, mutated in place.
    """
    if query_points.dim() == 3:
        assert query_points.shape[0] == 1

    query_points[..., 0] = (
        query_points[..., 0] * image_change[0] + image_change[2]
    )
    query_points[..., 1] = (
        query_points[..., 1] * image_change[1] + image_change[3]
    )

    return query_points


# Change from rescaled images to the original images
def rescale_tracks_single(
    tracks: torch.Tensor,
    image_change: tuple[float, float, float, float] | np.ndarray,
) -> torch.Tensor:
    """Map tracks from rescaled back to original image coordinates.

    Args:
        tracks: ``(K, 2)`` or ``(1, K, 2)``. Mutated in place.
        image_change: ``(x_scale, y_scale, x_shift, y_shift)`` (4-tuple or
            length-4 ``np.ndarray``) describing the rescale that produced
            ``tracks``; this function inverts it.

    Returns:
        The same ``tracks`` tensor, mutated in place.
    """
    if tracks.dim() == 3:
        assert tracks.shape[0] == 1

    tracks[..., 0] = (tracks[..., 0] - image_change[2]) / image_change[0]
    tracks[..., 1] = (tracks[..., 1] - image_change[3]) / image_change[1]

    return tracks


def rescale_tracks(
    predictions_dict: dict,
    image_changes_full: dict[
        int, tuple[float, float, float, float] | np.ndarray
    ],
    indexes: Iterable[int] | None = None,
    reverse: bool = False,
) -> dict:
    """Rescale tracks (and ``"tracks_virtual"`` if present) in
    ``predictions_dict``.

    Args:
        predictions_dict: Dict with ``"indexes"``, ``"tracks"``, and optionally
            ``"tracks_virtual"``. The track tensors are mutated in place.
        image_changes_full: Per-image ``(x_scale, y_scale, x_shift, y_shift)``
            either as a 4-tuple or a length-4 ``np.ndarray``.
        indexes: Iterable of star indexes to process. ``None`` means all.
        reverse: If ``False``, rescaled -> original. If ``True``, original ->
            rescaled.

    Returns:
        The same ``predictions_dict``, mutated in place.
    """
    if indexes is None:
        range(len(predictions_dict["indexes"]))

    # image_changes_full: (x_scale, y_scale, x_shift, y_shift)
    for idx in indexes:
        index = predictions_dict["indexes"][idx]
        for i, idx_inner in enumerate(index):
            # rescale the tracks
            if not reverse:
                predictions_dict["tracks"][idx][0, i] = rescale_tracks_single(
                    predictions_dict["tracks"][idx][0, i],
                    image_changes_full[idx_inner],
                )
            else:
                predictions_dict["tracks"][idx][0, i] = (
                    standardize_query_points(
                        predictions_dict["tracks"][idx][0, i],
                        image_changes_full[idx_inner],
                    )
                )

    if "tracks_virtual" in predictions_dict:
        for idx in indexes:
            index = predictions_dict["indexes"][idx]
            for i, idx_inner in enumerate(index):
                # rescale the tracks
                if not reverse:
                    predictions_dict["tracks_virtual"][idx][0, i] = (
                        rescale_tracks_single(
                            predictions_dict["tracks_virtual"][idx][0, i],
                            image_changes_full[idx_inner],
                        )
                    )
                else:
                    predictions_dict["tracks_virtual"][idx][0, i] = (
                        standardize_query_points(
                            predictions_dict["tracks_virtual"][idx][0, i],
                            image_changes_full[idx_inner],
                        )
                    )

    return predictions_dict
