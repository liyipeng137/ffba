import logging
from pathlib import Path

import faiss
import numpy as np
import pycolmap
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrackSnapping:
    def __init__(self, snapping_thres: float = 1.0) -> None:
        self.snapping_thres = snapping_thres

    def main(
        self,
        database_dir: str | Path,
        predictions_dict: dict,
        images_shape_ori: list[tuple[int, int]],
        images_list: list[str],
    ) -> None:
        """
        Snap predicted tracks to the nearest keypoints in the COLMAP database.

        Loads keypoints from the COLMAP database at ``database_dir``, scales the
        per-image snapping threshold to original image resolution, and rewrites
        ``predictions_dict`` in-place via :meth:`_snap_tracks`.

        Args:
            database_dir: Path to the COLMAP database holding image keypoints.
            predictions_dict: Tracks dictionary produced by upstream inference;
                modified in place.
            images_shape_ori: Per-image (H, W) of the original images, used to
                rescale ``self.snapping_thres`` from the 1024-px tracker space.
            images_list: Image filenames in the same order as
                ``images_shape_ori``; resolved against database image names.
        """
        database = pycolmap.Database.open(database_dir)
        images = database.read_all_images()

        image_name_to_idx_db = {
            images[idx].name: idx for idx in range(len(images))
        }
        image_name_to_standard_path = [str(Path(x)) for x in images_list]

        query_points_full = []
        for idx in range(len(image_name_to_standard_path)):
            image_name = image_name_to_standard_path[idx]
            image_id = images[image_name_to_idx_db[image_name]].image_id
            query_keypoints = database.read_keypoints(image_id)
            query_points_full.append(
                torch.from_numpy(query_keypoints)[:, :2].float().unsqueeze(0)
            )

        logger.info("Finish loading keypoints from the database")

        snapping_thres_dict = {}
        for image_idx in range(len(images_shape_ori)):
            scaling_factor = (
                images_shape_ori[image_idx][0] + images_shape_ori[image_idx][1]
            ) / 1024
            snapping_thres_dict[image_idx] = (
                self.snapping_thres * scaling_factor
            )

        logger.info("Start track snapping")
        self._snap_tracks(
            predictions_dict, query_points_full, snapping_thres_dict
        )

    def _snap_tracks(
        self,
        predictions_dict: dict,
        query_points_full: list[torch.Tensor],
        snapping_thres: float | dict[int, float],
    ) -> None:
        """
        Snap each predicted track to its nearest database keypoint per image.

        Builds an FAISS L2 index per image, finds the closest database keypoint
        to every track observation, and overwrites positions whose squared
        distance is below the per-image threshold. Observations that are not
        snapped have their visibility score halved. Mutates ``predictions_dict``
        in-place by adding the ``"scores"`` and ``"is_close"`` entries.

        Args:
            predictions_dict: Track dictionary with ``"indexes"``,
                ``"tracks"``, and ``"vis"`` entries; updated in place.
            query_points_full: Per-image keypoint tensors of shape (1, K, 2).
            snapping_thres: Either a single threshold applied to every image or
                a per-image-index mapping (already in pixel units, pre-square).
        """
        predictions_dict["scores"] = {}
        for idx in range(len(predictions_dict["indexes"])):
            predictions_dict["scores"][idx] = torch.where(
                predictions_dict["vis"][idx] > 0.05,
                predictions_dict["vis"][idx],
                0.0,
            )

        N = len(query_points_full)
        snapping_thres_dict = {}
        if isinstance(snapping_thres, float):
            for i in range(N):
                snapping_thres_dict[i] = snapping_thres
        elif not isinstance(snapping_thres, dict):
            raise ValueError(
                "Snapping threshold should be a float or a dictionary"
            )
        else:
            snapping_thres_dict = snapping_thres

        indexes_all = []
        for idx in range(N):
            index = faiss.IndexFlatL2(2)
            index.add(query_points_full[idx][0])
            indexes_all.append(index)

        counter = 0
        for key in snapping_thres_dict:
            snapping_thres_dict[key] = snapping_thres_dict[key] ** 2

        indexes = range(len(predictions_dict["indexes"]))
        counter_valid = 0
        predictions_dict["is_close"] = {}
        logger.info("Indexing and snapping points...")
        for idx in tqdm(indexes):
            is_close_all = []
            for i, idx_inner in enumerate(predictions_dict["indexes"][idx]):
                distances, indices = indexes_all[idx_inner].search(
                    predictions_dict["tracks"][idx][0, i], 1
                )
                if query_points_full[idx_inner].shape == (1, 0, 2):
                    is_close = np.zeros(
                        (predictions_dict["tracks"][idx][0, i].shape[0],),
                        dtype=bool,
                    )
                else:
                    is_close = (
                        snapping_thres_dict[idx_inner] > distances
                    ).reshape(-1)
                    counter += (
                        is_close
                        * (predictions_dict["scores"][idx][0, i] > 0)
                        .cpu()
                        .numpy()
                    ).sum()

                predictions_dict["tracks"][idx][0, i, is_close] = (
                    query_points_full[idx_inner][0, indices[is_close, 0]]
                )

                counter_valid += (
                    predictions_dict["scores"][idx][0, i] > 0
                ).sum()

                predictions_dict["scores"][idx][0, i, ~is_close] /= 2

                is_close_all.append(is_close)

            predictions_dict["is_close"][idx] = torch.from_numpy(
                np.stack(is_close_all, axis=0)
            ).unsqueeze(0)

        logger.info(f"Total points snapped {counter} / {counter_valid}")
