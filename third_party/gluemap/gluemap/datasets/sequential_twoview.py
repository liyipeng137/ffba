import argparse
import os

import torch

from gluemap.datasets.twoview import BaseTwoViewDataset
from gluemap.datasets.utils import (
    establish_neighbors_sequential,
    retrieve_global_neighbors,
)


class SequentialTwoViewDataset(BaseTwoViewDataset):
    """Two-view dataset for sequentially captured images (e.g. video).

    Differs from :class:`BaseTwoViewDataset` in that pairs combine a
    fixed sliding-window neighborhood with FAISS-retrieved global
    neighbors, and the image list can be subsampled in time.
    """

    # -----------------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------------
    def _get_image_list(
        self, args: argparse.Namespace
    ) -> tuple[list[str], list[str]]:
        """Like the parent, but additionally subsamples the used list.

        Args:
            args: Parsed CLI/config namespace. Reads ``sample_frequency``
                in addition to the parent's keys; values ``> 1`` keep
                every Nth image.

        Returns:
            ``(img_list_full, img_list_used)`` — the full discovery
            order and the (possibly subsampled) subset used downstream.
        """
        img_list_full, img_list_used = super()._get_image_list(args)
        if args.sample_frequency > 1:
            img_list_used = img_list_used[:: args.sample_frequency]

        return img_list_full, img_list_used

    def _construct_pairs(
        self, args: argparse.Namespace, img_list: list[str]
    ) -> None:
        """Populate ``self.pairs`` and ``self.sequential_edges``.

        Loads cached SALAD descriptors (required — no fallback), builds
        a sequential window via :func:`establish_neighbors_sequential`,
        merges with FAISS global retrieval via
        :func:`retrieve_global_neighbors`, and records the immediate
        sequential neighbor pairs in ``self.sequential_edges``.

        Args:
            args: Parsed CLI/config namespace. Reads ``curr_processed``,
                ``num_neighbors_sequential``, ``num_neighbors``, and
                ``images_path``.
            img_list: Full list of image paths discovered during
                preprocessing — used only by the parent contract; the
                sequential descriptor lookup keys off ``self.images_list``.

        Raises:
            FileNotFoundError: If ``salad_descriptors.pt`` is missing.
        """
        descriptors_path = os.path.join(
            args.curr_processed, "salad_descriptors.pt"
        )

        if os.path.exists(descriptors_path):
            descriptors_db = torch.load(descriptors_path, weights_only=False)
            # find the corresponding images and only keep the short list
            image_indexes = [
                img_list.index(os.path.join(args.images_path, img))
                for img in self.images_list
            ]
            descriptors_db = descriptors_db[image_indexes]

        else:
            raise FileNotFoundError(
                f"SALAD descriptors not found at {descriptors_path}. "
                "Please run SALAD descriptor extraction first."
            )

        neighbors = establish_neighbors_sequential(
            self.images_list, num_neighbors=args.num_neighbors_sequential
        )  # N, num_neighbors + 1

        # retrieve global neighbors based on SALAD descriptors
        self.pairs = retrieve_global_neighbors(
            args, [neighbors], [descriptors_db]
        )

        # All sequential neighbors within the window
        seq_set = set()
        for k in range(neighbors.shape[0]):
            center = int(neighbors[k, 0])
            for m in range(1, neighbors.shape[1]):
                nbr = int(neighbors[k, m])
                seq_set.add((min(center, nbr), max(center, nbr)))
        self.sequential_edges = list(seq_set)
