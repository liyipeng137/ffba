import argparse
import logging
import os

import imagesize
import numpy as np
import torch
from tqdm import tqdm

from gluemap.utils.load_fn import (
    load_and_preprocess_images,
    load_and_preprocess_images_1024,
    load_and_preprocess_images_inner,
)

logger = logging.getLogger(__name__)


class DemoBaseDataset:
    """Base class shared by the two-view and star datasets.

    Holds the common preprocessing settings (target image size, patch
    alignment, camera model) and implements lazy/eager image loading
    along with the ``_preload_images`` step that decides whether the
    full set fits in memory.
    """

    def __init__(self, args: argparse.Namespace, patch_size: int = 16) -> None:
        """Initialize size-related settings from ``args`` and ``patch_size``.

        Args:
            args: Parsed CLI/config namespace. Reads optional
                ``camera_model``; defaults to ``SIMPLE_PINHOLE``.
            patch_size: Patch side length the network expects. ``14``
                bumps ``image_size`` to ``518`` (DINO/Pi3-style models);
                anything else sets ``image_size`` to ``512``.
        """
        self.image_size = 512
        self.patch_size = patch_size
        if patch_size == 14:
            self.image_size = 518
        else:
            self.image_size = 512

        self.camera_model = (
            args.camera_model
            if hasattr(args, "camera_model")
            else "SIMPLE_PINHOLE"
        )

        # list of (i, j) tuples, i < j, for immediate sequential neighbors
        self.sequential_edges: list[tuple[int, int]] = []

    def load_images(
        self,
        indexes: np.ndarray | list[int],
        load_1024: bool = False,
    ) -> (
        tuple[torch.Tensor, list[torch.Tensor], list[list[float]], list[str]]
        | tuple[
            torch.Tensor,
            list[torch.Tensor],
            list[list[float]],
            torch.Tensor,
            list[list[float]],
            list[str],
        ]
    ):
        """Load (or fetch from cache) a subset of images by index.

        When ``_preload_images`` populated the in-memory caches
        (``self.images``, ``self.images_ori``, ...), the requested
        subset is gathered from those tensors. Otherwise the images
        are read and preprocessed on the fly via
        :func:`gluemap.utils.load_fn.load_and_preprocess_images` (or
        :func:`load_and_preprocess_images_inner` when only the
        ``images_ori`` cache is populated).

        Args:
            indexes: Iterable of integer indices into ``self.images_list``.
            load_1024: When ``True``, additionally produce the 1024x1024
                variant used by the VGGSfM tracker.

        Returns:
            Without ``load_1024`` (4-tuple): ``(images, images_ori,
            images_change, image_paths)``. With ``load_1024`` (6-tuple):
            ``(images, images_ori, images_change, images_1024,
            images_change_1024, image_paths)``. See
            :func:`gluemap.utils.load_fn.load_and_preprocess_images_inner`
            for the ``images_change`` schema.
        """
        image_paths = [
            os.path.join(self.images_path[i], self.images_list[i])
            for i in indexes
        ]

        # for the case with few images, load images beforehand is more efficient
        if hasattr(self, "has_preloaded"):
            images = torch.stack(
                [self.images[i] for i in indexes], dim=0
            ).float()
            images_ori = [self.images_ori[i] for i in indexes]
            images_change = [self.images_change[i] for i in indexes]

            if load_1024:
                images_1024 = torch.stack(
                    [self.images_1024[i] for i in indexes], dim=0
                ).float()
                images_change_1024 = [
                    self.images_change_1024[i] for i in indexes
                ]

        else:
            if not hasattr(self, "images_ori"):
                images, images_ori, images_change = load_and_preprocess_images(
                    image_paths,
                    image_size=self.image_size,
                    patch_size=self.patch_size,
                    force_square=self.force_square,
                )
            else:
                images, images_ori, images_change = (
                    load_and_preprocess_images_inner(
                        [self.images_ori[i] for i in indexes],
                        image_size=self.image_size,
                        patch_size=self.patch_size,
                        force_square=self.force_square,
                    )
                )

            if load_1024:
                images_1024, images_change_1024 = (
                    load_and_preprocess_images_1024(images_ori)
                )

        if not load_1024:
            return images, images_ori, images_change, image_paths
        else:
            return (
                images,
                images_ori,
                images_change,
                images_1024,
                images_change_1024,
                image_paths,
            )

    # -----------------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------------
    def _preload_images(self, args: argparse.Namespace) -> None:
        """Probe image shapes and, when feasible, eagerly preload all images.

        Reads each image's ``(height, width)`` via ``imagesize``,
        decides whether to force-square the canvas based on shape
        homogeneity, and builds ``self.intrinsics_mapping`` according
        to ``args.intrinsics_mode``: ``SHARED`` (one camera per unique
        shape), ``PER_FOLDER`` (one camera per ``(dirname, shape)``
        pair), or ``PER_CAMERA`` (one camera per image). When fewer
        than 200 images are present, the full preprocessed tensors are
        loaded into ``self.images``, ``self.images_ori``,
        ``self.images_change``, ``self.images_1024``, and
        ``self.images_change_1024`` and a ``has_preloaded`` flag is set
        so :meth:`load_images` can serve from cache.

        Args:
            args: Parsed CLI/config namespace. Reads ``images_path`` and
                optional ``intrinsics_mode`` (default ``SHARED``).
        """
        # get the image size of all the images
        images_shape_ori = []
        for img_path in tqdm(self.images_list):
            img_size = imagesize.get(os.path.join(args.images_path, img_path))
            images_shape_ori.append(
                img_size[::-1]
            )  # (width, height) -> (height, width)

        self.images_shape_ori = images_shape_ori

        # Check whether all images have the same size
        # If not, force all images to be square
        unique_shapes = set(images_shape_ori)
        if (
            len(unique_shapes) == 1
            and images_shape_ori[0][0] < images_shape_ori[0][1]
        ):
            self.force_square = False
        else:
            self.force_square = True

        self.intrinsics_mapping = {}
        mode = getattr(args, "intrinsics_mode", "SHARED")

        if mode == "SHARED":
            shape_to_intrinsics_idx = {}
            intrinsics_idx = 0
            for i, shape in enumerate(self.images_shape_ori):
                if shape not in shape_to_intrinsics_idx:
                    shape_to_intrinsics_idx[shape] = intrinsics_idx
                    intrinsics_idx += 1
                self.intrinsics_mapping[i] = shape_to_intrinsics_idx[shape]
        elif mode == "PER_FOLDER":
            key_to_intrinsics_idx: dict[tuple[str, tuple[int, int]], int] = {}
            intrinsics_idx = 0
            for i, shape in enumerate(self.images_shape_ori):
                folder = os.path.dirname(self.images_list[i])
                key = (folder, shape)
                if key not in key_to_intrinsics_idx:
                    key_to_intrinsics_idx[key] = intrinsics_idx
                    intrinsics_idx += 1
                self.intrinsics_mapping[i] = key_to_intrinsics_idx[key]
        elif mode == "PER_CAMERA":
            self.intrinsics_mapping = {
                i: i for i in range(len(self.images_list))
            }
        else:
            raise ValueError(f"Unknown intrinsics_mode: {mode!r}")

        # if the number of images is fewer than 50, directly load the images
        if len(self.images_list) < 50:
            self.has_preloaded = True
            self.images, self.images_ori, self.images_change = (
                load_and_preprocess_images(
                    [
                        os.path.join(self.images_path[i], self.images_list[i])
                        for i in range(len(self.images_list))
                    ],
                    image_size=self.image_size,
                    patch_size=self.patch_size,
                    force_square=self.force_square,
                )
            )
            self.images_1024, self.images_change_1024 = (
                load_and_preprocess_images_1024(self.images_ori)
            )
        else:
            logger.warning("images are not preloaded!")

        logger.info("preload image done")
