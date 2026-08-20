"""Tests for ``intrinsics_mode`` in ``DemoBaseDataset._preload_images``.

The dataset's ``_preload_images`` method builds ``self.intrinsics_mapping``
(image-index -> camera-bucket-id) using ``args.intrinsics_mode``:
``SHARED``, ``PER_FOLDER``, or ``PER_CAMERA``. These tests pin down the
expected mappings without touching disk: ``imagesize.get`` is mocked, and
the image list is sized above the in-memory preload threshold so no tensors
are loaded.
"""

import argparse
from unittest.mock import patch

from gluemap.datasets.base import DemoBaseDataset


def _make_dataset(images_list, images_shape_by_name, mode):
    """Construct a ``DemoBaseDataset`` and run its preload step.

    ``images_shape_by_name`` maps a path tail (the value in ``images_list``)
    to the ``(width, height)`` tuple ``imagesize.get`` would return. The
    list is padded to >=50 entries so the eager-preload branch is skipped.
    """
    pad_shape = (640, 480)
    pad_needed = max(0, 50 - len(images_list))
    padded_list = list(images_list) + [
        f"_pad/{i}.jpg" for i in range(pad_needed)
    ]
    shape_lookup = dict(images_shape_by_name)
    for i in range(pad_needed):
        shape_lookup[f"_pad/{i}.jpg"] = pad_shape

    root = "/fake/root"
    prefix = root + "/"

    def fake_imagesize_get(path):
        return shape_lookup[path[len(prefix) :]]

    args = argparse.Namespace(
        images_path=root,
        intrinsics_mode=mode,
    )
    ds = DemoBaseDataset(args)
    ds.images_list = padded_list
    ds.images_path = [args.images_path for _ in padded_list]

    with patch(
        "gluemap.datasets.base.imagesize.get", side_effect=fake_imagesize_get
    ):
        ds._preload_images(args)
    return ds


def test_shared_collapses_same_shape_across_folders():
    images = ["seqA/a.jpg", "seqA/b.jpg", "seqB/c.jpg"]
    shapes = {name: (640, 480) for name in images}

    ds = _make_dataset(images, shapes, mode="SHARED")

    # All three share the same (h, w); all map to bucket 0.
    assert ds.intrinsics_mapping[0] == 0
    assert ds.intrinsics_mapping[1] == 0
    assert ds.intrinsics_mapping[2] == 0


def test_per_camera_assigns_unique_bucket_per_image():
    images = ["seqA/a.jpg", "seqA/b.jpg", "seqB/c.jpg"]
    shapes = {name: (640, 480) for name in images}

    ds = _make_dataset(images, shapes, mode="PER_CAMERA")

    # Bucketing covers all (real + padding) entries; check real ones.
    assert ds.intrinsics_mapping[0] == 0
    assert ds.intrinsics_mapping[1] == 1
    assert ds.intrinsics_mapping[2] == 2
    # And matches identity over the whole list.
    n = len(ds.images_list)
    assert ds.intrinsics_mapping == {i: i for i in range(n)}


def test_per_folder_distinguishes_same_shape_across_folders():
    # seqA/{a,b} and seqB/{c,d} all share shape (640, 480). PER_FOLDER must
    # give seqA and seqB different bucket IDs.
    images = ["seqA/a.jpg", "seqA/b.jpg", "seqB/c.jpg", "seqB/d.jpg"]
    shapes = {name: (640, 480) for name in images}

    ds = _make_dataset(images, shapes, mode="PER_FOLDER")

    assert ds.intrinsics_mapping[0] == ds.intrinsics_mapping[1]
    assert ds.intrinsics_mapping[2] == ds.intrinsics_mapping[3]
    assert ds.intrinsics_mapping[0] != ds.intrinsics_mapping[2]


def test_per_folder_distinguishes_shapes_within_folder():
    # Same folder, two different (h, w) shapes -> two buckets.
    images = ["seqA/a.jpg", "seqA/b.jpg", "seqA/c.jpg"]
    shapes = {
        "seqA/a.jpg": (640, 480),
        "seqA/b.jpg": (640, 480),
        "seqA/c.jpg": (800, 600),
    }

    ds = _make_dataset(images, shapes, mode="PER_FOLDER")

    assert ds.intrinsics_mapping[0] == ds.intrinsics_mapping[1]
    assert ds.intrinsics_mapping[0] != ds.intrinsics_mapping[2]


def test_per_folder_bucket_ids_are_dense():
    # Downstream code (intrinsics_averaging, colmap writers) assumes
    # bucket IDs are a contiguous 0..K-1 range.
    images = [
        "seqA/a.jpg",  # (640, 480)
        "seqB/b.jpg",  # (640, 480)
        "seqA/c.jpg",  # (800, 600)
        "seqB/d.jpg",  # (800, 600)
    ]
    shapes = {
        "seqA/a.jpg": (640, 480),
        "seqB/b.jpg": (640, 480),
        "seqA/c.jpg": (800, 600),
        "seqB/d.jpg": (800, 600),
    }

    ds = _make_dataset(images, shapes, mode="PER_FOLDER")

    values = set(ds.intrinsics_mapping[i] for i in range(len(images)))
    # 4 distinct (folder, shape) pairs -> 4 dense IDs.
    assert values == {0, 1, 2, 3}


def test_per_folder_deterministic_ordering():
    images = ["seqA/a.jpg", "seqB/b.jpg", "seqA/c.jpg"]
    shapes = {name: (640, 480) for name in images}

    ds1 = _make_dataset(images, shapes, mode="PER_FOLDER")
    ds2 = _make_dataset(images, shapes, mode="PER_FOLDER")

    assert ds1.intrinsics_mapping == ds2.intrinsics_mapping
