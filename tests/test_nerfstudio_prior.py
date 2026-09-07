import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.nerfstudio_prior import (  # noqa: E402
    convert_nerfstudio_c2w_to_opencv_w2c,
    load_nerfstudio_prior,
)


def test_nerfstudio_identity_pose_converts_to_opencv_w2c():
    actual = convert_nerfstudio_c2w_to_opencv_w2c(np.eye(4))

    np.testing.assert_allclose(
        actual,
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
            ]
        ),
    )


def test_prior_loader_uses_json_order_and_shared_centered_intrinsics(tmp_path):
    images_dir = tmp_path / "image"
    images_dir.mkdir()
    for index, value in enumerate((32, 96, 160), start=1):
        Image.new("RGB", (8, 6), color=(value, 0, 0)).save(
            images_dir / f"frame_{index}.png"
        )

    order = (3, 1, 2)
    focals = (10.0, 14.0, 18.0)
    frames = []
    for frame_index, (image_number, focal) in enumerate(zip(order, focals)):
        transform = np.eye(4)
        transform[0, 3] = float(frame_index + 1)
        frames.append(
            {
                "file_path": f"./image/frame_{image_number}.png",
                "w": 8,
                "h": 6,
                "fl_x": focal,
                "fl_y": focal,
                "cx": 100.0,
                "cy": 100.0,
                "transform_matrix": transform.tolist(),
            }
        )
    transforms_path = tmp_path / "transforms.json"
    transforms_path.write_text(json.dumps({"camera_model": "OPENCV", "frames": frames}))

    prior = load_nerfstudio_prior(
        transforms_path,
        images_dir,
        num_images=3,
        subsample=2,
        num_workers=1,
        retrieval_long_side=8,
        retrieval_patch_multiple=2,
    )

    assert [Path(name).name for name in prior.image_names] == [
        "frame_3.png",
        "frame_2.png",
    ]
    assert prior.images.shape == (2, 3, 6, 8)
    assert prior.retrieval_images.shape == (2, 3, 6, 8)
    assert prior.image_size_hw == (6, 8)
    np.testing.assert_allclose(prior.intrinsic[:, 0, 0], [14.0, 14.0])
    np.testing.assert_allclose(prior.intrinsic[:, 1, 1], [14.0, 14.0])
    np.testing.assert_allclose(prior.intrinsic[:, 0, 2], [4.0, 4.0])
    np.testing.assert_allclose(prior.intrinsic[:, 1, 2], [3.0, 3.0])
    assert prior.audit["selected_original_indices"] == [0, 2]
    assert prior.audit["image_order"] == "transforms_json_frames"

    rotations = prior.extrinsic[:, :3, :3].astype(np.float64)
    translations = prior.extrinsic[:, :3, 3].astype(np.float64)
    centers = np.einsum("nij,nj->ni", -np.transpose(rotations, (0, 2, 1)), translations)
    np.testing.assert_allclose(centers[:, 0], [1.0, 3.0])


def test_prior_loader_rejects_images_outside_dataset(tmp_path):
    images_dir = tmp_path / "image"
    images_dir.mkdir()
    outside = tmp_path / "outside.png"
    Image.new("RGB", (8, 6)).save(outside)
    transforms_path = tmp_path / "transforms.json"
    transforms_path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "file_path": "./outside.png",
                        "w": 8,
                        "h": 6,
                        "fl_x": 10.0,
                        "fl_y": 10.0,
                        "transform_matrix": np.eye(4).tolist(),
                    }
                ]
            }
        )
    )

    try:
        load_nerfstudio_prior(transforms_path, images_dir, num_workers=1)
    except ValueError as exc:
        assert "outside --dataset" in str(exc)
    else:
        raise AssertionError("Expected an out-of-dataset frame to be rejected")
