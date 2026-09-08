"""Feed-forward model loading and inference helpers."""

import gc
import os
import sys
import time
from pathlib import Path

import torch

from utils.geometry import (
    depth_intrinsics_to_local_points,
    estimate_intrinsics_and_depth,
    remove_homogeneous_row,
    unproject_depth_map_to_point_map,
)

FEEDFORWARD_ROOT = Path(__file__).resolve().parents[2] / "feedforward"
VGGT_OMEGA_CKPT = str(
    Path(__file__).resolve().parents[2] / "checkpoints" / "vggt_omega_1b_512.pt"
)


def ensure_feedforward_on_path():
    if not FEEDFORWARD_ROOT.is_dir():
        raise FileNotFoundError(
            f"Feedforward model directory not found: {FEEDFORWARD_ROOT}"
        )

    feedforward_root_str = str(FEEDFORWARD_ROOT)
    if feedforward_root_str not in sys.path:
        sys.path.insert(0, feedforward_root_str)


def decode_vggt_omega_pose(pose_enc, image_size_hw):
    ensure_feedforward_on_path()
    from vggt_omega.utils.pose_enc import encoding_to_camera

    return encoding_to_camera(pose_enc, image_size_hw)


def load_model(model_name="pi3x", device=None):
    """Load and initialize a supported geometric foundation model."""
    ensure_feedforward_on_path()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if model_name == "pi3x":
        from pi3x_model.models.pi3x import Pi3X

        model = Pi3X.from_pretrained("yyfz233/Pi3X")
    elif model_name == "vggt_omega":
        from vggt_omega.models import VGGTOmega

        if not os.path.isfile(VGGT_OMEGA_CKPT):
            raise FileNotFoundError(
                f"VGGT-Omega checkpoint not found: {VGGT_OMEGA_CKPT}. "
                "Update VGGT_OMEGA_CKPT in utils/feedforward.py to point to your local weights."
            )
        print(f"Loading VGGT-Omega checkpoint: {VGGT_OMEGA_CKPT}")
        model = VGGTOmega()
        state_dict = torch.load(VGGT_OMEGA_CKPT, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError("Other model backbones are not implemented!")

    model.eval()
    model = model.to(device)
    return model, device


def run_inference_step_by_step(
    model, batches, size_hw, device, need_features=False, pi3x_intrinsics_method="lstsq"
):
    """
    Output:
     - extrinsic: (N, 3, 4)
     - intrinsic: (N, 3, 3)
     - depth: (N, H, W, 1)
     - depth_conf: (N, H, W)
    """
    ensure_feedforward_on_path()
    from pi3x_model.models.pi3x import Pi3X
    from pi3x_model.utils.transforms_utils import recover_intrinsics_from_output
    from vggt_omega.models import VGGTOmega

    predictions = []
    start = time.time()

    if isinstance(model, VGGTOmega):
        for i, images in enumerate(batches):
            prediction = dict()
            if images.shape[-2] % 16 != 0 or images.shape[-1] % 16 != 0:
                raise ValueError(
                    "VGGT-Omega requires image height and width to be multiples of 16. "
                    f"Got {tuple(images.shape[-2:])} for subset {i}."
                )

            with torch.no_grad():
                images = images.to(device)
                res = model(images)

            extri, intri = decode_vggt_omega_pose(
                res["pose_enc"],
                res["images"].shape[-2:],
            )

            prediction["depth"] = res["depth"].to(dtype=torch.float32, device="cpu")
            prediction["depth_conf"] = res["depth_conf"].to(
                dtype=torch.float32, device="cpu"
            )
            prediction["pose_enc"] = res["pose_enc"].to(
                dtype=torch.float32, device="cpu"
            )
            prediction["extrinsic"] = extri.to(
                dtype=torch.float32, device="cpu"
            ).squeeze(0)
            prediction["intrinsic"] = intri.to(
                dtype=torch.float32, device="cpu"
            ).squeeze(0)
            prediction["local_points"] = depth_intrinsics_to_local_points(
                prediction["depth"],
                prediction["intrinsic"],
            )
            prediction["world_points"] = unproject_depth_map_to_point_map(
                prediction["depth"].squeeze(0),
                prediction["extrinsic"],
                prediction["intrinsic"],
            )

            predictions.append(prediction)

            del res, images, extri, intri
            gc.collect()
            torch.cuda.empty_cache()

    elif isinstance(model, Pi3X):
        if pi3x_intrinsics_method not in ("lstsq", "moge"):
            raise ValueError(
                f"Unsupported pi3x_intrinsics_method: {pi3x_intrinsics_method}"
            )

        for i, images in enumerate(batches):
            prediction = dict()
            if images.shape[-2] % 14 != 0 or images.shape[-1] % 14 != 0:
                raise ValueError(
                    "Pi3X requires image height and width to be multiples of 14. "
                    f"Got {tuple(images.shape[-2:])} for subset {i}."
                )

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    images = images[None].to(device)
                    res = model(imgs=images)

            c2w = res["camera_poses"].squeeze(0)
            prediction["extrinsic"] = remove_homogeneous_row(torch.linalg.inv(c2w)).to(
                dtype=torch.float32, device="cpu"
            )
            prediction["world_points"] = (
                res["points"].to(dtype=torch.float32, device="cpu").squeeze(0)
            )
            prediction["depth_conf"] = (
                res["conf"].to(dtype=torch.float32, device="cpu").squeeze(0).squeeze(-1)
            )
            prediction["depth_conf"] = torch.sigmoid(prediction["depth_conf"])
            prediction["local_points"] = (
                res["local_points"].to(dtype=torch.float32, device="cpu").squeeze(0)
            )

            if pi3x_intrinsics_method == "moge":
                intrinsic = recover_intrinsics_from_output(res)
                prediction["intrinsic"] = torch.from_numpy(intrinsic).to(
                    dtype=torch.float32, device="cpu"
                )
                depth = res["local_points"].squeeze(0)[..., 2]
            else:
                intrinsic, depth = estimate_intrinsics_and_depth(
                    res["local_points"].squeeze(0)
                )
                prediction["intrinsic"] = intrinsic.to(
                    dtype=torch.float32, device="cpu"
                )

            prediction["depth"] = (
                depth.to(dtype=torch.float32, device="cpu").unsqueeze(0).unsqueeze(-1)
            )

            predictions.append(prediction)

            del res, images, c2w
            gc.collect()
            torch.cuda.empty_cache()

    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")

    end = time.time()
    print(f"[INFERENCE] Time used: {end - start}s. ")
    return predictions
