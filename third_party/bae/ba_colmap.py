import argparse
import os
from time import perf_counter

import pypose as pp
import torch
import torch.nn as nn
from pypose.autograd.function import psjac

from bae.optim import LM
from bae.utils.pysolvers import PCG
from datapipes.colmap_loader import read_colmap_data, save_colmap_cameras, save_colmap_result


@psjac
def project_colmap(points, camera_params, intrinsics):
    """Project COLMAP world points with shared PINHOLE intrinsics."""
    points_proj = pp.SE3(camera_params[..., :7]).Act(points)
    points_proj = points_proj[..., :2] / points_proj[..., 2].unsqueeze(-1)

    fx = intrinsics[..., 0].unsqueeze(-1)
    fy = intrinsics[..., 1].unsqueeze(-1)
    cx = intrinsics[..., 2].unsqueeze(-1)
    cy = intrinsics[..., 3].unsqueeze(-1)
    u = fx * points_proj[..., 0].unsqueeze(-1) + cx
    v = fy * points_proj[..., 1].unsqueeze(-1) + cy
    return torch.cat([u, v], dim=-1)


class ColmapResidual(nn.Module):
    def __init__(self, camera_params, points_3d, intrinsics, optimize_intrinsics=True):
        super().__init__()
        if intrinsics is None:
            raise ValueError("intrinsics must be provided for COLMAP mode")
        if intrinsics.dim() == 1:
            intrinsics = intrinsics.unsqueeze(0)
        if intrinsics.shape[-1] != 4:
            raise ValueError("intrinsics must have shape [4] or [1, 4]")

        self.pose = pp.Parameter(camera_params, sjac=True)
        self.points_3d = pp.Parameter(points_3d, sjac=True)
        self.pose.trim_SE3_grad = True

        if optimize_intrinsics:
            self.shared_intr = pp.Parameter(intrinsics, sjac=True)
        else:
            self.register_buffer("shared_intr", intrinsics)

    def forward(self, points_2d, camera_indices=None, point_indices=None):
        if isinstance(points_2d, dict):
            input_dict = points_2d
            points_2d = input_dict["points_2d"]
            camera_indices = input_dict["camera_indices"]
            point_indices = input_dict["point_indices"]

        zero_indices = torch.zeros_like(camera_indices)
        intrinsics_batched = self.shared_intr[zero_indices]
        points_proj = project_colmap(
            self.points_3d[point_indices],
            self.pose[camera_indices],
            intrinsics_batched,
        )
        return points_proj - points_2d


def least_square_error(
    camera_params,
    points_3d,
    camera_indices,
    point_indices,
    points_2d,
    intrinsics,
    optimize_intrinsics=True,
):
    model = ColmapResidual(
        camera_params,
        points_3d,
        intrinsics=intrinsics,
        optimize_intrinsics=optimize_intrinsics,
    )
    loss = model(points_2d, camera_indices, point_indices)
    return torch.sum(loss**2, dim=-1).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Bundle adjustment entrypoint for COLMAP data")
    parser.add_argument(
        "--input-dir",
        default="colmap_data",
        help="Directory containing COLMAP model files (.txt or .bin)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save optimized COLMAP txt (if omitted, no save)",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu")
    parser.add_argument("--iters", type=int, default=20, help="Optimization iterations")
    parser.add_argument(
        "--optimize-intrinsics",
        action="store_true",
        default=True,
        help="Optimize shared PINHOLE intrinsics",
    )
    parser.add_argument(
        "--no-optimize-intrinsics",
        dest="optimize_intrinsics",
        action="store_false",
        help="Disable shared intrinsics optimization",
    )
    return parser.parse_args()


def _format_intrinsics(intr: torch.Tensor | None) -> str:
    if intr is None:
        return "None"
    if intr.dim() > 1:
        intr = intr.squeeze(0)
    vals = intr.detach().cpu().tolist()
    return f"fx={vals[0]:.6f}, fy={vals[1]:.6f}, cx={vals[2]:.6f}, cy={vals[3]:.6f}"


def main():
    args = parse_args()

    dataset = read_colmap_data(input_dir=args.input_dir)
    metadata = dataset.get("metadata", {})
    original_intrinsics = metadata.get("intrinsics", None)

    trimmed_dataset = {
        k: v.to(args.device) for k, v in dataset.items() if isinstance(v, torch.Tensor)
    }
    trimmed_dataset["metadata"] = metadata
    if original_intrinsics is not None:
        trimmed_dataset["intrinsics"] = original_intrinsics.to(args.device)

    input_dict = {
        "points_2d": trimmed_dataset["points_2d"],
        "camera_indices": trimmed_dataset["camera_index_of_observations"],
        "point_indices": trimmed_dataset["point_index_of_observations"],
    }

    model = ColmapResidual(
        trimmed_dataset["camera_params"].clone(),
        trimmed_dataset["points_3d"].clone(),
        intrinsics=trimmed_dataset.get("intrinsics", None),
        optimize_intrinsics=args.optimize_intrinsics,
    ).to(args.device)

    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
    solver = PCG(tol=1e-4, maxiter=250)
    optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

    initial_loss = least_square_error(
        model.pose,
        model.points_3d,
        trimmed_dataset["camera_index_of_observations"],
        trimmed_dataset["point_index_of_observations"],
        trimmed_dataset["points_2d"],
        intrinsics=trimmed_dataset.get("intrinsics", None),
        optimize_intrinsics=args.optimize_intrinsics,
    ).item()
    print("Initial loss:", initial_loss)

    start = perf_counter()
    for idx in range(args.iters):
        loss = optimizer.step(input_dict)
        print("Iteration", idx, "loss", loss.item(), "time", perf_counter() - start)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    print("Time", perf_counter() - start)

    ending_loss = least_square_error(
        model.pose,
        model.points_3d,
        trimmed_dataset["camera_index_of_observations"],
        trimmed_dataset["point_index_of_observations"],
        trimmed_dataset["points_2d"],
        intrinsics=trimmed_dataset.get("intrinsics", None),
        optimize_intrinsics=args.optimize_intrinsics,
    ).item()
    print("Ending loss:", ending_loss)

    optimized_intrinsics = None
    if args.optimize_intrinsics and hasattr(model, "shared_intr") and model.shared_intr is not None:
        optimized_intrinsics = model.shared_intr.detach().cpu().squeeze(0)
    elif trimmed_dataset.get("intrinsics", None) is not None:
        optimized_intrinsics = trimmed_dataset["intrinsics"].detach().cpu()

    print("Intrinsics before:", _format_intrinsics(original_intrinsics))
    print("Intrinsics after :", _format_intrinsics(optimized_intrinsics))

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        save_colmap_result(
            os.path.join(args.out_dir, "images_optimized.txt"),
            os.path.join(args.out_dir, "points3D_optimized.txt"),
            dataset,
            model.pose.detach().cpu(),
            model.points_3d.detach().cpu(),
        )
        if optimized_intrinsics is not None:
            save_colmap_cameras(
                os.path.join(args.out_dir, "cameras_optimized.txt"),
                dataset,
                optimized_intrinsics,
            )


if __name__ == "__main__":
    main()
