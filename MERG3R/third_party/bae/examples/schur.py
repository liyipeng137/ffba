from time import perf_counter
from datetime import datetime
from pathlib import Path
import inspect
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pypose as pp
import torch
import torch.nn as nn
import warp as wp
from pypose.autograd.function import psjac

from bae.optim.optimizer import Schur
from datapipes.bal_loader import get_problem
from bae.optim import LM
from bae.utils.pysolvers import PCG
from bae.optim.strategy import TrustRegion


TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"
# other options:
# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"
# TARGET_DATASET = "venice"
# TARGET_PROBLEM = "problem-1778-993923-pre"
# TARGET_DATASET = "final"
# TARGET_PROBLEM = "problem-13682-4456117-pre"

DEVICE = "cuda"
OPTIMIZE_INTRINSICS = True
NUM_CAMERA_PARAMS = 10 if OPTIMIZE_INTRINSICS else 7
REPORT_WARP_MEMPOOL = True


def _format_bytes(num_bytes: int) -> str:
    sign = "-" if num_bytes < 0 else ""
    size = float(abs(num_bytes))
    units = ["B", "KiB", "MiB", "GiB", "TiB"]

    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            break
        size /= 1024.0
        
    if unit == "B":
        return f"{sign}{int(size)} {unit}"

    return f"{sign}{size:.2f} {unit}"


@psjac
def project(points, camera_params):
    projection = pp.SE3(camera_params[..., :7]).Act(points)
    projection = -projection[..., :2] / projection[..., [2]]

    f = camera_params[..., [-3]]
    k1 = camera_params[..., [-2]]
    k2 = camera_params[..., [-1]]

    n = torch.sum(projection**2, axis=-1, keepdim=True)
    r = 1 + k1 * n + k2 * n**2
    return projection * r * f


class Residual(nn.Module):
    def __init__(self, camera_params, points):
        super().__init__()
        self.pose = pp.Parameter(camera_params, sjac=True)
        self.points = pp.Parameter(points, sjac=True)
        self.pose.trim_SE3_grad = True

    def forward(self, observes, cidx, pidx):
        points_proj = project(self.points[pidx], self.pose[cidx])
        return points_proj - observes


def least_square_error(camera_params, points, cidx, pidx, observes):
    model = Residual(camera_params, points)
    loss = model(observes, cidx, pidx)
    return torch.sum(loss**2, dim=-1).mean()


def main():
    dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET)
    print(f"Fetched {TARGET_PROBLEM} from {TARGET_DATASET}")
    file_name = f'{TARGET_DATASET}.{TARGET_PROBLEM}'
    cuda_device = torch.device(DEVICE) if DEVICE.startswith("cuda") else None
    memory_snapshot_path = None
    warp_device = None
    warp_mempool_start_current = None
    warp_mempool_start_high = None
    dataset = {
        key: value.to(DEVICE)
        for key, value in dataset.items()
        if isinstance(value, torch.Tensor)
    }
    input = {
        "observes": dataset["points_2d"],
        "cidx": dataset["camera_index_of_observations"],
        "pidx": dataset["point_index_of_observations"],
    }

    if DEVICE.startswith("cuda") and torch.cuda.is_available():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = Path("memory_traces")
        snapshot_dir.mkdir(exist_ok=True)
        memory_snapshot_path = snapshot_dir / f"{file_name}_cuda_memory_{timestamp}.pickle"
        memory_history_kwargs = {
            "enabled": "all",
            "context": "all",
            "stacks": "python",
            "device": cuda_device,
        }
        if "clear_history" in inspect.signature(torch.cuda.memory._record_memory_history).parameters:
            memory_history_kwargs["clear_history"] = True
        torch.cuda.memory._record_memory_history(**memory_history_kwargs)

    if REPORT_WARP_MEMPOOL and DEVICE.startswith("cuda"):
        try:
            if wp.is_cuda_available():
                warp_device = wp.get_device("cuda:0" if DEVICE == "cuda" else DEVICE)
                if not wp.is_mempool_enabled(warp_device):
                    wp.set_mempool_enabled(warp_device, True)
                warp_mempool_start_current = wp.get_mempool_used_mem_current(warp_device)
                warp_mempool_start_high = wp.get_mempool_used_mem_high(warp_device)
        except Exception as e:
            print(f"Warning: failed to query Warp mempool stats: {e}")

    model = Residual(
        dataset["camera_params"][:, :NUM_CAMERA_PARAMS].clone(),
        dataset["points_3d"].clone(),
    ).to(DEVICE)

    strategy = TrustRegion(up=2.0, down=0.5**4)
    solver = PCG(tol=1e-4, maxiter=250)
    optimizer = Schur(model, strategy=strategy, solver=solver, reject=30)

    print('Loss:', least_square_error(
        model.pose,
        model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())

    print("Initial loss", optimizer.model.loss(input, None).item())

    start = perf_counter()
    for idx in range(20):
        loss = optimizer.step(input)
        print("Iteration", idx, "loss", loss.item(), "time", perf_counter() - start)

    torch.cuda.synchronize()
    end = perf_counter()
    print("Time", end - start)

    if memory_snapshot_path:
        torch.cuda.memory._dump_snapshot(str(memory_snapshot_path))
        print(f"CUDA memory snapshot saved to {memory_snapshot_path}")

    if cuda_device is not None and torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
        try:
            peak_reserved = torch.cuda.max_memory_reserved(cuda_device)
        except AttributeError:
            peak_reserved = torch.cuda.max_memory_cached(cuda_device)
        print(f"Peak CUDA memory allocated: {_format_bytes(peak_allocated)}")
        print(f"Peak CUDA memory reserved: {_format_bytes(peak_reserved)}")

    if warp_device is not None and warp_mempool_start_current is not None:
        try:
            warp_current = wp.get_mempool_used_mem_current(warp_device)
            warp_high = wp.get_mempool_used_mem_high(warp_device)
            print(f"Warp CUDA mempool current: {_format_bytes(warp_current)} (Δ {_format_bytes(warp_current - warp_mempool_start_current)})")
            print(f"Warp CUDA mempool high-water: {_format_bytes(warp_high)} (Δ {_format_bytes(warp_high - warp_mempool_start_high)})")
        except Exception as e:
            print(f"Warning: failed to query Warp mempool stats: {e}")

    print('Ending loss:', least_square_error(
        model.pose,
        model.points,
        dataset["camera_index_of_observations"],
        dataset["point_index_of_observations"],
        dataset["points_2d"],
    ).item())


if __name__ == "__main__":
    main()
