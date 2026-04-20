from time import perf_counter
import os
import torch
import pypose as pp

from ba_helpers import Reproj, least_square_error
from datapipes.bal_loader import get_problem, read_bal_data
from bae.sparse.py_ops import *
try:
    from bae.sparse.solve import *
except ImportError:
    pass
from bae.optim import LM
from bae.utils.pysolvers import PCG, CuDSS

# TARGET_DATASET = "ladybug"
# TARGET_PROBLEM = "problem-1723-156502-pre"
# TARGET_PROBLEM = "problem-49-7776-pre"
# TARGET_PROBLEM = "problem-1695-155710-pre"  
# TARGET_PROBLEM = "problem-969-105826-pre"
TARGET_DATASET = "trafalgar"
TARGET_PROBLEM = "problem-257-65132-pre"
# TARGET_DATASET = "dubrovnik"
# TARGET_PROBLEM = "problem-356-226730-pre"



DEVICE = 'cuda'
OPTIMIZE_INTRINSICS = True

USE_QUATERNIONS = True

LOAD_COLMAP = True
SAVE_COLMAP = True

from datapipes.colmap_loader import read_colmap_data
if LOAD_COLMAP:
    dataset = read_colmap_data(
        "colmap_data/cameras.txt",
        "colmap_data/images.txt",
        "colmap_data/points3D.txt",
    )
else:
    file_name = f"{TARGET_DATASET}.{TARGET_PROBLEM}"
    dataset = get_problem(TARGET_PROBLEM, TARGET_DATASET, use_quat=USE_QUATERNIONS)

if OPTIMIZE_INTRINSICS:
    NUM_CAMERA_PARAMS = 10 if USE_QUATERNIONS else 9
else:
    NUM_CAMERA_PARAMS = 7 if USE_QUATERNIONS else 6

if LOAD_COLMAP:
    NUM_CAMERA_PARAMS = 7
    OPTIMIZE_INTRINSICS = True

print(f'Fetched {TARGET_PROBLEM} from {TARGET_DATASET}')

metadata = dataset.get("metadata", {})
original_intrinsics = metadata.get("intrinsics", None) if LOAD_COLMAP else None

trimmed_dataset = dataset
trimmed_dataset = {k: v.to(DEVICE) for k, v in trimmed_dataset.items() if type(v) == torch.Tensor}
trimmed_dataset["metadata"] = metadata
if LOAD_COLMAP and original_intrinsics is not None:
    trimmed_dataset["intrinsics"] = original_intrinsics.to(DEVICE)

input = {
    "points_2d": trimmed_dataset['points_2d'],
    "camera_indices": trimmed_dataset['camera_index_of_observations'],
    "point_indices": trimmed_dataset['point_index_of_observations']
}

model = Reproj(
    trimmed_dataset['camera_params'][:, :NUM_CAMERA_PARAMS].clone(),
    trimmed_dataset['points_3d'].clone(),
    load_colmap=LOAD_COLMAP,
    intrinsics=trimmed_dataset.get("intrinsics", None),
    optimize_intrinsics=OPTIMIZE_INTRINSICS if LOAD_COLMAP else False,
).to(DEVICE)
strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5**4)
solver = PCG(tol=1e-4, maxiter=250)  # or CuDSS()
optimizer = LM(model, strategy=strategy, solver=solver, reject=30)

print('Loss:', least_square_error(
    model.pose,
    model.points_3d,
    trimmed_dataset['camera_index_of_observations'],
    trimmed_dataset['point_index_of_observations'],
    trimmed_dataset['points_2d'],
    load_colmap=LOAD_COLMAP,
    intrinsics=trimmed_dataset.get("intrinsics", None),
    optimize_intrinsics=OPTIMIZE_INTRINSICS if LOAD_COLMAP else False,
).item())

print("Initial loss", optimizer.model.loss(input, None).item())

start = perf_counter()
for idx in range(20):
    loss = optimizer.step(input)
    print('Iteration', idx, 'loss', loss.item(), 'time', perf_counter() - start)
    if LOAD_COLMAP and hasattr(model, "shared_intr") and idx % 5 == 0:
        print("Shared intrinsics:", model.shared_intr.data)

torch.cuda.synchronize()
end = perf_counter()
print('Time', end - start)

print('Ending loss:', least_square_error(
    model.pose,
    model.points_3d,
    trimmed_dataset['camera_index_of_observations'],
    trimmed_dataset['point_index_of_observations'],
    trimmed_dataset['points_2d'],
    load_colmap=LOAD_COLMAP,
    intrinsics=trimmed_dataset.get("intrinsics", None),
    optimize_intrinsics=OPTIMIZE_INTRINSICS if LOAD_COLMAP else False,
).item())

if SAVE_COLMAP and LOAD_COLMAP and "metadata" in dataset:
    if OPTIMIZE_INTRINSICS and hasattr(model, "shared_intr") and model.shared_intr is not None:
        optimized_intrinsics = model.shared_intr.detach().cpu().squeeze(0)
    else:
        optimized_intrinsics = trimmed_dataset.get("intrinsics", dataset["metadata"].get("intrinsics"))

    from datapipes.colmap_loader import save_colmap_cameras, save_colmap_result

    os.makedirs("colmap_data_ba", exist_ok=True)
    save_colmap_result(
        "colmap_data_ba/images_optimized.txt",
        "colmap_data_ba/points3D_optimized.txt",
        dataset,
        model.pose.detach().cpu(),
        model.points_3d.detach().cpu(),
    )
    save_colmap_cameras(
        "colmap_data_ba/cameras_optimized.txt",
        dataset,
        optimized_intrinsics,
    )
