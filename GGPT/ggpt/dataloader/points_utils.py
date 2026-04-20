from dataclasses import dataclass
import torch
import os
import numpy as np

@dataclass
class OctreeNode:
    center: torch.Tensor      # (3,)
    half_size: float
    indices: torch.Tensor     # point indices
    children: list = None

def build_octree(points,
                 indices,
                 center,
                 half_size,
                 MAX,
                 min_half_size=1e-3):
    """
    points: (N, 3) tensor
    indices: (M,) tensor
    center: (3,)
    half_size: float
    """
    if len(indices) <= MAX or half_size <= min_half_size:
        return OctreeNode(center, half_size, indices, children=None)

    children = []
    quarter = half_size / 2.0

    offsets = torch.tensor([
        [-1, -1, -1],
        [-1, -1,  1],
        [-1,  1, -1],
        [-1,  1,  1],
        [ 1, -1, -1],
        [ 1, -1,  1],
        [ 1,  1, -1],
        [ 1,  1,  1],
    ], dtype=points.dtype, device=points.device)

    for o in offsets:
        child_center = center + o * quarter
        mask = (
            (points[indices, 0] >= child_center[0] - quarter) &
            (points[indices, 0] <  child_center[0] + quarter) &
            (points[indices, 1] >= child_center[1] - quarter) &
            (points[indices, 1] <  child_center[1] + quarter) &
            (points[indices, 2] >= child_center[2] - quarter) &
            (points[indices, 2] <  child_center[2] + quarter)
        )
        child_indices = indices[mask]
        if len(child_indices) > 0:
            child = build_octree(
                points,
                child_indices,
                child_center,
                quarter,
                MAX,
                min_half_size
            )
            children.append(child)

    return OctreeNode(center, half_size, indices=None, children=children)


def make_root(points):
    min_xyz = torch.min(points, dim=0)[0]
    max_xyz = torch.max(points, dim=0)[0]
    # min_xyz = points.mean(dim=0) - 3 * torch.std(points, dim=0)
    # max_xyz = points.mean(dim=0) + 3 * torch.std(points, dim=0) #REMOVE outliers
    center = (min_xyz + max_xyz) / 2.0
    half_size = torch.max(max_xyz - min_xyz) / 2.0
    return center, half_size


def collect_leaves(node, leaves):
    if node.children is None:
        leaves.append(node)
    else:
        for c in node.children:
            collect_leaves(c, leaves)


def chunk_by_octree(points, MAX):
    indices = torch.arange(points.shape[0], device=points.device)
    center, half_size = make_root(points)
    root = build_octree(points, indices, center, half_size, MAX=MAX)
    leaves = []
    collect_leaves(root, leaves)
    #chunks = [leaf.indices for leaf in leaves]
    #return chunks
    return leaves


def pca_transform(points, eigvecs=None, mean=None):
    reshape_flag = None
    if points.dim() != 2:
        reshape_flag = points.shape[:-1]
        points = points.reshape(-1, 3)
    
    if eigvecs is not None and mean is not None:
        points_pca = (points - mean) @ eigvecs
    else:
        mean = torch.mean(points, dim=0)
        X = points - mean
        C = (X.T @ X) / X.shape[0]  # covariance
        eigvals, eigvecs = torch.linalg.eigh(C)
        order = torch.argsort(eigvals, descending=True)
        eigvecs = eigvecs[:, order]
        # deterministic sign fixing
        for i in range(3):
            j = torch.argmax(torch.abs(eigvecs[:, i]))
            if eigvecs[j, i] < 0:
                eigvecs[:, i] *= -1

        points_pca = X @ eigvecs
    if reshape_flag is not None:
        points_pca = points_pca.reshape(*reshape_flag, 3)
    return points_pca, eigvecs, mean




