from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class Merg3rCoarseState:
    high_images: torch.Tensor
    low_image_names: list[str]
    high_image_names: list[str]
    low_image_size_hw: tuple[int, int]
    high_image_size_hw: tuple[int, int]
    final_predictions: dict
    extrinsic: np.ndarray
    intrinsic_low: np.ndarray
    intrinsic_high: np.ndarray
    image_ids: np.ndarray
    pairs: np.ndarray
    pair_graph_stats: dict
    raw_depth: np.ndarray | None
    raw_depth_conf: np.ndarray | None
    retrieval_sim_matrix: np.ndarray | None
    image_pyramid: dict | None
    initial_geometry_source: str = "feedforward"
    source_metadata: dict | None = None
