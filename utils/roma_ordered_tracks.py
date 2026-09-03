"""Motion-aware RoMaV2 tracking for ordered FFBA image sequences.

The module deliberately keeps the learned matcher separate from the numerical
tracking code.  Pure planning/sampling helpers are unit-testable without CUDA,
while :func:`run_roma_ordered_prior_tracks` owns model inference and artifact
publication for the pipeline.
"""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROMAV2_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "vidmap"
    / "third_party"
    / "RoMaV2"
    / "src"
)


@dataclass
class RomaOrderedConfig:
    device: str = "cuda"
    compile: bool = False
    lowres_size: int = 560
    lowres_batch_size: int = 4
    highres_max_size: int = 1200
    max_keypoints: int = 1500
    max_births_per_frame: int = 384
    min_track_length: int = 3
    short_track_rescue_min_observations: int = 16
    aliked_keypoints: int = 750
    aliked_detection_threshold: float = 0.005
    min_confidence: float = 0.05
    nms_radius: float = 3.0
    birth_nms_radius: float = 6.0
    max_track_sigma_px: float = 8.0
    max_anchor_gap: int = 12
    continuity_motion_target: float = 0.06
    geometry_motion_target: float = 0.12
    min_stage_a_overlap: float = 0.10
    min_stage_a_grid_coverage: float = 0.15
    direct_consistency_px: float = 4.0
    stage_a_samples: int = 512
    stage_a_depth_rel_threshold: float = 0.15
    spatial_grid_size: int = 8
    spatial_prefilter_oversample: int = 8
    epipolar_diagnostics: bool = True


@dataclass(frozen=True)
class PlanEdge:
    source: int
    target: int
    kind: str
    cumulative_motion: float
    stage_a_overlap: float | None = None
    stage_a_grid_coverage: float | None = None
    stage_a_parallax_deg: float | None = None


@dataclass
class DenseMatchField:
    """Dense source-grid to target-pixel map in working-image coordinates."""

    source: int
    target: int
    matches: np.ndarray
    certainty: np.ndarray
    covariance: np.ndarray
    source_size_wh: tuple[int, int]
    target_size_wh: tuple[int, int]


@dataclass
class RomaOrderedTrackingResult:
    tracks: list[list[tuple[int, np.ndarray]]]
    pairs: np.ndarray
    stats: dict
    plan: list[PlanEdge]
    audit_path: Path


@dataclass
class _TrackRecord:
    observations: dict[int, np.ndarray] = field(default_factory=dict)
    certainty: dict[int, float] = field(default_factory=dict)
    covariance: dict[int, np.ndarray] = field(default_factory=dict)
    provenance: dict[int, str] = field(default_factory=dict)


def _summary(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def _validate_config(config: RomaOrderedConfig):
    positive = (
        "lowres_size",
        "lowres_batch_size",
        "highres_max_size",
        "max_keypoints",
        "max_births_per_frame",
        "max_anchor_gap",
        "stage_a_samples",
        "spatial_grid_size",
        "spatial_prefilter_oversample",
    )
    for name in positive:
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.aliked_keypoints < 0:
        raise ValueError("aliked_keypoints must be nonnegative")
    if int(config.min_track_length) < 2:
        raise ValueError("min_track_length must be at least 2")
    if int(config.short_track_rescue_min_observations) < 0:
        raise ValueError("short_track_rescue_min_observations must be nonnegative")
    if not 0.0 <= config.min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    for name in ("min_stage_a_overlap", "min_stage_a_grid_coverage"):
        if not 0.0 <= float(getattr(config, name)) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    for name in (
        "nms_radius",
        "birth_nms_radius",
        "max_track_sigma_px",
        "continuity_motion_target",
        "geometry_motion_target",
        "direct_consistency_px",
        "stage_a_depth_rel_threshold",
    ):
        if float(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")


def _bilinear_sample(array: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Sample an ``H x W x ...`` array at field coordinates."""
    xy = np.asarray(xy, dtype=np.float64)
    if xy.size == 0:
        return np.empty((0, *array.shape[2:]), dtype=array.dtype)
    height, width = array.shape[:2]
    x = np.clip(xy[:, 0], 0.0, max(width - 1, 0))
    y = np.clip(xy[:, 1], 0.0, max(height - 1, 0))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    for _ in range(array.ndim - 2):
        wx = wx[..., None]
        wy = wy[..., None]
    return (
        array[y0, x0] * (1 - wx) * (1 - wy)
        + array[y0, x1] * wx * (1 - wy)
        + array[y1, x0] * (1 - wx) * wy
        + array[y1, x1] * wx * wy
    )


def _source_to_field(xy, source_size_wh, field_shape):
    width, height = source_size_wh
    field_height, field_width = field_shape
    xy = np.asarray(xy, dtype=np.float64)
    return np.stack(
        [
            xy[:, 0] * field_width / max(width, 1) - 0.5,
            xy[:, 1] * field_height / max(height, 1) - 0.5,
        ],
        axis=-1,
    )


def sample_dense_field(field: DenseMatchField, source_xy: np.ndarray):
    field_xy = _source_to_field(
        source_xy,
        field.source_size_wh,
        field.certainty.shape,
    )
    return (
        _bilinear_sample(field.matches, field_xy).astype(np.float32),
        _bilinear_sample(field.certainty, field_xy).astype(np.float32),
        _bilinear_sample(field.covariance, field_xy).astype(np.float32),
    )


def field_source_points(field: DenseMatchField):
    field_height, field_width = field.certainty.shape
    width, height = field.source_size_wh
    yy, xx = np.meshgrid(
        np.arange(field_height, dtype=np.float32),
        np.arange(field_width, dtype=np.float32),
        indexing="ij",
    )
    return np.stack(
        [
            (xx + 0.5) * width / field_width,
            (yy + 0.5) * height / field_height,
        ],
        axis=-1,
    )


def _grid_cell_ids(points, image_size_wh, grid_size):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    width, height = (float(value) for value in image_size_wh)
    cell_x = np.clip(
        (points[:, 0] * grid_size / max(width, 1)).astype(np.int64),
        0,
        grid_size - 1,
    )
    cell_y = np.clip(
        (points[:, 1] * grid_size / max(height, 1)).astype(np.int64),
        0,
        grid_size - 1,
    )
    return cell_y * grid_size + cell_x


def spatial_prefilter_indices(
    points,
    scores,
    max_points,
    image_size_wh,
    *,
    grid_size=8,
    oversample=8,
):
    """Keep high-quality candidates per cell before spatial selection.

    A global top-k destroys low-score image regions before spatial balancing
    sees them.  This bounded prefilter gives every populated cell its own
    candidate budget, then lets :func:`spatial_select_indices` make the final
    density-aware choice.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if points.shape[0] != scores.shape[0]:
        raise ValueError("points and scores must have the same length")
    max_points = int(max_points)
    grid_size = int(grid_size)
    oversample = int(oversample)
    if max_points <= 0 or points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    if grid_size <= 0 or oversample <= 0:
        raise ValueError("grid_size and oversample must be positive")

    width, height = (float(value) for value in image_size_wh)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(scores)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    cell_ids = _grid_cell_ids(
        points[valid_indices], image_size_wh, grid_size
    )
    order = np.argsort(cell_ids, kind="stable")
    sorted_cells = cell_ids[order]
    counts = np.bincount(sorted_cells, minlength=grid_size * grid_size)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    per_cell_limit = max(
        1,
        int(math.ceil(max_points / (grid_size * grid_size))) * oversample,
    )
    kept = []
    for cell in range(grid_size * grid_size):
        local = order[offsets[cell] : offsets[cell + 1]]
        if local.size == 0:
            continue
        original = valid_indices[local]
        if original.size > per_cell_limit:
            best = np.argpartition(scores[original], -per_cell_limit)[
                -per_cell_limit:
            ]
            original = original[best]
        kept.append(original)
    if not kept:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(kept).astype(np.int64, copy=False)


def spatial_select_indices(
    points,
    scores,
    max_points,
    image_size_wh,
    *,
    radius=3.0,
    grid_size=8,
    existing_points=None,
):
    """Deterministic grid-round-robin selection with spatial suppression."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if points.shape[0] != scores.shape[0]:
        raise ValueError("points and scores must have the same length")
    max_points = int(max_points)
    if max_points <= 0 or points.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    width, height = (float(value) for value in image_size_wh)
    valid = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(scores)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return np.empty(0, dtype=np.int64)

    points_valid = points[valid_indices]
    cell_ids = _grid_cell_ids(points_valid, image_size_wh, grid_size)
    buckets = {}
    for local_idx, cell in enumerate(cell_ids.tolist()):
        buckets.setdefault(cell, []).append(local_idx)
    for cell in buckets:
        buckets[cell].sort(
            key=lambda idx: (
                -float(scores[valid_indices[idx]]),
                int(valid_indices[idx]),
            )
        )

    radius = max(float(radius), 0.0)
    spatial_cell = max(radius, 1.0)
    occupied = {}
    occupancy = np.zeros(grid_size * grid_size, dtype=np.int64)

    def add_occupied(point):
        key = (int(math.floor(point[0] / spatial_cell)), int(math.floor(point[1] / spatial_cell)))
        occupied.setdefault(key, []).append(np.asarray(point, dtype=np.float32))

    if existing_points is not None:
        existing = np.asarray(existing_points, dtype=np.float32).reshape(-1, 2)
        existing_valid = (
            np.isfinite(existing).all(axis=1)
            & (existing[:, 0] >= 0)
            & (existing[:, 0] < width)
            & (existing[:, 1] >= 0)
            & (existing[:, 1] < height)
        )
        if np.any(existing_valid):
            occupancy += np.bincount(
                _grid_cell_ids(existing[existing_valid], image_size_wh, grid_size),
                minlength=grid_size * grid_size,
            )
        for point in existing[existing_valid]:
            if np.isfinite(point).all():
                add_occupied(point)

    def is_clear(point):
        if radius <= 0:
            return True
        key_x = int(math.floor(point[0] / spatial_cell))
        key_y = int(math.floor(point[1] / spatial_cell))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for other in occupied.get((key_x + dx, key_y + dy), ()):
                    if float(np.linalg.norm(point - other)) < radius:
                        return False
        return True

    selected = []
    cursors = {cell: 0 for cell in buckets}
    while len(selected) < max_points:
        added = False
        available_cells = [
            cell for cell, bucket in buckets.items() if cursors[cell] < len(bucket)
        ]
        available_cells.sort(
            key=lambda cell: (
                int(occupancy[cell]),
                -float(
                    scores[
                        valid_indices[buckets[cell][cursors[cell]]]
                    ]
                ),
                int(cell),
            )
        )
        for cell in available_cells:
            bucket = buckets[cell]
            cursor = cursors[cell]
            while cursor < len(bucket):
                local_idx = bucket[cursor]
                cursor += 1
                point = points_valid[local_idx]
                if is_clear(point):
                    selected.append(int(valid_indices[local_idx]))
                    add_occupied(point)
                    occupancy[cell] += 1
                    added = True
                    break
            cursors[cell] = cursor
            if len(selected) >= max_points:
                break
            if added:
                break
        if not added:
            break
    return np.asarray(selected, dtype=np.int64)


def build_ordered_tracking_plan(
    adjacent_stats,
    geometry,
    num_images,
    *,
    max_anchor_gap=12,
    continuity_motion_target=0.06,
    geometry_motion_target=0.12,
    min_overlap=0.10,
    min_grid_coverage=0.15,
):
    """Build a hard sequential backbone plus two dynamic direct anchors."""
    if num_images <= 1:
        return []
    motions = np.asarray(
        [float(item.get("motion_median_normalized", 0.0)) for item in adjacent_stats],
        dtype=np.float64,
    )
    if motions.shape != (num_images - 1,):
        raise ValueError("adjacent_stats must contain one record per consecutive pair")
    motions = np.nan_to_num(motions, nan=0.0, posinf=0.0, neginf=0.0)
    prefix = np.concatenate([[0.0], np.cumsum(np.maximum(motions, 0.0))])
    edges = []
    for target in range(1, num_images):
        edges.append(
            PlanEdge(
                target - 1,
                target,
                "sequential",
                float(motions[target - 1]),
            )
        )
        candidates = []
        for source in range(max(0, target - int(max_anchor_gap)), target - 1):
            detail = geometry.get((source, target), {})
            overlap = float(detail.get("projected_overlap", 0.0))
            coverage = float(detail.get("projected_grid_coverage", 0.0))
            if overlap < min_overlap or coverage < min_grid_coverage:
                continue
            cumulative = float(prefix[target] - prefix[source])
            if cumulative <= 0:
                continue
            candidates.append((source, cumulative, detail))
        selected_sources = set()
        for kind, target_motion in (
            ("continuity_direct", continuity_motion_target),
            ("geometry_direct", geometry_motion_target),
        ):
            available = [item for item in candidates if item[0] not in selected_sources]
            if not available:
                continue

            def rank(item):
                source, cumulative, detail = item
                motion_distance = abs(math.log(max(cumulative, 1e-8) / target_motion))
                parallax = float(detail.get("parallax_median_deg", 0.0))
                parallax_bonus = min(max(parallax, 0.0) / 3.0, 1.0)
                if kind == "continuity_direct":
                    parallax_bonus *= 0.25
                return (
                    motion_distance - 0.15 * parallax_bonus,
                    -float(detail.get("projected_overlap", 0.0)),
                    -source,
                )

            source, cumulative, detail = min(available, key=rank)
            selected_sources.add(source)
            edges.append(
                PlanEdge(
                    source,
                    target,
                    kind,
                    cumulative,
                    float(detail.get("projected_overlap", 0.0)),
                    float(detail.get("projected_grid_coverage", 0.0)),
                    float(detail.get("parallax_median_deg", 0.0)),
                )
            )
    return edges


def _camera_centers(extrinsic):
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.swapaxes(rotation, 1, 2), translation)


def _regular_samples(height, width, max_samples):
    count = min(int(max_samples), int(height * width))
    side = max(int(math.sqrt(count)), 1)
    xs = np.linspace(0.5, width - 0.5, min(side, width), dtype=np.float64)
    ys = np.linspace(0.5, height - 0.5, min(side, height), dtype=np.float64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)


def compute_stage_a_temporal_geometry(
    extrinsic,
    intrinsics,
    depth,
    *,
    max_gap,
    max_samples=512,
    depth_rel_threshold=0.15,
    grid_size=8,
):
    """Compute coarse directed overlap/parallax metadata for past anchors."""
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    num_images, height, width = depth.shape
    samples = _regular_samples(height, width, max_samples)
    centers = _camera_centers(extrinsic)
    geometry = {}
    for target in range(1, num_images):
        for source in range(max(0, target - int(max_gap)), target - 1):
            source_depth = _bilinear_sample(depth[source], samples)
            valid_source = np.isfinite(source_depth) & (source_depth > 1e-6)
            if not np.any(valid_source):
                geometry[(source, target)] = {
                    "projected_overlap": 0.0,
                    "projected_grid_coverage": 0.0,
                    "projected_visible_ratio": 0.0,
                    "parallax_median_deg": 0.0,
                }
                continue
            xy = samples[valid_source]
            z = source_depth[valid_source]
            intrinsic = intrinsics[source]
            rays = np.stack(
                [
                    (xy[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0],
                    (xy[:, 1] - intrinsic[1, 2]) / intrinsic[1, 1],
                    np.ones(len(xy)),
                ],
                axis=-1,
            )
            camera_points = rays * z[:, None]
            rotation_s = extrinsic[source, :3, :3]
            translation_s = extrinsic[source, :3, 3]
            world = (camera_points - translation_s) @ rotation_s
            rotation_t = extrinsic[target, :3, :3]
            translation_t = extrinsic[target, :3, 3]
            camera_t = world @ rotation_t.T + translation_t
            target_z = camera_t[:, 2]
            intrinsic_t = intrinsics[target]
            safe_z = np.where(np.abs(target_z) > 1e-8, target_z, 1.0)
            target_xy = np.stack(
                [
                    intrinsic_t[0, 0] * camera_t[:, 0] / safe_z + intrinsic_t[0, 2],
                    intrinsic_t[1, 1] * camera_t[:, 1] / safe_z + intrinsic_t[1, 2],
                ],
                axis=-1,
            )
            visible = (
                np.isfinite(target_xy).all(axis=1)
                & np.isfinite(target_z)
                & (target_z > 1e-6)
                & (target_xy[:, 0] >= 0)
                & (target_xy[:, 0] < width)
                & (target_xy[:, 1] >= 0)
                & (target_xy[:, 1] < height)
            )
            target_depth = _bilinear_sample(depth[target], target_xy)
            relative_error = np.abs(target_depth - target_z) / np.maximum(
                np.maximum(np.abs(target_depth), np.abs(target_z)),
                1e-6,
            )
            consistent = (
                visible
                & np.isfinite(target_depth)
                & (target_depth > 1e-6)
                & np.isfinite(relative_error)
                & (relative_error <= float(depth_rel_threshold))
            )
            cell_x = np.clip((xy[:, 0] * grid_size / width).astype(int), 0, grid_size - 1)
            cell_y = np.clip((xy[:, 1] * grid_size / height).astype(int), 0, grid_size - 1)
            source_cells = np.unique(cell_y * grid_size + cell_x)
            consistent_cells = np.unique((cell_y * grid_size + cell_x)[consistent])
            rays_source = world - centers[source]
            rays_target = world - centers[target]
            norm_source = np.linalg.norm(rays_source, axis=1)
            norm_target = np.linalg.norm(rays_target, axis=1)
            valid_rays = consistent & (norm_source > 1e-9) & (norm_target > 1e-9)
            if np.any(valid_rays):
                cosine = np.einsum(
                    "ij,ij->i",
                    rays_source[valid_rays] / norm_source[valid_rays, None],
                    rays_target[valid_rays] / norm_target[valid_rays, None],
                )
                parallax = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            else:
                parallax = np.empty(0)
            geometry[(source, target)] = {
                "projected_overlap": float(consistent.sum() / max(len(xy), 1)),
                "projected_grid_coverage": float(
                    len(consistent_cells) / max(len(source_cells), 1)
                ),
                "projected_visible_ratio": float(visible.sum() / max(len(xy), 1)),
                "parallax_median_deg": float(np.median(parallax)) if parallax.size else 0.0,
            }
    return geometry


class _RomaRuntime:
    def __init__(self, config: RomaOrderedConfig):
        if str(ROMAV2_SOURCE) not in sys.path:
            sys.path.insert(0, str(ROMAV2_SOURCE))
        from romav2 import RoMaV2  # noqa: PLC0415
        from romav2 import local_correlation  # noqa: PLC0415

        torch.set_float32_matmul_precision("highest")
        local_correlation.local_corr = None
        self.config = config
        self.device = torch.device(config.device)
        self.model = RoMaV2(
            RoMaV2.Cfg(setting="precise", compile=bool(config.compile))
        ).eval()
        self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.aliked = None

    def close(self):
        if self.aliked is not None:
            self.aliked.cpu()
            self.aliked = None
        self.model.cpu()
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resize_preserve(self, image, max_size):
        height, width = image.shape[-2:]
        scale = min(float(max_size) / max(height, width), 1.0)
        out_height = max(int(round(height * scale)), 8)
        out_width = max(int(round(width * scale)), 8)
        out_height -= out_height % 4
        out_width -= out_width % 4
        return F.interpolate(
            image[None].to(self.device),
            size=(out_height, out_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    @staticmethod
    def _confidence(raw_confidence):
        if str(ROMAV2_SOURCE) not in sys.path:
            sys.path.insert(0, str(ROMAV2_SOURCE))
        from romav2.geometry import prec_mat_from_prec_params  # noqa: PLC0415

        overlap = raw_confidence[..., :1].sigmoid()[..., 0]
        precision = prec_mat_from_prec_params(raw_confidence[..., 1:4])
        eye = torch.eye(2, device=precision.device, dtype=precision.dtype)
        covariance = torch.linalg.inv(precision.float() + 1e-6 * eye)
        return overlap, covariance

    @torch.inference_mode()
    def lowres_adjacent_stats(self, images):
        size = int(self.config.lowres_size)
        batch_size = int(self.config.lowres_batch_size)
        previous_bidirectional = self.model.bidirectional
        self.model.bidirectional = False
        resized = F.interpolate(
            images,
            size=(size, size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        stats = []
        try:
            for start in range(0, len(images) - 1, batch_size):
                end = min(start + batch_size, len(images) - 1)
                image_a = resized[start:end].to(self.device)
                image_b = resized[start + 1 : end + 1].to(self.device)
                raw = self.model(image_a, image_b)
                overlap, _ = self._confidence(raw["confidence_AB"])
                warp = raw["warp_AB"]
                field_height, field_width = overlap.shape[1:3]
                yy, xx = torch.meshgrid(
                    torch.arange(field_height, device=self.device) + 0.5,
                    torch.arange(field_width, device=self.device) + 0.5,
                    indexing="ij",
                )
                source_xy = torch.stack(
                    [xx * size / field_width, yy * size / field_height], dim=-1
                )
                target_xy = (warp + 1.0) * (size / 2.0)
                displacement = torch.linalg.norm(target_xy - source_xy, dim=-1) / float(
                    math.hypot(size, size)
                )
                in_bounds = (
                    (target_xy[..., 0] >= 0)
                    & (target_xy[..., 0] < size)
                    & (target_xy[..., 1] >= 0)
                    & (target_xy[..., 1] < size)
                )
                valid = (overlap >= self.config.min_confidence) & in_bounds
                for batch_idx in range(end - start):
                    values = displacement[batch_idx][valid[batch_idx]].detach().cpu().numpy()
                    valid_ratio = float(valid[batch_idx].float().mean().item())
                    stats.append(
                        {
                            "source": int(start + batch_idx),
                            "target": int(start + batch_idx + 1),
                            "motion_median_normalized": (
                                float(np.median(values)) if values.size else 0.0
                            ),
                            "motion_p90_normalized": (
                                float(np.percentile(values, 90))
                                if values.size
                                else 0.0
                            ),
                            "valid_ratio": valid_ratio,
                            "certainty_median": float(
                                overlap[batch_idx][valid[batch_idx]].median().item()
                            )
                            if torch.any(valid[batch_idx])
                            else 0.0,
                        }
                    )
                del raw, warp, overlap, image_a, image_b
        finally:
            self.model.bidirectional = previous_bidirectional
            del resized
        return stats

    @torch.inference_mode()
    def match_highres(self, images, source, target):
        source_image = images[source]
        target_image = images[target]
        source_height, source_width = source_image.shape[-2:]
        target_height, target_width = target_image.shape[-2:]
        low_size = 800
        image_a_low = F.interpolate(
            source_image[None].to(self.device),
            size=(low_size, low_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        image_b_low = F.interpolate(
            target_image[None].to(self.device),
            size=(low_size, low_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        image_a_high = self._resize_preserve(source_image, self.config.highres_max_size)
        image_b_high = self._resize_preserve(target_image, self.config.highres_max_size)
        if image_a_high.shape[-2:] != image_b_high.shape[-2:]:
            raise ValueError("RoMa ordered tracking requires equal work-image shapes")
        previous_bidirectional = self.model.bidirectional
        self.model.bidirectional = False
        try:
            raw = self.model(image_a_low, image_b_low, image_a_high, image_b_high)
        finally:
            self.model.bidirectional = previous_bidirectional
        overlap, covariance = self._confidence(raw["confidence_AB"])
        warp = raw["warp_AB"]
        matches = torch.stack(
            [
                (warp[..., 0] + 1.0) * (target_width / 2.0),
                (warp[..., 1] + 1.0) * (target_height / 2.0),
            ],
            dim=-1,
        )
        process_height, process_width = image_b_high.shape[-2:]
        scale = torch.tensor(
            [target_width / process_width, target_height / process_height],
            device=covariance.device,
            dtype=covariance.dtype,
        )
        covariance = covariance * scale[None, None, :, None] * scale[None, None, None, :]
        result = DenseMatchField(
            int(source),
            int(target),
            matches[0].float().cpu().numpy(),
            overlap[0].float().cpu().numpy(),
            covariance[0].float().cpu().numpy(),
            (int(source_width), int(source_height)),
            (int(target_width), int(target_height)),
        )
        del raw, warp, overlap, covariance, matches
        return result

    @torch.inference_mode()
    def aliked_points(self, image):
        if self.config.aliked_keypoints == 0:
            return np.empty((0, 2), dtype=np.float32)
        if self.aliked is None:
            from lightglue import ALIKED  # noqa: PLC0415

            self.aliked = (
                ALIKED(
                    max_num_keypoints=int(self.config.aliked_keypoints),
                    detection_threshold=float(self.config.aliked_detection_threshold),
                )
                .eval()
                .to(self.device)
            )
        features = self.aliked.extract(image[None].to(self.device))
        return features["keypoints"][0].float().cpu().numpy()


class _OrderedTrackBuilder:
    def __init__(self, config, images, runtime):
        self.config = config
        self.images = images
        self.runtime = runtime
        self.records: list[_TrackRecord] = []
        self.active: list[int] = []
        self.stats = {
            "sequential": [],
            "direct": [],
            "epipolar": [],
        }

    @property
    def effective_min_track_length(self):
        return min(int(self.config.min_track_length), int(self.images.shape[0]))

    def _active_source_points(self, source):
        ids = [
            track_id
            for track_id in self.active
            if source in self.records[track_id].observations
        ]
        if not ids:
            return ids, np.empty((0, 2), dtype=np.float32)
        return ids, np.stack([self.records[track_id].observations[source] for track_id in ids])

    def _birth_points(self, field, existing):
        remaining = max(int(self.config.max_keypoints) - len(existing), 0)
        if len(existing) > 0:
            remaining = min(remaining, int(self.config.max_births_per_frame))
        possible_length = int(self.images.shape[0]) - int(field.source)
        if possible_length < self.effective_min_track_length:
            remaining = 0
        if remaining == 0:
            return np.empty((0, 2), dtype=np.float32)
        source_points = field_source_points(field).reshape(-1, 2)
        certainty = field.certainty.reshape(-1)
        matches = field.matches.reshape(-1, 2)
        covariance = field.covariance.reshape(-1, 2, 2)
        trace = np.maximum(
            covariance[:, 0, 0] + covariance[:, 1, 1],
            1e-6,
        )
        sigma = np.sqrt(trace)
        width_t, height_t = field.target_size_wh
        valid = (
            (certainty >= self.config.min_confidence)
            & np.isfinite(matches).all(axis=1)
            & np.isfinite(covariance).all(axis=(1, 2))
            & (matches[:, 0] >= 0)
            & (matches[:, 0] < width_t)
            & (matches[:, 1] >= 0)
            & (matches[:, 1] < height_t)
            & (sigma <= self.config.max_track_sigma_px)
        )
        quality = np.full(certainty.shape, np.nan, dtype=np.float32)
        quality[valid] = certainty[valid] / np.maximum(sigma[valid], 0.25)
        dense_indices = spatial_prefilter_indices(
            source_points,
            quality,
            remaining,
            field.source_size_wh,
            grid_size=self.config.spatial_grid_size,
            oversample=self.config.spatial_prefilter_oversample,
        )
        candidates = source_points[dense_indices]
        scores = quality[dense_indices]
        salient = self.runtime.aliked_points(self.images[field.source])
        if salient.size:
            salient_matches, salient_certainty, salient_covariance = (
                sample_dense_field(field, salient)
            )
            salient_trace = np.maximum(
                salient_covariance[:, 0, 0]
                + salient_covariance[:, 1, 1],
                1e-6,
            )
            salient_sigma = np.sqrt(salient_trace)
            valid_salient = (
                (salient_certainty >= self.config.min_confidence)
                & np.isfinite(salient_matches).all(axis=1)
                & np.isfinite(salient_covariance).all(axis=(1, 2))
                & (salient_matches[:, 0] >= 0)
                & (salient_matches[:, 0] < width_t)
                & (salient_matches[:, 1] >= 0)
                & (salient_matches[:, 1] < height_t)
                & (salient_sigma <= self.config.max_track_sigma_px)
            )
            candidates = np.concatenate([salient[valid_salient], candidates], axis=0)
            scores = np.concatenate(
                [
                    1.05
                    * salient_certainty[valid_salient]
                    / np.maximum(salient_sigma[valid_salient], 0.25),
                    scores,
                ],
                axis=0,
            )
        selected = spatial_select_indices(
            candidates,
            scores,
            remaining,
            field.source_size_wh,
            radius=self.config.birth_nms_radius,
            grid_size=self.config.spatial_grid_size,
            existing_points=existing,
        )
        return candidates[selected].astype(np.float32)

    def _thin_target(self, track_ids, points, confidence, covariance):
        if len(track_ids) == 0:
            return np.empty(0, dtype=np.int64)
        trace = np.maximum(covariance[:, 0, 0] + covariance[:, 1, 1], 1e-6)
        lengths = np.asarray([len(self.records[track_id].observations) for track_id in track_ids])
        scores = confidence / np.sqrt(trace) * (1.0 + 0.02 * np.minimum(lengths, 20))
        return spatial_select_indices(
            points,
            scores,
            self.config.max_keypoints,
            (int(self.images.shape[-1]), int(self.images.shape[-2])),
            radius=self.config.nms_radius,
            grid_size=self.config.spatial_grid_size,
        )

    def advance_sequential(self, field):
        source, target = field.source, field.target
        continuing_ids, continuing_xy = self._active_source_points(source)
        birth_xy = self._birth_points(field, continuing_xy)
        new_ids = []
        for xy in birth_xy:
            record = _TrackRecord()
            record.observations[source] = xy.astype(np.float32)
            record.certainty[source] = 1.0
            record.covariance[source] = np.zeros((2, 2), dtype=np.float32)
            record.provenance[source] = "birth"
            self.records.append(record)
            new_ids.append(len(self.records) - 1)
        track_ids = continuing_ids + new_ids
        if not track_ids:
            self.active = []
            return
        source_xy = np.stack(
            [self.records[track_id].observations[source] for track_id in track_ids]
        )
        target_xy, step_confidence, step_covariance = sample_dense_field(field, source_xy)
        accumulated_confidence = []
        accumulated_covariance = []
        for track_id, confidence, covariance in zip(
            track_ids, step_confidence, step_covariance, strict=False
        ):
            record = self.records[track_id]
            previous_confidence = float(record.certainty[source])
            previous_covariance = record.covariance[source]
            accumulated_confidence.append(min(previous_confidence, float(confidence)))
            accumulated_covariance.append(previous_covariance + covariance)
        accumulated_confidence = np.asarray(accumulated_confidence, dtype=np.float32)
        accumulated_covariance = np.asarray(accumulated_covariance, dtype=np.float32)
        width, height = field.target_size_wh
        sigma = np.sqrt(
            np.maximum(accumulated_covariance[:, 0, 0] + accumulated_covariance[:, 1, 1], 0.0)
        )
        valid = (
            np.isfinite(target_xy).all(axis=1)
            & np.isfinite(accumulated_covariance).all(axis=(1, 2))
            & (target_xy[:, 0] >= 0)
            & (target_xy[:, 0] < width)
            & (target_xy[:, 1] >= 0)
            & (target_xy[:, 1] < height)
            & (accumulated_confidence >= self.config.min_confidence)
            & (sigma <= self.config.max_track_sigma_px)
        )
        valid_indices = np.flatnonzero(valid)
        kept_local = self._thin_target(
            [track_ids[index] for index in valid_indices],
            target_xy[valid_indices],
            accumulated_confidence[valid_indices],
            accumulated_covariance[valid_indices],
        )
        kept_indices = valid_indices[kept_local]
        self.active = []
        for index in kept_indices:
            track_id = track_ids[int(index)]
            record = self.records[track_id]
            record.observations[target] = target_xy[index].astype(np.float32)
            record.certainty[target] = float(accumulated_confidence[index])
            record.covariance[target] = accumulated_covariance[index].astype(np.float32)
            record.provenance[target] = "sequential"
            self.active.append(track_id)
        self.stats["sequential"].append(
            {
                "source": int(source),
                "target": int(target),
                "continuing": int(len(continuing_ids)),
                "born": int(len(new_ids)),
                "accepted_before_thinning": int(valid.sum()),
                "active_after_thinning": int(len(self.active)),
                "certainty": _summary(accumulated_confidence[kept_indices]),
                "sigma_px": _summary(sigma[kept_indices]),
            }
        )
        if self.config.epipolar_diagnostics:
            self.stats["epipolar"].append(_epipolar_diagnostic(field))

    def refine_direct(self, field, kind):
        source, target = field.source, field.target
        track_ids = [
            track_id
            for track_id in self.active
            if source in self.records[track_id].observations
            and target in self.records[track_id].observations
        ]
        if not track_ids:
            self.stats["direct"].append(
                {"source": source, "target": target, "kind": kind, "eligible": 0}
            )
            return
        source_xy = np.stack(
            [self.records[track_id].observations[source] for track_id in track_ids]
        )
        direct_xy, direct_confidence, direct_covariance = sample_dense_field(field, source_xy)
        current_xy = np.stack(
            [self.records[track_id].observations[target] for track_id in track_ids]
        )
        source_covariance = np.stack(
            [self.records[track_id].covariance[source] for track_id in track_ids]
        )
        current_covariance = np.stack(
            [self.records[track_id].covariance[target] for track_id in track_ids]
        )
        total_direct_covariance = source_covariance + direct_covariance
        direct_sigma = np.sqrt(
            np.maximum(total_direct_covariance[:, 0, 0] + total_direct_covariance[:, 1, 1], 0.0)
        )
        disagreement = np.linalg.norm(direct_xy - current_xy, axis=1)
        agreement_threshold = np.maximum(
            float(self.config.direct_consistency_px),
            direct_sigma + 1.0,
        )
        width, height = field.target_size_wh
        valid = (
            np.isfinite(direct_xy).all(axis=1)
            & np.isfinite(total_direct_covariance).all(axis=(1, 2))
            & (direct_xy[:, 0] >= 0)
            & (direct_xy[:, 0] < width)
            & (direct_xy[:, 1] >= 0)
            & (direct_xy[:, 1] < height)
            & (direct_confidence >= self.config.min_confidence)
            & (direct_sigma <= self.config.max_track_sigma_px)
            & (disagreement <= agreement_threshold)
        )
        current_trace = current_covariance[:, 0, 0] + current_covariance[:, 1, 1]
        direct_trace = total_direct_covariance[:, 0, 0] + total_direct_covariance[:, 1, 1]
        replace = valid & (direct_trace < current_trace)
        for index in np.flatnonzero(replace):
            record = self.records[track_ids[int(index)]]
            record.observations[target] = direct_xy[index].astype(np.float32)
            record.certainty[target] = min(
                record.certainty[source], float(direct_confidence[index])
            )
            record.covariance[target] = total_direct_covariance[index].astype(np.float32)
            record.provenance[target] = kind
        disagreement_only = (~valid) & np.isfinite(disagreement)
        high_confidence_disagreement = disagreement_only & (
            direct_confidence >= self.config.min_confidence
        )
        for index in np.flatnonzero(high_confidence_disagreement):
            record = self.records[track_ids[int(index)]]
            record.covariance[target] = (record.covariance[target] * 1.25).astype(np.float32)
        self.stats["direct"].append(
            {
                "source": int(source),
                "target": int(target),
                "kind": kind,
                "eligible": int(len(track_ids)),
                "consistent": int(valid.sum()),
                "replaced": int(replace.sum()),
                "disagreement_px": _summary(disagreement),
                "direct_sigma_px": _summary(direct_sigma),
            }
        )
        if self.config.epipolar_diagnostics:
            diagnostic = _epipolar_diagnostic(field)
            diagnostic["kind"] = kind
            self.stats["epipolar"].append(diagnostic)

    def finalize(self):
        long_record_ids = {
            record_id
            for record_id, record in enumerate(self.records)
            if len(record.observations) >= self.effective_min_track_length
        }
        rescue_record_ids, rescue_stats = rescue_short_tracks_for_frame_coverage(
            self.records,
            long_record_ids,
            num_images=int(self.images.shape[0]),
            min_observations=int(
                self.config.short_track_rescue_min_observations
            ),
            image_size_wh=(int(self.images.shape[-1]), int(self.images.shape[-2])),
            grid_size=int(self.config.spatial_grid_size),
            radius=float(self.config.birth_nms_radius),
        )
        kept_record_ids = long_record_ids | rescue_record_ids
        tracks = []
        kept_records = []
        dropped_short_tracks = 0
        dropped_short_observations = 0
        for record_id, record in enumerate(self.records):
            if record_id not in kept_record_ids:
                if len(record.observations) < self.effective_min_track_length:
                    dropped_short_tracks += 1
                    dropped_short_observations += len(record.observations)
                continue
            observations = [
                (int(image_idx), np.asarray(xy, dtype=np.float32))
                for image_idx, xy in sorted(record.observations.items())
            ]
            tracks.append(observations)
            kept_records.append(record)
        self.stats["finalize"] = {
            "configured_min_track_length": int(self.config.min_track_length),
            "effective_min_track_length": int(self.effective_min_track_length),
            "input_records": int(len(self.records)),
            "kept_tracks": int(len(tracks)),
            "dropped_short_tracks": int(dropped_short_tracks),
            "dropped_short_observations": int(dropped_short_observations),
            "short_track_rescue": rescue_stats,
        }
        return tracks, kept_records


def _record_tracking_quality(record):
    values = []
    for image_idx, covariance in record.covariance.items():
        if record.provenance.get(image_idx) == "birth":
            continue
        covariance = np.asarray(covariance, dtype=np.float64)
        sigma = math.sqrt(max(float(np.trace(covariance)), 1e-6))
        values.append(float(record.certainty.get(image_idx, 0.0)) / max(sigma, 0.25))
    return min(values) if values else 0.0


def rescue_short_tracks_for_frame_coverage(
    records,
    kept_record_ids,
    *,
    num_images,
    min_observations,
    image_size_wh,
    grid_size=8,
    radius=6.0,
):
    """Rescue only enough two-view tracks to keep weak frames connected.

    The normal minimum-length filter removes the short birth cohorts that make
    patchy point clouds.  A completely hard cutoff can also disconnect blurred
    or fast-motion frames.  This fallback retains a tiny, spatially balanced
    subset of triangulatable short tracks only where long-track coverage is
    below ``min_observations``.
    """
    min_observations = int(min_observations)
    kept_record_ids = set(int(record_id) for record_id in kept_record_ids)
    coverage = np.zeros(int(num_images), dtype=np.int64)
    points_by_image = [[] for _ in range(int(num_images))]
    for record_id in kept_record_ids:
        for image_idx, xy in records[record_id].observations.items():
            coverage[int(image_idx)] += 1
            points_by_image[int(image_idx)].append(np.asarray(xy, dtype=np.float32))
    coverage_before = coverage.copy()
    deficient_before = np.flatnonzero(coverage < min_observations)
    if min_observations == 0 or deficient_before.size == 0:
        return set(), {
            "enabled": bool(min_observations > 0),
            "min_observations": min_observations,
            "rescued_tracks": 0,
            "rescued_observations": 0,
            "deficient_frames_before": deficient_before.astype(int).tolist(),
            "deficient_frames_after": deficient_before.astype(int).tolist(),
            "coverage_before": _summary(coverage_before),
            "coverage_after": _summary(coverage),
        }

    candidates_by_image = [[] for _ in range(int(num_images))]
    quality = {}
    for record_id, record in enumerate(records):
        if record_id in kept_record_ids or len(record.observations) < 2:
            continue
        quality[record_id] = _record_tracking_quality(record)
        for image_idx in record.observations:
            candidates_by_image[int(image_idx)].append(record_id)

    rescued = set()
    # Start with the weakest frames so a two-view rescue can also satisfy its
    # better-covered neighbor before that neighbor is visited.
    frame_order = np.argsort(coverage, kind="stable")
    for image_idx in frame_order:
        deficit = max(min_observations - int(coverage[image_idx]), 0)
        if deficit == 0:
            continue
        candidate_ids = [
            record_id
            for record_id in candidates_by_image[int(image_idx)]
            if record_id not in rescued
        ]
        if not candidate_ids:
            continue
        candidate_points = np.stack(
            [records[record_id].observations[int(image_idx)] for record_id in candidate_ids]
        )
        candidate_scores = np.asarray(
            [quality[record_id] for record_id in candidate_ids], dtype=np.float32
        )
        existing = np.asarray(points_by_image[int(image_idx)], dtype=np.float32).reshape(-1, 2)
        selected_local = spatial_select_indices(
            candidate_points,
            candidate_scores,
            deficit,
            image_size_wh,
            radius=radius,
            grid_size=grid_size,
            existing_points=existing,
        )
        # A weak frame may only contain a compact textured region.  Fill any
        # remaining tiny coverage deficit without NMS; the hard cap keeps this
        # fallback far below the former fill-to-1500 behavior.
        if len(selected_local) < deficit:
            selected_set = set(int(index) for index in selected_local)
            remaining_local = np.asarray(
                [index for index in range(len(candidate_ids)) if index not in selected_set],
                dtype=np.int64,
            )
            if remaining_local.size:
                selected_points = candidate_points[selected_local]
                fallback_existing = np.concatenate(
                    [existing, selected_points], axis=0
                )
                fallback = spatial_select_indices(
                    candidate_points[remaining_local],
                    candidate_scores[remaining_local],
                    deficit - len(selected_local),
                    image_size_wh,
                    radius=0.0,
                    grid_size=grid_size,
                    existing_points=fallback_existing,
                )
                selected_local = np.concatenate(
                    [selected_local, remaining_local[fallback]]
                )
        for local_idx in selected_local:
            record_id = candidate_ids[int(local_idx)]
            if record_id in rescued:
                continue
            rescued.add(record_id)
            for other_image_idx, xy in records[record_id].observations.items():
                other_image_idx = int(other_image_idx)
                coverage[other_image_idx] += 1
                points_by_image[other_image_idx].append(
                    np.asarray(xy, dtype=np.float32)
                )

    deficient_after = np.flatnonzero(coverage < min_observations)
    return rescued, {
        "enabled": True,
        "min_observations": min_observations,
        "rescued_tracks": int(len(rescued)),
        "rescued_observations": int(
            sum(len(records[record_id].observations) for record_id in rescued)
        ),
        "rescued_track_length": _summary(
            len(records[record_id].observations) for record_id in rescued
        ),
        "deficient_frames_before": deficient_before.astype(int).tolist(),
        "deficient_frames_after": deficient_after.astype(int).tolist(),
        "coverage_before": _summary(coverage_before),
        "coverage_after": _summary(coverage),
    }


def _epipolar_diagnostic(field, max_points=2000):
    base = {
        "source": int(field.source),
        "target": int(field.target),
        "samples": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "status": "unavailable",
    }
    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        return base
    source = field_source_points(field).reshape(-1, 2)
    target = field.matches.reshape(-1, 2)
    certainty = field.certainty.reshape(-1)
    width, height = field.target_size_wh
    valid = (
        (certainty >= 0.05)
        & np.isfinite(target).all(axis=1)
        & (target[:, 0] >= 0)
        & (target[:, 0] < width)
        & (target[:, 1] >= 0)
        & (target[:, 1] < height)
    )
    indices = np.flatnonzero(valid)
    if indices.size < 8:
        base["status"] = "insufficient_matches"
        return base
    if indices.size > max_points:
        best = np.argpartition(certainty[indices], -max_points)[-max_points:]
        indices = indices[best]
    try:
        _matrix, mask = cv2.findFundamentalMat(
            source[indices].astype(np.float32),
            target[indices].astype(np.float32),
            cv2.USAC_MAGSAC,
            1.0,
            0.999,
            10000,
        )
    except (cv2.error, TypeError):
        mask = None
    base["samples"] = int(len(indices))
    if mask is None:
        base["status"] = "estimation_failed"
        return base
    inliers = int(np.asarray(mask).reshape(-1).astype(bool).sum())
    base.update(
        {
            "inliers": inliers,
            "inlier_ratio": float(inliers / len(indices)),
            "status": "ok",
        }
    )
    return base


def _plan_pairs(plan):
    pairs = {tuple(sorted((edge.source, edge.target))) for edge in plan}
    return np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)


def _write_audit(output_dir, plan, adjacent_stats, geometry, records, stats, config):
    audit_dir = Path(output_dir) / "roma_ordered_tracking"
    audit_dir.mkdir(parents=True, exist_ok=True)
    plan_path = audit_dir / "tracking_plan.json"
    geometry_records = [
        {"source": int(source), "target": int(target), **detail}
        for (source, target), detail in sorted(geometry.items())
    ]
    with open(plan_path, "w") as f:
        json.dump(
            {
                "config": asdict(config),
                "adjacent_motion": adjacent_stats,
                "stage_a_geometry": geometry_records,
                "edges": [asdict(edge) for edge in plan],
                "tracking": stats,
            },
            f,
            indent=2,
        )

    offsets = [0]
    image_indices = []
    xy = []
    certainty = []
    covariance = []
    provenance = []
    for record in records:
        for image_idx in sorted(record.observations):
            image_indices.append(int(image_idx))
            xy.append(record.observations[image_idx])
            certainty.append(record.certainty[image_idx])
            cov = record.covariance[image_idx]
            covariance.append([cov[0, 0], cov[1, 1], cov[0, 1]])
            provenance.append(record.provenance[image_idx])
        offsets.append(len(image_indices))
    audit_path = audit_dir / "track_observations.npz"
    np.savez_compressed(
        audit_path,
        track_offsets=np.asarray(offsets, dtype=np.int64),
        image_indices=np.asarray(image_indices, dtype=np.int32),
        xy=np.asarray(xy, dtype=np.float32).reshape(-1, 2),
        certainty=np.asarray(certainty, dtype=np.float32),
        covariance=np.asarray(covariance, dtype=np.float32).reshape(-1, 3),
        provenance=np.asarray(provenance, dtype="U20"),
    )
    return audit_path


def run_roma_ordered_prior_tracks(
    images,
    extrinsic,
    intrinsics,
    depth,
    output_dir,
    config: RomaOrderedConfig,
):
    """Run the two-pass RoMaV2 ordered frontend and return FFBA prior tracks."""
    _validate_config(config)
    images = images.detach().cpu()
    num_images = int(images.shape[0])
    if num_images < 2:
        raise ValueError("RoMa ordered tracking requires at least two images")
    started = time.time()
    geometry_started = time.time()
    geometry = compute_stage_a_temporal_geometry(
        extrinsic,
        intrinsics,
        depth,
        max_gap=config.max_anchor_gap,
        max_samples=config.stage_a_samples,
        depth_rel_threshold=config.stage_a_depth_rel_threshold,
        grid_size=config.spatial_grid_size,
    )
    geometry_seconds = time.time() - geometry_started
    runtime = _RomaRuntime(config)
    try:
        lowres_started = time.time()
        adjacent_stats = runtime.lowres_adjacent_stats(images)
        lowres_seconds = time.time() - lowres_started
        plan = build_ordered_tracking_plan(
            adjacent_stats,
            geometry,
            num_images,
            max_anchor_gap=config.max_anchor_gap,
            continuity_motion_target=config.continuity_motion_target,
            geometry_motion_target=config.geometry_motion_target,
            min_overlap=config.min_stage_a_overlap,
            min_grid_coverage=config.min_stage_a_grid_coverage,
        )
        builder = _OrderedTrackBuilder(config, images, runtime)
        edges_by_target = {}
        for edge in plan:
            edges_by_target.setdefault(edge.target, []).append(edge)
        highres_started = time.time()
        for target in range(1, num_images):
            edges = edges_by_target[target]
            sequential = next(edge for edge in edges if edge.kind == "sequential")
            field = runtime.match_highres(images, sequential.source, sequential.target)
            builder.advance_sequential(field)
            del field
            for edge in edges:
                if edge.kind == "sequential":
                    continue
                field = runtime.match_highres(images, edge.source, edge.target)
                builder.refine_direct(field, edge.kind)
                del field
        highres_seconds = time.time() - highres_started
        tracks, records = builder.finalize()
    finally:
        runtime.close()

    pairs = _plan_pairs(plan)
    lengths = [len(track) for track in tracks]
    provenance_counts = {}
    for record in records:
        for value in record.provenance.values():
            provenance_counts[value] = provenance_counts.get(value, 0) + 1
    stats = {
        "frontend": "romav2_ordered",
        "num_images": num_images,
        "num_tracks": int(len(tracks)),
        "num_observations": int(sum(lengths)),
        "track_length": _summary(lengths),
        "plan": {
            "num_edges": int(len(plan)),
            "sequential_edges": int(sum(edge.kind == "sequential" for edge in plan)),
            "continuity_direct_edges": int(
                sum(edge.kind == "continuity_direct" for edge in plan)
            ),
            "geometry_direct_edges": int(
                sum(edge.kind == "geometry_direct" for edge in plan)
            ),
            "num_undirected_pairs": int(len(pairs)),
        },
        "provenance_observations": dict(sorted(provenance_counts.items())),
        "adjacent_motion": adjacent_stats,
        **builder.stats,
        "timing": {
            "stage_a_geometry": geometry_seconds,
            "lowres_adjacent_motion": lowres_seconds,
            "highres_tracking": highres_seconds,
            "total": time.time() - started,
        },
    }
    audit_path = _write_audit(
        output_dir,
        plan,
        adjacent_stats,
        geometry,
        records,
        stats,
        config,
    )
    stats["audit_path"] = str(audit_path)
    return RomaOrderedTrackingResult(tracks, pairs, stats, plan, audit_path)
