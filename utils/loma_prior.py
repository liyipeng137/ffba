"""Indexed LoMa prior matches with independent recall and SIFT-guided selection.

Torch and the vendored model are imported only by the inference adapter. Pair
construction, annotation, geometry verification and DB export also run on CPU.
"""

from collections import Counter
from dataclasses import dataclass
from contextlib import contextmanager
from functools import partial
import importlib
from pathlib import Path
import sys
import time

import numpy as np


def _pairs(values, num_images):
    result = set()
    for i, j in np.asarray(values, dtype=np.int64).reshape(-1, 2):
        i, j = sorted((int(i), int(j)))
        if not (0 <= i < num_images and 0 <= j < num_images):
            raise ValueError(f"Out-of-range pair {(i, j)} for {num_images} images")
        if i != j:
            result.add((i, j))
    return result


def _graph_stats(pairs, num_images):
    adjacency = [set() for _ in range(num_images)]
    for i, j in pairs:
        adjacency[i].add(j)
        adjacency[j].add(i)
    unseen = set(range(num_images))
    components = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        component, pending = [], [seed]
        while pending:
            node = pending.pop()
            component.append(node)
            neighbors = adjacency[node] & unseen
            unseen.difference_update(neighbors)
            pending.extend(sorted(neighbors))
        components.append(sorted(component))
    return {
        "degree": [len(neighbors) for neighbors in adjacency],
        "zero_degree_images": [
            i for i, neighbors in enumerate(adjacency) if not neighbors
        ],
        "num_components": len(components),
        "components": components,
    }


def build_loma_candidate_pairs(
    pose_pairs, temporal_pairs, retrieval_matrix, dino_topk=30
):
    similarity = np.asarray(retrieval_matrix, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("LoMa requires a square DINO retrieval matrix")
    if dino_topk <= 0:
        raise ValueError("loma_dino_candidates must be positive")
    n = len(similarity)
    pose = _pairs(pose_pairs, n)
    temporal = _pairs(temporal_pairs, n)
    retrieval = {}
    for i, row in enumerate(similarity):
        valid = np.flatnonzero(np.isfinite(row) & (np.arange(n) != i))
        order = valid[np.argsort(-row[valid], kind="stable")[:dino_topk]]
        for rank, j in enumerate(order, start=1):
            pair = tuple(sorted((i, int(j))))
            retrieval.setdefault(pair, []).append(
                {
                    "source": i,
                    "target": int(j),
                    "rank": rank,
                    "similarity": float(row[j]),
                }
            )
    records = []
    for pair in sorted(pose | temporal | set(retrieval)):
        records.append(
            {
                "pair": list(pair),
                "sources": [
                    name
                    for name, group in (
                        ("pose", pose),
                        ("temporal", temporal),
                        ("dino", retrieval),
                    )
                    if pair in group
                ],
                "dino_retrieval": retrieval.get(pair, []),
            }
        )
    return records, {
        "strategy": "pose_union_dino_union_temporal",
        "num_images": n,
        "dino_candidates": int(dino_topk),
        "num_pairs": len(records),
        "pose_pairs": len(pose),
        "temporal_pairs": len(temporal),
        "dino_pairs": len(retrieval),
        "sift_intersection_pairs": len(pose | temporal),
        "prior_only_pairs": len(set(retrieval) - pose - temporal),
        **_graph_stats([record["pair"] for record in records], n),
    }


def annotate_loma_pairs_with_sift(pair_records, sift_pairs, sift_schedule):
    executed = {
        tuple(sorted(map(int, pair))) for pair in np.asarray(sift_pairs).reshape(-1, 2)
    }
    records = {tuple(record["pair"]): record for record in sift_schedule["pairs"]}
    if set(records) != executed:
        raise ValueError("SIFT audit must cover exactly the executed candidate pairs")
    config = sift_schedule["config"]
    annotated = []
    for pair_record in pair_records:
        result = dict(pair_record)
        pair = tuple(result["pair"])
        record = records.get(pair)
        if pair not in executed:
            result.update(
                sift_support="untried", sift=None, sift_insufficient_reasons=[]
            )
        else:
            reasons = []
            if record["inlier_count"] < config["min_pair_inliers"]:
                reasons.append("low_inlier_count")
            for side in ("source", "target"):
                if record[f"{side}_grid_coverage"] < config["min_grid_coverage"]:
                    reasons.append(f"low_{side}_coverage")
            result.update(
                sift_support="insufficient" if reasons else "sufficient",
                sift=dict(record),
                sift_insufficient_reasons=reasons,
            )
        annotated.append(result)
    return annotated


def select_loma_pairs(
    pair_records,
    retrieval_matrix,
    *,
    mode="sift_guided",
    sufficient_neighbors=3,
    insufficient_neighbors=5,
    untried_neighbors=5,
    temporal_window=2,
):
    """Select from the annotated pool, retaining either endpoint's choice.

    Temporal edges do not consume quotas. Each image selects independently;
    incoming choices never consume its outgoing quota. Temporal separation is
    a soft diversity preference, not an overlap or geometric validity test.
    """
    quotas = {
        "sufficient": sufficient_neighbors,
        "insufficient": insufficient_neighbors,
        "untried": untried_neighbors,
    }
    if mode not in {"all", "sift_guided"}:
        raise ValueError(f"Unsupported LoMa pair selection: {mode}")
    if any(value < 0 for value in quotas.values()) or temporal_window < 0:
        raise ValueError("LoMa neighbor counts and temporal window must be nonnegative")
    similarity = np.asarray(retrieval_matrix, dtype=np.float64)
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("LoMa selection requires a square DINO retrieval matrix")
    n = len(similarity)
    records = [
        {**record, "selected": False, "executed": False, "selection": []}
        for record in pair_records
    ]
    adjacency = [[] for _ in range(n)]
    seen = set()
    for index, record in enumerate(records):
        i, j = record["pair"]
        if not 0 <= i < j < n or (i, j) in seen:
            raise ValueError("LoMa selection requires unique canonical candidate pairs")
        seen.add((i, j))
        if record["sift_support"] not in quotas:
            raise ValueError(f"Unknown SIFT support: {record['sift_support']}")
        adjacency[i].append((j, index))
        adjacency[j].append((i, index))
        if mode == "all" or "temporal" in record["sources"]:
            record["selected"] = True
            record["selection"].append(
                {"reason": "all_candidates" if mode == "all" else "temporal"}
            )

    directed_counts = []
    for i, neighbors in enumerate(adjacency):
        counts = dict.fromkeys(quotas, 0)
        selected_neighbors = [
            j for j, idx in neighbors if "temporal" in records[idx]["sources"]
        ]

        def rank_key(candidate, support):
            j, idx = candidate
            sift = records[idx]["sift"]
            coverage = (
                min(sift["source_grid_coverage"], sift["target_grid_coverage"])
                if sift
                else 0.0
            )
            inliers = sift["inlier_count"] if sift else 0
            sim = float(similarity[i, j])
            sim = sim if np.isfinite(sim) else -np.inf
            # Reliable edges prioritize verified SIFT support. Weak/untried
            # edges prioritize retrieval plausibility, never the lowest count.
            return (
                (-coverage, -inliers, -sim, j)
                if support == "sufficient"
                else (-sim, -coverage, -inliers, j)
            )

        if mode == "sift_guided":
            for support, limit in quotas.items():
                candidates = sorted(
                    [
                        (j, idx)
                        for j, idx in neighbors
                        if "temporal" not in records[idx]["sources"]
                        and records[idx]["sift_support"] == support
                    ],
                    key=lambda item: rank_key(item, support),
                )
                for rank in range(1, min(limit, len(candidates)) + 1):
                    diverse = [
                        item
                        for item in candidates
                        if all(
                            abs(item[0] - other) > temporal_window
                            for other in selected_neighbors
                        )
                    ]
                    j, idx = (diverse or candidates)[0]
                    candidates.remove((j, idx))
                    selected_neighbors.append(j)
                    counts[support] += 1
                    sim = float(similarity[i, j])
                    records[idx]["selected"] = True
                    records[idx]["selection"].append(
                        {
                            "reason": support,
                            "source": i,
                            "target": j,
                            "rank": rank,
                            "dino_similarity": sim if np.isfinite(sim) else None,
                            "temporally_separated": bool(diverse),
                        }
                    )
        directed_counts.append(counts)

    selected = [record for record in records if record["selected"]]
    for record in records:
        if not record["selected"]:
            record["selection"].append({"reason": "not_selected_within_class_quota"})
    stats = {
        "mode": mode,
        "quotas_per_image": quotas,
        "temporal_window": temporal_window,
        "ranking": "sufficient: coverage/inliers/DINO; others: DINO/coverage/inliers; image ID tie-break",
        "diversity": "prefer neighbors farther than temporal_window from earlier choices; soft fallback",
        "num_candidate_pairs": len(records),
        "num_selected_pairs": len(selected),
        "num_skipped_pairs": len(records) - len(selected),
        "temporal_pairs": sum("temporal" in record["sources"] for record in records),
        "selected_by_sift_support": dict(
            Counter(record["sift_support"] for record in selected)
        ),
        "directed_choices_per_image": directed_counts,
        "graph": _graph_stats([record["pair"] for record in selected], n),
    }
    return records, stats


def normalized_to_work_pixels(keypoints, height, width):
    """LoMa align_corners=False coordinates -> COLMAP pixel-center convention.

    The center of the upper-left pixel is (0.5, 0.5). No -0.5 shift, snapping,
    rounding or clamping: the official path API uses this same inverse mapping.
    """
    points = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError("LoMa returned non-finite keypoints")
    return (points + 1) * np.array([width, height], dtype=np.float32) / 2


class LoMaBackend:
    """Native LoMa inference with optional batching and CPU/CUDA feature cache."""

    def __init__(self, device, feature_cache="cpu"):
        import torch

        requested = torch.device(device)
        if requested.type == "cuda":
            if requested.index is None:
                requested = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(requested)
            torch.cuda.reset_peak_memory_stats(requested)
        source = Path(__file__).resolve().parents[1] / "third_party" / "LoMa" / "src"
        if not (source / "loma" / "loma.py").is_file():
            raise FileNotFoundError(f"Vendored LoMa missing: {source}")
        existing = sys.modules.get("loma")
        if existing is not None and not Path(
            existing.__file__
        ).resolve().is_relative_to(source.resolve()):
            raise RuntimeError(
                "Another loma package is already imported; use the vendored LoMa in a fresh process"
            )
        sys.path.insert(0, str(source))
        module = importlib.import_module("loma.loma")
        runtime = importlib.import_module("loma.device")
        # Upstream uses module-global devices, so .to(cpu) alone is insufficient.
        if runtime.device.type != requested.type:
            raise RuntimeError(
                f"Vendored LoMa selected {runtime.device}; requested {device}. "
                "Run on its selected accelerator or in a CPU-only runtime."
            )
        self.torch = torch
        self.device = requested
        self.feature_cache = feature_cache
        self.phase_seconds = Counter()
        self._events = []
        self.extraction_stats = {}
        self.model = module.LoMa(module.LoMaB()).eval().to(requested)
        self.filter_matches = module.filter_matches
        self.metadata = {
            "architecture": "LoMa-B",
            "source": str(Path(module.__file__).resolve()),
            "weights_url": self.model.cfg.weights_url,
            "num_keypoints": self.model.cfg.num_keypoints,
            "filter_threshold": self.model.cfg.filter_threshold,
            "descriptor": self.model.cfg.descriptor,
            "compile": self.model.cfg.compile,
            "torch_version": torch.__version__,
            "device": str(requested),
            "cache": f"{feature_cache}, native descriptor dtype, per-run only",
            "preprocessing": "native path API: DaD resize and DeDoDe 784x784; original work coordinates",
        }

    @contextmanager
    def _measure(self, name):
        if self.device.type == "cuda":
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            yield
            end.record()
            self._events.append((name, start, end))
        else:
            start = time.perf_counter()
            yield
            self.phase_seconds[name] += time.perf_counter() - start

    def collect_events(self):
        # Results are already copied to CPU at batch boundaries. No global
        # synchronization between detector/descriptor/matcher kernels.
        for name, start, end in self._events:
            end.synchronize()
            self.phase_seconds[name] += start.elapsed_time(end) / 1000
        self._events.clear()

    def _upload(self, tensors):
        tensor = self.torch.cat(tensors, dim=0)
        if tensor.device.type == "cpu" and self.device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.device.type == "cuda")

    def _feature(self, points, descriptions, height, width):
        with self._measure("d2h"):
            cpu_points = points.detach().cpu()
            cached_descriptions = (
                descriptions.detach().cpu()
                if self.feature_cache == "cpu"
                else descriptions.detach()
            )
        return {
            "keypoints": normalized_to_work_pixels(
                cpu_points[0].numpy(), height, width
            ),
            "normalized": cpu_points
            if self.feature_cache == "cpu"
            else points.detach(),
            "descriptors": cached_descriptions,
            "image_size_hw": (height, width),
        }

    def close(self):
        self._events.clear()
        self.model = None

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def memory_stats(self):
        if self.device.type != "cuda":
            return {"cuda_peak_bytes": None}
        return {
            "cuda_allocated_bytes": self.torch.cuda.memory_allocated(self.device),
            "cuda_reserved_bytes": self.torch.cuda.memory_reserved(self.device),
            "cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(
                self.device
            ),
            "cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(
                self.device
            ),
        }

    def reset_memory_peak(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def extract(self, path):
        start = time.perf_counter()
        with self.torch.inference_mode():
            points, descriptions, height, width = self.model.detect_and_describe(
                str(path)
            )
        feature = self._feature(points, descriptions, height, width)
        self.collect_events()
        self.phase_seconds["native_extract_wall"] += time.perf_counter() - start
        return feature

    def extract_batched(self, paths, batch_size, workers):
        from utils.loma_execution import (
            bucket_batches,
            image_shape,
            prepare_image,
            prefetch_batches,
        )

        detector = self.model._detector
        shape = partial(
            image_shape,
            resize=detector.resize,
            keep_aspect_ratio=detector.keep_aspect_ratio,
        )
        prepare = partial(
            prepare_image,
            resize=detector.resize,
            keep_aspect_ratio=detector.keep_aspect_ratio,
        )
        shapes = [shape(path) for path in paths]
        batches = bucket_batches(range(len(paths)), lambda i: shapes[i], batch_size)
        stats = {"mode": "batched", "batch_size_histogram": {}, "buckets": []}
        self.extraction_stats = stats
        histogram = Counter()
        buckets = Counter()
        bucket_images = Counter()
        iterator = prefetch_batches(
            batches, lambda i: prepare(paths[i]), workers, stats
        )
        try:
            for prepared in iterator:
                ids = [i for i, _ in prepared]
                try:
                    with self.torch.inference_mode():
                        self.phase_seconds["preprocess_tasks"] += sum(
                            data["seconds"] for _, data in prepared
                        )
                        with self._measure("h2d"):
                            detector_images = self._upload(
                                [
                                    self.torch.from_numpy(data["detector"])[None]
                                    for _, data in prepared
                                ]
                            )
                            descriptor_images = self._upload(
                                [
                                    self.torch.from_numpy(data["descriptor"])[None]
                                    for _, data in prepared
                                ]
                            )
                        with self._measure("detector"):
                            points = detector.detect(
                                {"image": detector_images},
                                num_keypoints=self.model.cfg.num_keypoints,
                            )["keypoints"]
                        with self._measure("descriptor"):
                            descriptions = self.model._descriptor.describe_keypoints(
                                descriptor_images, points
                            )["descriptions"]
                        features = [
                            self._feature(
                                points[k : k + 1],
                                descriptions[k : k + 1],
                                *data["image_size_hw"],
                            )
                            for k, (_, data) in enumerate(prepared)
                        ]
                    self.collect_events()
                    histogram[len(ids)] += 1
                    buckets[shapes[ids[0]]] += 1
                    bucket_images[shapes[ids[0]]] += len(ids)
                    del (
                        detector_images,
                        descriptor_images,
                        points,
                        descriptions,
                        prepared,
                    )
                    for i, feature in zip(ids, features, strict=True):
                        yield i, feature
                except Exception as exc:
                    raise RuntimeError(
                        f"LoMa extraction batch images={ids}, batch={len(ids)}, "
                        f"completed_images={sum(size * count for size, count in histogram.items())}, "
                        f"cache={self.feature_cache}, memory={self.memory_stats()}: {exc}"
                    ) from exc
        finally:
            iterator.close()
            stats["batch_size_histogram"] = dict(histogram)
            stats["buckets"] = [
                {
                    "detector_size_wh": list(size),
                    "images": bucket_images[size],
                    "forward_calls": count,
                }
                for size, count in buckets.items()
            ]

    def match_batch(self, pairs):
        """Match one homogeneous shape bucket; return one output per input pair."""
        if not pairs:
            return []
        if not len(pairs[0][0]["keypoints"]) or not len(pairs[0][1]["keypoints"]):
            return [
                (np.empty((0, 2), dtype=np.uint32), np.empty(0, dtype=np.float32))
                for _ in pairs
            ]
        with self.torch.inference_mode():
            with self._measure("h2d_or_cache_gather"):
                inputs = [
                    self._upload([pair[side][field] for pair in pairs])
                    for field in ("normalized", "descriptors")
                    for side in (0, 1)
                ]
            with self._measure("matcher"):
                result = self.model(*inputs)
                indices, _, scores, _ = self.filter_matches(
                    result["scores"], self.model.cfg.filter_threshold
                )
            # Transfer O(B*N) outputs, not O(B*N*N) assignment matrices.
            with self._measure("d2h"):
                indices = indices.cpu().numpy()
                scores = scores.float().cpu().numpy()
        self.collect_events()
        outputs = []
        for ids, values in zip(indices, scores, strict=True):
            valid = ids >= 0
            outputs.append(
                (
                    np.column_stack((np.flatnonzero(valid), ids[valid])).astype(
                        np.uint32
                    ),
                    values[valid],
                )
            )
        return outputs

    def match(self, feature0, feature1):
        if not len(feature0["keypoints"]) or not len(feature1["keypoints"]):
            return np.empty((0, 2), dtype=np.uint32), np.empty(0, dtype=np.float32)
        with self.torch.inference_mode():
            with self._measure("h2d_or_cache_gather"):
                inputs = (
                    feature0["normalized"].to(self.device),
                    feature1["normalized"].to(self.device),
                    feature0["descriptors"].to(self.device),
                    feature1["descriptors"].to(self.device),
                )
            with self._measure("matcher"):
                result = self.model(*inputs)
                indices, _, scores, _ = self.filter_matches(
                    result["scores"], self.model.cfg.filter_threshold
                )
                valid = indices[0] >= 0
                matches = self.torch.stack(
                    (self.torch.where(valid)[0], indices[0][valid]), dim=-1
                )
            with self._measure("d2h"):
                output = (
                    matches.cpu().numpy().astype(np.uint32),
                    scores[0][valid].float().cpu().numpy(),
                )
        self.collect_events()
        return output


def _camera(pycolmap, intrinsic, image_size_hw, camera_id=1):
    h, w = image_size_hw
    matrix = np.asarray(intrinsic)
    return pycolmap.Camera(
        camera_id=camera_id,
        model="SIMPLE_PINHOLE",
        width=w,
        height=h,
        params=[
            float((matrix[0, 0] + matrix[1, 1]) / 2),
            float(matrix[0, 2]),
            float(matrix[1, 2]),
        ],
    )


@dataclass
class LoMaPriorResult:
    keypoints: list
    geometries: dict
    pair_records: list
    stats: dict

    def observation_counts(self):
        masks = [np.zeros(len(points), dtype=bool) for points in self.keypoints]
        for (i, j), geometry in self.geometries.items():
            matches = np.asarray(geometry.inlier_matches)
            if len(matches):
                masks[i][matches[:, 0]] = True
                masks[j][matches[:, 1]] = True
        return np.asarray([int(mask.sum()) for mask in masks], dtype=np.int64)

    def subset(self, kept_indices):
        mapping = {int(old): new for new, old in enumerate(kept_indices)}
        geometries = {
            (mapping[i], mapping[j]): geometry
            for (i, j), geometry in self.geometries.items()
            if i in mapping and j in mapping
        }
        records = []
        for record in self.pair_records:
            i, j = record["pair"]
            if i in mapping and j in mapping:
                records.append(
                    {
                        **record,
                        "original_pair": [i, j],
                        "pair": [mapping[i], mapping[j]],
                    }
                )
        return LoMaPriorResult(
            [self.keypoints[i] for i in kept_indices], geometries, records, self.stats
        )


def run_loma_prior(
    image_paths,
    image_size_hw,
    intrinsics,
    pair_records,
    device="cuda",
    backend=None,
    *,
    match_batch_size=1,
    extract_batch_size=1,
    preprocess_workers=0,
    geometry_workers=1,
    feature_cache="cpu",
):
    from utils.loma_execution import validate_execution

    validate_execution(
        device,
        match_batch_size,
        extract_batch_size,
        preprocess_workers,
        geometry_workers,
        feature_cache,
    )
    start = time.perf_counter()
    owned = backend is None
    if owned:
        try:
            backend = LoMaBackend(device, feature_cache=feature_cache)
        except Exception as exc:
            raise RuntimeError(
                f"LoMa model loading failed on {device}, cache={feature_cache}: {exc}"
            ) from exc
    try:
        return _run_loma_prior(
            image_paths,
            image_size_hw,
            intrinsics,
            pair_records,
            backend,
            start,
            match_batch_size,
            extract_batch_size,
            preprocess_workers,
            geometry_workers,
            feature_cache,
        )
    finally:
        if owned:
            backend.close()


def _verify_pair(pair, points0, points1, matches, intrinsic0, intrinsic1, size):
    import pycolmap

    start = time.perf_counter()
    try:
        geometry = pycolmap.TwoViewGeometry()
        if len(matches):
            options = pycolmap.TwoViewGeometryOptions()
            options.ransac.num_threads = 1
            geometry = pycolmap.estimate_two_view_geometry(
                _camera(pycolmap, intrinsic0, size),
                points0,
                _camera(pycolmap, intrinsic1, size),
                points1,
                matches,
                options,
            )
        return geometry, time.perf_counter() - start
    except Exception as exc:
        raise RuntimeError(f"LoMa pair {pair} verification failed: {exc}") from exc


def _run_loma_prior(
    image_paths,
    image_size_hw,
    intrinsics,
    pair_records,
    backend,
    start,
    match_batch_size,
    extract_batch_size,
    preprocess_workers,
    geometry_workers,
    feature_cache,
):
    import pycolmap
    from utils.loma_execution import GeometryQueue, bucket_batches

    backend.synchronize()
    timing = {"model_load": time.perf_counter() - start}

    def memory_snapshot():
        snapshot = backend.memory_stats() if hasattr(backend, "memory_stats") else {}
        if hasattr(backend, "reset_memory_peak"):
            backend.reset_memory_peak()
        return snapshot

    memory = {"model_load": memory_snapshot()}
    print(
        f"[LOMA-PRIOR] Execution: match_batch={match_batch_size}, "
        f"extract_batch={extract_batch_size}, preprocess_workers={preprocess_workers}, "
        f"geometry_workers={geometry_workers}, feature_cache={feature_cache}",
        flush=True,
    )
    t0 = time.perf_counter()
    cache = [None] * len(image_paths)
    if len(intrinsics) != len(cache):
        raise ValueError("LoMa intrinsics/image count mismatch")
    extraction = (
        backend.extract_batched(image_paths, extract_batch_size, preprocess_workers)
        if extract_batch_size > 1 or preprocess_workers > 0
        else ((i, backend.extract(path)) for i, path in enumerate(image_paths))
    )
    try:
        for position, (i, feature) in enumerate(extraction, start=1):
            if not 0 <= i < len(cache) or cache[i] is not None:
                raise ValueError(
                    f"LoMa extraction returned duplicate/invalid image index {i}"
                )
            if tuple(feature["image_size_hw"]) != tuple(image_size_hw):
                raise ValueError(
                    f"LoMa work image size differs from geometry: {image_paths[i]}"
                )
            cache[i] = feature
            if position % 100 == 0 or position == len(image_paths):
                print(
                    f"[LOMA-PRIOR] Extracted {position}/{len(image_paths)} images",
                    flush=True,
                )
    finally:
        extraction.close()
    if any(feature is None for feature in cache):
        raise ValueError("LoMa extraction did not return every image")
    backend.synchronize()
    timing["feature_extraction"] = time.perf_counter() - t0
    memory["feature_extraction"] = memory_snapshot()
    cache_bytes = sum(
        t.numel() * t.element_size()
        for feature in cache
        for name in ("normalized", "descriptors")
        if (t := feature.get(name)) is not None
    )
    timing.update(matching=0.0, geometric_verification=0.0, schema_version=2)
    options = pycolmap.TwoViewGeometryOptions()
    options.ransac.num_threads = 1
    geometries = {}
    records = [dict(record) for record in pair_records]
    # Reject invalid/duplicate input before dispatch; restoration assumes one job per pair.
    canonical = _pairs([record["pair"] for record in records], len(cache))
    if len(canonical) != len(records) or any(
        tuple(record["pair"]) not in canonical for record in records
    ):
        raise ValueError("LoMa execution requires unique canonical pairs")

    def consume(position, output):
        geometry, seconds = output
        timing["geometric_verification"] += seconds
        inliers = len(geometry.inlier_matches)
        if inliers:
            geometries[tuple(records[position]["pair"])] = geometry
        records[position].update(
            executed=True, inlier_matches=inliers, geometry_config=int(geometry.config)
        )

    def shape_key(position):
        pair = records[position]["pair"]
        return tuple(
            (
                len(cache[i]["keypoints"]),
                tuple(cache[i]["descriptors"].shape[2:])
                if "descriptors" in cache[i]
                else (),
                str(cache[i]["descriptors"].dtype)
                if "descriptors" in cache[i]
                else "synthetic",
            )
            for i in pair
        )

    batches = (
        bucket_batches(range(len(records)), shape_key, match_batch_size)
        if match_batch_size > 1
        else ([i] for i in range(len(records)))
    )
    batch_histogram = Counter()
    bucket_stats = {}
    completed = 0
    match_start = time.perf_counter()
    with GeometryQueue(
        geometry_workers, max(2 * match_batch_size, geometry_workers), consume
    ) as queue:
        for batch in batches:
            queue.collect()
            t0 = time.perf_counter()
            inputs = [
                (cache[records[p]["pair"][0]], cache[records[p]["pair"][1]])
                for p in batch
            ]
            try:
                outputs = (
                    backend.match_batch(inputs)
                    if match_batch_size > 1
                    else [backend.match(*inputs[0])]
                )
                backend.synchronize()
            except Exception as exc:
                raise RuntimeError(
                    f"LoMa matching pairs={[records[p]['pair'] for p in batch]}, batch={len(batch)}, completed_pairs={completed}/{len(records)}, cache={feature_cache}, cache_bytes={cache_bytes}: {exc}"
                ) from exc
            timing["matching"] += time.perf_counter() - t0
            if len(outputs) != len(batch):
                raise ValueError(
                    "LoMa batch returned a different number of pair results"
                )
            batch_histogram[len(batch)] += 1
            key = str(shape_key(batch[0]))
            bucket = bucket_stats.setdefault(
                key, {"pairs": 0, "calls": 0, "batch_size_histogram": {}}
            )
            bucket["pairs"] += len(batch)
            bucket["calls"] += 1
            bucket["batch_size_histogram"][len(batch)] = (
                bucket["batch_size_histogram"].get(len(batch), 0) + 1
            )
            for position, (matches, scores) in zip(batch, outputs, strict=True):
                i, j = records[position]["pair"]
                matches = np.asarray(matches, dtype=np.uint32).reshape(-1, 2)
                if len(scores) != len(matches):
                    raise ValueError(f"LoMa score/match count mismatch for {(i, j)}")
                if len(matches) and (
                    matches[:, 0].max() >= len(cache[i]["keypoints"])
                    or matches[:, 1].max() >= len(cache[j]["keypoints"])
                ):
                    raise ValueError(
                        f"LoMa returned invalid feature indices for {(i, j)}"
                    )
                records[position].update(
                    raw_matches=len(matches),
                    match_score_mean=float(np.mean(scores)) if len(scores) else None,
                )
                queue.submit(
                    position,
                    _verify_pair,
                    (i, j),
                    cache[i]["keypoints"],
                    cache[j]["keypoints"],
                    matches,
                    intrinsics[i],
                    intrinsics[j],
                    image_size_hw,
                )
            previous = completed
            completed += len(batch)
            if completed // 100 != previous // 100 or completed == len(records):
                print(
                    f"[LOMA-PRIOR] Matched {completed}/{len(records)} pairs", flush=True
                )
    timing["match_and_verify_wall"] = time.perf_counter() - match_start
    timing["geometry_queue_wait"] = queue.stats["queue_wait_seconds"]
    timing["semantics"] = {
        "matching": "sum of batch wall durations including transfers",
        "geometric_verification": "sum of CPU task wall durations; overlaps across workers and with matching",
        "match_and_verify_wall": "elapsed pipeline wall duration; do not add overlapping task durations",
        "backend_phases": "CUDA event seconds on GPU; CPU wall seconds otherwise. preprocess_tasks sums worker durations. native_extract_wall includes native preprocessing and inference without a kernel breakdown",
    }
    timing["backend_phases"] = dict(getattr(backend, "phase_seconds", {}))
    geometries = dict(sorted(geometries.items()))
    stats = {
        "provider": "loma",
        "model": backend.metadata,
        "pycolmap_version": pycolmap.__version__,
        "geometry_options": options.todict(),
        "num_pairs": len(records),
        "num_verified_pairs": len(geometries),
        "verified_graph": _graph_stats(geometries, len(cache)),
        "num_keypoints": [len(feature["keypoints"]) for feature in cache],
        "timing": timing,
        "execution": {
            "match_batch_size": match_batch_size,
            "extract_batch_size": extract_batch_size,
            "preprocess_workers": preprocess_workers,
            "geometry_workers": geometry_workers,
            "feature_cache": feature_cache,
            "feature_cache_bytes": cache_bytes,
            "matching_batch_histogram": dict(batch_histogram),
            "matching_buckets": bucket_stats,
            "extraction": getattr(backend, "extraction_stats", {})
            or {"mode": "native_path", "forward_calls": len(cache)},
            "geometry": queue.stats,
        },
        "track_assembly": "existing pycolmap triangulation; no custom union or pruning",
    }
    result = LoMaPriorResult(
        [feature["keypoints"] for feature in cache], geometries, records, stats
    )
    stats["observations_per_image"] = result.observation_counts().tolist()
    stats["unique_matched_observations"] = sum(stats["observations_per_image"])
    stats["by_sift_support"] = {}
    for support in ("sufficient", "insufficient", "untried"):
        subset_records = [r for r in records if r["sift_support"] == support]
        subset_geometry = {
            tuple(r["pair"]): geometries[tuple(r["pair"])]
            for r in subset_records
            if tuple(r["pair"]) in geometries
        }
        unique = LoMaPriorResult(
            result.keypoints, subset_geometry, [], {}
        ).observation_counts()
        stats["by_sift_support"][support] = {
            "pairs": len(subset_records),
            "verified_pairs": len(subset_geometry),
            "raw_matches": sum(r["raw_matches"] for r in subset_records),
            "inlier_matches": sum(r["inlier_matches"] for r in subset_records),
            "unique_observations": int(unique.sum()),
        }
    stats["support_counts_overlap"] = True
    memory["match_and_verify"] = memory_snapshot()
    stats["memory"] = {
        name: max(stage[name] for stage in memory.values() if name in stage)
        for name in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
        if any(name in stage for stage in memory.values())
    }
    stats["memory"]["stages"] = memory
    stats["memory"]["scope"] = (
        "process CUDA allocator; per-stage peaks reset at stage boundaries; includes other live allocations"
    )
    timing["total"] = time.perf_counter() - start
    return result


def write_loma_database(path, image_names, image_size_hw, intrinsic, result):
    import pycolmap

    path = Path(path)
    if len(image_names) != len(result.keypoints):
        raise ValueError("LoMa DB image/keypoint count mismatch")
    path.unlink(missing_ok=True)
    database = pycolmap.Database.open(str(path))
    try:
        database.write_camera(_camera(pycolmap, intrinsic, image_size_hw))
        for i, name in enumerate(image_names):
            database.write_image(
                pycolmap.Image(image_id=i + 1, camera_id=1, name=name),
                use_image_id=True,
            )
            database.write_keypoints(
                i + 1, np.asarray(result.keypoints[i], dtype=np.float32)
            )
        for (i, j), geometry in sorted(result.geometries.items()):
            database.write_matches(
                i + 1, j + 1, np.asarray(geometry.inlier_matches, dtype=np.uint32)
            )
            database.write_two_view_geometry(i + 1, j + 1, geometry)
    finally:
        database.close()
    return {
        "database": str(path),
        "num_pairs": len(result.geometries),
        "num_keypoints": [len(points) for points in result.keypoints],
        "unique_matched_observations": int(result.observation_counts().sum()),
        "snap": {"enabled": False},
        "feature_ids": "stable per-image indices",
    }


def summarize_final_tracks(reconstruction, sift_keypoint_counts, result):
    """Final learned track support by pair class; classes may overlap."""
    support_bits = {"sufficient": 1, "insufficient": 2, "untried": 4}
    membership = [np.zeros(len(points), dtype=np.uint8) for points in result.keypoints]
    for record in result.pair_records:
        pair = tuple(record["pair"])
        geometry = result.geometries.get(pair)
        if geometry is None:
            continue
        matches = np.asarray(geometry.inlier_matches)
        for column, image_idx in enumerate(pair):
            membership[image_idx][matches[:, column]] |= support_bits[
                record["sift_support"]
            ]
    lengths = Counter()
    by_support = {name: Counter() for name in ("sufficient", "insufficient", "untried")}
    for point in reconstruction.points3D.values():
        supports = 0
        learned = 0
        elements = list(point.track.elements)
        for element in elements:
            idx = int(element.point2D_idx) - sift_keypoint_counts[int(element.image_id)]
            if idx >= 0:
                learned += 1
                supports |= int(membership[int(element.image_id) - 1][idx])
        if learned:
            length = len(elements)
            lengths[length] += 1
            for support, bit in support_bits.items():
                if supports & bit:
                    by_support[support][length] += 1
    return {
        "track_length_histogram": dict(sorted(lengths.items())),
        "by_sift_support_length_histogram": {
            key: dict(sorted(value.items())) for key, value in by_support.items()
        },
        "support_classes_may_overlap": True,
        "attribution": "associated observations, not causal gain from each pair class",
    }
