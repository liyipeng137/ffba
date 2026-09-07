"""Indexed LoMa prior matches; selection is independent of SIFT success.

Torch and the vendored model are imported only by the inference adapter. Pair
construction, annotation, geometry verification and DB export also run on CPU.
"""

from collections import Counter
from dataclasses import dataclass
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
    """Native path preprocessing, once per image; descriptors cached on CPU."""

    def __init__(self, device):
        import torch

        requested = torch.device(device)
        if requested.type == "cuda":
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
            "cache": "CPU, native descriptor dtype, per-run only",
            "preprocessing": "native path API: DaD resize and DeDoDe 784x784; original work coordinates",
        }

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def memory_stats(self):
        if self.device.type != "cuda":
            return {"cuda_peak_bytes": None}
        return {
            "cuda_peak_allocated_bytes": self.torch.cuda.max_memory_allocated(
                self.device
            ),
            "cuda_peak_reserved_bytes": self.torch.cuda.max_memory_reserved(
                self.device
            ),
        }

    def extract(self, path):
        with self.torch.inference_mode():
            points, descriptions, height, width = self.model.detect_and_describe(
                str(path)
            )
        points = points.detach().cpu()
        return {
            "keypoints": normalized_to_work_pixels(points[0].numpy(), height, width),
            "normalized": points,
            "descriptors": descriptions.detach().cpu(),
            "image_size_hw": (height, width),
        }

    def match(self, feature0, feature1):
        if not len(feature0["keypoints"]) or not len(feature1["keypoints"]):
            return np.empty((0, 2), dtype=np.uint32), np.empty(0, dtype=np.float32)
        with self.torch.inference_mode():
            result = self.model(
                feature0["normalized"].to(self.device),
                feature1["normalized"].to(self.device),
                feature0["descriptors"].to(self.device),
                feature1["descriptors"].to(self.device),
            )
            indices, _, scores, _ = self.filter_matches(
                result["scores"], self.model.cfg.filter_threshold
            )
            valid = indices[0] >= 0
            matches = self.torch.stack(
                (self.torch.where(valid)[0], indices[0][valid]), dim=-1
            )
            return matches.cpu().numpy().astype(np.uint32), scores[0][
                valid
            ].float().cpu().numpy()


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
    image_paths, image_size_hw, intrinsics, pair_records, device="cuda", backend=None
):
    import pycolmap

    start = time.perf_counter()
    if backend is None:
        backend = LoMaBackend(device)
    backend.synchronize()
    timing = {"model_load": time.perf_counter() - start}
    t0 = time.perf_counter()
    cache = []
    for position, path in enumerate(image_paths, start=1):
        feature = backend.extract(path)
        if tuple(feature["image_size_hw"]) != tuple(image_size_hw):
            raise ValueError(f"LoMa work image size differs from geometry: {path}")
        cache.append(feature)
        if position % 100 == 0 or position == len(image_paths):
            print(
                f"[LOMA-PRIOR] Extracted {position}/{len(image_paths)} images",
                flush=True,
            )
    backend.synchronize()
    timing["feature_extraction"] = time.perf_counter() - t0
    timing.update(matching=0.0, geometric_verification=0.0)
    cameras = [
        _camera(pycolmap, intrinsic, image_size_hw, i + 1)
        for i, intrinsic in enumerate(intrinsics)
    ]
    if len(cameras) != len(cache):
        raise ValueError("LoMa intrinsics/image count mismatch")
    options = pycolmap.TwoViewGeometryOptions()
    geometries = {}
    records = []
    for position, record in enumerate(pair_records, start=1):
        i, j = record["pair"]
        t0 = time.perf_counter()
        matches, scores = backend.match(cache[i], cache[j])
        backend.synchronize()
        timing["matching"] += time.perf_counter() - t0
        matches = np.asarray(matches, dtype=np.uint32).reshape(-1, 2)
        if len(matches) and (
            matches[:, 0].max() >= len(cache[i]["keypoints"])
            or matches[:, 1].max() >= len(cache[j]["keypoints"])
        ):
            raise ValueError(f"LoMa returned invalid feature indices for {(i, j)}")
        t0 = time.perf_counter()
        geometry = pycolmap.TwoViewGeometry()
        if len(matches):
            geometry = pycolmap.estimate_two_view_geometry(
                cameras[i],
                cache[i]["keypoints"],
                cameras[j],
                cache[j]["keypoints"],
                matches,
                options,
            )
        timing["geometric_verification"] += time.perf_counter() - t0
        inliers = len(geometry.inlier_matches)
        if inliers:
            geometries[(i, j)] = geometry
        records.append(
            {
                **record,
                "executed": True,
                "raw_matches": len(matches),
                "inlier_matches": inliers,
                "geometry_config": int(geometry.config),
                "match_score_mean": float(np.mean(scores)) if len(scores) else None,
            }
        )
        if position % 100 == 0 or position == len(pair_records):
            print(
                f"[LOMA-PRIOR] Matched {position}/{len(pair_records)} pairs", flush=True
            )
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
    stats["memory"] = backend.memory_stats() if hasattr(backend, "memory_stats") else {}
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
