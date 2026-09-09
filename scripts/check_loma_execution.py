"""Small real-GPU LoMa comparison, independent of SIFT/BAE.

Run from the repository root with PYTHONPATH=.; reports numerical differences,
not an automatic geometry-quality acceptance decision. No persistent feature cache.
"""

import argparse
from functools import partial
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from utils.loma_execution import bucket_batches, prepare_image
from utils.loma_prior import LoMaBackend


def compare_matches(reference, current, maps=None):
    rows = []
    for pair in reference:
        before, before_scores = reference[pair]
        after, after_scores = current[pair]
        if maps is not None:
            after = np.column_stack(
                (maps[pair[0]][after[:, 0]], maps[pair[1]][after[:, 1]])
            )
        a = {
            tuple(row): float(score)
            for row, score in zip(before, before_scores, strict=True)
        }
        b = {
            tuple(row): float(score)
            for row, score in zip(after, after_scores, strict=True)
            if (row >= 0).all()
        }
        common = a.keys() & b.keys()
        rows.append(
            {
                "pair": list(pair),
                "reference_count": len(before),
                "current_count": len(after),
                "common": len(common),
                "lost": len(a) - len(common),
                "added_or_unmapped": len(after) - len(common),
                "max_common_score_delta": max(
                    (abs(a[k] - b[k]) for k in common), default=None
                ),
            }
        )
    return rows


def compare_features(reference, current, tolerance):
    from scipy.spatial import cKDTree

    mapping, rows = [], []
    for i, (a, b) in enumerate(zip(reference, current, strict=True)):
        x, y = a["keypoints"], b["keypoints"]
        remap = np.full(len(y), -1, dtype=np.int64)
        row = {
            "image": i,
            "reference_count": len(x),
            "current_count": len(y),
            "mutual_within_tolerance": 0,
        }
        if len(x) and len(y):
            distances, indices = cKDTree(x).query(y)
            reverse = cKDTree(y).query(x)[1]
            valid = (distances <= tolerance) & (reverse[indices] == np.arange(len(y)))
            remap[valid] = indices[valid]
            row.update(
                mutual_within_tolerance=int(valid.sum()),
                nearest_pixel_distance_median=float(np.median(distances)),
                nearest_pixel_distance_max=float(distances.max()),
            )
            if valid.any():
                left = a["descriptors"][0].float().cpu().numpy()[indices[valid]]
                right = b["descriptors"][0].float().cpu().numpy()[valid]
                row["aligned_descriptor_mean_abs_delta"] = float(
                    np.abs(left - right).mean()
                )
                row["aligned_descriptor_max_abs_delta"] = float(
                    np.abs(left - right).max()
                )
        mapping.append(remap)
        rows.append(row)
    return mapping, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images_dir",
        type=Path,
        required=True,
        help="Use pipeline work images for a comparable preprocessing check",
    )
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--match_batch_sizes", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--extract_batch_sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--preprocess_workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--keypoint_tolerance_px", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("loma_execution_check.json")
    )
    args = parser.parse_args()
    if (
        args.num_images < 2
        or args.repeats < 1
        or min(args.match_batch_sizes + args.extract_batch_sizes) < 1
        or args.preprocess_workers < 0
        or args.keypoint_tolerance_px <= 0
    ):
        parser.error(
            "Require >=2 images, positive batch sizes/repeats/tolerance and nonnegative workers"
        )
    if args.device.split(":")[0] != "cuda":
        parser.error("This smoke check requires CUDA for the cache comparison")
    paths = sorted(
        p
        for p in args.images_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and p.is_file()
    )[: args.num_images]
    if len(paths) < 2:
        parser.error("At least two images are required")
    report = {
        "status": "running",
        "settings": vars(args).copy(),
        "images": [],
        "matching": {},
        "extraction": {},
    }
    for path in paths:
        with Image.open(path) as image:
            report["images"].append(
                {
                    "path": str(path.resolve()),
                    "size_wh": list(image.size),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    pairs = list(itertools.combinations(range(len(paths)), 2))
    backend = None

    def measured(function):
        backend.synchronize()
        backend.reset_memory_peak()
        start = time.perf_counter()
        output = function()
        backend.synchronize()
        return output, {
            "wall_seconds": time.perf_counter() - start,
            **backend.memory_stats(),
        }

    def match(features, batch_size):
        outputs = {}
        batches = bucket_batches(
            pairs,
            lambda pair: tuple(
                (
                    len(features[i]["keypoints"]),
                    str(features[i]["descriptors"].dtype),
                    tuple(features[i]["descriptors"].shape[2:]),
                )
                for i in pair
            ),
            batch_size,
        )
        for batch in batches:
            inputs = [(features[i], features[j]) for i, j in batch]
            values = (
                [backend.match(*inputs[0])]
                if batch_size == 1
                else backend.match_batch(inputs)
            )
            outputs.update(zip(batch, values, strict=True))
        return outputs

    try:
        start = time.perf_counter()
        backend = LoMaBackend(args.device)
        backend.synchronize()
        report["model_load_seconds"] = time.perf_counter() - start
        report["model"] = backend.metadata
        print("Checking native/new input tensors...", flush=True)
        for path in paths:
            prepared = prepare_image(
                path,
                backend.model._detector.resize,
                backend.model._detector.keep_aspect_ratio,
            )
            np.testing.assert_array_equal(
                prepared["detector"],
                backend.model._detector.load_image(path)[0].cpu().numpy(),
            )
            np.testing.assert_array_equal(
                prepared["descriptor"],
                backend.model._descriptor.read_image(path)[0].cpu().numpy(),
            )
        report["preprocessing_inputs_exact"] = True
        del prepared
        print("Extracting one fixed native feature set...", flush=True)
        reference, report["native_extraction"] = measured(
            lambda: [backend.extract(path) for path in paths]
        )
        report["feature_layouts"] = [
            {
                name: {
                    "shape": list(feature[name].shape),
                    "dtype": str(feature[name].dtype),
                }
                for name in ("normalized", "descriptors")
            }
            for feature in reference
        ]
        # One CPU snapshot is held throughout M/C; no re-extraction between matcher tests.
        baseline, report["native_matching_cold"] = measured(lambda: match(reference, 1))
        for cache in ("cpu", "cuda"):
            features = (
                reference
                if cache == "cpu"
                else [
                    {
                        **f,
                        **{
                            name: f[name].to(backend.device)
                            for name in ("normalized", "descriptors")
                        },
                    }
                    for f in reference
                ]
            )
            for a, b in zip(reference, features, strict=True):
                for name in ("normalized", "descriptors"):
                    assert a[name].dtype == b[name].dtype
                    assert backend.torch.equal(a[name], b[name].cpu())
            for size in args.match_batch_sizes:
                print(f"Matching: cache={cache}, batch={size}", flush=True)
                _, cold = measured(partial(match, features, size))
                runs = []
                for _ in range(args.repeats):
                    output, timing = measured(partial(match, features, size))
                    runs.append({**timing, "pairs": compare_matches(baseline, output)})
                report["matching"][f"{cache}_batch_{size}"] = {
                    "first_call": cold,
                    "repeats": runs,
                }
            del features, a, b
        for size in args.extract_batch_sizes:
            print(
                f"Extraction: batch={size}, workers={args.preprocess_workers}",
                flush=True,
            )

            def extract():
                features = [None] * len(paths)
                for i, feature in backend.extract_batched(
                    paths, size, args.preprocess_workers
                ):
                    assert features[i] is None
                    features[i] = feature
                assert all(f is not None for f in features)
                return features

            features, timing = measured(extract)
            maps, differences = compare_features(
                reference, features, args.keypoint_tolerance_px
            )
            output, matching_time = measured(partial(match, features, 1))
            report["extraction"][str(size)] = {
                **timing,
                "features": differences,
                "remapped_matches": compare_matches(baseline, output, maps),
                "matching_time": matching_time,
                "execution": backend.extraction_stats.copy(),
            }
            del features
        report["status"] = "completed"
        report["interpretation"] = (
            "Numerical/timing report only; batches can change floating-point/top-k results. Compare remapped features and matches, then validate geometry and BAE on fixed pipeline inputs. Small-image timings do not establish 789-image speedup."
        )
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        if backend is not None:
            backend.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
