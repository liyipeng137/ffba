"""Replay only refinement on frozen pipeline artifacts for track-selection A/B.

Run with PYTHONPATH=. in the GPU environment. A baseline source snapshot is
required for the two controls; only its two named functions are loaded. The
source run is read-only and every variant must use a fresh output directory.
"""

import argparse
import ast
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import torch

from utils import gluemap_refine_core as ref
from utils.gluemap_spv_refine import GluemapSpvRefineConfig, _make_refine_args


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_database(source, target):
    # PyCOLMAP opens databases for writing even in the triangulation path.
    # SQLite backup includes committed WAL content and isolates each replay.
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as source_db:
        with sqlite3.connect(target) as target_db:
            source_db.backup(target_db)


def restore_function(snapshot, name):
    tree = ast.parse(snapshot.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected one {name} in {snapshot}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(snapshot), "exec"),
         ref.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["baseline", "no_two_view", "all_sources"],
                        required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    cli = parser.parse_args()
    source = cli.source_run.resolve()
    output = cli.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ref._ensure_gluemap_imports()
    pycolmap = ref._lazy_import_pycolmap()
    config_json = json.loads((source / "pipeline_config.json").read_text())
    allowed = {field.name for field in fields(GluemapSpvRefineConfig)}
    config = GluemapSpvRefineConfig(**{
        k: v for k, v in config_json["args"].items() if k in allowed
    })
    args = _make_refine_args(config)
    if args.ba_backend != "bae":
        raise ValueError("Frozen replay currently supports real-only BAE")
    if cli.variant in {"baseline", "no_two_view"}:
        restore_function(cli.baseline_source, "run_select_tracks")
    if cli.variant == "baseline":
        restore_function(cli.baseline_source, "triangulate_from_seed_reconstruction")

    artifacts = np.load(source / "intrinsics_refine_inputs.npz")
    names = artifacts["image_names"].tolist()
    coarse = pycolmap.Reconstruction(str(source / "coarse"))
    by_name = {image.name: image for image in coarse.images.values()}
    extrinsic = np.stack([by_name[name].cam_from_world().matrix() for name in names])
    first_camera = coarse.cameras[by_name[names[0]].camera_id]
    image_size = (first_camera.height, first_camera.width)
    intrinsics = [torch.from_numpy(artifacts["shared_global_intrinsic"].copy()).double()[None]]
    mapping = {i: int(v) for i, v in enumerate(artifacts["intrinsics_mapping"])}
    for filename in ("database_sift.db", "database_merged.db"):
        copy_database(source / filename, output / filename)
    features = ref.load_database_keypoint_features(output / "database_sift.db", names)
    inputs = [source / "database_merged.db", source / "intrinsics_refine_inputs.npz",
              cli.baseline_source, Path(ref.__file__)]
    inputs.extend(sorted((source / "coarse").glob("*.bin")))
    manifest = {
        "variant": cli.variant, "source_run": str(source),
        "ignore_two_view_tracks": cli.variant != "baseline",
        "selection_scope": "all_sources" if cli.variant == "all_sources" else "sift_first",
        "input_sha256": {str(p): file_hash(p) for p in inputs},
        "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "pycolmap_version": pycolmap.__version__, "args": vars(args),
        "scope": "backend replay; no feature extraction or matching",
        "database_isolation": "per-replay SQLite backup; source databases are read-only",
    }
    (output / "replay_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "args"}), flush=True)
    start = time.perf_counter()
    reconstruction, _, stats = ref.run_merg3r_augmented_refinement_loop(
        args, pycolmap, output, names, image_size, "SIMPLE_PINHOLE", extrinsic,
        intrinsics, mapping, None, features, output / "database_merged.db",
    )
    elapsed = time.perf_counter() - start
    # The shared loop's production metadata describes the new policy; controls
    # replace the functions above and explicitly record their effective policy.
    for iteration in stats["iterations"]:
        iteration["triangulation"]["ignore_two_view_tracks"] = manifest["ignore_two_view_tracks"]
        iteration["select_tracks"]["selection_scope"] = manifest["selection_scope"]
    refined = output / "refined_gluemap_aba"
    refined.mkdir()
    reconstruction.write(str(refined))
    result = {"augmented_refinement": stats, "timing": {"augmented_refinement": elapsed},
              "output": {"refined_dir": str(refined)}}
    (output / "refine_stats.json").write_text(json.dumps(result, indent=2))
    (output / "exit_code").write_text("0\n")
    print(f"Replay complete: {cli.variant}, {elapsed:.2f}s", flush=True)


if __name__ == "__main__":
    main()
