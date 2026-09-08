"""One configuration source for the formal pipeline, without GPU imports."""

import argparse
from copy import deepcopy
from pathlib import Path
import math
from functools import lru_cache
from types import SimpleNamespace

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys instead of silently accepting the last value."""


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Configuration keys must be unique strings: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def _read(path):
    data = yaml.load(Path(path).read_text(), Loader=UniqueKeyLoader)
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    return data


def _merge(base, override, prefix=""):
    for key, value in override.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown configuration field: {path}")
        default = base[key]
        if isinstance(default, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be a mapping")
            _merge(default, value, path)
        else:
            if value is None and path == "bae.final_huber_delta":
                base[key] = value
                continue
            expected_type = FIELD_TYPES.get(path, type(default).__name__)
            valid = type(value).__name__ == expected_type
            if expected_type == "float":
                valid = type(value) in (float, int)
            if not valid:
                raise ValueError(f"Invalid type for {path}: expected {expected_type}")
            base[key] = value
    return base


def _lookup(config, path):
    value = config
    for key in path.split("."):
        value = value[key]
    return value


def flatten(config):
    values = {name: _lookup(config, path) for path, name in FIELD_MAP.items()}
    values.update(FIXED)
    values["input_order"] = config["pipeline"]["input_order"]
    values["mode"] = config["pipeline"]["mode"]
    values["prior_provider"] = {"standard": "vggsfm", "lite": "loma"}[values["mode"]]
    values["work_image_workers"] = values["image_pyramid_workers"]
    # Temporal settings are disabled by input semantics, not by DINO ordering.
    if values["input_order"] == "unordered":
        values["sift_temporal_window"] = 0
    return values


def validate(config):
    leaves = {}

    def walk(value, prefix=""):
        if isinstance(value, dict):
            for name, item in value.items():
                walk(item, f"{prefix}.{name}" if prefix else name)
        else:
            leaves[prefix] = value

    walk(config)
    expected = set(FIELD_MAP) | {"pipeline.mode", "pipeline.input_order"}
    if set(leaves) != expected:
        raise ValueError(
            f"Configuration schema mismatch: unknown={sorted(set(leaves) - expected)}, missing={sorted(expected - set(leaves))}"
        )
    for path, value in leaves.items():
        kind = FIELD_TYPES[path]
        if path == "bae.final_huber_delta" and value is None:
            continue
        valid = type(value).__name__ == kind
        if kind == "float":
            valid = type(value) in (int, float)
        if not valid or (type(value) is float and not math.isfinite(value)):
            raise ValueError(f"Invalid value/type for {path}: expected finite {kind}")
    choices = {
        "pipeline.mode": {"standard", "lite"},
        "pipeline.input_order": {"ordered", "unordered"},
        "prior.vggsfm.group_strategy": {"projected_overlap", "sift_pose_dino"},
        "prior.loma.feature_cache": {"cpu", "cuda"},
        "initialization.feedforward.intrinsics_method": {"moge", "lstsq"},
        "initialization.feedforward.splitting_type": {
            "interleave",
            "zigzag",
            "threshold",
            "original",
            "original_threshold",
        },
        "refinement.filter_reproj_error_type": {"angular", "pixel", "normalized"},
        "bae.fix_gauge": {"none", "two_cams", "three_points", "two_cams_full"},
        "bae.robust_loss": {"none", "huber"},
    }
    for path, allowed in choices.items():
        if _lookup(config, path) not in allowed:
            raise ValueError(f"{path} must be one of {sorted(allowed)}")
    values = flatten(config)
    positive = [
        "subsample",
        "stage1_downscale_n",
        "stage1_multiple",
        "prior_dino_long_side",
        "prior_dino_batch_size",
        "subset_size",
        "pair_k_pose",
        "sift_schedule_grid_size",
        "sift_schedule_min_inliers_per_cell",
        "neighbors_per_center",
        "vggsfm_group_batch_size",
        "vggsfm_max_center_gap",
        "projected_overlap_dino_candidates",
        "projected_overlap_samples",
        "projected_overlap_reproj_threshold",
        "vggsfm_query_points",
        "loma_dino_candidates",
        "loma_match_batch_size",
        "loma_extract_batch_size",
        "loma_geometry_workers",
        "num_refinement_iterations",
        "augmented_ba_max_filter_iterations",
        "augmented_ba_normalized_reproj_threshold",
        "bae_max_num_iterations",
        "bae_huber_delta",
        "image_pyramid_workers",
        "filter_reproj_error_threshold",
    ]
    for name in positive:
        if values[name] <= 0:
            raise ValueError(f"{name} must be positive")
    for name in [
        "stage2_scale_factor",
        "sift_temporal_window",
        "sift_schedule_min_pair_inliers",
        "loma_sufficient_neighbors",
        "loma_insufficient_neighbors",
        "loma_untried_neighbors",
        "loma_preprocess_workers",
        "select_track_min_support",
        "min_frame_observations",
    ]:
        if values[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in [
        "alpha",
        "sift_schedule_min_grid_coverage",
        "projected_overlap_conf_quantile",
        "vggsfm_vis_threshold",
        "vggsfm_score_threshold",
    ]:
        if not 0 <= values[name] <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0 <= values["overlap"] < values["subset_size"]:
        raise ValueError("overlap must be nonnegative and smaller than subset_size")
    if not 0 < values["pair_pose_rotation_threshold"] <= 180:
        raise ValueError("pose_rotation_threshold must be in (0, 180]")
    if (
        values["final_bae_huber_delta"] is not None
        and values["final_bae_huber_delta"] <= 0
    ):
        raise ValueError("final_huber_delta must be positive or null")
    if values["mode"] == "lite" and values["bae_max_observations"] > 0:
        raise ValueError(
            "LoMa lite has no observation cap; set bae.max_observations to 0"
        )
    return config


def load_config(path=None, *, mode=None, input_order=None):
    config = _read(DEFAULT_CONFIG)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        _merge(config, _read(path))
    if mode is not None:
        config["pipeline"]["mode"] = mode
    if input_order is not None:
        config["pipeline"]["input_order"] = input_order
    return validate(config)


@lru_cache(maxsize=1)
def default_values():
    return flatten(load_config())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SIFT + sparse VGGSfM / LoMa + BAE reconstruction"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prior_transforms_json")
    parser.add_argument("--input_order", choices=["ordered", "unordered"])
    parser.add_argument("--mode", choices=["standard", "lite"])
    cli = parser.parse_args(argv)
    try:
        config = load_config(cli.config, mode=cli.mode, input_order=cli.input_order)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    values = flatten(config)
    values.update(
        dataset=str(Path(cli.dataset).expanduser().resolve()),
        output_dir=str(Path(cli.output_dir).expanduser().resolve()),
        prior_transforms_json=(
            str(Path(cli.prior_transforms_json).expanduser().resolve())
            if cli.prior_transforms_json
            else None
        ),
        config_path=str(cli.config.resolve()),
        resolved_config=deepcopy(config),
    )
    return SimpleNamespace(**values)


def save_resolved_config(args, output_dir, *, group_resolution=None):
    config = deepcopy(args.resolved_config)
    config["run"] = {
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "prior_transforms_json": args.prior_transforms_json,
        "initial_geometry_source": "prior_pose"
        if args.prior_transforms_json
        else "feedforward",
        "effective_temporal_window": args.sift_temporal_window,
        "ba_backend": "bae",
        "ignore_two_view_tracks": True,
        "selection_scope": "all_sources",
        "loma_torch_threads": 8,
    }
    config["run"]["group_resolution"] = group_resolution
    import subprocess

    try:
        config["run"]["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=DEFAULT_CONFIG.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        config["run"]["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=DEFAULT_CONFIG.parent, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        config["run"]["git_commit"] = None
    Path(output_dir, "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )


FIELD_MAP = {
    "input.num_images": "num_images",
    "input.subsample": "subsample",
    "input.stage1_downscale_n": "stage1_downscale_n",
    "input.stage1_multiple": "stage1_multiple",
    "input.stage2_scale_factor": "stage2_scale_factor",
    "retrieval.long_side": "prior_dino_long_side",
    "retrieval.batch_size": "prior_dino_batch_size",
    "retrieval.alpha": "alpha",
    "initialization.feedforward.intrinsics_method": "pi3x_intrinsics_method",
    "initialization.feedforward.subset_size": "subset_size",
    "initialization.feedforward.overlap": "overlap",
    "initialization.feedforward.splitting_type": "splitting_type",
    "sift.pose_neighbors": "pair_k_pose",
    "sift.pose_rotation_threshold": "pair_pose_rotation_threshold",
    "sift.temporal_window": "sift_temporal_window",
    "sift.grid_size": "sift_schedule_grid_size",
    "sift.min_inliers_per_cell": "sift_schedule_min_inliers_per_cell",
    "sift.min_pair_inliers": "sift_schedule_min_pair_inliers",
    "sift.min_grid_coverage": "sift_schedule_min_grid_coverage",
    "prior.vggsfm.weights": "path_tracker",
    "prior.vggsfm.neighbors_per_center": "neighbors_per_center",
    "prior.vggsfm.group_strategy": "vggsfm_group_strategy",
    "prior.vggsfm.group_batch_size": "vggsfm_group_batch_size",
    "prior.vggsfm.max_center_gap": "vggsfm_max_center_gap",
    "prior.vggsfm.dino_candidates": "projected_overlap_dino_candidates",
    "prior.vggsfm.overlap_samples": "projected_overlap_samples",
    "prior.vggsfm.overlap_reproj_threshold": "projected_overlap_reproj_threshold",
    "prior.vggsfm.overlap_conf_quantile": "projected_overlap_conf_quantile",
    "prior.vggsfm.query_points": "vggsfm_query_points",
    "prior.vggsfm.aliked_detection_threshold": "aliked_detection_threshold",
    "prior.vggsfm.vis_threshold": "vggsfm_vis_threshold",
    "prior.vggsfm.score_threshold": "vggsfm_score_threshold",
    "prior.vggsfm.fine_tracking": "vggsfm_fine_tracking",
    "prior.vggsfm.snap_threshold": "prior_snap_threshold",
    "prior.vggsfm.keypoint_merge_threshold": "prior_keypoint_merge_threshold",
    "prior.loma.dino_candidates": "loma_dino_candidates",
    "prior.loma.sufficient_neighbors": "loma_sufficient_neighbors",
    "prior.loma.insufficient_neighbors": "loma_insufficient_neighbors",
    "prior.loma.untried_neighbors": "loma_untried_neighbors",
    "prior.loma.match_batch_size": "loma_match_batch_size",
    "prior.loma.extract_batch_size": "loma_extract_batch_size",
    "prior.loma.preprocess_workers": "loma_preprocess_workers",
    "prior.loma.geometry_workers": "loma_geometry_workers",
    "prior.loma.feature_cache": "loma_feature_cache",
    "refinement.min_frame_observations": "min_frame_observations",
    "refinement.iterations": "num_refinement_iterations",
    "refinement.post_ba_filter_iterations": "augmented_ba_max_filter_iterations",
    "refinement.post_ba_normalized_reproj_threshold": "augmented_ba_normalized_reproj_threshold",
    "refinement.tri_min_angle": "tri_min_angle",
    "refinement.tri_create_max_angle_error": "tri_create_max_angle_error",
    "refinement.select_track_min_support": "select_track_min_support",
    "refinement.filter_reproj_error_type": "filter_reproj_error_type",
    "refinement.filter_reproj_error_threshold": "filter_reproj_error_threshold",
    "bae.max_iterations": "bae_max_num_iterations",
    "bae.max_observations": "bae_max_observations",
    "bae.optimize_intrinsics": "bae_optimize_intrinsics",
    "bae.fix_gauge": "bae_fix_gauge",
    "bae.robust_loss": "bae_robust_loss",
    "bae.huber_delta": "bae_huber_delta",
    "bae.final_huber_delta": "final_bae_huber_delta",
    "runtime.image_workers": "image_pyramid_workers",
    "runtime.debug_print": "debug_print",
    "runtime.device": "device",
}

FIXED = {
    "image_pyramid": True,
    "sequence_type": "shortest_path",
    "alignment_type": "weighted_iterative",
    "prior_match_topology": "star",
    "ba_backend": "bae",
    "vggsfm_schedule_mode": "sift_first_sparse",
    "loma_pair_selection": "sift_guided",
    "multi_dirs": False,
}

FIELD_TYPES = {
    "input.num_images": "int",
    "input.subsample": "int",
    "input.stage1_downscale_n": "int",
    "input.stage1_multiple": "int",
    "input.stage2_scale_factor": "int",
    "retrieval.long_side": "int",
    "retrieval.batch_size": "int",
    "retrieval.alpha": "float",
    "initialization.feedforward.intrinsics_method": "str",
    "initialization.feedforward.subset_size": "int",
    "initialization.feedforward.overlap": "int",
    "initialization.feedforward.splitting_type": "str",
    "sift.pose_neighbors": "int",
    "sift.pose_rotation_threshold": "float",
    "sift.temporal_window": "int",
    "sift.grid_size": "int",
    "sift.min_inliers_per_cell": "int",
    "sift.min_pair_inliers": "int",
    "sift.min_grid_coverage": "float",
    "prior.vggsfm.weights": "str",
    "prior.vggsfm.neighbors_per_center": "int",
    "prior.vggsfm.group_strategy": "str",
    "prior.vggsfm.group_batch_size": "int",
    "prior.vggsfm.max_center_gap": "int",
    "prior.vggsfm.dino_candidates": "int",
    "prior.vggsfm.overlap_samples": "int",
    "prior.vggsfm.overlap_reproj_threshold": "float",
    "prior.vggsfm.overlap_conf_quantile": "float",
    "prior.vggsfm.query_points": "int",
    "prior.vggsfm.aliked_detection_threshold": "float",
    "prior.vggsfm.vis_threshold": "float",
    "prior.vggsfm.score_threshold": "float",
    "prior.vggsfm.fine_tracking": "bool",
    "prior.vggsfm.snap_threshold": "float",
    "prior.vggsfm.keypoint_merge_threshold": "float",
    "prior.loma.dino_candidates": "int",
    "prior.loma.sufficient_neighbors": "int",
    "prior.loma.insufficient_neighbors": "int",
    "prior.loma.untried_neighbors": "int",
    "prior.loma.match_batch_size": "int",
    "prior.loma.extract_batch_size": "int",
    "prior.loma.preprocess_workers": "int",
    "prior.loma.geometry_workers": "int",
    "prior.loma.feature_cache": "str",
    "refinement.min_frame_observations": "int",
    "refinement.iterations": "int",
    "refinement.post_ba_filter_iterations": "int",
    "refinement.post_ba_normalized_reproj_threshold": "float",
    "refinement.tri_min_angle": "float",
    "refinement.tri_create_max_angle_error": "float",
    "refinement.select_track_min_support": "int",
    "refinement.filter_reproj_error_type": "str",
    "refinement.filter_reproj_error_threshold": "float",
    "bae.max_iterations": "int",
    "bae.max_observations": "int",
    "bae.optimize_intrinsics": "bool",
    "bae.fix_gauge": "str",
    "bae.robust_loss": "str",
    "bae.huber_delta": "float",
    "bae.final_huber_delta": "float",
    "runtime.image_workers": "int",
    "runtime.debug_print": "bool",
    "runtime.device": "str",
    "pipeline.mode": "str",
    "pipeline.input_order": "str",
}
