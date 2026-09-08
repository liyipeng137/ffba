from dataclasses import dataclass, field, fields
from pathlib import Path
from types import SimpleNamespace
from ffba.config import default_values
import numpy as np

CAMERA_MODEL = "SIMPLE_PINHOLE"
S_DATABASE_MODE = "sift"
QUERY_SOURCE = "aliked"
TRACKER_INPUT = "1024"


@dataclass
class GluemapSpvRefineConfig:
    path_tracker: str = field(default_factory=lambda: default_values()["path_tracker"])
    device: str = field(default_factory=lambda: default_values()["device"])
    neighbors_per_center: int = field(
        default_factory=lambda: default_values()["neighbors_per_center"]
    )
    pair_pose_rotation_threshold: float = field(
        default_factory=lambda: default_values()["pair_pose_rotation_threshold"]
    )
    vggsfm_group_strategy: str = field(
        default_factory=lambda: default_values()["vggsfm_group_strategy"]
    )
    vggsfm_group_batch_size: int = field(
        default_factory=lambda: default_values()["vggsfm_group_batch_size"]
    )
    projected_overlap_dino_candidates: int = field(
        default_factory=lambda: default_values()["projected_overlap_dino_candidates"]
    )
    projected_overlap_samples: int = field(
        default_factory=lambda: default_values()["projected_overlap_samples"]
    )
    projected_overlap_reproj_threshold: float = field(
        default_factory=lambda: default_values()["projected_overlap_reproj_threshold"]
    )
    projected_overlap_conf_quantile: float = field(
        default_factory=lambda: default_values()["projected_overlap_conf_quantile"]
    )
    vggsfm_schedule_mode: str = field(
        default_factory=lambda: default_values()["vggsfm_schedule_mode"]
    )
    sift_temporal_window: int = field(
        default_factory=lambda: default_values()["sift_temporal_window"]
    )
    sift_schedule_grid_size: int = field(
        default_factory=lambda: default_values()["sift_schedule_grid_size"]
    )
    sift_schedule_min_inliers_per_cell: int = field(
        default_factory=lambda: default_values()["sift_schedule_min_inliers_per_cell"]
    )
    sift_schedule_min_pair_inliers: int = field(
        default_factory=lambda: default_values()["sift_schedule_min_pair_inliers"]
    )
    sift_schedule_min_grid_coverage: float = field(
        default_factory=lambda: default_values()["sift_schedule_min_grid_coverage"]
    )
    vggsfm_max_center_gap: int = field(
        default_factory=lambda: default_values()["vggsfm_max_center_gap"]
    )
    vggsfm_query_points: int = field(
        default_factory=lambda: default_values()["vggsfm_query_points"]
    )
    aliked_detection_threshold: float = field(
        default_factory=lambda: default_values()["aliked_detection_threshold"]
    )
    vggsfm_vis_threshold: float = field(
        default_factory=lambda: default_values()["vggsfm_vis_threshold"]
    )
    vggsfm_score_threshold: float = field(
        default_factory=lambda: default_values()["vggsfm_score_threshold"]
    )
    vggsfm_fine_tracking: bool = field(
        default_factory=lambda: default_values()["vggsfm_fine_tracking"]
    )
    prior_snap_threshold: float = field(
        default_factory=lambda: default_values()["prior_snap_threshold"]
    )
    prior_keypoint_merge_threshold: float = field(
        default_factory=lambda: default_values()["prior_keypoint_merge_threshold"]
    )
    prior_match_topology: str = field(
        default_factory=lambda: default_values()["prior_match_topology"]
    )
    min_frame_observations: int = field(
        default_factory=lambda: default_values()["min_frame_observations"]
    )
    ba_backend: str = field(default_factory=lambda: default_values()["ba_backend"])
    bae_max_num_iterations: int = field(
        default_factory=lambda: default_values()["bae_max_num_iterations"]
    )
    bae_max_observations: int = field(
        default_factory=lambda: default_values()["bae_max_observations"]
    )
    bae_optimize_intrinsics: bool = field(
        default_factory=lambda: default_values()["bae_optimize_intrinsics"]
    )
    bae_fix_gauge: str = field(
        default_factory=lambda: default_values()["bae_fix_gauge"]
    )
    bae_robust_loss: str = field(
        default_factory=lambda: default_values()["bae_robust_loss"]
    )
    bae_huber_delta: float = field(
        default_factory=lambda: default_values()["bae_huber_delta"]
    )
    final_bae_huber_delta: float | None = field(
        default_factory=lambda: default_values()["final_bae_huber_delta"]
    )
    num_refinement_iterations: int = field(
        default_factory=lambda: default_values()["num_refinement_iterations"]
    )
    augmented_ba_max_filter_iterations: int = field(
        default_factory=lambda: default_values()["augmented_ba_max_filter_iterations"]
    )
    augmented_ba_normalized_reproj_threshold: float = field(
        default_factory=lambda: default_values()[
            "augmented_ba_normalized_reproj_threshold"
        ]
    )
    tri_min_angle: float = field(
        default_factory=lambda: default_values()["tri_min_angle"]
    )
    tri_create_max_angle_error: float = field(
        default_factory=lambda: default_values()["tri_create_max_angle_error"]
    )
    select_track_min_support: int = field(
        default_factory=lambda: default_values()["select_track_min_support"]
    )
    filter_reproj_error_type: str = field(
        default_factory=lambda: default_values()["filter_reproj_error_type"]
    )
    filter_reproj_error_threshold: float = field(
        default_factory=lambda: default_values()["filter_reproj_error_threshold"]
    )
    debug_print: bool = field(default_factory=lambda: default_values()["debug_print"])
    work_image_workers: int = field(
        default_factory=lambda: default_values()["work_image_workers"]
    )
    prior_provider: str = field(
        default_factory=lambda: default_values()["prior_provider"]
    )
    loma_dino_candidates: int = field(
        default_factory=lambda: default_values()["loma_dino_candidates"]
    )
    loma_pair_selection: str = field(
        default_factory=lambda: default_values()["loma_pair_selection"]
    )
    loma_sufficient_neighbors: int = field(
        default_factory=lambda: default_values()["loma_sufficient_neighbors"]
    )
    loma_insufficient_neighbors: int = field(
        default_factory=lambda: default_values()["loma_insufficient_neighbors"]
    )
    loma_untried_neighbors: int = field(
        default_factory=lambda: default_values()["loma_untried_neighbors"]
    )
    loma_match_batch_size: int = field(
        default_factory=lambda: default_values()["loma_match_batch_size"]
    )
    loma_extract_batch_size: int = field(
        default_factory=lambda: default_values()["loma_extract_batch_size"]
    )
    loma_preprocess_workers: int = field(
        default_factory=lambda: default_values()["loma_preprocess_workers"]
    )
    loma_geometry_workers: int = field(
        default_factory=lambda: default_values()["loma_geometry_workers"]
    )
    loma_feature_cache: str = field(
        default_factory=lambda: default_values()["loma_feature_cache"]
    )
    input_order: str = field(default_factory=lambda: default_values()["input_order"])

    @classmethod
    def from_namespace(cls, args):
        return cls(**{item.name: getattr(args, item.name) for item in fields(cls)})


@dataclass
class GluemapSpvRefineResult:
    image_names: list[str]
    extrinsic: np.ndarray
    pairs: np.ndarray
    intrinsic: np.ndarray
    intrinsics_mapping: dict[int, int]
    stats: dict
    refined_dir: Path
    virtual_refined_dir: Path | None


def _make_refine_args(config: GluemapSpvRefineConfig):
    if (
        config.prior_provider == "vggsfm"
        and config.vggsfm_schedule_mode != "sift_first_sparse"
    ):
        raise ValueError("The formal VGGSfM pipeline always uses sparse centers")
    if config.prior_provider not in {"vggsfm", "loma"}:
        raise ValueError(f"Unsupported prior provider: {config.prior_provider}")
    if config.prior_provider == "loma":
        from ffba.matching.loma_execution import validate_execution

        validate_execution(
            config.device,
            config.loma_match_batch_size,
            config.loma_extract_batch_size,
            config.loma_preprocess_workers,
            config.loma_geometry_workers,
            config.loma_feature_cache,
        )
        if config.ba_backend != "bae":
            raise ValueError("LoMa V1 requires --ba_backend bae")
        if config.bae_max_observations > 0:
            raise ValueError(
                "LoMa V1 has no observation cap; use --bae_max_observations 0"
            )
        if config.loma_dino_candidates <= 0:
            raise ValueError("loma_dino_candidates must be positive")
        if config.loma_pair_selection != "sift_guided":
            raise ValueError("The formal LoMa pipeline uses sift_guided selection")
        if (
            min(
                config.loma_sufficient_neighbors,
                config.loma_insufficient_neighbors,
                config.loma_untried_neighbors,
            )
            < 0
        ):
            raise ValueError("LoMa neighbor counts must be nonnegative")
    if config.ba_backend != "bae":
        raise ValueError("The formal pipeline supports BAE only")
    return SimpleNamespace(
        path_tracker=config.path_tracker,
        input_order=config.input_order,
        prior_provider=config.prior_provider,
        loma_dino_candidates=config.loma_dino_candidates,
        loma_pair_selection=config.loma_pair_selection,
        loma_sufficient_neighbors=config.loma_sufficient_neighbors,
        loma_insufficient_neighbors=config.loma_insufficient_neighbors,
        loma_untried_neighbors=config.loma_untried_neighbors,
        loma_match_batch_size=config.loma_match_batch_size,
        loma_extract_batch_size=config.loma_extract_batch_size,
        loma_preprocess_workers=config.loma_preprocess_workers,
        loma_geometry_workers=config.loma_geometry_workers,
        loma_feature_cache=config.loma_feature_cache,
        track_mode="SP",
        neighbors_per_center=config.neighbors_per_center,
        pair_pose_rotation_threshold=config.pair_pose_rotation_threshold,
        group_strategy=config.vggsfm_group_strategy,
        vggsfm_schedule_mode=config.vggsfm_schedule_mode,
        sift_temporal_window=config.sift_temporal_window,
        sift_schedule_grid_size=config.sift_schedule_grid_size,
        sift_schedule_min_inliers_per_cell=config.sift_schedule_min_inliers_per_cell,
        sift_schedule_min_pair_inliers=config.sift_schedule_min_pair_inliers,
        sift_schedule_min_grid_coverage=config.sift_schedule_min_grid_coverage,
        vggsfm_max_center_gap=config.vggsfm_max_center_gap,
        vggsfm_group_batch_size=config.vggsfm_group_batch_size,
        skip_doppelgangers=True,
        valid_dg_threshold=0.8,
        star_sequential_window=0,
        build_virtual_tracks=False,
        vggsfm_query_points=config.vggsfm_query_points,
        vggsfm_query_source=QUERY_SOURCE,
        vggsfm_tracker_input=TRACKER_INPUT,
        aliked_detection_threshold=config.aliked_detection_threshold,
        vggsfm_vis_threshold=config.vggsfm_vis_threshold,
        vggsfm_score_threshold=config.vggsfm_score_threshold,
        vggsfm_fine_tracking=config.vggsfm_fine_tracking,
        s_database_mode=S_DATABASE_MODE,
        prior_snap_to_sift=config.prior_provider != "loma",
        prior_snap_threshold=config.prior_snap_threshold,
        prior_keep_unsnapped=True,
        prior_keypoint_merge_threshold=config.prior_keypoint_merge_threshold,
        prior_match_topology=config.prior_match_topology,
        drop_low_coverage_frames=True,
        min_frame_observations=config.min_frame_observations,
        device=config.device,
        camera_model=CAMERA_MODEL,
        ba_backend=config.ba_backend,
        bae_max_num_iterations=config.bae_max_num_iterations,
        bae_max_observations=config.bae_max_observations,
        bae_optimize_intrinsics=config.bae_optimize_intrinsics,
        bae_fix_gauge=config.bae_fix_gauge,
        bae_robust_loss=config.bae_robust_loss,
        bae_huber_delta=config.bae_huber_delta,
        final_bae_huber_delta=config.final_bae_huber_delta,
        num_refinement_iterations=config.num_refinement_iterations,
        augmented_ba_max_filter_iterations=config.augmented_ba_max_filter_iterations,
        augmented_ba_normalized_reproj_threshold=config.augmented_ba_normalized_reproj_threshold,
        tri_min_angle=config.tri_min_angle,
        tri_create_max_angle_error=config.tri_create_max_angle_error,
        enable_select_tracks=True,
        select_track_min_support=config.select_track_min_support,
        enable_reprojection_filter=True,
        filter_reproj_error_type=config.filter_reproj_error_type,
        filter_reproj_error_threshold=config.filter_reproj_error_threshold,
        debug_print=config.debug_print,
    )
