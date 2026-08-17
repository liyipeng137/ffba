import csv
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from algos.utils import export_prediction_depth_maps
from utils import gluemap_refine_core as ref

CAMERA_MODEL = "SIMPLE_PINHOLE"
S_DATABASE_MODE = "sift"
QUERY_SOURCE = "aliked"
GROUP_STRATEGY = "pose"
TRACK_MODE = "SPV"
TRACKER_INPUT = "1024"


@dataclass
class GluemapSpvRefineConfig:
    path_tracker: str
    device: str = "cuda"
    neighbors_per_center: int = 25
    pair_pose_rotation_threshold: float = 30.0
    vggsfm_query_points: int = 1024
    aliked_detection_threshold: float = 0.005
    vggsfm_vis_threshold: float = 0.5
    vggsfm_score_threshold: float = 0.0
    vggsfm_fine_tracking: bool = False
    prior_snap_threshold: float = 1.0
    prior_keypoint_merge_threshold: float = 1e-3
    prior_match_topology: str = "all_pairs"
    min_frame_observations: int = 10
    ba_backend: str = "ceres"
    ba_max_num_iterations: int = 100
    bae_max_num_iterations: int = 20
    bae_optimize_intrinsics: bool = False
    bae_fix_gauge: str = "two_cams"
    bae_robust_loss: str = "none"
    bae_huber_delta: float = 1.0
    num_refinement_iterations: int = 2
    augmented_ba_max_filter_iterations: int = 3
    augmented_ba_normalized_reproj_threshold: float = 1e-2
    tri_min_angle: float = 1.0
    tri_create_max_angle_error: float = 0.5
    select_track_min_support: int = 512
    filter_reproj_error_type: str = "angular"
    filter_reproj_error_threshold: float = 0.5
    virtual_init_angular_error_threshold: float | None = None
    virtual_verify_mode: str = "n2"
    save_virtual_tracks_debug: bool = False
    debug_print: bool = True
    work_image_workers: int = 16


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
    use_virtual_tracks = config.ba_backend == "ceres"
    return SimpleNamespace(
        path_tracker=config.path_tracker,
        track_mode="SPV" if use_virtual_tracks else "SP",
        neighbors_per_center=config.neighbors_per_center,
        pair_pose_rotation_threshold=config.pair_pose_rotation_threshold,
        group_strategy=GROUP_STRATEGY,
        skip_doppelgangers=True,
        valid_dg_threshold=0.8,
        star_sequential_window=0,
        build_virtual_tracks=use_virtual_tracks,
        save_virtual_tracks_debug=config.save_virtual_tracks_debug,
        vggsfm_query_points=config.vggsfm_query_points,
        vggsfm_query_source=QUERY_SOURCE,
        vggsfm_tracker_input=TRACKER_INPUT,
        aliked_detection_threshold=config.aliked_detection_threshold,
        vggsfm_vis_threshold=config.vggsfm_vis_threshold,
        vggsfm_score_threshold=config.vggsfm_score_threshold,
        vggsfm_fine_tracking=config.vggsfm_fine_tracking,
        s_database_mode=S_DATABASE_MODE,
        prior_snap_to_sift=True,
        prior_snap_threshold=config.prior_snap_threshold,
        prior_keep_unsnapped=True,
        prior_keypoint_merge_threshold=config.prior_keypoint_merge_threshold,
        prior_match_topology=config.prior_match_topology,
        drop_low_coverage_frames=True,
        min_frame_observations=config.min_frame_observations,
        device=config.device,
        camera_model=CAMERA_MODEL,
        ba_backend=config.ba_backend,
        ba_max_num_iterations=config.ba_max_num_iterations,
        bae_max_num_iterations=config.bae_max_num_iterations,
        bae_optimize_intrinsics=config.bae_optimize_intrinsics,
        bae_fix_gauge=config.bae_fix_gauge,
        bae_robust_loss=config.bae_robust_loss,
        bae_huber_delta=config.bae_huber_delta,
        num_refinement_iterations=config.num_refinement_iterations,
        augmented_ba_max_filter_iterations=(config.augmented_ba_max_filter_iterations),
        augmented_ba_normalized_reproj_threshold=(
            config.augmented_ba_normalized_reproj_threshold
        ),
        tri_min_angle=config.tri_min_angle,
        tri_create_max_angle_error=config.tri_create_max_angle_error,
        enable_select_tracks=True,
        select_track_min_support=config.select_track_min_support,
        enable_reprojection_filter=True,
        filter_reproj_error_type=config.filter_reproj_error_type,
        filter_reproj_error_threshold=config.filter_reproj_error_threshold,
        virtual_init_angular_error_threshold=(
            config.virtual_init_angular_error_threshold
        ),
        virtual_verify_mode=config.virtual_verify_mode,
        debug_print=config.debug_print,
    )


def _debug(args, message):
    if args.debug_print:
        print(f"[PIPELINE-REFINE] {message}", flush=True)


def _save_one_work_image(item):
    idx, image, images_dir = item
    name = f"frame_{idx:06d}.png"
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(images_dir / name)
    return name


def _save_work_images(images, output_dir, num_workers=16):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images_cpu = images.detach().cpu().float().clamp(0, 1)

    num_workers = int(num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1")
    if int(images_cpu.shape[0]) == 0:
        raise ValueError("Cannot save work images from an empty tensor")
    worker_count = min(num_workers, int(images_cpu.shape[0]))

    work_items = [(idx, image, images_dir) for idx, image in enumerate(images_cpu)]
    if worker_count == 1:
        image_names = [_save_one_work_image(item) for item in work_items]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            image_names = list(executor.map(_save_one_work_image, work_items))

    print(
        "[PIPELINE-REFINE] Saved work images: "
        f"images={len(image_names)}, workers={worker_count}, images_dir={images_dir}"
    )
    return images_dir, image_names


def _write_json(path, payload):
    with open(path, "w") as f:
        # default=str keeps the stats dump from crashing after expensive compute
        # if a value (e.g. a Path or numpy scalar) is not JSON-serializable.
        json.dump(payload, f, indent=2, default=str)


def _link_or_copy_image(source, destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _draw_text_safe(draw, position, text, fill):
    try:
        draw.text(position, text, fill=fill)
    except UnicodeEncodeError:
        draw.text(position, text.encode("ascii", "replace").decode(), fill=fill)


def _paste_contained(sheet, image_path, box, border_color, border_width=3):
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(sheet)
    draw.rectangle(box, outline=border_color, width=border_width)
    inner_width = max(x1 - x0 - 2 * border_width, 1)
    inner_height = max(y1 - y0 - 2 * border_width, 1)
    with Image.open(image_path) as image:
        image = ImageOps.contain(
            image.convert("RGB"),
            (inner_width, inner_height),
            method=Image.Resampling.LANCZOS,
        )
    paste_x = x0 + border_width + (inner_width - image.width) // 2
    paste_y = y0 + border_width + (inner_height - image.height) // 2
    sheet.paste(image, (paste_x, paste_y))


def _save_group_contact_sheet(
    contact_sheet_path,
    images_dir,
    image_names,
    center,
    neighbors,
):
    columns = 5
    tile_width = 320
    tile_image_height = 220
    tile_label_height = 44
    margin = 12
    center_image_height = 340
    center_label_height = 32
    rows = max((len(neighbors) + columns - 1) // columns, 1)
    sheet_width = columns * tile_width + (columns + 1) * margin
    center_region_height = center_image_height + center_label_height + 2 * margin
    sheet_height = center_region_height + rows * (
        tile_image_height + tile_label_height + margin
    )
    sheet = Image.new("RGB", (sheet_width, sheet_height), color=(28, 28, 28))
    draw = ImageDraw.Draw(sheet)

    center_path = images_dir / image_names[center]
    center_box = (
        margin,
        margin,
        sheet_width - margin,
        margin + center_image_height,
    )
    _paste_contained(
        sheet, center_path, center_box, border_color=(255, 80, 80), border_width=5
    )
    _draw_text_safe(
        draw,
        (margin, margin + center_image_height + 7),
        f"CENTER idx={center:06d}  {image_names[center]}",
        fill=(255, 220, 220),
    )

    grid_y = center_region_height
    for neighbor in neighbors:
        rank = int(neighbor["rank"])
        image_idx = int(neighbor["image_index"])
        row = (rank - 1) // columns
        column = (rank - 1) % columns
        x0 = margin + column * (tile_width + margin)
        y0 = grid_y + row * (tile_image_height + tile_label_height + margin)
        image_box = (x0, y0, x0 + tile_width, y0 + tile_image_height)
        border_color = (90, 210, 120) if neighbor["rotation_valid"] else (255, 170, 70)
        _paste_contained(
            sheet,
            images_dir / image_names[image_idx],
            image_box,
            border_color=border_color,
        )
        label_y = y0 + tile_image_height + 4
        _draw_text_safe(
            draw,
            (x0, label_y),
            f"R{rank:02d} idx={image_idx:06d} angle={neighbor['rotation_angle_deg']:.1f}",
            fill=(235, 235, 235),
        )
        _draw_text_safe(
            draw,
            (x0, label_y + 17),
            f"distance={neighbor['camera_center_distance']:.4f}",
            fill=(190, 190, 190),
        )

    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(contact_sheet_path, quality=92)


def export_vggsfm_groups(
    coarse_state,
    output_dir,
    neighbors_per_center=25,
    pair_pose_rotation_threshold=30.0,
    num_workers=16,
):
    """Export the current pose groups for visual inspection, then return."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir, image_names = _save_work_images(
        coarse_state.high_images,
        output_dir,
        num_workers=num_workers,
    )

    extrinsic = np.asarray(coarse_state.extrinsic, dtype=np.float64)
    pairs = np.asarray(coarse_state.pairs, dtype=np.int64)
    centers = ref.camera_centers_from_w2c(extrinsic)
    viewing_axes = ref.camera_viewing_axes_from_w2c(extrinsic)
    group_args = SimpleNamespace(
        neighbors_per_center=int(neighbors_per_center),
        pair_pose_rotation_threshold=float(pair_pose_rotation_threshold),
    )
    groups, group_stats = ref.build_vggsfm_groups(
        group_args,
        pairs,
        len(image_names),
        image_names,
        tuple(coarse_state.high_image_size_hw),
        centers=centers,
        viewing_axes=viewing_axes,
    )

    audit_dir = (
        output_dir / "vggsfm_group_audit" / (f"pose_k{int(neighbors_per_center)}")
    )
    groups_dir = audit_dir / "groups"
    contact_sheets_dir = audit_dir / "contact_sheets"
    groups_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets_dir.mkdir(parents=True, exist_ok=True)

    rotation_threshold = float(pair_pose_rotation_threshold)

    def _export_one_group(group):
        center = int(group[0])
        group_dir = groups_dir / f"center_{center:06d}"
        group_dir.mkdir(parents=True, exist_ok=True)
        center_filename = f"00_center_idx{center:06d}.png"
        link_modes = [
            _link_or_copy_image(
                images_dir / image_names[center],
                group_dir / center_filename,
            )
        ]
        neighbors = []
        for rank, image_idx_value in enumerate(group[1:], start=1):
            image_idx = int(image_idx_value)
            dot = float(np.dot(viewing_axes[center], viewing_axes[image_idx]))
            rotation_angle = float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
            distance = float(np.linalg.norm(centers[center] - centers[image_idx]))
            filename = f"{rank:02d}_neighbor_idx{image_idx:06d}.png"
            link_modes.append(
                _link_or_copy_image(
                    images_dir / image_names[image_idx],
                    group_dir / filename,
                )
            )
            neighbors.append(
                {
                    "rank": rank,
                    "image_index": image_idx,
                    "work_image_name": image_names[image_idx],
                    "source_image_name": coarse_state.high_image_names[image_idx],
                    "rotation_angle_deg": rotation_angle,
                    "camera_center_distance": distance,
                    "rotation_valid": rotation_angle < rotation_threshold,
                    "group_image": str(
                        (Path("groups") / group_dir.name / filename).as_posix()
                    ),
                }
            )

        contact_sheet_path = contact_sheets_dir / f"center_{center:06d}.jpg"
        _save_group_contact_sheet(
            contact_sheet_path,
            images_dir,
            image_names,
            center,
            neighbors,
        )
        return {
            "center_index": center,
            "center_work_image_name": image_names[center],
            "center_source_image_name": coarse_state.high_image_names[center],
            "group_dir": str((Path("groups") / group_dir.name).as_posix()),
            "center_group_image": str(
                (Path("groups") / group_dir.name / center_filename).as_posix()
            ),
            "contact_sheet": str(
                (Path("contact_sheets") / contact_sheet_path.name).as_posix()
            ),
            "image_materialization": (
                "hardlink" if all(mode == "hardlink" for mode in link_modes) else "copy"
            ),
            "neighbors": neighbors,
        }

    worker_count = min(max(int(num_workers), 1), max(len(groups), 1))
    if worker_count == 1:
        group_entries = [_export_one_group(group) for group in groups]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            group_entries = list(executor.map(_export_one_group, groups))

    manifest = {
        "mode": "export_vggsfm_groups_only",
        "strategy": GROUP_STRATEGY,
        "neighbors_per_center": int(neighbors_per_center),
        "pair_pose_rotation_threshold": rotation_threshold,
        "num_images": len(image_names),
        "work_images_dir": str(images_dir),
        "group_stats": group_stats,
        "groups": group_entries,
    }
    manifest_path = audit_dir / "groups.json"
    _write_json(manifest_path, manifest)

    labels_path = audit_dir / "labels.csv"
    if not labels_path.exists():
        with open(labels_path, "w", newline="") as labels_file:
            writer = csv.writer(labels_file)
            writer.writerow(
                [
                    "center_index",
                    "neighbor_index",
                    "rank",
                    "covisibility_label_0_3_or_u",
                    "duplicate_view_0_1",
                    "extreme_view_0_1",
                    "notes",
                ]
            )
            for group_entry in group_entries:
                for neighbor in group_entry["neighbors"]:
                    writer.writerow(
                        [
                            group_entry["center_index"],
                            neighbor["image_index"],
                            neighbor["rank"],
                            "",
                            "",
                            "",
                            "",
                        ]
                    )

    print(
        "[PIPELINE-GROUP-EXPORT] Exported VGGSfM pose groups: "
        f"groups={len(group_entries)}, audit_dir={audit_dir}, "
        f"manifest={manifest_path}, labels={labels_path}",
        flush=True,
    )
    return manifest_path


def run_gluemap_spv_refinement(coarse_state, output_dir, config):
    ref._ensure_gluemap_imports()
    pycolmap = ref._lazy_import_pycolmap()
    args = _make_refine_args(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    t0 = time.time()
    images_dir, image_names = _save_work_images(
        coarse_state.high_images,
        output_dir,
        num_workers=config.work_image_workers,
    )
    save_work_images_seconds = time.time() - t0
    image_size_hw = tuple(coarse_state.high_image_size_hw)
    depth_image_size_hw = tuple(coarse_state.low_image_size_hw)
    initial_intrinsics_high_all = np.asarray(
        coarse_state.intrinsic_high, dtype=np.float64
    )
    initial_intrinsics_low_all = np.asarray(
        coarse_state.intrinsic_low, dtype=np.float64
    )
    extrinsic = np.asarray(coarse_state.extrinsic, dtype=np.float64)
    pairs = np.asarray(coarse_state.pairs, dtype=np.int64)
    metadata = {"image_size_hw": image_size_hw}

    stats = {
        "track_mode": args.track_mode,
        "camera_model": CAMERA_MODEL,
        "s_database_mode": S_DATABASE_MODE,
        "vggsfm_query_source": QUERY_SOURCE,
        "vggsfm_tracker_input": TRACKER_INPUT,
        "group_strategy": GROUP_STRATEGY,
        "ba_backend": config.ba_backend,
        "bae_optimize_intrinsics": config.bae_optimize_intrinsics,
        "bae_fix_gauge": config.bae_fix_gauge,
        "bae_robust_loss": config.bae_robust_loss,
        "bae_huber_delta": config.bae_huber_delta,
        "timing": {"save_work_images": save_work_images_seconds},
        "work_images": {
            "images_dir": str(images_dir),
            "image_names": image_names,
            "num_workers": int(config.work_image_workers),
            "source_low_image_names": list(coarse_state.low_image_names),
            "source_high_image_names": list(coarse_state.high_image_names),
            "low_image_size_hw": list(depth_image_size_hw),
            "high_image_size_hw": list(image_size_hw),
            "image_pyramid": coarse_state.image_pyramid,
        },
    }

    _debug(
        args,
        "Loaded coarse state: "
        f"images={len(image_names)}, pairs={pairs.shape[0]}, "
        f"high_image_size_hw={image_size_hw}, "
        f"low_image_size_hw={depth_image_size_hw}, "
        f"camera_model={CAMERA_MODEL}",
    )

    _debug(
        args,
        "Running VGGSfM prior tracking: "
        f"group_strategy={GROUP_STRATEGY}, "
        f"neighbors_per_center={args.neighbors_per_center}, "
        f"query_points={args.vggsfm_query_points}, "
        f"query_source={QUERY_SOURCE}, tracker_input={TRACKER_INPUT}",
    )
    t0 = time.time()
    prior_tracks, prior_stats = ref.run_vggsfm_prior_tracks(
        args,
        coarse_state.high_images,
        None,
        pairs,
        metadata,
        extrinsic,
        image_names,
    )
    stats["timing"]["vggsfm_prior_tracks"] = time.time() - t0
    stats["vggsfm"] = prior_stats
    group_stats = prior_stats["group_stats"]
    valid_neighbors = group_stats.get("selected_rotation_valid_neighbors")
    unfiltered_neighbors = group_stats.get("selected_unfiltered_neighbors")
    neighbor_priority_summary = ""
    if valid_neighbors is not None and unfiltered_neighbors is not None:
        neighbor_priority_summary = (
            f"neighbor_order={group_stats['neighbor_order']}, "
            f"valid_neighbors_mean={valid_neighbors['mean']:.2f}, "
            f"unfiltered_neighbors_mean={unfiltered_neighbors['mean']:.2f}, "
        )
    workload = prior_stats["workload"]
    query_track_stats = prior_stats["query_track_stats"]
    track_length = query_track_stats["track_length"]
    _debug(
        args,
        "VGGSfM prior done: "
        f"groups={prior_stats['num_groups']}, "
        f"tracks={prior_stats['num_tracks']}, "
        f"observations={prior_stats['num_observations']}, "
        f"{neighbor_priority_summary}"
        f"attempted_query_views={workload['attempted_query_views']}, "
        f"forming_track_rate={query_track_stats['forming_track_rate']:.3f}, "
        f"track_length_median={track_length['median']:.2f}, "
        f"track_length_p90={track_length['p90']:.2f}, "
        f"fmap_precompute={prior_stats['precompute_fmaps']['seconds']:.2f}s, "
        f"fmap_cache={prior_stats['precompute_fmaps']['storage_dtype']}@"
        f"{prior_stats['precompute_fmaps']['storage_device']}, "
        f"fmap_resident="
        f"{prior_stats['precompute_fmaps']['resident_on_tracker_device']}, "
        f"group_tracking={prior_stats['group_tracking_time']:.2f}s, "
        f"time={stats['timing']['vggsfm_prior_tracks']:.2f}s",
    )

    prefilter_image_names = list(image_names)
    _debug(args, "Preparing prefilter SIFT database")
    t0 = time.time()
    prefilter_intrinsics_mapping = {idx: 0 for idx in range(len(image_names))}
    features, sift_prefilter_stats = ref.prepare_sift_database_for_refine(
        args,
        output_dir,
        output_dir,
        image_names,
        pairs,
        CAMERA_MODEL,
        prefilter_intrinsics_mapping,
    )
    stats["timing"]["prepare_sift_database_prefilter"] = time.time() - t0
    stats["s_database"] = {
        "mode": S_DATABASE_MODE,
        "prefilter": sift_prefilter_stats,
        "prefilter_database_ready": True,
    }
    s_counts = np.asarray(
        sift_prefilter_stats["observations_per_image"], dtype=np.int64
    )
    p_counts = ref.count_track_observations(prior_tracks, len(image_names))
    _debug(args, ref.format_count_summary("S observations/frame", s_counts))
    _debug(args, ref.format_count_summary("P observations/frame", p_counts))
    _debug(
        args, ref.format_count_summary("S+P observations/frame", s_counts + p_counts)
    )

    (
        image_names,
        images,
        extrinsic,
        features,
        pairs,
        prior_tracks,
        coverage_stats,
    ) = ref.filter_low_coverage_frames(
        image_names,
        coarse_state.high_images,
        extrinsic,
        features,
        pairs,
        prior_tracks,
        s_counts,
        p_counts,
        args.min_frame_observations,
        enabled=True,
    )
    stats["frame_filtering"] = coverage_stats
    stats["num_images_after_filter"] = len(image_names)
    _debug(
        args,
        "Frame filtering: "
        f"min_obs={coverage_stats['min_frame_observations']}, "
        f"dropped={len(coverage_stats['dropped_indices'])}, "
        f"remaining={len(image_names)}",
    )
    if coverage_stats["dropped_indices"]:
        _debug(
            args,
            "Dropped frames: "
            + ", ".join(
                f"{idx}:{name}"
                for idx, name in zip(
                    coverage_stats["dropped_indices"],
                    coverage_stats["dropped_names"],
                    strict=False,
                )
            ),
        )

    kept_indices = np.asarray(coverage_stats["kept_indices"], dtype=np.int64)
    initial_intrinsics_high = initial_intrinsics_high_all[kept_indices]
    initial_intrinsics_low = initial_intrinsics_low_all[kept_indices]
    depth = coarse_state.raw_depth[kept_indices]
    depth_conf = (
        coarse_state.raw_depth_conf[kept_indices]
        if coarse_state.raw_depth_conf is not None
        else None
    )

    t0 = time.time()
    depth_predictions = {"depth": depth}
    depth_conf_threshold = None
    if depth_conf is not None:
        depth_predictions["depth_conf"] = depth_conf
        depth_conf_threshold = 2.0
    stats["depth_export"] = export_prediction_depth_maps(
        depth_predictions,
        image_names,
        output_dir / "pred_depth",
        conf_threshold=depth_conf_threshold,
    )
    stats["timing"]["depth_export"] = time.time() - t0

    t0 = time.time()
    (
        averaged_intrinsics,
        global_intrinsics,
        intrinsics_mapping,
    ) = ref.average_intrinsics_with_gluemap(initial_intrinsics_high, CAMERA_MODEL)
    stats["timing"]["intrinsics_averaging"] = time.time() - t0
    intrinsic = averaged_intrinsics[0]
    stats["intrinsics"] = ref.summarize_intrinsics(
        initial_intrinsics_high, averaged_intrinsics, CAMERA_MODEL
    )
    ref.save_intrinsics_artifacts(
        output_dir,
        initial_intrinsics_high,
        averaged_intrinsics,
        intrinsics_mapping,
        image_names,
    )
    _debug(
        args,
        "Intrinsics averaged: "
        f"fx={intrinsic[0, 0]:.2f}, fy={intrinsic[1, 1]:.2f}, "
        f"cx={intrinsic[0, 2]:.2f}, cy={intrinsic[1, 2]:.2f}, "
        f"time={stats['timing']['intrinsics_averaging']:.2f}s",
    )

    if args.build_virtual_tracks:
        _debug(
            args,
            f"Building virtual tracks: verify_mode={args.virtual_verify_mode}",
        )
        t0 = time.time()
        (
            virtual_predictions_dict,
            stats["virtual_tracks"],
        ) = ref.build_virtual_track_diagnostics(
            args,
            output_dir,
            depth,
            depth_conf,
            extrinsic,
            initial_intrinsics_low,
            global_intrinsics,
            intrinsics_mapping,
            pairs,
            image_names,
            image_size_hw,
            depth_image_size_hw=depth_image_size_hw,
        )
        stats["timing"]["virtual_tracks"] = time.time() - t0
        vt = stats["virtual_tracks"]
        final_vt = vt["update_virtual_tracks_global"]["virtual"]
        _debug(
            args,
            "Virtual tracks done: "
            f"verify_mode={vt['verify_mode']}, "
            f"groups={vt['num_groups']}, "
            f"valid_obs={final_vt['valid_observations']}, "
            f"time={stats['timing']['virtual_tracks']:.2f}s",
        )
    else:
        virtual_predictions_dict = None
        stats["virtual_tracks"] = {
            "enabled": False,
            "reason": "BAE backend uses SP real tracks only",
        }
        stats["timing"]["virtual_tracks"] = 0.0
        _debug(args, "Skipping virtual tracks for BAE SP refinement")

    if coverage_stats["dropped_indices"]:
        _debug(args, "Filtering SIFT database after frame filtering")
        t0 = time.time()
        features, sift_final_stats = ref.filter_sift_database_for_refine(
            output_dir / "database_sift.db",
            output_dir / "database_sift.db",
            prefilter_image_names,
            coverage_stats["kept_indices"],
            pairs,
            intrinsics_mapping,
        )
        stats["timing"]["filter_sift_database_final"] = time.time() - t0
        stats["timing"]["prepare_sift_database_final"] = 0.0
    else:
        sift_final_stats = stats["s_database"]["prefilter"]
        stats["timing"]["filter_sift_database_final"] = 0.0
        stats["timing"]["prepare_sift_database_final"] = 0.0
    stats["s_database"]["final"] = sift_final_stats
    _debug(
        args,
        "SIFT DB ready: "
        f"keypoints={sift_final_stats['num_keypoints_total']}, "
        f"pairs={sift_final_stats['num_pairs']}, "
        f"matches={sift_final_stats['num_matches']}",
    )

    _debug(args, "Writing VGGSfM prior database")
    t0 = time.time()
    stats["prior_database"] = ref.write_tracks_database(
        str(output_dir / "database_vggsfm_prior.db"),
        image_names,
        image_size_hw,
        intrinsic,
        CAMERA_MODEL,
        prior_tracks,
        features=features,
        snap_to_features=True,
        snap_threshold=args.prior_snap_threshold,
        keep_unsnapped=True,
        merge_threshold=args.prior_keypoint_merge_threshold,
        snap_target="sift",
        match_topology=args.prior_match_topology,
    )
    stats["timing"]["write_prior_db"] = time.time() - t0
    prior_db_stats = stats["prior_database"]
    snap_stats = prior_db_stats["snap"]
    _debug(
        args,
        "Prior DB written: "
        f"tracks={prior_db_stats['num_tracks']}, "
        f"topology={prior_db_stats['keypoint_merge']['match_topology']}, "
        f"pairs={prior_db_stats['num_pairs']}, "
        f"raw_kp={prior_db_stats['keypoint_merge']['raw_total']}, "
        f"merged_kp={prior_db_stats['keypoint_merge']['merged_total']}, "
        f"snapped={snap_stats['snapped_observations']}",
    )

    from gluemap.utils.colmap import merge_colmap_databases  # noqa: PLC0415

    _debug(args, "Merging VGGSfM prior and SIFT databases")
    t0 = time.time()
    merge_colmap_databases(
        str(output_dir / "database_vggsfm_prior.db"),
        str(output_dir / "database_sift.db"),
        str(output_dir / "database_merged.db"),
        primary_features_first=False,
    )
    stats["timing"]["merge_databases"] = time.time() - t0

    coarse_dir = output_dir / "coarse"
    _debug(args, f"Writing coarse reconstruction: {coarse_dir}")
    t0 = time.time()
    ref.write_coarse_reconstruction(
        coarse_dir,
        image_names,
        image_size_hw,
        extrinsic,
        intrinsic,
        CAMERA_MODEL,
    )
    stats["timing"]["write_coarse"] = time.time() - t0

    _debug(
        args,
        "Running augmented refinement: "
        f"iterations={args.num_refinement_iterations}, "
        f"ba_max_iters={args.ba_max_num_iterations}",
    )
    t0 = time.time()
    (
        reconstruction,
        virtual_reconstruction,
        augmented_stats,
    ) = ref.run_merg3r_augmented_refinement_loop(
        args,
        pycolmap,
        output_dir,
        image_names,
        image_size_hw,
        CAMERA_MODEL,
        extrinsic,
        global_intrinsics,
        intrinsics_mapping,
        virtual_predictions_dict,
        features,
        output_dir / "database_merged.db",
    )
    stats["timing"]["augmented_refinement"] = time.time() - t0
    stats["augmented_refinement"] = augmented_stats

    refined_dir = output_dir / "refined_gluemap_aba"
    refined_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(refined_dir))
    virtual_dir = None
    if virtual_reconstruction is not None:
        virtual_dir = output_dir / "virtual_gluemap_aba"
        virtual_dir.mkdir(parents=True, exist_ok=True)
        virtual_reconstruction.write(str(virtual_dir))

    stats["timing"]["total"] = time.time() - t_start
    stats["output"] = {
        "coarse_dir": str(coarse_dir),
        "database_merged": str(output_dir / "database_merged.db"),
        "refined_dir": str(refined_dir),
        "virtual_refined_dir": (str(virtual_dir) if virtual_dir is not None else None),
    }
    _write_json(output_dir / "refine_stats.json", stats)

    final_sources = augmented_stats["final"]["real_by_source"]
    _debug(
        args,
        "Augmented refinement done: "
        f"real_points={augmented_stats['final']['real']['points3D']}, "
        f"s_only={final_sources['s_only']}, "
        f"p_only={final_sources['p_only']}, "
        f"mixed={final_sources['mixed']}, "
        f"virtual_points={augmented_stats['final']['virtual']['points3D']}, "
        f"time={stats['timing']['augmented_refinement']:.2f}s",
    )

    return GluemapSpvRefineResult(
        image_names=image_names,
        extrinsic=extrinsic,
        pairs=pairs,
        intrinsic=intrinsic,
        intrinsics_mapping=intrinsics_mapping,
        stats=stats,
        refined_dir=refined_dir,
        virtual_refined_dir=virtual_dir,
    )
