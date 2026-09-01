import csv
import gc
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from algos.utils import export_prediction_depth_maps
from utils import gluemap_refine_core as ref
from utils.pano_rig import (
    build_pose_audit,
    center_driven_keep_indices,
    configure_rig_database,
    filter_metadata,
    write_rig_reconstruction,
)

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
    vggsfm_group_strategy: str = GROUP_STRATEGY
    vggsfm_group_batch_size: int = 2
    projected_overlap_dino_candidates: int = 30
    projected_overlap_samples: int = 2048
    projected_overlap_reproj_threshold: float = 4.0
    projected_overlap_conf_quantile: float = 0.2
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
    bae_max_observations: int = 0
    bae_optimize_intrinsics: bool = False
    bae_fix_gauge: str = "two_cams"
    bae_robust_loss: str = "none"
    bae_huber_delta: float = 1.0
    final_bae_huber_delta: float | None = None
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
    stop_before_bae: bool = True


@dataclass
class GluemapSpvRefineResult:
    image_names: list[str]
    extrinsic: np.ndarray
    pairs: np.ndarray
    intrinsic: np.ndarray
    intrinsics_mapping: dict[int, int]
    stats: dict
    pre_bae_dir: Path
    refined_dir: Path | None
    virtual_refined_dir: Path | None


def _make_refine_args(config: GluemapSpvRefineConfig):
    use_virtual_tracks = config.ba_backend == "ceres" and not config.stop_before_bae
    return SimpleNamespace(
        path_tracker=config.path_tracker,
        track_mode="SPV" if use_virtual_tracks else "SP",
        neighbors_per_center=config.neighbors_per_center,
        pair_pose_rotation_threshold=config.pair_pose_rotation_threshold,
        group_strategy=config.vggsfm_group_strategy,
        vggsfm_group_batch_size=config.vggsfm_group_batch_size,
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
        bae_max_observations=config.bae_max_observations,
        bae_optimize_intrinsics=config.bae_optimize_intrinsics,
        bae_fix_gauge=config.bae_fix_gauge,
        bae_robust_loss=config.bae_robust_loss,
        bae_huber_delta=config.bae_huber_delta,
        final_bae_huber_delta=config.final_bae_huber_delta,
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


def _cuda_memory_snapshot(device):
    allocated_bytes = int(torch.cuda.memory_allocated(device))
    reserved_bytes = int(torch.cuda.memory_reserved(device))
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": allocated_bytes,
        "reserved_bytes": reserved_bytes,
        "driver_free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _format_cuda_memory_snapshot(snapshot):
    gib = 1024**3
    return (
        f"allocated={snapshot['allocated_bytes'] / gib:.2f} GiB, "
        f"reserved={snapshot['reserved_bytes'] / gib:.2f} GiB, "
        f"driver_free={snapshot['driver_free_bytes'] / gib:.2f}/"
        f"{snapshot['total_bytes'] / gib:.2f} GiB"
    )


def _save_one_work_image(item):
    _idx, image, images_dir, name = item
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(images_dir / name)
    return name


def _save_work_images(images, output_dir, num_workers=16, image_names=None):
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    images_cpu = images.detach().cpu().float().clamp(0, 1)

    num_workers = int(num_workers)
    if num_workers <= 0:
        raise ValueError("num_workers must be >= 1")
    if int(images_cpu.shape[0]) == 0:
        raise ValueError("Cannot save work images from an empty tensor")
    if image_names is None:
        image_names = [
            f"frame_{idx:06d}.png" for idx in range(int(images_cpu.shape[0]))
        ]
    image_names = [str(name) for name in image_names]
    if len(image_names) != int(images_cpu.shape[0]):
        raise ValueError("Work image name count does not match image tensor")
    if len(set(image_names)) != len(image_names):
        raise ValueError("Work image names must be unique")
    worker_count = min(num_workers, int(images_cpu.shape[0]))

    work_items = [
        (idx, image, images_dir, image_names[idx])
        for idx, image in enumerate(images_cpu)
    ]
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
        if "projected_overlap" in neighbor:
            _draw_text_safe(
                draw,
                (x0, label_y),
                f"R{rank:02d} idx={image_idx:06d} score={neighbor['projected_overlap']:.3f}",
                fill=(235, 235, 235),
            )
            sources = "+".join(neighbor.get("candidate_sources", []))
            _draw_text_safe(
                draw,
                (x0, label_y + 17),
                f"grid={neighbor['projected_grid_coverage']:.3f} src={sources}",
                fill=(190, 190, 190),
            )
        else:
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
    selection_strategy="pose",
    retrieval_sim_matrix=None,
    projected_overlap_dino_candidates=30,
    projected_overlap_samples=2048,
    projected_overlap_reproj_threshold=4.0,
    projected_overlap_conf_quantile=0.2,
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
    candidate_details = None
    if selection_strategy == "pose":
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
        audit_name = f"pose_k{int(neighbors_per_center)}"
    elif selection_strategy == "projected_overlap":
        if coarse_state.tracking_groups is not None:
            groups = coarse_state.tracking_groups
            group_stats = coarse_state.tracking_group_stats
        elif retrieval_sim_matrix is None:
            raise ValueError(
                "retrieval_sim_matrix is required for projected-overlap audit"
            )
        else:
            groups, group_stats, candidate_details = ref.build_projected_overlap_groups(
                pairs=pairs,
                extrinsic=extrinsic,
                intrinsics=np.asarray(coarse_state.intrinsic_low, dtype=np.float64),
                depth=coarse_state.raw_depth,
                depth_conf=coarse_state.raw_depth_conf,
                retrieval_sim_matrix=retrieval_sim_matrix,
                max_neighbors=int(neighbors_per_center),
                rotation_threshold=float(pair_pose_rotation_threshold),
                dino_candidates=int(projected_overlap_dino_candidates),
                max_samples=int(projected_overlap_samples),
                reprojection_threshold=float(projected_overlap_reproj_threshold),
                confidence_quantile=float(projected_overlap_conf_quantile),
            )
        audit_name = f"projected_overlap_hybrid_k{int(neighbors_per_center)}"
    else:
        raise ValueError(
            "selection_strategy must be 'pose' or 'projected_overlap', "
            f"got {selection_strategy!r}"
        )

    audit_dir = output_dir / "vggsfm_group_audit" / audit_name
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
            neighbor_entry = {
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
            if candidate_details is not None:
                detail = next(
                    item
                    for item in candidate_details[center]
                    if item["image_index"] == image_idx
                )
                neighbor_entry.update(
                    {
                        key: value
                        for key, value in detail.items()
                        if key not in {"image_index", "selected_rank"}
                    }
                )
            neighbors.append(neighbor_entry)

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
        "strategy": selection_strategy,
        "neighbors_per_center": int(neighbors_per_center),
        "pair_pose_rotation_threshold": rotation_threshold,
        "num_images": len(image_names),
        "work_images_dir": str(images_dir),
        "group_stats": group_stats,
        "groups": group_entries,
    }
    candidate_scores_path = None
    if candidate_details is not None:
        candidate_scores_path = audit_dir / "candidate_scores.json"
        _write_json(
            candidate_scores_path,
            {
                "strategy": "projected_overlap_hybrid",
                "centers": [
                    {
                        "center_index": int(center),
                        "center_source_image_name": coarse_state.high_image_names[
                            center
                        ],
                        "candidates": candidate_details[center],
                    }
                    for center in sorted(candidate_details)
                ],
            },
        )
        manifest["candidate_scores"] = str(candidate_scores_path)
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
        "[PIPELINE-GROUP-EXPORT] Exported VGGSfM groups: "
        f"strategy={selection_strategy}, groups={len(group_entries)}, "
        f"audit_dir={audit_dir}, "
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
    rig = coarse_state.rig
    if rig is None:
        raise ValueError("The pano branch requires explicit rig metadata")
    if not config.stop_before_bae and config.ba_backend != "bae":
        raise ValueError(
            "Full cubemap rig refinement requires --ba_backend=bae so the "
            "five sensor poses share one optimized rig pose per timestamp"
        )

    t_start = time.time()
    t0 = time.time()
    images_dir, image_names = _save_work_images(
        coarse_state.high_images,
        output_dir,
        num_workers=config.work_image_workers,
        image_names=rig.image_names,
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
        "group_strategy": args.group_strategy,
        "ba_backend": config.ba_backend,
        "bae_optimize_intrinsics": config.bae_optimize_intrinsics,
        "bae_fix_gauge": config.bae_fix_gauge,
        "bae_robust_loss": config.bae_robust_loss,
        "bae_huber_delta": config.bae_huber_delta,
        "final_bae_huber_delta": config.final_bae_huber_delta,
        "stop_before_bae": bool(config.stop_before_bae),
        "rig": rig.to_dict(),
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
        f"group_strategy={args.group_strategy}, "
        f"group_batch_size={args.vggsfm_group_batch_size}, "
        f"neighbors_per_center={args.neighbors_per_center}, "
        f"query_points={args.vggsfm_query_points}, "
        f"query_source={QUERY_SOURCE}, tracker_input={TRACKER_INPUT}",
    )
    tracking_groups = None
    tracking_group_stats = None
    if args.group_strategy == "projected_overlap":
        t0 = time.time()
        if coarse_state.tracking_groups is not None:
            tracking_groups = coarse_state.tracking_groups
            tracking_group_stats = coarse_state.tracking_group_stats
        else:
            if coarse_state.retrieval_sim_matrix is None:
                raise ValueError(
                    "retrieval_sim_matrix is required when "
                    "vggsfm_group_strategy='projected_overlap'"
                )
            tracking_groups, tracking_group_stats, _ = (
                ref.build_projected_overlap_groups(
                    pairs=pairs,
                    extrinsic=extrinsic,
                    intrinsics=initial_intrinsics_low_all,
                    depth=coarse_state.raw_depth,
                    depth_conf=coarse_state.raw_depth_conf,
                    retrieval_sim_matrix=coarse_state.retrieval_sim_matrix,
                    max_neighbors=int(args.neighbors_per_center),
                    rotation_threshold=float(args.pair_pose_rotation_threshold),
                    dino_candidates=int(config.projected_overlap_dino_candidates),
                    max_samples=int(config.projected_overlap_samples),
                    reprojection_threshold=float(
                        config.projected_overlap_reproj_threshold
                    ),
                    confidence_quantile=float(
                        config.projected_overlap_conf_quantile
                    ),
                )
            )
        stats["timing"]["vggsfm_group_build"] = time.time() - t0
        _debug(
            args,
            "Built projected-overlap VGGSfM groups: "
            f"groups={len(tracking_groups)}, "
            f"group_size={tracking_group_stats['group_size']}",
        )
    elif args.group_strategy != "pose":
        raise ValueError(
            "vggsfm_group_strategy must be 'pose' or 'projected_overlap', "
            f"got {args.group_strategy!r}"
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
        groups=tracking_groups,
        group_stats=tracking_group_stats,
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
    prefilter_intrinsics_mapping = {
        idx: int(sensor_idx) for idx, sensor_idx in enumerate(rig.image_sensor_indices)
    }
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

    center_indices = np.asarray(rig.center_image_indices, dtype=np.int64)
    center_total_counts = s_counts[center_indices] + p_counts[center_indices]
    kept_frame_indices, forced_keep_indices = center_driven_keep_indices(
        rig,
        center_total_counts,
        args.min_frame_observations,
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
        keep_indices=forced_keep_indices,
    )
    rig = filter_metadata(rig, kept_frame_indices)
    coverage_stats.update(
        {
            "strategy": "center_driven_whole_rig_frame",
            "center_observations": center_total_counts.tolist(),
            "kept_frame_indices": kept_frame_indices.tolist(),
            "kept_frame_source_indices": list(rig.frame_source_indices),
            "dropped_frame_indices": sorted(
                set(range(len(center_total_counts))) - set(kept_frame_indices.tolist())
            ),
        }
    )
    stats["frame_filtering"] = coverage_stats
    stats["num_images_after_filter"] = len(image_names)
    _debug(
        args,
        "Center-driven rig frame filtering: "
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
    center_depth_indices = np.asarray(rig.frame_source_indices, dtype=np.int64)
    depth = coarse_state.raw_depth[center_depth_indices]
    depth_conf = (
        coarse_state.raw_depth_conf[center_depth_indices]
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
        [image_names[idx] for idx in rig.center_image_indices],
        output_dir / "pred_depth",
        conf_threshold=depth_conf_threshold,
    )
    stats["timing"]["depth_export"] = time.time() - t0

    t0 = time.time()
    intrinsics_mapping = {
        idx: int(sensor_idx) for idx, sensor_idx in enumerate(rig.image_sensor_indices)
    }
    sensor_intrinsics = []
    for sensor_idx in range(len(rig.sensor_names)):
        image_idx = rig.image_sensor_indices.index(sensor_idx)
        sensor_intrinsics.append(initial_intrinsics_high[image_idx].copy())
    averaged_intrinsics = np.asarray(
        [sensor_intrinsics[sensor_idx] for sensor_idx in rig.image_sensor_indices]
    )
    global_intrinsics = [
        torch.from_numpy(intrinsic).to(torch.float64).unsqueeze(0)
        for intrinsic in sensor_intrinsics
    ]
    stats["timing"]["intrinsics_averaging"] = time.time() - t0
    intrinsic = sensor_intrinsics[rig.center_sensor_index]
    stats["intrinsics"] = ref.summarize_intrinsics(
        initial_intrinsics_high, averaged_intrinsics, CAMERA_MODEL
    )
    stats["intrinsics"].update(
        {
            "method": "fixed_from_panorama_extraction_contract",
            "mapping": "one_camera_per_rig_sensor",
            "optimized": False,
        }
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
        "Fixed panorama intrinsics prepared: "
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
            "reason": (
                "Rig-aware BAE uses real SIFT/VGGSfM tracks; only the center "
                "face has feed-forward depth"
            ),
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
        sensor_intrinsics,
        CAMERA_MODEL,
        prior_tracks,
        features=features,
        snap_to_features=True,
        snap_threshold=args.prior_snap_threshold,
        keep_unsnapped=True,
        merge_threshold=args.prior_keypoint_merge_threshold,
        snap_target="sift",
        match_topology=args.prior_match_topology,
        intrinsics_mapping=intrinsics_mapping,
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
    configure_rig_database(
        output_dir / "database_merged.db",
        image_names,
        image_size_hw,
        sensor_intrinsics,
        rig,
        camera_model=CAMERA_MODEL,
    )

    coarse_dir = output_dir / "pre_bae_rig"
    _debug(args, f"Writing rig-aware pre-BAE reconstruction: {coarse_dir}")
    t0 = time.time()
    pre_bae_reconstruction = write_rig_reconstruction(
        coarse_dir,
        image_names,
        image_size_hw,
        extrinsic[np.asarray(rig.center_image_indices, dtype=np.int64)],
        sensor_intrinsics,
        rig,
        camera_model=CAMERA_MODEL,
    )
    stats["timing"]["write_coarse"] = time.time() - t0
    stats["pre_bae_reconstruction"] = {
        "directory": str(coarse_dir),
        "num_rigs": int(len(pre_bae_reconstruction.rigs)),
        "num_cameras": int(len(pre_bae_reconstruction.cameras)),
        "num_frames": int(len(pre_bae_reconstruction.frames)),
        "num_images": int(len(pre_bae_reconstruction.images)),
        "num_points3D": int(len(pre_bae_reconstruction.points3D)),
        "pose_semantics": "sensor_from_world = sensor_from_rig * rig_from_world",
    }
    pose_audit = build_pose_audit(extrinsic, rig)
    pose_audit_path = output_dir / "pre_bae_rig_pose_audit.json"
    _write_json(pose_audit_path, pose_audit)
    stats["pre_bae_reconstruction"]["pose_audit"] = str(pose_audit_path)
    stats["pre_bae_reconstruction"]["max_projection_center_spread"] = pose_audit[
        "max_projection_center_spread"
    ]
    stats["pre_bae_reconstruction"][
        "max_absolute_orientation_error_degrees"
    ] = pose_audit[
        "max_absolute_orientation_error_degrees"
    ]

    if config.stop_before_bae:
        stats["timing"]["total"] = time.time() - t_start
        stats["status"] = "stopped_before_bae_as_requested"
        stats["output"] = {
            "pre_bae_dir": str(coarse_dir),
            "database_merged": str(output_dir / "database_merged.db"),
            "refined_dir": None,
        }
        _write_json(output_dir / "refine_stats.json", stats)
        _debug(
            args,
            "Stopped before BAE after writing the fixed rig model and merged DB",
        )
        return GluemapSpvRefineResult(
            image_names=image_names,
            extrinsic=extrinsic,
            pairs=pairs,
            intrinsic=intrinsic,
            intrinsics_mapping=intrinsics_mapping,
            stats=stats,
            pre_bae_dir=coarse_dir,
            refined_dir=None,
            virtual_refined_dir=None,
        )

    if args.device.startswith("cuda") and torch.cuda.is_available():
        cuda_device = torch.device(args.device)
        torch.cuda.synchronize(cuda_device)
        cuda_memory_before = _cuda_memory_snapshot(cuda_device)
        print(
            "[PIPELINE-REFINE] CUDA memory before augmented refinement "
            f"cleanup: {_format_cuda_memory_snapshot(cuda_memory_before)}",
            flush=True,
        )
        gc_collected = int(gc.collect())
        torch.cuda.empty_cache()
        torch.cuda.synchronize(cuda_device)
        cuda_memory_after = _cuda_memory_snapshot(cuda_device)
        print(
            "[PIPELINE-REFINE] CUDA memory after augmented refinement "
            f"cleanup: {_format_cuda_memory_snapshot(cuda_memory_after)}, "
            f"gc_collected={gc_collected}",
            flush=True,
        )
        stats["augmented_refinement_cuda_cleanup"] = {
            "enabled": True,
            "device": str(cuda_device),
            "gc_collected": gc_collected,
            "before": cuda_memory_before,
            "after": cuda_memory_after,
            "reserved_bytes_released": int(
                cuda_memory_before["reserved_bytes"]
                - cuda_memory_after["reserved_bytes"]
            ),
            "driver_free_bytes_gained": int(
                cuda_memory_after["driver_free_bytes"]
                - cuda_memory_before["driver_free_bytes"]
            ),
        }
    else:
        stats["augmented_refinement_cuda_cleanup"] = {
            "enabled": False,
            "reason": "CUDA device is not active",
        }

    _debug(
        args,
        "Running augmented refinement: "
        f"iterations={args.num_refinement_iterations}, "
        f"ba_max_iters={args.ba_max_num_iterations}, "
        f"filter_reproj_threshold={args.filter_reproj_error_threshold}, "
        f"bae_huber_delta={args.bae_huber_delta}, "
        f"final_bae_huber_delta={args.final_bae_huber_delta}",
    )
    sensor_from_rig_by_sensor = []
    for rotation in rig.sensor_from_rig_rotations:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(rotation, dtype=np.float64)
        sensor_from_rig_by_sensor.append(transform)
    bae_rig_config = ref.build_bae_rig_config(
        image_names,
        rig.image_frame_indices,
        rig.image_sensor_indices,
        sensor_from_rig_by_sensor,
        shared_focal=True,
    )
    stats["bae_rig_config"] = {
        "enabled": True,
        "num_rig_frames": int(rig.num_frames),
        "num_images": int(rig.num_images),
        "num_sensors": int(len(rig.sensor_names)),
        "shared_focal": bool(bae_rig_config["shared_focal"]),
        "num_intrinsics_groups": 1,
        "pose_semantics": bae_rig_config["pose_semantics"],
    }
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
        bae_rig_config=bae_rig_config,
        rig_seed_reconstruction=pre_bae_reconstruction,
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

    final_images_by_name = {
        image.name: image for image in reconstruction.images.values()
    }
    missing_final_images = [
        image_name
        for image_name in image_names
        if image_name not in final_images_by_name
    ]
    if missing_final_images:
        raise ValueError(
            "Final BAE reconstruction is missing rig images: "
            f"{missing_final_images[:5]}"
        )
    final_extrinsic = np.asarray(
        [
            np.asarray(final_images_by_name[name].cam_from_world().matrix())
            for name in image_names
        ],
        dtype=np.float64,
    )
    post_bae_pose_audit = build_pose_audit(final_extrinsic, rig)
    post_bae_pose_audit_path = output_dir / "post_bae_rig_pose_audit.json"
    _write_json(post_bae_pose_audit_path, post_bae_pose_audit)
    stats["post_bae_rig_pose_audit"] = {
        "path": str(post_bae_pose_audit_path),
        "max_projection_center_spread": post_bae_pose_audit[
            "max_projection_center_spread"
        ],
        "max_absolute_orientation_error_degrees": post_bae_pose_audit[
            "max_absolute_orientation_error_degrees"
        ],
    }

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
        extrinsic=final_extrinsic,
        pairs=pairs,
        intrinsic=intrinsic,
        intrinsics_mapping=intrinsics_mapping,
        stats=stats,
        pre_bae_dir=coarse_dir,
        refined_dir=refined_dir,
        virtual_refined_dir=virtual_dir,
    )
