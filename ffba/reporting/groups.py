import csv
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ffba import api as ref
from ffba.refinement import _save_work_images, _write_json


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
        if retrieval_sim_matrix is None:
            raise ValueError(
                "retrieval_sim_matrix is required for projected-overlap audit"
            )
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
