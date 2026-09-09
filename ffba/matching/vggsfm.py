"""matching / vggsfm for the formal SIFT + prior + BAE pipeline."""

import time
from collections import defaultdict
import numpy as np
import torch


def sample_query_points(keypoints, max_points):
    if keypoints.shape[0] <= max_points:
        return keypoints
    indices = np.linspace(0, keypoints.shape[0] - 1, max_points, dtype=np.int64)
    return keypoints[indices]


def apply_image_change(points, image_change):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 0] = points[..., 0] * image_change[0] + image_change[2]
    points[..., 1] = points[..., 1] * image_change[1] + image_change[3]
    return points


def invert_image_change(points, image_change):
    points = np.asarray(points, dtype=np.float32).copy()
    points[..., 0] = (points[..., 0] - image_change[2]) / image_change[0]
    points[..., 1] = (points[..., 1] - image_change[3]) / image_change[1]
    return points


def prepare_vggsfm_tracker_images(args, images):
    from ffba.runtime import _ensure_gluemap_imports

    if args.vggsfm_tracker_input == "native":
        return (
            images,
            None,
            {
                "tracker_input": "native",
                "tracker_image_size_hw": [
                    int(images.shape[-2]),
                    int(images.shape[-1]),
                ],
            },
        )

    _ensure_gluemap_imports()
    from gluemap.utils.load_fn import (  # noqa: PLC0415
        load_and_preprocess_images_1024,
    )

    images_cpu = [images[idx].detach().cpu() for idx in range(images.shape[0])]
    images_1024, image_changes_1024 = load_and_preprocess_images_1024(images_cpu)
    image_changes_1024 = np.asarray(image_changes_1024, dtype=np.float32)
    return (
        images_1024,
        image_changes_1024,
        {
            "tracker_input": "1024",
            "tracker_image_size_hw": [
                int(images_1024.shape[-2]),
                int(images_1024.shape[-1]),
            ],
        },
    )


@torch.no_grad()
def build_vggsfm_query_points(
    args, tracker_images, features=None, tracker_image_changes=None
):
    from ffba.runtime import _ensure_gluemap_imports

    _ensure_gluemap_imports()
    from lightglue import ALIKED  # noqa: PLC0415

    t0 = time.time()
    extractor = (
        ALIKED(
            max_num_keypoints=args.vggsfm_query_points,
            detection_threshold=args.aliked_detection_threshold,
        )
        .eval()
        .to(args.device)
    )
    query_points = []
    for idx in range(tracker_images.shape[0]):
        image = tracker_images[idx : idx + 1].to(args.device)
        feats = extractor.extract(image)
        keypoints = feats["keypoints"][0].detach().cpu().numpy().astype(np.float32)
        query_points.append(keypoints)

    return query_points, {
        "query_source": "aliked",
        "aliked_detection_threshold": args.aliked_detection_threshold,
        "query_counts": [int(points.shape[0]) for points in query_points],
        "query_extraction_time": time.time() - t0,
    }


@torch.no_grad()
def precompute_vggsfm_tracker_fmaps(tracker, args, tracker_images, chunk_size=32):
    t0 = time.time()
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")

    num_images = int(tracker_images.shape[0])
    tracker_device = torch.device(args.device)
    use_cuda_cache = tracker_device.type == "cuda"
    cache_budget_fraction = 0.5
    tracker_fmaps = None
    storage_dtype = torch.float32
    free_cuda_bytes_before_cache = None
    estimated_cache_bytes = None
    fallback_reason = None

    # Keep the CUDA allocator cache warm across VGGSfM preprocessing chunks.
    # if use_cuda_cache:
    #     torch.cuda.empty_cache()

    for start in range(0, num_images, chunk_size):
        end = min(start + chunk_size, num_images)
        images_chunk = tracker_images[start:end].to(tracker_device, non_blocking=True)
        fmaps_chunk = tracker.process_images_to_fmaps(images_chunk)

        if tracker_fmaps is None:
            full_shape = (num_images, *fmaps_chunk.shape[1:])
            if use_cuda_cache:
                storage_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                element_size = torch.empty((), dtype=storage_dtype).element_size()
                estimated_cache_bytes = int(np.prod(full_shape)) * element_size
                free_cuda_bytes_before_cache, _ = torch.cuda.mem_get_info(
                    tracker_device
                )
                cache_budget_bytes = int(
                    free_cuda_bytes_before_cache * cache_budget_fraction
                )
                if estimated_cache_bytes > cache_budget_bytes:
                    fallback_reason = (
                        "estimated compressed fmap cache exceeds 50% of "
                        "currently free CUDA memory"
                    )
                else:
                    try:
                        tracker_fmaps = torch.empty(
                            full_shape,
                            dtype=storage_dtype,
                            device=tracker_device,
                        )
                    except torch.OutOfMemoryError:
                        fallback_reason = "CUDA allocation failed"
                        # torch.cuda.empty_cache()

            if tracker_fmaps is None:
                # Preserve the previous FP32 CPU-cache behavior when the
                # compressed resident cache would leave too little workspace
                # for the tracker itself.
                storage_dtype = torch.float32
                tracker_fmaps = torch.empty(
                    full_shape,
                    dtype=storage_dtype,
                    device="cpu",
                )

        tracker_fmaps[start:end].copy_(fmaps_chunk.detach(), non_blocking=True)
        del images_chunk, fmaps_chunk
        # Keep the CUDA allocator cache warm for the next chunk.
        # if use_cuda_cache:
        #     torch.cuda.empty_cache()

    cache_bytes = tracker_fmaps.numel() * tracker_fmaps.element_size()
    resident_on_tracker_device = tracker_fmaps.device.type == tracker_device.type and (
        tracker_device.index is None
        or tracker_fmaps.device.index == tracker_device.index
    )
    return tracker_fmaps, {
        "enabled": True,
        "chunk_size": int(chunk_size),
        "seconds": time.time() - t0,
        "shape": [int(v) for v in tracker_fmaps.shape],
        "storage_device": str(tracker_fmaps.device),
        "storage_dtype": str(tracker_fmaps.dtype).removeprefix("torch."),
        "resident_on_tracker_device": resident_on_tracker_device,
        "cache_bytes": int(cache_bytes),
        "estimated_compressed_cache_bytes": estimated_cache_bytes,
        "free_cuda_bytes_before_cache": free_cuda_bytes_before_cache,
        "cuda_cache_budget_fraction": (
            cache_budget_fraction if use_cuda_cache else None
        ),
        "fallback_reason": fallback_reason,
    }


@torch.no_grad()
def run_vggsfm_prior_tracks(
    args,
    images,
    features,
    pairs,
    metadata,
    extrinsic,
    image_names,
    groups=None,
    group_stats=None,
):
    from ffba.geometry import camera_viewing_axes_from_w2c
    from ffba.reporting.statistics import summarize_distribution
    from ffba.runtime import _ensure_gluemap_imports, debug

    if not args.path_tracker:
        raise ValueError("--path_tracker is required")
    group_batch_size = int(getattr(args, "vggsfm_group_batch_size", 1))
    if group_batch_size <= 0:
        raise ValueError("vggsfm_group_batch_size must be >= 1")

    _ensure_gluemap_imports()
    from vggsfm.vggsfm_tracker import TrackerPredictor  # noqa: PLC0415

    tracker = TrackerPredictor().eval().to(args.device)
    tracker.load_state_dict(
        torch.load(args.path_tracker, map_location="cpu", weights_only=False)
    )

    viewing_axes = camera_viewing_axes_from_w2c(extrinsic)
    if groups is None or group_stats is None:
        raise ValueError("VGGSfM tracking requires preselected sparse-center groups")
    observations = 0
    tracker_images, tracker_image_changes, tracker_stats = (
        prepare_vggsfm_tracker_images(args, images)
    )
    query_points_per_image, query_stats = build_vggsfm_query_points(
        args,
        tracker_images,
        features,
        tracker_image_changes=tracker_image_changes,
    )
    tracker_fmaps, fmaps_stats = precompute_vggsfm_tracker_fmaps(
        tracker,
        args,
        tracker_images,
    )
    tracker_parameter = next(tracker.parameters())
    tracker_device = tracker_parameter.device
    tracker_dtype = tracker_parameter.dtype

    neighbor_rank_stats = [
        {
            "rank": rank,
            "pair_count": 0,
            "rotation_valid_pairs": 0,
            "unfiltered_pairs": 0,
            "unclassified_pairs": 0,
            "pairs_executed": 0,
            "pairs_with_accepted_observations": 0,
            "attempted_queries": 0,
            "visibility_pass": 0,
            "score_pass": 0,
            "in_bounds_pass": 0,
            "accepted_observations": 0,
        }
        for rank in range(1, int(args.neighbors_per_center) + 1)
    ]
    rotation_threshold = getattr(args, "pair_pose_rotation_threshold", None)
    total_neighbor_slots = 0
    for group in groups:
        center = int(group[0])
        total_neighbor_slots += max(len(group) - 1, 0)
        for rank, image_idx in enumerate(group[1:], start=1):
            rank_stats = neighbor_rank_stats[rank - 1]
            rank_stats["pair_count"] += 1
            if rotation_threshold is None:
                rank_stats["unclassified_pairs"] += 1
                continue
            dot = float(np.dot(viewing_axes[center], viewing_axes[int(image_idx)]))
            angle = float(np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0))))
            if angle < rotation_threshold:
                rank_stats["rotation_valid_pairs"] += 1
            else:
                rank_stats["unfiltered_pairs"] += 1

    total_queries = 0
    valid_center_queries = 0
    attempted_query_views = 0
    tracks_by_group = {}
    track_lengths_by_group = {}
    t_group = time.time()
    tracking_buckets = defaultdict(list)
    zero_query_centers = []
    for group_order, group in enumerate(groups):
        group = [int(image_idx) for image_idx in group]
        center = int(group[0])
        query_np = sample_query_points(
            query_points_per_image[center], args.vggsfm_query_points
        )
        if query_np.shape[0] == 0:
            zero_query_centers.append(center)
            continue
        num_queries = int(query_np.shape[0])
        total_queries += num_queries
        attempted_query_views += num_queries * max(len(group) - 1, 0)
        tracking_buckets[(len(group), num_queries)].append(
            (group_order, group, query_np)
        )

    bucket_stats = []
    num_forward_calls = 0
    effective_batch_size_histogram = defaultdict(int)
    for bucket_index, ((group_size, query_count), jobs) in enumerate(
        sorted(tracking_buckets.items()),
        start=1,
    ):
        group_count = len(jobs)
        batch_count = (group_count + group_batch_size - 1) // group_batch_size
        tail_batch_size = group_count % group_batch_size or min(
            group_batch_size, group_count
        )
        bucket_stat = {
            "bucket_index": int(bucket_index),
            "group_size": int(group_size),
            "neighbors_per_group": int(group_size - 1),
            "query_points": int(query_count),
            "group_count": int(group_count),
            "batch_count": int(batch_count),
            "tail_batch_size": int(tail_batch_size),
        }
        bucket_stats.append(bucket_stat)
        debug(
            args,
            "VGGSfM batch bucket "
            f"{bucket_index}/{len(tracking_buckets)}: "
            f"group_size={group_size}, query_points={query_count}, "
            f"groups={group_count}, batches={batch_count}, "
            f"tail_batch_size={tail_batch_size}",
        )

    debug(
        args,
        "VGGSfM batch bucketing done: "
        f"buckets={len(bucket_stats)}, batch_size={group_batch_size}, "
        f"runnable_groups={sum(len(jobs) for jobs in tracking_buckets.values())}, "
        f"zero_query_groups={len(zero_query_centers)}, "
        f"query_points_per_bucket="
        f"{[item['query_points'] for item in bucket_stats]}",
    )

    h, w = metadata["image_size_hw"]
    for (group_size, query_count), jobs in sorted(tracking_buckets.items()):
        for batch_start in range(0, len(jobs), group_batch_size):
            batch_jobs = jobs[batch_start : batch_start + group_batch_size]
            actual_batch_size = len(batch_jobs)
            num_forward_calls += 1
            effective_batch_size_histogram[actual_batch_size] += 1

            group_indices_np = np.asarray(
                [group for _group_order, group, _query_np in batch_jobs],
                dtype=np.int64,
            )
            group_tensor = None
            if args.vggsfm_fine_tracking:
                tracker_image_indices = torch.as_tensor(
                    group_indices_np,
                    dtype=torch.long,
                    device=tracker_images.device,
                )
                group_tensor = tracker_images[tracker_image_indices].to(tracker_device)
            tracker_fmap_indices = torch.as_tensor(
                group_indices_np,
                dtype=torch.long,
                device=tracker_fmaps.device,
            )
            group_fmaps = tracker_fmaps[tracker_fmap_indices].to(
                device=tracker_device,
                dtype=tracker_dtype,
                non_blocking=True,
            )
            query = torch.from_numpy(
                np.stack([query_np for _group_order, _group, query_np in batch_jobs])
            ).to(tracker_device, dtype=torch.float32)
            pred_track_batch, _, pred_vis_batch, pred_score_batch = tracker(
                group_tensor,
                query,
                fmaps=group_fmaps,
                fine_tracking=args.vggsfm_fine_tracking,
            )
            del group_fmaps, group_tensor, query
            pred_track_batch = pred_track_batch.detach().cpu().numpy()
            pred_vis_batch = pred_vis_batch.detach().cpu().numpy()
            pred_score_batch = pred_score_batch.detach().cpu().numpy()

            for batch_index, (group_order, group, query_np) in enumerate(batch_jobs):
                center = int(group[0])
                num_queries = int(query_np.shape[0])
                pred_track = pred_track_batch[batch_index]
                pred_vis = pred_vis_batch[batch_index]
                pred_score = pred_score_batch[batch_index]

                center_points = query_np.astype(np.float32, copy=True)
                if tracker_image_changes is not None:
                    center_points = invert_image_change(
                        center_points, tracker_image_changes[center]
                    )
                center_valid = (
                    (center_points[:, 0] >= 0)
                    & (center_points[:, 0] < w)
                    & (center_points[:, 1] >= 0)
                    & (center_points[:, 1] < h)
                )
                valid_center_queries += int(center_valid.sum())

                accepted_neighbor_masks = {}
                mapped_neighbor_tracks = {}
                for local_idx, image_idx in enumerate(group[1:], start=1):
                    rank_stats = neighbor_rank_stats[local_idx - 1]
                    rank_stats["pairs_executed"] += 1
                    rank_stats["attempted_queries"] += num_queries

                    visibility_values = np.asarray(pred_vis[local_idx]).reshape(-1)
                    score_values = np.asarray(pred_score[local_idx]).reshape(-1)
                    if (
                        visibility_values.size != num_queries
                        or score_values.size != num_queries
                    ):
                        raise ValueError(
                            "VGGSfM visibility/score output does not match query "
                            f"count: queries={num_queries}, "
                            f"visibility={visibility_values.shape}, "
                            f"score={score_values.shape}"
                        )
                    visibility_pass = visibility_values >= args.vggsfm_vis_threshold
                    score_pass = visibility_pass & (
                        score_values >= args.vggsfm_score_threshold
                    )
                    neighbor_points = pred_track[local_idx].astype(
                        np.float32, copy=True
                    )
                    if tracker_image_changes is not None:
                        neighbor_points = invert_image_change(
                            neighbor_points,
                            tracker_image_changes[int(image_idx)],
                        )
                    in_bounds_pass = (
                        score_pass
                        & (neighbor_points[:, 0] >= 0)
                        & (neighbor_points[:, 0] < w)
                        & (neighbor_points[:, 1] >= 0)
                        & (neighbor_points[:, 1] < h)
                    )
                    accepted = in_bounds_pass & center_valid

                    rank_stats["visibility_pass"] += int(visibility_pass.sum())
                    rank_stats["score_pass"] += int(score_pass.sum())
                    rank_stats["in_bounds_pass"] += int(in_bounds_pass.sum())
                    rank_stats["accepted_observations"] += int(accepted.sum())
                    if accepted.any():
                        rank_stats["pairs_with_accepted_observations"] += 1
                    accepted_neighbor_masks[local_idx] = accepted
                    mapped_neighbor_tracks[local_idx] = neighbor_points

                group_tracks = []
                group_track_lengths = []
                for point_idx in range(num_queries):
                    if not center_valid[point_idx]:
                        continue
                    obs = [(center, center_points[point_idx])]
                    for local_idx, image_idx in enumerate(group[1:], start=1):
                        if not accepted_neighbor_masks[local_idx][point_idx]:
                            continue
                        obs.append(
                            (
                                int(image_idx),
                                mapped_neighbor_tracks[local_idx][point_idx],
                            )
                        )
                    if len(obs) >= 2:
                        observations += len(obs)
                        group_tracks.append(obs)
                        group_track_lengths.append(len(obs))
                tracks_by_group[group_order] = group_tracks
                track_lengths_by_group[group_order] = group_track_lengths
    tracks = []
    track_lengths = []
    for group_order in range(len(groups)):
        tracks.extend(tracks_by_group.get(group_order, []))
        track_lengths.extend(track_lengths_by_group.get(group_order, []))
    group_tracking_time = time.time() - t_group
    debug(
        args,
        "VGGSfM batch tracking done: "
        f"forward_calls={num_forward_calls}, "
        f"effective_batch_size_histogram="
        f"{dict(sorted(effective_batch_size_histogram.items()))}",
    )

    for rank_stats in neighbor_rank_stats:
        attempted = rank_stats["attempted_queries"]
        for count_key, rate_key in (
            ("visibility_pass", "visibility_pass_rate"),
            ("score_pass", "score_pass_rate"),
            ("in_bounds_pass", "in_bounds_pass_rate"),
            ("accepted_observations", "accepted_observation_rate"),
        ):
            rank_stats[rate_key] = (
                float(rank_stats[count_key] / attempted) if attempted > 0 else 0.0
            )

    accepted_neighbor_observations = int(
        sum(rank["accepted_observations"] for rank in neighbor_rank_stats)
    )
    queries_forming_tracks = len(tracks)
    return tracks, {
        "num_groups": len(groups),
        "num_tracks": len(tracks),
        "num_observations": observations,
        "neighbors_per_center": args.neighbors_per_center,
        "group_strategy": args.group_strategy,
        "batching": {
            "configured_batch_size": int(group_batch_size),
            "num_buckets": int(len(bucket_stats)),
            "num_forward_calls": int(num_forward_calls),
            "zero_query_group_count": int(len(zero_query_centers)),
            "zero_query_centers": [int(center) for center in zero_query_centers],
            "effective_batch_size_histogram": {
                str(batch_size): int(count)
                for batch_size, count in sorted(effective_batch_size_histogram.items())
            },
            "buckets": bucket_stats,
        },
        "group_stats": group_stats,
        "query_points": args.vggsfm_query_points,
        "workload": {
            "total_neighbor_slots": int(total_neighbor_slots),
            "actual_query_points": int(total_queries),
            "attempted_query_views": int(attempted_query_views),
        },
        "neighbor_rank_stats": neighbor_rank_stats,
        "query_track_stats": {
            "total_queries": int(total_queries),
            "valid_center_queries": int(valid_center_queries),
            "invalid_center_queries": int(total_queries - valid_center_queries),
            "queries_forming_tracks": int(queries_forming_tracks),
            "queries_without_accepted_neighbor": int(
                valid_center_queries - queries_forming_tracks
            ),
            "forming_track_rate": (
                float(queries_forming_tracks / total_queries)
                if total_queries > 0
                else 0.0
            ),
            "forming_track_rate_valid_centers": (
                float(queries_forming_tracks / valid_center_queries)
                if valid_center_queries > 0
                else 0.0
            ),
            "track_length": summarize_distribution(track_lengths),
            "accepted_neighbor_observations": accepted_neighbor_observations,
            "observation_count_consistent": bool(
                accepted_neighbor_observations == observations - len(tracks)
            ),
        },
        "precompute_fmaps": fmaps_stats,
        "group_tracking_time": group_tracking_time,
        **tracker_stats,
        **query_stats,
    }
