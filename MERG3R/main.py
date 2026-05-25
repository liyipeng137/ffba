from pydoc import describe
import torch
import numpy as np
import os
import argparse
import json

from algos.utils import *
from algos.sequence import create_sequence
from algos.bundle_adjustment import global_bundle_adjustment
from algos.alignment import align_extrinsics
from algos.tracking import  extract_matches_lightglue, graph_extract_matches_lightglue
from algos.loma_tracking import graph_extract_matches_loma
from algos.hloc_tracking import graph_extract_matches_hloc
from algos.dense_debug import export_dense_debug_outputs
from algos.dense_correction import apply_inverse_depth_affine_correction
from algos.lingbot_depth_refine import run_lingbot_depth_refinement
from algos.infinidepth_refine import DEFAULT_INFINIDEPTH_MODEL_PATH, run_infinidepth_refinement
from bae_pipe import run_bae_refinement


import gc


def drop_untracked_frames(sequence, final_predictions, images, track, points_id, enabled=True):
    if not enabled:
        return images, track, points_id, []

    num_frames = len(track)
    keep_indices = []
    dropped_indices = []
    for frame_idx, (track_i, points_id_i) in enumerate(zip(track, points_id)):
        has_track = np.asarray(track_i).size > 0 and np.asarray(points_id_i).size > 0
        if has_track:
            keep_indices.append(frame_idx)
        else:
            dropped_indices.append(frame_idx)

    if not dropped_indices:
        return images, track, points_id, []
    if not keep_indices:
        raise ValueError("All frames have empty tracks after matching; cannot continue global BA.")

    keep_np = np.asarray(keep_indices, dtype=np.int64)
    keep_torch = torch.as_tensor(keep_indices, dtype=torch.long, device=images.device)

    image_ids = final_predictions.get("image_ids", None)
    if image_ids is not None:
        dropped_frame_ids = np.asarray(image_ids)[dropped_indices].tolist()
    else:
        dropped_frame_ids = dropped_indices
    dropped_names = [sequence.image_names[i] for i in dropped_indices if i < len(sequence.image_names)]
    print(
        "[TRACKING] Warning: dropping "
        f"{len(dropped_indices)} frames with no tracks before global BA. "
        f"indices={dropped_indices}, frame_ids={dropped_frame_ids}, names={dropped_names}"
    )

    for key, value in list(final_predictions.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (num_frames,):
            final_predictions[key] = value[keep_np]
        elif isinstance(value, torch.Tensor) and value.shape[:1] == (num_frames,):
            final_predictions[key] = value[keep_torch.to(value.device)]

    images = images[keep_torch]
    sequence.images = sequence.images[keep_torch.to(sequence.images.device)]
    sequence.image_names = [sequence.image_names[i] for i in keep_indices]
    track = [track[i] for i in keep_indices]
    points_id = [points_id[i] for i in keep_indices]

    return images, track, points_id, dropped_frame_ids


def parse_args():
    parser = argparse.ArgumentParser("Test parallel inference on VGGT. ")
    ################ MERG3R ################
    parser.add_argument("--subset_size", type=int, default=100)
    parser.add_argument("--num_images", type=int, default=-1)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--write_local", action="store_true")
    parser.add_argument("--ba_type", type=str, default="gradient")
    parser.add_argument("--alignment_type", type=str, default="weighted_iterative")
    parser.add_argument("--global_ba", action="store_true", default=True)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--high-dataset", dest="high_dataset", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence_type", type=str, default="shortest_path")
    parser.add_argument("--lr", type=float, default=3e-3)  # 3e-4
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--max_reproj", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=10, help="stride of save colmap points")
    parser.add_argument("--model", type=str, default="pi3x", choices=['vggt', 'pi3', 'pi3x', 'vggt_omega'])
    parser.add_argument("--pi3x_intrinsics_method", type=str, default="moge", choices=["lstsq", "moge"])
    parser.add_argument("--multi_dirs", action="store_true")
    parser.add_argument("--point_vis_threshold", type=float, default=20.0)
    parser.add_argument("--tracking_type", type=str, default="graph", choices=['graph', 'video'])
    parser.add_argument("--tracking_matcher", type=str, default="loma", choices=['lightglue', 'loma', 'hloc'])
    parser.add_argument("--loma_arch", type=str, default="LoMa-B", choices=['LoMa-B', 'LoMa-B128', 'LoMa-L', 'LoMa-G', 'LoMa-R'])
    parser.add_argument("--loma_filter_threshold", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--splitting_type", type=str, default="interleave", choices=['interleave', 'zigzag', 'threshold', "original", "original_threshold"])
    parser.add_argument("--format", type=str, default="txt", choices=['txt', 'bin'])
    ################ MERG3R ################

    ################ BAE ################
    parser.add_argument("--bae_refine", action="store_true", help="Run BAE sparse refinement after MERG3R global BA.")
    parser.add_argument("--bae_iters", type=int, default=20)
    parser.add_argument("--bae_optimize_intrinsics", action="store_true")
    ################ BAE ################

    ################ SAVE DENSE ################
    parser.add_argument(
        "--dense_max_points",
        type=int,
        default=50_000_000,
        help="Maximum points to export in dense_model_points.ply. Set <=0 to disable.",
    )
    parser.add_argument(
        "--save_dense_depth",
        dest="save_dense_depth",
        action="store_true",
        default=False,
        help="Save per-camera depth maps projected from the merged dense point cloud.",
    )
    parser.add_argument(
        "--no_save_dense_depth",
        dest="save_dense_depth",
        action="store_false",
        help="Disable per-camera depth maps projected from the merged dense point cloud.",
    )
    parser.add_argument(
        "--dense_depth_dir",
        type=str,
        default=None,
        help="Directory for projected dense depth maps. Defaults to <output_dir>/dense_depth.",
    )
    parser.add_argument(
        "--dense_depth_max_points",
        type=int,
        default=-1,
        help="Maximum points from the shared dense_model_points set to use for projected depth. Set <=0 to use all shared points.",
    )
    parser.add_argument(
        "--dense_depth_stride",
        type=int,
        default=1,
        help="Pixel stride when collecting the shared dense_model_points set if dense depth export is enabled.",
    )
    parser.add_argument(
        "--dense_depth_chunk_size",
        type=int,
        default=1_000_000,
        help="Projection chunk size for dense depth export. Set -1 to disable chunking.",
    )
    parser.add_argument(
        "--dense_depth_camera_batch_size",
        type=int,
        default=4,
        help="Number of cameras projected together on CUDA for dense depth export.",
    )
    parser.add_argument(
        "--dense_depth_backend",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Projection backend for dense depth export.",
    )
    ################ SAVE DENSE ################

    ################ LINGBOT ################
    parser.add_argument(
        "--lingbot_refine",
        action="store_true",
        help="Run LingBot-Depth refinement after projected dense depth export.",
    )
    parser.add_argument(
        "--lingbot_output_dir",
        type=str,
        default=None,
        help="Directory for LingBot-Depth refined outputs. Defaults to <output_dir>/lingbot_depth.",
    )
    ################ LINGBOT ################ 

    ################ INFINIDEPTH ################
    parser.add_argument(
        "--infinidepth_refine",
        action="store_true",
        help="Run InfiniDepth depth-sensor refinement using single-frame depth as sparse sensor depth.",
    )
    parser.add_argument(
        "--infinidepth_output_dir",
        type=str,
        default=None,
        help="Directory for InfiniDepth refined outputs. Defaults to <output_dir>/infinidepth_depth.",
    )
    parser.add_argument(
        "--infinidepth_model_path",
        type=str,
        default=DEFAULT_INFINIDEPTH_MODEL_PATH,
        help="Path to infinidepth_depthsensor.ckpt.",
    )
    parser.add_argument(
        "--infinidepth_input_height",
        type=int,
        default=768,
        help="InfiniDepth internal input height. Output depth is still saved at original RGB resolution.",
    )
    parser.add_argument(
        "--infinidepth_input_width",
        type=int,
        default=1024,
        help="InfiniDepth internal input width. Output depth is still saved at original RGB resolution.",
    )
    ################ INFINIDEPTH ################

    ################ DENSE DEBUG ################
    parser.add_argument(
        "--dense_debug",
        dest="dense_debug",
        action="store_true",
        default=True,
        help="Export sparse-dense residual diagnostics after global BA/BAE.",
    )
    parser.add_argument(
        "--no_dense_debug",
        dest="dense_debug",
        action="store_false",
        help="Disable sparse-dense residual diagnostics.",
    )
    parser.add_argument(
        "--dense_debug_max_points",
        type=int,
        default=2_000_000,
        help="Maximum points to export in dense_debug/dense_before_correction.ply. Set <=0 to disable.",
    )
    ################ DENSE DEBUG ################

    ################ DENSE CORRECTION ################
    parser.add_argument(
        "--dense_correction",
        type=str,
        default="none",
        choices=["none", "invz_affine"],
        help="Dense local point correction method to run after global BA/BAE.",
    )
    parser.add_argument("--dense_correction_min_anchors", type=int, default=50)
    parser.add_argument("--dense_correction_alpha", type=float, default=0.75)
    parser.add_argument("--dense_correction_conf_quantile", type=float, default=0.2)
    parser.add_argument("--dense_correction_edge_quantile", type=float, default=0.8)
    parser.add_argument("--dense_correction_residual_mad_k", type=float, default=3.5)
    parser.add_argument("--dense_correction_scale_min", type=float, default=0.5)
    parser.add_argument("--dense_correction_scale_max", type=float, default=2.0)
    parser.add_argument("--dense_correction_bias_abs_max", type=float, default=0.25)
    parser.add_argument("--dense_correction_depth_ratio_min", type=float, default=0.7)
    parser.add_argument("--dense_correction_depth_ratio_max", type=float, default=1.3)
    ################ DENSE CORRECTION ################

    args = parser.parse_args()

    
    return args


def main():
    
    args = parse_args()

    args_dict = vars(args)
    if args.dense_correction != "none" and not args.global_ba:
        raise ValueError("--dense_correction requires --global_ba so sparse tracks and BA points are available")
    if args.dense_correction_depth_ratio_min <= 0 or args.dense_correction_depth_ratio_max <= 0:
        raise ValueError("Dense correction depth ratio bounds must be positive.")
    if args.dense_correction_depth_ratio_min > args.dense_correction_depth_ratio_max:
        raise ValueError("--dense_correction_depth_ratio_min must be <= --dense_correction_depth_ratio_max")
    if not 0 <= args.dense_correction_alpha <= 1:
        raise ValueError("--dense_correction_alpha must be in [0, 1]")
    if not 0 <= args.dense_correction_conf_quantile <= 1:
        raise ValueError("--dense_correction_conf_quantile must be in [0, 1]")
    if not 0 <= args.dense_correction_edge_quantile <= 1:
        raise ValueError("--dense_correction_edge_quantile must be in [0, 1]")
    if args.dense_correction_scale_min > args.dense_correction_scale_max:
        raise ValueError("--dense_correction_scale_min must be <= --dense_correction_scale_max")
    if args.dense_depth_stride < 1:
        raise ValueError("--dense_depth_stride must be >= 1")
    if args.dense_depth_chunk_size == 0 or args.dense_depth_chunk_size < -1:
        raise ValueError("--dense_depth_chunk_size must be positive, or -1 for no chunking")
    if args.dense_depth_camera_batch_size < 1:
        raise ValueError("--dense_depth_camera_batch_size must be >= 1")
    if args.infinidepth_input_height <= 0 or args.infinidepth_input_width <= 0:
        raise ValueError("--infinidepth_input_height and --infinidepth_input_width must be positive")
    if args.lingbot_refine and not args.save_dense_depth:
        raise ValueError("--lingbot_refine requires projected dense depth export; remove --no_save_dense_depth")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this pipeline. No CUDA device was detected.")
    device = "cuda"

    print(f"[MAIN] Output to {args.output_dir}")
    subsample = args.subsample

    os.makedirs(args.output_dir, exist_ok=True)
    # Save to JSON
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    images, image_names = process_images(args.dataset, subsample, device, args.num_images, args.multi_dirs, args.model)
    # if args.model == "vggt_omega":
    #     image_names = save_tensor_images(
    #         images,
    #         image_names,
    #         os.path.join(args.output_dir, "resized_images"),
    #     )

    size_hw = images.shape[-2:]

    
    start_time = time.time()

    # Reset and start GPU memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    print("[MAIN] Create sequence. ")

    seq_start = time.time()
    sequence = create_sequence(images, image_names, 
                                sequence_type=args.sequence_type, 
                                subset_size=args.subset_size, 
                                overlap=args.overlap, 
                                save_path=args.output_dir, 
                                alpha=args.alpha, 
                                splitting_type=args.splitting_type)
    seq_end = time.time()

    batches = sequence.image_split

    for i, img in enumerate(batches):
        batches[i] = img.to("cpu")

    print(f"[MAIN] Number of subsets {len(batches)}. ")
    gc.collect()
    torch.cuda.empty_cache()

    
    model, _ = load_model(args.model, device=device)
    inf_start = time.time()

    sequence.predictions = run_inference_step_by_step(
        model,
        batches,
        size_hw,
        device,
        need_features=False,
        pi3x_intrinsics_method=args.pi3x_intrinsics_method,
    )
    inf_end = time.time()
    del model


    gc.collect()
    torch.cuda.empty_cache()
    # Convert each prediction to extrinsics
    for i, prediction in enumerate(sequence.predictions):
        prediction['images'] = batches[i]

    # Squeeze and convert to numpy
    for prediction in sequence.predictions:
        for key in prediction.keys():
            if isinstance(prediction[key], torch.Tensor):
                prediction[key] = prediction[key].cpu().numpy() 
                if prediction[key].shape[0] == 1:
                    prediction[key] = prediction[key].squeeze(0) # remove batch dimension
    
    
    final_predictions, _, _ = align_extrinsics(sequence, method=args.alignment_type, ba=False)
   
    restore_predictions_order(final_predictions)
    for key in sequence.subset_to_img_ids.keys():
        if isinstance(sequence.subset_to_img_ids[key], torch.Tensor):
            sequence.subset_to_img_ids[key] = sequence.subset_to_img_ids[key].cpu().numpy()
    print("[MAIN] DONE MERGING")

    tracking_start = 0
    tracking_end = 0

    ba_start = 0
    ba_end = 0
    bae_start = 0
    bae_end = 0
    bae_stats = None
    valid_track_mask = None
    track = None
    points_id = None
    dropped_untracked_frame_ids = []
    if args.global_ba:

        print("START TRACKING")

        final_predictions['world_points'] = unproject_depth_map_to_point_map(final_predictions['depth'], final_predictions['extrinsic'], final_predictions['intrinsic'])

        tracking_start = time.time()

        if args.sequence_type == 'video':
            track, points_id, points_3d, points_conf = extract_matches_lightglue(images, final_predictions['world_points'], final_predictions['depth_conf'],
                                                                                    final_predictions['extrinsic'], final_predictions['intrinsic'],
                                                                                    steps=[1, 2, 3, 5, 7, 10], max_num_keypoints=4096, device=device)
        elif args.sequence_type == 'shortest_path':

            if args.tracking_type =="graph":
                if args.tracking_matcher == "hloc":
                    track, points_id, points_3d, points_conf = graph_extract_matches_hloc(
                        images,
                        final_predictions['extrinsic'],
                        final_predictions['intrinsic'],
                        k=5,
                        workspace_dir=os.path.join(args.output_dir, "hloc_tracking"),
                        skip_geometric_verification=False,
                        overwrite=True,
                    )
                elif args.tracking_matcher == "loma":
                    track, points_id, points_3d, points_conf = graph_extract_matches_loma(images, final_predictions['world_points'], 
                                                                            final_predictions['depth_conf'], 
                                                                            final_predictions['extrinsic'], final_predictions['intrinsic'], k=5,
                                                                            max_num_keypoints=4096, device=device,
                                                                            arch=args.loma_arch,
                                                                            filter_threshold=args.loma_filter_threshold)
                else:
                    track, points_id, points_3d, points_conf = graph_extract_matches_lightglue(images, final_predictions['world_points'], 
                                                                            final_predictions['depth_conf'], 
                                                                            final_predictions['extrinsic'], final_predictions['intrinsic'], k=5,
                                                                            max_num_keypoints=4096, device=device)
            elif args.tracking_type == 'video':
                reordered_images = images[sequence.video_path]
                reordered_points = final_predictions['world_points'][sequence.video_path]
                reordered_conf = final_predictions['depth_conf'][sequence.video_path]
                reordered_extrinsic = final_predictions['extrinsic'][sequence.video_path]
                reordered_intrinsic = final_predictions['intrinsic'][sequence.video_path]

                track, points_id, points_3d, points_conf = extract_matches_lightglue(reordered_images, reordered_points, reordered_conf, 
                                                                                    reordered_extrinsic, reordered_intrinsic,
                                                                                    steps=[1, 2, 3, 5, 7, 10], max_num_keypoints=4096, device=device)
            
                inverse_path = np.argsort(sequence.video_path).tolist()

                # Reorder both lists
                track = [track[i] for i in inverse_path]
                points_id = [points_id[i] for i in inverse_path]

        tracking_end = time.time()
        images, track, points_id, dropped_untracked_frame_ids = drop_untracked_frames(
            sequence,
            final_predictions,
            images,
            track,
            points_id,
            enabled=True,
        )
        del final_predictions['world_points']
        final_predictions['track'] = track
        final_predictions['points_id'] = points_id
        final_predictions['points'] = points_3d


        print("START GLOBAL BA")
        
        ba_start = time.time()
        _, valid_track_mask =  global_bundle_adjustment(final_predictions, 
                                                        track, 
                                                        points_id, 
                                                        points_3d, 
                                                        points_conf, 
                                                        max_reproj_error=args.max_reproj, 
                                                        lr=args.lr,
                                                        epoch=args.epoch
                                                        )
        final_predictions['valid_track_mask'] = valid_track_mask
        shared_intrinsic = np.mean(final_predictions['intrinsic'], axis=0, keepdims=True)
        final_predictions['intrinsic'] = np.repeat(
            shared_intrinsic,
            final_predictions['intrinsic'].shape[0],
            axis=0,
        )

        ba_end = time.time()

        if args.bae_refine:
            print("START BAE REFINEMENT")
            bae_start = time.time()
            bae_stats = run_bae_refinement(
                final_predictions,
                track,
                points_id,
                valid_track_mask=valid_track_mask,
                iters=args.bae_iters,
                device=device,
                optimize_intrinsics=args.bae_optimize_intrinsics,
            )
            bae_end = time.time()
            print(
                "[BAE] DONE "
                f"initial_loss={bae_stats['initial_loss']:.6f}, "
                f"ending_loss={bae_stats['ending_loss']:.6f}, "
                f"observations={bae_stats['num_observations']}, "
                f"cameras={bae_stats['num_cameras']}, "
                f"dropped_cameras={bae_stats['num_dropped_cameras']}, "
                f"points={bae_stats['num_points']}"
            )
    elif args.bae_refine:
        raise ValueError("--bae_refine requires --global_ba so tracks and sparse points are available")

    
    torch.cuda.synchronize()
    end_time = time.time()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)  # in MiB
    elapsed = end_time - start_time

    single_frame_depth_start = time.time()
    single_frame_depth_stats = export_prediction_depth_maps(
        final_predictions,
        sequence.image_names,
        os.path.join(args.output_dir, "single_frame_depth"),
        conf_threshold=5.0,
    )
    single_frame_depth_end = time.time()

    low_intrinsic = np.asarray(final_predictions['intrinsic'], dtype=np.float32).copy()
    high_output_images = None
    high_output_image_names = None
    high_output_intrinsic = None
    high_output_enabled = args.high_dataset is not None
    if high_output_enabled:
        high_output_images, high_output_image_names = load_images_matching_names(
            args.high_dataset,
            sequence.image_names,
            device=sequence.images.device,
        )
        high_output_intrinsic = scale_intrinsics_between_image_sets(
            low_intrinsic,
            sequence.images,
            high_output_images,
        )
        print(
            "[MAIN] High-resolution output enabled: "
            f"low={tuple(sequence.images.shape[-2:])}, high={tuple(high_output_images.shape[-2:])}"
        )

    dense_debug_stats = None
    # if args.dense_debug and args.global_ba:
    #     dense_debug_stats = export_dense_debug_outputs(
    #         args.output_dir,
    #         final_predictions,
    #         sequence.images,
    #         sequence.image_names,
    #         track,
    #         points_id,
    #         valid_track_mask=valid_track_mask,
    #         dense_max_points=args.dense_debug_max_points if args.dense_debug_max_points > 0 else None,
    #     )

    dense_correction_stats = None
    if args.dense_correction == "invz_affine":
        dense_correction_stats = apply_inverse_depth_affine_correction(
            final_predictions,
            sequence.images,
            sequence.image_names,
            track,
            points_id,
            valid_track_mask=valid_track_mask,
            output_dir=args.output_dir,
            min_anchors_per_frame=args.dense_correction_min_anchors,
            alpha=args.dense_correction_alpha,
            conf_quantile=args.dense_correction_conf_quantile,
            edge_quantile=args.dense_correction_edge_quantile,
            residual_mad_k=args.dense_correction_residual_mad_k,
            scale_min=args.dense_correction_scale_min,
            scale_max=args.dense_correction_scale_max,
            bias_abs_max=args.dense_correction_bias_abs_max,
            depth_ratio_min=args.dense_correction_depth_ratio_min,
            depth_ratio_max=args.dense_correction_depth_ratio_max,
        )

    dense_world_points = None
    dense_world_colors = None
    dense_world_stats = None
    dense_collect_start = 0
    dense_collect_end = 0
    dense_ply_start = 0
    dense_ply_end = 0
    dense_depth_start = 0
    dense_depth_end = 0
    if 'local_points' in final_predictions:
        dense_collect_start = time.time()
        dense_world_points, dense_world_colors, dense_world_stats = collect_dense_world_points(
            final_predictions['local_points'],
            final_predictions['extrinsic'],
            sequence.images,
            final_predictions['depth_conf'],
            conf_threshold=args.point_vis_threshold,
            stride=args.dense_depth_stride if args.save_dense_depth else 1,
            max_points=args.dense_max_points if args.dense_max_points > 0 else None,
            include_colors=True,
        )
        dense_collect_end = time.time()
        if len(dense_world_points) == 0:
            print("[OUTPUT WRITING] No dense points passed filtering for dense_model_points.ply")
        else:
            dense_ply_start = time.time()
            export_dense_world_points_ply(
                os.path.join(args.output_dir, "dense_model_points.ply"),
                dense_world_points,
                dense_world_colors,
                stats=dense_world_stats,
            )
            dense_ply_end = time.time()

    dense_depth_stats = None
    dense_depth_dir = args.dense_depth_dir or os.path.join(args.output_dir, "dense_depth")
    if args.save_dense_depth and dense_world_points is not None:
        dense_depth_start = time.time()
        dense_depth_world_points = dense_world_points
        if args.dense_depth_max_points > 0 and len(dense_depth_world_points) > args.dense_depth_max_points:
            sample_indices = np.linspace(
                0,
                len(dense_depth_world_points) - 1,
                args.dense_depth_max_points,
                dtype=np.int64,
            )
            dense_depth_world_points = dense_depth_world_points[sample_indices]
        dense_depth_stats = export_dense_projected_depth_maps(
            dense_depth_dir,
            None,
            final_predictions['extrinsic'],
            final_predictions['intrinsic'],
            sequence.image_names,
            None,
            chunk_size=args.dense_depth_chunk_size,
            world_points=dense_depth_world_points,
            image_size=tuple(final_predictions['local_points'].shape[1:3]),
            backend=args.dense_depth_backend,
            camera_batch_size=args.dense_depth_camera_batch_size,
        )
        dense_depth_end = time.time()

    lingbot_start = 0
    lingbot_end = 0
    lingbot_stats = None
    lingbot_refined_ply_start = 0
    lingbot_refined_ply_end = 0
    lingbot_refined_ply_stats = None
    if args.lingbot_refine:
        if dense_depth_stats is None:
            raise RuntimeError("LingBot-Depth refinement requires projected dense depth outputs, but none were produced.")
        lingbot_start = time.time()
        lingbot_output_dir = args.lingbot_output_dir or os.path.join(args.output_dir, "lingbot_depth")
        lingbot_images = high_output_images if high_output_enabled else sequence.images
        lingbot_image_names = high_output_image_names if high_output_enabled else sequence.image_names
        lingbot_intrinsic = high_output_intrinsic if high_output_enabled else low_intrinsic
        shared_intrinsic = np.mean(np.asarray(lingbot_intrinsic, dtype=np.float32), axis=0)
        lingbot_stats = run_lingbot_depth_refinement(
            lingbot_images,
            lingbot_image_names,
            # os.path.join(dense_depth_dir, "depth_npy"),
            os.path.join(args.output_dir, "single_frame_depth", "depth_npy"),
            lingbot_output_dir,
            shared_intrinsic,
            device=device,
            depth_image_names=sequence.image_names,
        )
        lingbot_end = time.time()
        if high_output_enabled:
            print(
                "[OUTPUT WRITING] Skipping dense_lingbot_refined_points.ply in high-resolution "
                "refine mode; high-resolution refined depths were written by LingBot-Depth."
            )
        else:
            lingbot_refined_ply_start = time.time()
            lingbot_refined_ply_stats = export_depth_npy_world_points_ply(
                os.path.join(args.output_dir, "dense_lingbot_refined_points.ply"),
                os.path.join(lingbot_output_dir, "depth_npy"),
                sequence.image_names,
                sequence.images,
                final_predictions['extrinsic'],
                shared_intrinsic,
                stride=args.dense_depth_stride,
                max_points=args.dense_max_points if args.dense_max_points > 0 else None,
                valid_mask_depth_npy_dir=os.path.join(dense_depth_dir, "depth_npy"),
                valid_mask_image_names=sequence.image_names,
            )
            lingbot_refined_ply_end = time.time()


    elif args.infinidepth_refine:
        infinidepth_start = 0
        infinidepth_end = 0
        infinidepth_stats = None
        infinidepth_start = time.time()
        infinidepth_output_dir = args.infinidepth_output_dir or os.path.join(args.output_dir, "infinidepth_depth")
        infinidepth_images = high_output_images if high_output_enabled else sequence.images
        infinidepth_image_names = high_output_image_names if high_output_enabled else sequence.image_names
        infinidepth_intrinsic = high_output_intrinsic if high_output_enabled else low_intrinsic
        infinidepth_stats = run_infinidepth_refinement(
            infinidepth_images,
            infinidepth_image_names,
            os.path.join(args.output_dir, "single_frame_depth", "depth_npy"),
            infinidepth_output_dir,
            infinidepth_intrinsic,
            model_path=args.infinidepth_model_path,
            device=device,
            depth_image_names=sequence.image_names,
            confidence_dir=os.path.join(args.output_dir, "single_frame_depth", "confidence"),
            input_size=(args.infinidepth_input_height, args.infinidepth_input_width),
        )
        infinidepth_end = time.time()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / (1024**2))

    with open(os.path.join(args.output_dir, "computation_stats.txt",), "w")as f:
        f.write(f"Runtime: {elapsed:.4f} seconds\n")
        f.write(f"Sequence Time: {seq_end - seq_start} seconds\n")
        f.write(f"Inference Time: {inf_end - inf_start} seconds\n")
        f.write(f"Tracking Time: {tracking_end - tracking_start} seconds\n")
        f.write(f"Dropped Untracked Frames: {len(dropped_untracked_frame_ids)}\n")
        if dropped_untracked_frame_ids:
            f.write(f"Dropped Untracked Frame IDs: {dropped_untracked_frame_ids}\n")
        f.write(f"BA Time: {ba_end - ba_start} seconds\n")
        f.write(f"BAE Time: {bae_end - bae_start} seconds\n")
        if bae_stats is not None:
            f.write(f"BAE Initial Loss: {bae_stats['initial_loss']}\n")
            f.write(f"BAE Ending Loss: {bae_stats['ending_loss']}\n")
            f.write(f"BAE Observations: {bae_stats['num_observations']}\n")
            f.write(f"BAE Cameras: {bae_stats['num_cameras']}\n")
            f.write(f"BAE Dropped Cameras: {bae_stats['num_dropped_cameras']}\n")
            f.write(f"BAE Points: {bae_stats['num_points']}\n")
        if single_frame_depth_stats is not None:
            f.write(f"Single Frame Depth Frames: {single_frame_depth_stats['num_depth_frames']}\n")
            f.write(f"Single Frame Depth Nonzero Pixels: {single_frame_depth_stats['num_nonzero_depth_pixels']}\n")
            f.write(f"Single Frame Depth Conf Percentile: {single_frame_depth_stats['conf_threshold']}\n")
            f.write(f"Single Frame Depth Conf Threshold Value: {single_frame_depth_stats['conf_threshold_value']}\n")
            f.write(f"Single Frame Depth Masked By Conf: {single_frame_depth_stats['num_masked_by_conf']}\n")
            f.write(f"Single Frame Depth Output Dir: {single_frame_depth_stats['output_dir']}\n")
        if dense_debug_stats is not None:
            f.write(f"Dense Debug Observations: {dense_debug_stats['num_observations']}\n")
            f.write(f"Dense Debug Frames With Anchors: {dense_debug_stats['num_frames_with_anchors']}\n")
        if dense_correction_stats is not None:
            f.write(f"Dense Correction Method: {dense_correction_stats['method']}\n")
            f.write(f"Dense Correction Corrected Frames: {dense_correction_stats['corrected_frames']}\n")
            f.write(f"Dense Correction Skipped Frames: {dense_correction_stats['skipped_frames']}\n")
            f.write(f"Dense Correction Clamped Frames: {dense_correction_stats['clamped_frames']}\n")
        if dense_depth_stats is not None:
            f.write(f"Dense Depth Backend: {dense_depth_stats['backend']}\n")
            f.write(f"Dense Depth Frames: {dense_depth_stats['num_depth_frames']}\n")
            f.write(f"Dense Depth Merged Points: {dense_depth_stats['num_merged_points']}\n")
            f.write(f"Dense Depth Nonzero Pixels: {dense_depth_stats['num_nonzero_depth_pixels']}\n")
            f.write(f"Dense Depth Output Dir: {dense_depth_stats['output_dir']}\n")
        if high_output_enabled:
            f.write(f"High Dataset: {args.high_dataset}\n")
            f.write(f"High Output Image Shape: {tuple(high_output_images.shape[-2:])}\n")
        if lingbot_stats is not None:
            f.write(f"LingBot Depth Model: {lingbot_stats['model']}\n")
            f.write(f"LingBot Depth Processed: {lingbot_stats['num_processed']}\n")
            f.write(f"LingBot Depth Skipped: {lingbot_stats['num_skipped']}\n")
            f.write(f"LingBot Depth Output Dir: {lingbot_stats['output_dir']}\n")
        if infinidepth_stats is not None:
            f.write(f"InfiniDepth Model: {infinidepth_stats['model']}\n")
            f.write(f"InfiniDepth Processed: {infinidepth_stats['num_processed']}\n")
            f.write(f"InfiniDepth Skipped: {infinidepth_stats['num_skipped']}\n")
            f.write(f"InfiniDepth Input Size: {infinidepth_stats['input_size']}\n")
            f.write(f"InfiniDepth Output Resolution Mode: {infinidepth_stats['output_resolution_mode']}\n")
            f.write(f"InfiniDepth Output Dir: {infinidepth_stats['output_dir']}\n")
        if lingbot_refined_ply_stats is not None:
            f.write(f"LingBot Refined PLY Points: {lingbot_refined_ply_stats['num_points']}\n")
            f.write(f"LingBot Refined PLY Valid Before Cap: {lingbot_refined_ply_stats['num_valid_points_before_cap']}\n")
            f.write(f"LingBot Refined PLY Missing Depths: {lingbot_refined_ply_stats['num_missing_depths']}\n")
            f.write(f"LingBot Refined PLY Projected Mask: {lingbot_refined_ply_stats['valid_mask_depth_npy_dir']}\n")
        f.write(f"Dense World Collect Time: {dense_collect_end - dense_collect_start} seconds\n")
        f.write(f"Dense PLY Time: {dense_ply_end - dense_ply_start} seconds\n")
        f.write(f"Single Frame Depth Time: {single_frame_depth_end - single_frame_depth_start} seconds\n")
        f.write(f"Dense Depth Time: {dense_depth_end - dense_depth_start} seconds\n")
        f.write(f"LingBot Depth Time: {lingbot_end - lingbot_start} seconds\n")
        f.write(f"LingBot Refined PLY Time: {lingbot_refined_ply_end - lingbot_refined_ply_start} seconds\n")
        f.write(f"InfiniDepth Time: {infinidepth_end - infinidepth_start} seconds\n")
        f.write(f"Peak GPU memory: {peak_mem:.2f} MiB\n")

    write_recon_to_colmap(
        args.output_dir,
        final_predictions,
        sequence.images,
        sequence.image_names,
        stride=args.stride,
        conf_threshold=args.point_vis_threshold,
        format=args.format,
        shared_camera=args.global_ba,
    )

    if high_output_enabled:
        high_colmap_intrinsic = high_output_intrinsic
        if args.global_ba:
            high_colmap_intrinsic = np.mean(high_colmap_intrinsic, axis=0, keepdims=True)
        high_h, high_w = tuple(high_output_images.shape[-2:])
        high_cameras_path = os.path.join(args.output_dir, "colmap", "high_cameras.txt")
        write_colmap_cameras_txt(high_cameras_path, high_colmap_intrinsic, high_w, high_h)
        print(f"[OUTPUT WRITING] Wrote high-resolution camera intrinsics to {high_cameras_path}")
    

if __name__ == "__main__":
    main()
