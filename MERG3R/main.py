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
    parser.add_argument("--subset_size", type=int, default=55)
    parser.add_argument("--num_images", type=int, default=-1)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--write_local", action="store_true")
    parser.add_argument("--ba_type", type=str, default="gradient")
    parser.add_argument("--alignment_type", type=str, default="weighted_iterative")
    parser.add_argument("--global_ba", action="store_true")
    parser.add_argument("--bae_refine", action="store_true", help="Run BAE sparse refinement after MERG3R global BA.")
    parser.add_argument("--bae_iters", type=int, default=20)
    parser.add_argument("--bae_optimize_intrinsics", action="store_true")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sequence_type", type=str, default="video")
    parser.add_argument("--lr", type=float, default=1e-4)  # 3e-4
    parser.add_argument("--epoch", type=int, default=300)
    parser.add_argument("--max_reproj", type=float, default=8.0)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument(
        "--dense_max_points",
        type=int,
        default=20_000_000,
        help="Maximum points to export in dense_model_points.ply. Set <=0 to disable.",
    )
    parser.add_argument("--model", type=str, default="pi3x", choices=['vggt', 'pi3', 'pi3x'])
    parser.add_argument("--pi3x_ckpt", type=str, default=None, help="Optional local Pi3X checkpoint path. If omitted, loads yyfz233/Pi3X.")
    parser.add_argument("--multi_dirs", action="store_true")
    parser.add_argument("--point_vis_threshold", type=float, default=50.0)
    parser.add_argument("--tracking_type", type=str, default="graph", choices=['graph', 'video'])
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--splitting_type", type=str, default="interleave", choices=['interleave', 'zigzag', 'threshold', "original", "original_threshold"])
    parser.add_argument("--format", type=str, default="txt", choices=['txt', 'bin'])
    args = parser.parse_args()

    
    return args


def main():
    
    args = parse_args()

    args_dict = vars(args)
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this pipeline. No CUDA device was detected.")
    device = "cuda"

    print(f"[MAIN] Output to {args.output_dir}")
    subsample = args.subsample

    os.makedirs(args.output_dir, exist_ok=True)
    # Save to JSON
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    images, image_names = process_images(args.dataset, subsample, device, args.num_images, args.multi_dirs)

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

    
    model, _ = load_model(args.model, device=device, pi3x_ckpt=args.pi3x_ckpt)
    inf_start = time.time()

    sequence.predictions = run_inference_step_by_step(model, batches, size_hw, device, need_features=False)
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

    if 'local_points' in final_predictions:
        export_dense_local_point_map_ply(
            os.path.join(args.output_dir, "dense_model_points.ply"),
            final_predictions['local_points'],
            final_predictions['extrinsic'],
            sequence.images,
            final_predictions['depth_conf'],
            conf_threshold=args.point_vis_threshold,
            stride=1,
            max_points=args.dense_max_points if args.dense_max_points > 0 else None,
        )

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
    

if __name__ == "__main__":
    main()
