import os
import glob
import time
import argparse

import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
import cv2
import matplotlib.pyplot as plt

import vggt_slam.slam_utils as utils
from vggt_slam.solver import Solver
from vggt_slam.pi3_solver import Pi3Solver, load_pi3x_model
from vggt_slam.submap import Submap

from vggt.models.vggt import VGGT

parser = argparse.ArgumentParser(description="VGGT-SLAM demo")
parser.add_argument("--base_model", type=str, default="pi3x", choices=["vggt", "pi3x"], help="Base inference model to use")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images")
parser.add_argument("--vis_map", action="store_true", help="Visualize point cloud in viser as it is being build, otherwise only show the final map")
parser.add_argument("--vis_voxel_size", type=float, default=None, help="Voxel size for downsampling the point cloud in the viewer (e.g. 0.05 for 5 cm). Default: no downsampling")
parser.add_argument("--run_os", action="store_true", help="Enable open-set semantic search with Perception Encoder CLIP and SAM3")
parser.add_argument("--vis_flow", action="store_true", help="Visualize optical flow from RAFT for keyframe selection")
parser.add_argument("--log_results", action="store_true", help="save txt file with results")
parser.add_argument("--skip_dense_log", action="store_true", help="by default, logging poses and logs dense point clouds. If this flag is set, dense logging is skipped")
parser.add_argument("--log_path", type=str, default="poses.txt", help="Path to save the log file")
parser.add_argument("--submap_size", type=int, default=32, help="Number of new frames per submap, does not include overlapping frames or loop closure frames")
parser.add_argument("--overlapping_window_size", type=int, default=3, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW. Number of overlapping frames, which are used in SL(4) estimation")
parser.add_argument("--max_loops", type=int, default=1, help="ONLY DEFAULT OF 1 SUPPORTED RIGHT NOW or 0 to disable loop closures.")
parser.add_argument("--min_disparity", type=float, default=50, help="Minimum disparity to generate a new keyframe")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--lc_thres", type=float, default=0.95, help="Threshold for image retrieval. Range: [0, 1.0]. Higher = more loop closures")
parser.add_argument("--export_pcd", type=str, default=None, help="Path to export the final merged point cloud (.pcd or .ply). If not set, no export.")
parser.add_argument("--export_transforms_json", type=str, default=None, help="Path to export LingBot-compatible transforms.json from optimized graph poses.")
parser.add_argument(
    "--export_transforms_path_mode",
    type=str,
    default="basename",
    choices=["basename", "absolute"],
    help="How image file_path is written in exported transforms.json.",
)
parser.add_argument("--export_dense_frames_dir", type=str, default=None, help="Directory to export graph-optimized dense points per unique image frame.")
parser.add_argument(
    "--export_dense_include_overlap_duplicates",
    action="store_true",
    help="Export duplicated overlap-frame dense nodes as separate .npz files. transforms.json still keeps unique images.",
)
parser.add_argument("--pi3_ckpt", type=str, default=None, help="Optional path to a Pi3X checkpoint. If not set, loads yyfz233/Pi3X")
parser.add_argument("--lingbot_transforms_json", type=str, default=None, help="Optional LingBot transforms.json used as a coarse prior for Pi3 loop candidate detection")
parser.add_argument("--loop_window_radius", type=int, default=2, help="Pi3 loop verification window radius around the candidate center frame")
parser.add_argument("--min_loop_frame_gap", type=int, default=45, help="Minimum global frame index gap between current and historical frames to consider a loop candidate")
parser.add_argument("--loop_translation_thresh", type=float, default=0.2, help="Maximum LingBot prior translation distance for loop candidate gating")
parser.add_argument("--loop_rotation_thresh_deg", type=float, default=15.0, help="Maximum LingBot prior rotation difference in degrees for loop candidate gating")
parser.add_argument("--fx", type=float, default=290.5493469238281, help="Shared Pi3X focal length fx in pixels for original images")
parser.add_argument("--fy", type=float, default=429.0402526855469, help="Shared Pi3X focal length fy in pixels for original images")
parser.add_argument("--cx", type=float, default=175.0, help="Shared Pi3X principal point cx in pixels for original images")
parser.add_argument("--cy", type=float, default=238.0, help="Shared Pi3X principal point cy in pixels for original images")

def main():
    """
    Main function that wraps the entire pipeline of VGGT-SLAM.
    """
    args = parser.parse_args()
    manual_intrinsics = [args.fx, args.fy, args.cx, args.cy]
    has_partial_intrinsics = any(v is not None for v in manual_intrinsics) and not all(v is not None for v in manual_intrinsics)
    if has_partial_intrinsics:
        raise ValueError("If using manual intrinsics, provide all of --fx --fy --cx --cy.")
    if args.base_model == "pi3x" and not all(v is not None for v in manual_intrinsics):
        raise ValueError("Pi3X mode requires shared original-image intrinsics: --fx --fy --cx --cy.")
    if args.min_loop_frame_gap is None:
        args.min_loop_frame_gap = 2 * args.submap_size
    if args.base_model == "pi3x" and args.max_loops > 0 and args.lingbot_transforms_json is None:
        print("Pi3X loop closure requires --lingbot_transforms_json; forcing max_loops=0.")
        args.max_loops = 0

    use_optical_flow_downsample = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if args.base_model == "pi3x":
        solver = Pi3Solver(
            init_conf_threshold=args.conf_threshold,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            lc_thres=args.lc_thres,
            vis_voxel_size=args.vis_voxel_size,
            lingbot_transforms_json=args.lingbot_transforms_json,
            loop_window_radius=args.loop_window_radius,
            min_loop_frame_gap=args.min_loop_frame_gap,
            loop_translation_thresh=args.loop_translation_thresh,
            loop_rotation_thresh_deg=args.loop_rotation_thresh_deg,
        )
    else:
        solver = Solver(
            init_conf_threshold=args.conf_threshold,
            lc_thres=args.lc_thres,
            vis_voxel_size=args.vis_voxel_size
        )

    print(f"Initializing and loading {args.base_model} model...")


    # if args.run_os:
    #     from sam3.model_builder import build_sam3_image_model
    #     from sam3.model.sam3_image_processor import Sam3Processor
    #     import core.vision_encoder.pe as pe
    #     import core.vision_encoder.transforms as transforms

    #     sam3_model = build_sam3_image_model()
    #     processor = Sam3Processor(sam3_model, confidence_threshold=0.50)

    #     clip_model = pe.CLIP.from_config("PE-Core-L14-336", pretrained=True)  # Downloads from HF
    #     clip_model = clip_model.cuda()
    #     clip_tokenizer = transforms.get_text_tokenizer(clip_model.context_length)
    #     clip_preprocess = transforms.get_image_transform(clip_model.image_size)
    # else:
    clip_model, clip_preprocess = None, None
    clip_tokenizer = None

    if args.base_model == "pi3x":
        model = load_pi3x_model(torch.device(device), args.pi3_ckpt)
    else:
        model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

        model.eval()
        model = model.to(torch.bfloat16)  # use half precision
        model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = [f for f in glob.glob(os.path.join(args.image_folder, "*")) 
               if "depth" not in os.path.basename(f).lower() and "txt" not in os.path.basename(f).lower() 
               and "db" not in os.path.basename(f).lower()]

    image_names = utils.sort_images_by_number(image_names)
    downsample_factor = 1
    image_names = utils.downsample_images(image_names, downsample_factor)
    print(f"Found {len(image_names)} images")

    image_names_subset = []
    count = 0
    image_count = 0
    total_time_start = time.time()
    keyframe_time = utils.Accumulator()
    backend_time = utils.Accumulator()
    for image_name in tqdm(image_names):
        if use_optical_flow_downsample:
            with keyframe_time:
                img = cv2.imread(image_name)
                enough_disparity = solver.flow_tracker.compute_disparity(img, args.min_disparity, args.vis_flow)
                if enough_disparity:
                    image_names_subset.append(image_name)
                    image_count += 1
        else:
            image_names_subset.append(image_name)

        # Run submap processing if enough images are collected or if it's the last group of images.
        if len(image_names_subset) == args.submap_size + args.overlapping_window_size or image_name == image_names[-1]:
            count += 1
            print(image_names_subset)
            t1 = time.time()
            predictions = solver.run_predictions(image_names_subset, model, args.max_loops, clip_model, clip_preprocess)
            print("Solver total time", time.time()-t1)
            print(count, "submaps processed")

            solver.add_points(predictions)

            with backend_time:
                solver.graph.optimize()

            loop_closure_detected = len(predictions["detected_loops"]) > 0
            if args.vis_map:
                if loop_closure_detected:
                    solver.update_all_submap_vis()
                else:
                    solver.update_latest_submap_vis()
            
            # Reset for next submap.
            image_names_subset = image_names_subset[-args.overlapping_window_size:]

    total_time = time.time() - total_time_start
    average_fps = total_time / image_count
    print(image_count, "frames processed")
    print("Total time:", total_time)
    model_label = "Pi3X" if args.base_model == "pi3x" else "VGGT"
    print(f"Total time for {model_label} calls: {solver.vggt_timer.total_time:.4f}s")
    print(f"Average {model_label} time per frame:", solver.vggt_timer.total_time / image_count)
    print("Average loop closure time per frame:", solver.loop_closure_timer.total_time / image_count)
    print("Average keyframe selection time per frame:", keyframe_time.total_time / image_count)
    print("Average backend time per frame:", backend_time.total_time / image_count)
    print("Average semantic time per frame:", solver.clip_timer.total_time / image_count)
    print("Average total time per frame:", total_time / image_count)
    print("Average FPS:", 1 / average_fps)
        
    print("Total number of submaps in map", solver.map.get_num_submaps())
    print("Total number of loop closures in map", solver.graph.get_num_loops())


    if not args.vis_map:
        # just show the map after all submaps have been processed
        solver.update_all_submap_vis()

    if args.log_results:
        solver.map.write_poses_to_file(args.log_path, solver.graph, kitti_format=False)

        if not args.skip_dense_log:
            # Log the dense point cloud for each submap.
            solver.map.save_framewise_pointclouds(solver.graph, args.log_path.replace(".txt", "_logs"))

    if args.export_pcd is not None:
        print(f"Exporting point cloud to {args.export_pcd} ...")
        solver.map.write_points_to_file(solver.graph, args.export_pcd)
        print("Point cloud exported.")


    if args.export_transforms_json is not None:
        print(f"Exporting LingBot-compatible transforms.json to {args.export_transforms_json} ...")
        solver.map.write_lingbot_transforms_json(
            args.export_transforms_json,
            solver.graph,
            path_mode=args.export_transforms_path_mode,
            unique_images=True,
        )
        print("transforms.json exported.")

    if args.export_dense_frames_dir is not None:
        print(f"Exporting per-frame dense points to {args.export_dense_frames_dir} ...")
        solver.map.save_framewise_dense_points(
            solver.graph,
            args.export_dense_frames_dir,
            unique_images=not args.export_dense_include_overlap_duplicates,
        )
        print("Per-frame dense points exported.")

    # export_submap_local_dir = "./local_dir"
    # if export_submap_local_dir is not None:
    #     print(f"Exporting per-submap local point clouds to {export_submap_local_dir} ...")
    #     exported_files = solver.map.write_submaps_to_dir(
    #         solver.graph,
    #         export_submap_local_dir,
    #         coordinate_mode="local",
    #     )
    #     print(f"Exported {len(exported_files)} local submap point clouds.")

    # export_submap_world_dir = "./world_dir"
    # if export_submap_world_dir is not None:
    #     print(f"Exporting per-submap world point clouds to {export_submap_world_dir} ...")
    #     exported_files = solver.map.write_submaps_to_dir(
    #         solver.graph,
    #         export_submap_world_dir,
    #         coordinate_mode="world",
    #     )
    #     print(f"Exported {len(exported_files)} world submap point clouds.")



if __name__ == "__main__":
    main()
