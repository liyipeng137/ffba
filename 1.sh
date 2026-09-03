# /bin/bash
set -euo pipefail

cd /kiri/FeedForwardWithBA


DATASET=/kiri/ff_data/k_room
TRACKER=/root/.cache/torch/hub/checkpoints/vggsfm_v2_tracker.pt
RUN_ROOT=/kiri/tmp/ab_covis_$(date +%Y%m%d_%H%M%S)

A_OUT=$RUN_ROOT/A_pose
B_OUT=$RUN_ROOT/B_ordered_motion

mkdir -p "$A_OUT" "$B_OUT"

{
  date -Iseconds
  git rev-parse HEAD
  git status --short --branch
  python --version
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
} > "$RUN_ROOT/environment.txt"

# 请设置成你当前老版本实际使用的 K。
# 如果质量基线是 K=25，就把这里改成 25；A/B 必须保持相同。
K=16

COMMON_ARGS=(
  --dataset "$DATASET"
  --device cuda:0
  --num_images -1
  --subsample 1

  --image_pyramid
  --stage1_downscale_n 4
  --stage1_multiple 14
  --stage2_scale_factor 0
  --image_pyramid_workers 16

  --sequence_type shortest_path
  --subset_size 120
  --overlap 5
  --alpha 0.7
  --splitting_type interleave
  --alignment_type weighted_iterative

  --pair_k_pose 25
  --pair_pose_rotation_threshold 30.0

  --path_tracker "$TRACKER"
  --neighbors_per_center "$K"
  --vggsfm_group_batch_size 3
  --vggsfm_query_points 1024
  --aliked_detection_threshold 0.005
  --vggsfm_vis_threshold 0.5
  --vggsfm_score_threshold 0.0

  --prior_match_topology star
  --prior_snap_threshold 1.0
  --prior_keypoint_merge_threshold 0.001
  --min_frame_observations 10

  --ba_backend bae
  --bae_max_num_iterations 20
  --bae_max_observations 2000000
  --bae_optimize_intrinsics
  --bae_fix_gauge two_cams
  --bae_robust_loss huber
  --bae_huber_delta 1.0
  --final_bae_huber_delta 2.0

  --num_refinement_iterations 3
  --select_track_min_support 512
  --filter_reproj_error_type angular
  --filter_reproj_error_threshold 1.0
)

# A
# python -u run_merg3r_gluemap_pipeline.py   "${COMMON_ARGS[@]}"   --output_dir "$A_OUT"   --vggsfm_group_strategy pose   2>&1 | tee "$A_OUT/run.log"

# B
python -u run_merg3r_gluemap_pipeline.py \
  "${COMMON_ARGS[@]}" \
  --output_dir "$B_OUT" \
  --vggsfm_group_strategy ordered_motion \
  --ordered_temporal_window 8 \
  --ordered_min_projected_overlap 0.10 \
  --ordered_min_projected_grid_coverage 0.25 \
  --ordered_min_projected_visible_ratio 0.25 \
  --ordered_motion_target 0.08 \
  --ordered_sift_pair_budget_ratio 1.0 \
  --ordered_vggsfm_neighbor_budget_ratio 1.0 \
  --ordered_vggsfm_hard_max_neighbors 32 \
  --ordered_sift_target_matches 512 \
  --ordered_sift_target_grid_coverage 0.5 \
  --ordered_sift_deficit_weight 0.5 \
  2>&1 | tee "$B_OUT/run.log"