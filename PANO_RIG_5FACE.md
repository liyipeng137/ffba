# Five-face panorama rig FFBA

This is the active panorama input contract and end-to-end test procedure.

## Input contract

`--dataset` must contain five synchronized directories:

```text
dataset/
  center/
  left/
  right/
  up/
  down/
  pano_rig_manifest.json
```

Every directory must contain the same unique basenames. Each image is a square
90° cubemap face. The stable sensor order is
`center, left, right, up, down`, corresponding to
`Front, Left, Right, Up, Down` in `pytorch360convert.e2c`. The Back face is
intentionally omitted.

All five virtual cameras are co-located. The fixed pose convention is:

```text
sensor_from_world = sensor_from_rig[sensor] * rig_from_world[frame]
```

`center` is the rig reference sensor. Left/right differ from center by -90°/+90°
yaw; up/down differ by +90°/-90° pitch according to the exact matrices stored in
`utils/pano_rig.py` and in every stage summary.

## Prepare data from an ERP video

Install `pytorch360convert==0.2.3`, then run:

```bash
source /opt/conda/bin/activate
python scripts/prepare_pano_rig_from_erp.py \
  --input /kiri/dataset/local_test_erp.mp4 \
  --output-dir /kiri/dataset/local_test_cubemap5_50 \
  --num-frames 50 \
  --device cuda
```

The script uses FFmpeg/ffprobe for video probing and decoding and
`pytorch360convert.e2c` for ERP-to-cubemap projection. The default face size is
ERP width divided by four. It writes source frame indices, timestamps, tool
versions, face mapping, and decode settings to `pano_rig_manifest.json`.

The output directory must not already exist, so a partial or previous result is
never silently overwritten.

## End-to-end FFBA

```bash
source /opt/conda/bin/activate
python run_merg3r_gluemap_pipeline.py \
  --dataset /kiri/dataset/local_test_cubemap5_50 \
  --output_dir /kiri/dataset/local_test_cubemap5_ffba \
  --path_tracker /root/.cache/torch/hub/checkpoints/vggsfm_v2_tracker.pt \
  --num_images 50 \
  --pair_k_pose 25 \
  --neighbors_per_center 12 \
  --vggsfm_group_batch_size 1 \
  --no-stop_before_bae \
  --ba_backend bae \
  --bae_max_num_iterations 3 \
  --num_refinement_iterations 1 \
  --augmented_ba_max_filter_iterations 1 \
  --bae_robust_loss huber \
  --bae_huber_delta 1.0 \
  --select_track_min_support 64 \
  --filter_reproj_error_threshold 1.0
```

`--num_images` counts rig timestamps, not individual face images. The example
therefore processes 50 rig poses and 250 images.

These are the functional/scale-validation settings used on the 50-timestamp
sample. Increase BAE and outer refinement iterations for a quality-tuning run.

## Pipeline behavior

Stage A loads only `center/` and runs Pi3X/MERG3R to estimate one coarse
`rig_from_world` per timestamp plus center depth. Stage B then:

1. loads and preprocesses all five faces;
2. expands each center pose with the fixed `sensor_from_rig` matrices;
3. constructs a cross-frame, frustum-aware pair graph for all five faces;
4. runs SIFT and VGGSfM with every face eligible as a tracking group center;
5. merges prior and SIFT databases and triangulates tracks;
6. runs BAE with one optimizable SE(3) block per rig timestamp.

The rig extrinsics are registered as fixed tensors in BAE. A timestamp's five
image observations share one pose index. Gauge fixing operates on rig frames,
not independent face images. Intrinsics are represented per sensor, so all
timestamps of one face share a camera record.

Weakly textured faces can legitimately finish with zero selected 3D
observations after triangulation and reprojection filtering. They still remain
registered in the native COLMAP rig and retain the exact fixed relative pose.
Use the per-image observation counts to distinguish this data-quality condition
from a missing input or pairing error.

## Outputs and acceptance checks

Important outputs:

- `pipeline_stage_b_pano_summary.json`: five-face contract, matrices, pair graph,
  depths and expanded pose shapes.
- `pre_bae_rig/`: native COLMAP rig seed with one frame per timestamp and five
  registered images per frame.
- `refined_gluemap_aba/`: final sparse reconstruction.
- `refine_stats.json`: track counts, filters, BAE backend summary, pose blocks,
  losses and timings.
- `post_bae_rig_pose_audit.json`: post-optimization co-location and relative
  orientation audit.

A successful rig-aware run must satisfy:

- Stage A image count equals the timestamp count.
- Stage B image count is five times the timestamp count.
- `rig_enabled` is true and `num_pose_blocks == num_rig_frames`.
- the final reconstruction contains one rig, one frame per timestamp, and five
  registered sensors per frame;
- post-BAE projection-center spread and relative-orientation errors are near
  floating-point precision.

## Tests

```bash
source /opt/conda/bin/activate
pytest -q \
  tests/test_prepare_pano_rig_from_erp.py \
  tests/test_pano_rig.py \
  tests/test_bae_rig_constraints.py \
  tests/test_gluemap_refine_core.py
```

The tests cover sampling and face mapping, exact cubemap rotations, pair graph
coverage, native COLMAP rig round trips, shared BAE pose indices, multi-sensor
intrinsics, gauge fixing, rig pose writeback, and the non-rig compatibility path.
