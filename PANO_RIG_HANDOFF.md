# Panorama Rig Pipeline: Design and Progress

> Historical note: this file documents the earlier three-face, pre-BAE
> milestone. It is retained for implementation history only. The active
> five-face input contract and rig-aware BAE workflow are documented in
> [PANO_RIG_5FACE.md](PANO_RIG_5FACE.md).

Last updated: 2026-08-28  
Branch: `codex/pano-rig`  
Branch base: `1f919d7` (`Tune final refinement with a separate BAE Huber delta`)

## Goal of this branch

Build a panorama-specific reconstruction path. Compatibility with the previous
single-camera input path is not a goal on this branch.

The first milestone stops immediately before augmented triangulation/BAE. Its
purpose is to verify that center-only feed-forward poses can be expanded into a
valid fixed rig and that all three perspective streams can participate in
SIFT/VGGSfM tracking without losing the frame relationship.

## Input contract for v1

Prepare this directory structure:

```text
DATASET/
  left/
    000000.png
    000001.png
    ...
  center/
    000000.png
    000001.png
    ...
  right/
    000000.png
    000001.png
    ...
```

The contract is:

1. `left/`, `center/`, and `right/` contain the same unique basenames. Sorting
   the basenames lexicographically must produce temporal order.
2. The three files with one basename are synchronized virtual cameras from the
   same panorama/video timestamp.
3. Every source image is a distortion-free pinhole view. All images use the
   same source dimensions, aspect ratio, square-pixel convention, and horizontal
   FOV. Rectangular images are supported.
4. Sensor yaw is fixed to `left=-60°`, `center=0°`, `right=+60°`; pitch and roll
   are zero. Positive yaw points toward camera-right.
5. The three virtual views have the same optical center. Rig translation is
   exactly zero in v1.
6. The current test-data/default contract is `1920x1080` (16:9) with horizontal
   FOV `110°`. If extraction uses another horizontal FOV, pass
   `--pano_hfov_degrees`. Intrinsics are derived from this known extraction FOV
   and adjusted for the pipeline resize and center crop. For the uncropped
   1920x1080 input, this gives approximately `fx=fy=672.20`, `cx=960`,
   `cy=540`, and vertical FOV `77.55°`.
7. `--num_images` means number of rig timestamps and `--subsample` is applied
   identically to all three directories. Recursive input (`--multi_dirs`) is not
   supported.

At least two complete triplets are required. The default run mode is
`--stop-before-bae`; `--no-stop-before-bae` intentionally raises because the
rig-aware BAE implementation is the next milestone.

## Preparing input from an ERP video

Use `scripts/prepare_pano_rig_from_erp.py` when the source is a stitched 2:1
equirectangular panorama video. The script uniformly samples source frames and
generates all three synchronized pinhole views in one FFmpeg filter graph.

```bash
python scripts/prepare_pano_rig_from_erp.py \
  --input /kiri/dataset/local_test_erp.mp4 \
  --output-dir /kiri/dataset/local_test_pano_rig_50 \
  --num-frames 50
```

Defaults match this branch's input contract: PNG output, `1920x1080`, HFOV
`110°`, and yaw `left=-60°`, `center=0°`, `right=+60°`. The vertical FOV is
derived from the horizontal FOV and aspect ratio (`77.55°` for the default), so
the generated views have square pixels. FFmpeg yaw is used directly: negative
yaw samples decreasing ERP x (left), and positive yaw samples increasing ERP x
(right).

The script refuses to reuse an existing output directory, constructs the data
in a temporary sibling directory, verifies every triplet and image size, then
renames it into place. `pano_rig_manifest.json` records the exact decoded source
frame indices, timestamps, projection parameters, intrinsics, yaw convention,
and FFmpeg filter graph.

The default test command is:

```bash
python run_merg3r_gluemap_pipeline.py \
  --dataset /path/to/pano_dataset \
  --output_dir /path/to/pano_output \
  --path_tracker /path/to/vggsfm_v2_tracker.pt
```

`--pano_hfov_degrees 110` is implicit in this command. Keep the explicit flag
when recording experiments if the extraction settings may otherwise be unclear.

## Implemented flow

### Stage A: center-only feed-forward reconstruction

Only `DATASET/center` is passed to the existing pi3x/MERG3R feed-forward and
alignment path. Its output supplies one `rig_from_world` pose per timestamp and
center-only depth diagnostics.

### Stage B: three-view image set and pose expansion

All three directories are preprocessed independently with the same pyramid
configuration, then interleaved in frame-major order:

```text
frame 0 left, frame 0 center, frame 0 right,
frame 1 left, frame 1 center, frame 1 right, ...
```

The center camera is COLMAP's rig reference sensor. For each timestamp:

```text
sensor_from_world = sensor_from_rig * rig_from_world
```

`sensor_from_rig` is fixed to yaw `[-60°, 0°, +60°]` and zero translation.
Thus all three images in a frame have the same projection center while their
viewing axes differ by the configured yaw.

Depth remains center-only. Side depth is not synthesized or copied from the
center because that would have the wrong viewing-ray geometry.

### Pair selection and VGGSfM groups

Every image, including left and right, may be a VGGSfM group center. This is
intentional: queries originating in the side views add constraints in parts of
the scene that the center stream does not observe well.

Pair candidates obey these v1 rules:

- only images from different rig timestamps are paired;
- same-frame left/center/right pairs are excluded because co-located cameras
  provide no triangulation baseline;
- viewing-axis angle must be below `--pano_pair_max_axis_angle` (default 85°),
  connecting adjacent 60° views but excluding the 120° left/right combination;
- candidates are ranked by camera-center distance, frame gap, and angle;
- selection is round-robin over target sensor directions so same-direction
  images cannot consume the entire `--pair_k_pose` budget.

The current VGGSfM strategy is `pose`. `projected_overlap` is disabled because
Stage A has no geometrically valid side-view depth.

### Frame filtering

Frame filtering uses only the center image's SIFT plus VGGSfM observation count.
If center is below `--min_frame_observations`, the complete left/center/right
triplet is removed. All image, feature, pair, track, intrinsic, depth, and rig
indices are remapped together.

### Intrinsics and COLMAP rig output

V1 does not optimize intrinsics. It creates three COLMAP camera records, one per
rig sensor, even when all three parameter vectors are identical. This is needed
because a COLMAP rig sensor is identified by its camera ID.

After SIFT and VGGSfM prior databases are merged, the fresh merged database is
configured with one rig and one frame per kept timestamp. The pre-BAE COLMAP
model is written with the same rig/frame/image relationships.

## Outputs to inspect

The main coarse-pose inspection artifacts are:

- `pre_bae_rig/`: COLMAP binary model with one rig, three cameras, one frame per
  timestamp, and three images per frame. It intentionally has no 3D points yet.
- `pre_bae_rig_pose_audit.json`: every image projection center and viewing
  direction, plus aggregate same-frame center spread and yaw error.
- `database_merged.db`: merged SIFT + VGGSfM correspondences with rig and frame
  tables configured.
- `pipeline_stage_a_summary.json`: center-only feed-forward stage.
- `pipeline_stage_b_pano_summary.json`: input, rig expansion, fixed intrinsics,
  and pair-graph metadata.
- `refine_stats.json`: VGGSfM/SIFT/filtering statistics and explicit
  `stopped_before_bae_as_requested` status.
- `pred_depth/`: center images only. No side-view depth is exported.

For a first real run, check:

1. `max_projection_center_spread` in the pose audit is near numerical zero.
2. `max_absolute_yaw_error_degrees` is near numerical zero.
3. COLMAP shows three viewing directions at every center-derived rig location.
4. Pair/group statistics have no zero-degree images and left/right images are
   present as group centers.
5. Filtering only removes complete triplets.

These checks validate the construction and data plumbing. They do not validate
whether the center Stage A trajectory itself is accurate.

## Code changes

- `run_merg3r_gluemap_pipeline.py`
  - center-only Stage A;
  - three-directory Stage B loading;
  - exact FOV-derived pyramid intrinsics;
  - rig pose expansion and rig-aware pair graph;
  - default pre-BAE stop.
- `utils/pano_rig.py`
  - rig metadata and transforms;
  - pair selection;
  - center-driven filtering helpers;
  - COLMAP rig reconstruction/database writers;
  - pose audit generation.
- `utils/gluemap_spv_refine.py`
  - rig image names and camera mapping;
  - complete-triplet filtering;
  - center-only depth handling;
  - three-camera prior database;
  - rig-aware pre-BAE outputs.
- `utils/gluemap_refine_core.py`
  - forced keep indices for grouped filtering;
  - multi-camera support in the prior-track database writer.
- `tests/test_pano_rig.py`
  - transform, pair, filtering, intrinsics, reconstruction, and database tests.

## Validation completed

Using `/Users/lyp/pycodex/`:

- `pytest -q tests/test_pano_rig.py`: 6 passed, including 1920x1080 HFOV 110
  intrinsics before and after pyramid resize/crop.
- `pytest -q tests/test_gluemap_refine_core.py`: 15 passed.
- `ruff check` on changed Python files and the new test: passed.
- `python -m py_compile` on changed pipeline modules: passed.
- pycolmap `4.1.1` rig reconstruction write/reload and database rig/frame
  round-trip: covered by the tests and passed.

The local `pycodex` environment currently has no `torch`, so the pipeline entry
cannot be imported or run here. No CUDA, VGGSfM, SIFT, or end-to-end dataset run
has been completed yet.

The repository pins `pycolmap-cuda12==4.1.0` in GlueMap while the local API test
used pycolmap `4.1.1`. The rig APIs used here exist in 4.1.1; the CUDA target
environment still needs a real-run compatibility check.

## Next milestone: rig-aware BAE

BAE must optimize one pose per rig frame, not three independent image poses.
The intended state and residual mapping are:

```text
optimized: rig_from_world[frame]
fixed:     sensor_from_rig[sensor]
fixed v1:  intrinsics[camera]

observation(image, point)
  -> frame_id, sensor_id, camera_id
  -> sensor_from_rig[sensor] * rig_from_world[frame]
  -> project(point, intrinsics[camera])
```

Gauge fixing must select rig frames. Optimized poses must be written once per
frame. A later intrinsic phase should optimize shared parameters per rig sensor
camera, with explicit bounds or priors; it should not create one independent
intrinsic vector per image.

Before implementing BAE, the first real-data checkpoint should decide whether
the center-only trajectory plus fixed yaw gives plausible coarse side poses and
whether left/right-as-center groups improve side coverage without excessive bad
tracks.
