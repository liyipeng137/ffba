# Dual-fisheye 9-camera handoff

> Last verified: 2026-09-01  
> Branch at verification: `codex/pano-rig`  
> Base commit before this document: `6c01da6`  
> This document is the entry point for the next agent. Older notes remain useful
> research history, but implementation status in this file takes precedence.

## 1. Goal and current conclusion

The target workflow is:

```text
dual fisheye source
  -> synchronized front/back physical fisheye frames
  -> nine square 90-degree pinhole views
  -> Stage A feed-forward reconstruction on front_center only
  -> SIFT + VGGSfM on all nine views
  -> triangulation and BAE with one rig pose per timestamp
  -> fixed or explicitly known physical sensor geometry
```

The nine output views are:

```text
front_center
front_ul  front_ur  front_dl  front_dr
back_ul   back_ur   back_dl   back_dr
```

The four constraints behind this layout are:

1. every output image samples exactly one physical fisheye;
2. no front/back stitching or cross-lens blending is allowed;
3. every output is square with 90-degree horizontal and vertical FOV;
4. diagonal views stay away from the low-quality fisheye rim when possible.

Current high-level status:

- ERP `Cubemap5` prepare and the complete five-face FFBA/BAE pipeline work.
- Both dual-fisheye prepare paths can generate the nine output directories.
- The INSV path uses embedded lens calibration and is the stronger dewarping path.
- The paired-frame path can apply a validated relative 2D roll correction, but
  uses an approximate fisheye model.
- The main pipeline is still hard-coded to the ERP five-face contract and does
  not read either dual-fisheye manifest.
- Raw INSV IMU exists and has been inspected, but it is not yet decoded/fused or
  applied by `prepare_dual_fisheye_rig_from_insv.py`.
- A physically correct two-center nine-camera rig has not yet been wired into
  pair selection, COLMAP output, or BAE.

## 2. Authoritative files

### Active pipeline and ERP baseline

- `run_merg3r_gluemap_pipeline.py`: active FFBA runner.
- `utils/pano_rig.py`: fixed ERP Cubemap5 geometry, pair graph, native COLMAP
  rig writing, and pose audit.
- `scripts/prepare_pano_rig_from_erp.py`: ERP video to five cubemap faces.
- `PANO_RIG_5FACE.md`: active five-face contract and commands.
- `third_party/gluemap/gluemap/estimators/bae_solver.py`: shared-pose BAE logic.
- `utils/gluemap_refine_core.py`: pipeline-independent BAE rig config builder.

### Dual-fisheye preparation

- `scripts/prepare_dual_fisheye_rig_from_insv.py`: calibrated INSV to nine views.
- `scripts/prepare_dual_fisheye_rig_from_frames.py`: paired fisheye images plus
  `gyro.txt` to approximately leveled nine views.
- `tests/test_prepare_dual_fisheye_rig_from_insv.py`: calibrated projection and
  INSV metadata/layout unit tests.
- `tests/test_prepare_dual_fisheye_rig_from_frames.py`: IMU parsing, roll,
  circle detection, and synthetic end-to-end tests.
- `INSV_DUAL_FISHEYE_9CAM_PROGRESS.md`: detailed research notes and INSV field
  investigation; some phase labels predate the current prepare scripts.

### Rig tests

- `tests/test_pano_rig.py`: Cubemap5 rotations, pair graph, and COLMAP rig tests.
- `tests/test_bae_rig_constraints.py`: one pose block per frame, fixed sensor
  transforms, sparse Jacobian dimensions, and writeback behavior.

`PANO_RIG_HANDOFF.md` documents an older three-face milestone and must not be
used as the current input contract.

## 3. Shared nine-view directory contract

Both dual-fisheye prepare scripts create this logical structure:

```text
output_dir/
  front_center/
  front_ul/
  front_ur/
  front_dl/
  front_dr/
  back_ul/
  back_ur/
  back_dl/
  back_dr/
  masks/
  previews/first_frame_9view_contact_sheet.jpg
  prepare_report.json
```

Optional diagnostic source directories differ by input path and are described
below. All nine view directories contain the same basenames in the same order.

Stable sensor roles:

| Sensor | Physical pixel source | Intended role |
| --- | --- | --- |
| `front_center` | front fisheye | only Stage A feed-forward input |
| `front_ul/ur/dl/dr` | front fisheye | Stage B supplementary views |
| `back_ul/ur/dl/dr` | back fisheye | Stage B supplementary views |

`front_center` plus eight diagonal views is described as “octahedron-like”, not
as an exact regular polyhedron. A hole around each fisheye center direction is
allowed for the four-diagonal subsets. There is intentionally no `back_center`.

OpenCV camera coordinates are used inside the prepare implementations:

```text
+x right, +y down, +z optical forward
```

Do not infer a common rig transform solely from `ul/ur/dl/dr`. These names are
currently defined in each physical lens's local projected image convention.

## 4. `prepare_dual_fisheye_rig_from_insv.py`

### 4.1 Input and command

Input is one original dual-track INSV file. The current sample is:

```text
/kiri/dataset/insv_data/VID_20260824_182629_00_004.insv
```

Small preparation example:

```bash
source /opt/conda/bin/activate
python scripts/prepare_dual_fisheye_rig_from_insv.py \
  --input /kiri/dataset/insv_data/VID_20260824_182629_00_004.insv \
  --output-dir /kiri/dataset/dual_fisheye_test/insv_rig9_5 \
  --num-frames 5 \
  --face-size 1024 \
  --save-source-frames
```

The output directory must not exist. Default values include:

```text
num_frames=50
face_size=1024
front_stream=0
back_stream=1
output HFOV=VFOV=90 degrees
rim margin=64 source pixels
```

`--start-time` and `--end-time` restrict the eligible interval. Uniform
midpoint-bin sampling is then applied within that interval.

Output names are renumbered by output order:

```text
000000.png, 000001.png, ...
```

The original source frame index and generated CFR timestamp are retained in the
manifest for every output group.

### 4.2 Video probing and synchronization

The script uses `ffprobe` to require two square video streams with identical:

- resolution;
- declared frame count;
- average/nominal frame rate;
- generated CFR timestamp sequence.

It does not enumerate actual per-frame PTS. It generates timestamps from
`nb_frames`, `avg_frame_rate`, and `start_time`, and intentionally rejects a
rate mismatch. FFmpeg is used only to select/decode raw BGR frames; projection
is implemented with NumPy and OpenCV.

For the X5 sample, both streams are 3840x3840 HEVC, approximately 29.97 fps,
with 3513 synchronized frames.

The current default labels stream 0 as front and stream 1 as back. This is a
working convention, not a verified container semantic. A known landmark or an
official ERP export must be used to freeze the naming.

### 4.3 Embedded calibration and dewarping

The script reads the INSV v3 private trailer without rewriting it. It parses the
metadata record and requires `offset_v3`. Each physical lens block contains:

```text
xi, fx, fy, cx, cy,
yaw, pitch, roll,
tx, ty, tz,
k1, k2, k3, p1, p2,
width, height, lens_type
```

The encoded 3840x3840 intrinsics are derived using the X4/X5 crop convention
already checked against telemetry-parser/Gyroflow. Pixel projection uses the
Insta360 unified `xi` model plus radial/tangential distortion:

```text
target pinhole ray
  -> view_to_lens_rotation
  -> normalized unified-camera projection
  -> k1/k2/k3 + p1/p2 distortion
  -> source fisheye pixel
  -> one cv2.remap interpolation
```

This is the calibrated dewarping path. The resulting images are modeled as
`PINHOLE` with no remaining distortion parameters, although empirical visual
comparison is still required at the source rim.

Important limitation: the raw `offset_v3` yaw/pitch/roll and translation are
stored in the manifest but are not applied as a verified `lens_from_rig`.
Their Euler order, signs, multiplication convention, and translation unit remain
unverified. In particular, `front_center` currently uses identity in the local
front-lens coordinate system. This is a likely contributor to the visually
tilted `front_center` that was observed.

### 4.4 Nine-view layout search

`front_center` looks along the front lens's local optical axis. For each physical
lens, four diagonal view axes are generated at azimuths corresponding to
`ul/ur/dl/dr`.

Unless `--diagonal-tilt-degrees` is supplied, the script:

1. starts from the exact octahedral tilt `54.7356 degrees`;
2. searches down to `--minimum-diagonal-tilt-degrees` (default 30 degrees);
3. searches a per-view image roll in `[-45, 45)`;
4. chooses the largest tilt for which every complete 90-degree square stays
   inside the requested radial and raster margin.

This optimizes source-image safety, not SfM matching quality. The selected tilt,
roll, margins, and each `view_to_lens_rotation` are written to the manifest.

### 4.5 Outputs and manifest

Optional `--save-source-frames` adds:

```text
fisheye_front/
fisheye_back/
```

The manifest is:

```text
dual_fisheye_rig_manifest.json
```

It records:

- INSV source/stream information and tool versions;
- source frame indices and generated timestamps;
- embedded and scaled lens calibration;
- raw front/back translation difference and its norm;
- static masks and remap statistics;
- selected nine-view local geometry;
- two `physical_center_group` labels: `front` and `back`;
- warnings for unverified front/back naming, Euler convention, translation
  units, and missing rolling-shutter correction.

### 4.6 What is not implemented in this script

- raw gyro/accelerometer record decoding;
- IMU scale/calibration and X5 axis conversion;
- video/IMU timestamp alignment;
- gyro integration or gyro+accelerometer attitude fusion;
- absolute gravity leveling or per-frame roll/pitch correction;
- rolling-shutter compensation;
- verified `lens_from_rig` or final `sensor_from_rig` transforms;
- use of the physical front/back baseline in the output image geometry.

The sample INSV has already been verified to contain raw IMU:

```text
is_raw_gyro=true
record id 3 size=2,346,240 bytes
20 bytes/sample
117,312 samples
timestamp step approximately 1000 microseconds (about 1 kHz)
```

The raw record begins roughly 166 ms before the first video frame timestamp,
which is useful for attitude-filter initialization. Open-source X5 parsing uses
the `yzX` IMU orientation convention, but this must still be visually validated
against this file before projection is changed.

## 5. `prepare_dual_fisheye_rig_from_frames.py`

### 5.1 Input and command

This path accepts only:

```text
input_dir/
  camera1/<matching square fisheye images>
  camera2/<matching square fisheye images>
  gyro.txt
```

`gyro.txt` rows are:

```text
image_name,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
```

An optional header is accepted. Matching uses the image stem, so the gyro name
may contain a supported image extension or omit it.

Example:

```bash
source /opt/conda/bin/activate
python scripts/prepare_dual_fisheye_rig_from_frames.py \
  --input-dir /kiri/dataset/dual_fisheye_test/3dgs_frames \
  --output-dir /kiri/dataset/dual_fisheye_test/frames_rig9_5 \
  --num-frames 5 \
  --face-size 1024 \
  --output-fov-degrees 90 \
  --fisheye-fov-degrees 200 \
  --diagonal-tilt-degrees 40 \
  --save-leveled-fisheye
```

Without `--num-frames`, every gyro row is processed. With it, uniform
midpoint-bin sampling is used. Output names preserve the input stem instead of
being renumbered.

Current semantic assumptions are:

```text
camera1 = front physical fisheye
camera2 = back physical fisheye
```

Both directories must contain every gyro stem. All input images must be square
and share one resolution. Extra images not referenced by `gyro.txt` are ignored.

### 5.2 Relative roll leveling

This script uses only accelerometer x/z to compute a 2D image roll:

```text
raw angle = unwrap(degrees(atan2(acc_x, acc_z)))
relative roll = filtered angle - reference angle + roll offset
camera1 image rotation = +relative roll
camera2 image rotation = -relative roll
```

The opposite camera signs were validated visually on the supplied paired-frame
sample. They remain a dataset convention rather than a calibrated lens mapping.

The reference defaults to the first gyro row and can be overridden with
`--roll-reference-frame`. Therefore the implementation assumes only relative
roll: the reference frame receives zero correction even if it is itself tilted.
Use `--roll-offset-degrees` to compensate a known fixed reference tilt.

`--accel-median-window` optionally applies an odd-width temporal median filter.
The implementation does not check accelerometer norm or reject samples affected
by linear acceleration.

Despite the filename `gyro.txt`, `gyro_x/y/z` are stored but not used because no
timestamps are available. `acc_y` is also unused. There is no pitch/yaw or full
3D attitude estimation.

### 5.3 Approximate fisheye model

No lens calibration is provided. The script:

1. detects the non-black circular support on the first/middle/last selected
   image for each camera;
2. takes median center/radius values;
3. clips the selected radius to the largest circle inscribed in the square
   raster so roll rotation cannot sample outside the JPEG;
4. assumes an equidistant fisheye with configurable total FOV, default 200
   degrees;
5. maps target pinhole lens angle linearly to source radius.

This produces useful approximate rectilinear views, but it is not calibrated
dewarping. Residual distortion is expected, especially near diagonal-view rims.
The manifest deliberately calls the output camera `nominal_PINHOLE`.

The nine-view geometry is fixed rather than searched:

```text
output size=1024 by default
output HFOV=VFOV=90 degrees by default
diagonal tilt=40 degrees by default
rim margin=16 source pixels by default
```

The per-frame roll inverse is composed into the static projection remap, so the
formal nine-view output is sampled from the raw fisheye in one interpolation.
`leveled_camera1/` and `leveled_camera2/` are optional diagnostics only and are
not used as intermediate projection inputs.

### 5.4 Outputs and manifest

The manifest is:

```text
dual_fisheye_frames_rig_manifest.json
```

It records original IMU rows, per-frame roll corrections, detected circles,
nominal pinhole intrinsics, local view rotations, masks, and warnings. This
schema is not identical to the INSV manifest.

The real three-frame smoke test on
`/kiri/dataset/dual_fisheye_test/3dgs_frames` successfully wrote all nine views.
After using the conservative inscribed radius, the static remaps had a 1.0 valid
pixel ratio; visually relevant out-of-raster black corners were removed.

## 6. ERP Cubemap5 pipeline: implemented baseline

### 6.1 Active input contract

The main runner currently requires exactly:

```text
dataset/
  center/
  left/
  right/
  up/
  down/
```

All directories must contain identical basenames and identical square images.
Every face is a standard 90-degree cubemap face. Stable order is:

```text
center, left, right, up, down
Front,  Left, Right, Up,   Down
```

The Back cubemap face is omitted. The five cameras share one optical center and
have zero relative translation. Exact rotations match
`pytorch360convert.e2c`, including Up/Down image roll.

`scripts/prepare_pano_rig_from_erp.py` writes `pano_rig_manifest.json`, but the
main runner does not read that file. It recognizes the contract through the
five hard-coded directory names and CLI HFOV.

### 6.2 Current Stage A and Stage B

Stage A:

- loads `center/` only;
- runs Pi3X/MERG3R feed-forward reconstruction;
- obtains one coarse `rig_from_world` and center depth per timestamp.

Stage B:

- loads all five faces;
- expands the center pose with fixed `sensor_from_rig` rotations;
- stacks images frame-major in the five-sensor order;
- computes per-sensor image-pyramid intrinsics from 90-degree HFOV;
- runs SIFT and VGGSfM on all five high-resolution faces;
- triangulates merged tracks;
- runs augmented refinement and BAE.

Non-center faces have no feed-forward depth. Cubemap mode therefore currently
requires `--vggsfm_group_strategy pose`.

Frame filtering is center-driven: if a center frame is removed, all five images
for that timestamp are removed together.

### 6.3 Pair and group behavior

`build_rig_pose_pairs`:

- excludes every same-timestamp pair because Cubemap5 is co-located;
- filters cross-frame candidates by viewing-axis angle;
- defaults to a 95-degree maximum, connecting adjacent cubemap directions but
  not opposite directions;
- sorts candidates by camera-center distance, frame gap, and axis angle;
- interleaves target sensors round-robin so one face cannot consume the whole
  neighbor budget.

Every face may be a VGGSfM group center. Groups use pose-based selection because
only `center` has Stage A depth.

### 6.4 BAE rig constraints

The pose convention is:

```text
sensor_from_world = sensor_from_rig @ rig_from_world
```

For Cubemap5:

- one timestamp owns one optimizable SE(3) `rig_from_world` block;
- its five images share the same pose index;
- `sensor_from_rig` is registered as a fixed BAE tensor buffer;
- intrinsics are shared per sensor/camera ID across timestamps;
- gauge fixing operates on rig frames;
- optimized rig poses are expanded back to image poses for writeback;
- native COLMAP rig/frame relationships are retained and audited.

The low-level BAE config/solver is mostly sensor-count independent. It maps
image names to external frame IDs and fixed `sensor_from_rig` transforms. The
tested upstream integration is still Cubemap5-specific.

### 6.5 Verified 150-frame baseline

Existing artifacts:

```text
input  /kiri/dataset/local_test_cubemap5_150
output /kiri/dataset/local_test_cubemap5_ffba_150
log    /kiri/dataset/ffba_logs/local_test_cubemap5_ffba_150.log
```

Recorded result:

```text
150 rig timestamps
750 registered images
approximately 400,373 points
approximately 1,804,506 observations
S-only points approximately 227,253
P-only points approximately 173,120
maximum rig center spread approximately 1.61e-14
maximum relative orientation error approximately 2.41e-6 degrees
```

All faces were registered for all timestamps. `up` had final selected
observations in 128/150 timestamps; all other faces had observations in 150/150.
This is a data/track-coverage condition, not a rig-registration failure.

The user-provided quality-run command was:

```bash
python run_merg3r_gluemap_pipeline.py \
  --dataset /kiri/dataset/local_test_cubemap5_150 \
  --output_dir /kiri/dataset/local_test_cubemap5_ffba_150 \
  --path_tracker /root/.cache/torch/hub/checkpoints/vggsfm_v2_tracker.pt \
  --pair_k_pose 25 \
  --neighbors_per_center 16 \
  --vggsfm_group_batch_size 2 \
  --no-stop_before_bae \
  --ba_backend bae \
  --bae_max_num_iterations 30 \
  --num_refinement_iterations 3 \
  --augmented_ba_max_filter_iterations 1 \
  --bae_robust_loss huber \
  --bae_huber_delta 1.0 \
  --select_track_min_support 256 \
  --filter_reproj_error_threshold 2.0
```

## 7. Fish9 pipeline work that is not implemented

The presence of nine prepared image directories does not mean they can be
passed to the current runner. The following work remains.

### 7.1 Manifest-driven input

Current hard-coded assumptions to replace:

- `utils/pano_rig.py` owns a fixed five-name sensor tuple and five rotations;
- `prepare_pano_stage_b` explicitly opens `center/left/right/up/down`;
- image ordering and summaries are fixed to five faces;
- intrinsics are reconstructed from a single CLI HFOV rather than read from a
  sensor manifest;
- native COLMAP rig/database writers are exercised only through Cubemap5
  metadata;
- neither dual-fisheye manifest schema is consumed anywhere in the pipeline.

The next implementation should define one normalized internal rig manifest
adapter instead of teaching every downstream stage about both JSON schemas.

Minimum normalized fields:

```text
sensor names and stable order
reference/feed-forward sensor
per-sensor image directory and role
per-sensor K and camera model
image-to-timestamp mapping
physical center group
fixed sensor_from_rig, or an explicit known per-frame transform model
source frame index/timestamp
mask path and valid-region policy
```

### 7.2 Correct common rig geometry

Neither prepare manifest currently contains a verified final
`sensor_from_rig` for all nine views.

For body-fixed images, the intended composition is:

```text
sensor_from_rig = view_from_lens @ lens_from_rig
```

Required verification before using it as a hard BAE constraint:

- stream 0/1 front/back assignment;
- raw fisheye mirroring and image-axis direction;
- `offset_v3` yaw/pitch/roll Euler order and transform direction;
- translation units and approximately 32.5 mm physical baseline;
- whether the reference rig origin is front lens, back lens, or a midpoint;
- rendered `ul/ur/dl/dr` directions in the final common rig frame.

Do not copy local `view_to_lens_rotation` directly into BAE as
`sensor_from_rig`.

### 7.3 IMU leveling decision

There are two distinct corrections:

1. fixed lens/encoded-image orientation from `offset_v3`;
2. dynamic device roll/pitch from gyro+accelerometer.

The fixed correction should be validated first. It preserves a standard fixed
rig and may explain much of the observed image tilt.

For dynamic leveling, raw INSV IMU must be decoded, scaled, axis-corrected,
time-aligned, and fused into per-frame quaternions. Accelerometer supplies
gravity; gyro supplies smooth short-term rotation. Absolute yaw remains a gauge
choice, but roll/pitch no longer needs the first video frame to be level.

Dynamic leveling changes the camera rotation relative to the physical body for
each frame. With a one-center approximation, the common rotation can be absorbed
into the frame rig pose. With two physical centers, a standard constant
`sensor_from_rig` is not sufficient unless images remain body-fixed. The correct
model is a fixed physical center plus a known per-frame software rotation, or a
BAE config extended to image-specific known sensor rotations.

Do not silently combine gravity-leveled images with a fixed two-center rig.

Recommended modes for the eventual INSV prepare contract:

```text
body-fixed       calibrated dewarp + fixed lens orientation; standard rig
gravity-leveled  calibrated dewarp + fused per-frame leveling; known dynamic rotation
```

### 7.4 Fish9 pair graph

Cubemap5 excludes all same-frame pairs because every face shares a center. That
rule is wrong for a true dual-center rig.

Fish9 should distinguish:

- same timestamp, same physical center: zero baseline; normally exclude from
  triangulation pairs, though tracking overlap can still be useful;
- same timestamp, different physical centers: real baseline; include only if
  calibrated frusta overlap and image quality is acceptable;
- different timestamps: use actual centers, axes, FOV, temporal gap, and
  overlap; round-robin sensor coverage remains useful.

SIFT database pairs and VGGSfM groups need not use exactly the same graph, but
their differences must be explicit and audited.

### 7.5 Stage A, tracking, filtering, and BAE wiring

Required behavior:

- Stage A reads only `front_center`.
- Stage B loads all nine sensors and verifies synchronized basenames.
- Only `front_center` owns feed-forward depth.
- Every sensor can become a SIFT/VGGSfM participant and, if useful, a group
  center.
- Frame filtering removes an entire nine-image timestamp group.
- Camera/intrinsics IDs are shared per virtual sensor.
- Native COLMAP contains one rig, one frame per timestamp, and nine images per
  retained frame.
- BAE optimizes one physical rig pose per timestamp, not nine independent poses.
- Fixed physical extrinsics remain buffers, not parameters.
- Post-BAE audit checks front/back baseline, relative rotations, pose-block
  count, and registered/observed frame counts per sensor.

The existing BAE core can likely be reused once a correct config is built, but
a dedicated nine-sensor/two-center unit test and real integration test are still
required.

## 8. Recommended next-agent execution order

### Step 1: freeze conventions before pipeline edits

Using 3-5 known frames and the official ERP as a visual reference:

1. verify stream 0/1 front/back labels;
2. determine fixed lens-to-rig rotations from `offset_v3`;
3. render axis markers or known landmarks for every virtual sensor;
4. decide rig origin and translation unit;
5. write normalized `sensor_from_rig` transforms to a fixture.

Acceptance: all nine view axes and both physical centers are explainable in one
coordinate frame. Do not proceed with hard BAE constraints before this passes.

### Step 2: add INSV IMU diagnostics separately

Implement a read-only raw IMU extractor that outputs:

```text
timestamp_seconds, acc_xyz_g, gyro_xyz_deg_per_s
```

Then verify:

- sample count and monotonic timestamps;
- approximately 1 kHz interval;
- X5 axis/sign mapping;
- gravity norm while stationary;
- video-frame timestamp interpolation;
- first-frame attitude and several visually distinctive roll events.

Keep this separate from image remapping until the attitude plot/numeric audit is
credible.

### Step 3: normalize manifests and generalize the loader

Introduce a generic rig metadata structure populated by adapters for:

- existing ERP Cubemap5;
- calibrated INSV nine-view output;
- approximate paired-frame nine-view output.

Preserve the existing Cubemap5 tests and 150-frame behavior unchanged.

### Step 4: reach a pre-BAE Fish9 checkpoint

On 5-20 timestamps:

- Stage A only on `front_center`;
- all nine Stage B images loaded;
- pair graph and group contact sheets exported;
- SIFT/VGGSfM databases created;
- native COLMAP rig seed written;
- pose/baseline audit passes before triangulation.

### Step 5: enable Fish9 BAE

Add explicit tests for nine sensors and two center groups, then run a short BAE
configuration. Verify:

```text
rig_enabled=true
num_pose_blocks=num_retained_timestamps
all fixed relative rotations remain unchanged
front/back baseline remains unchanged
one native COLMAP frame contains nine registered images
```

### Step 6: 150-frame A/B

Use the same source timestamps where possible and compare Fish9 against the ERP
baseline on:

- registration and observation count by sensor;
- mask/valid-pixel ratio and SIFT density;
- pair count, geometric inlier rate, and graph connectivity;
- track length and triangulation angle;
- S-only/P-only/mixed track composition;
- BAE loss and reprojection/angular error;
- fixed-rig audit errors;
- seam/physical-lens-boundary regions;
- downstream Gaussian Splatting quality if sparse metrics pass.

## 9. Current verification status

Focused tests run on 2026-09-01:

```bash
source /opt/conda/bin/activate
pytest -q \
  tests/test_prepare_pano_rig_from_erp.py \
  tests/test_prepare_dual_fisheye_rig_from_insv.py \
  tests/test_prepare_dual_fisheye_rig_from_frames.py \
  tests/test_pano_rig.py \
  tests/test_bae_rig_constraints.py
```

Result:

```text
37 passed, 5 warnings
```

The warnings are PyTorch/pycolmap runtime warnings, not test failures.

Coverage includes:

- ERP sampling and cubemap face mapping;
- INSV metadata/`offset_v3` parsing and calibrated center projection;
- diagonal layout margin search;
- paired-frame gyro CSV parsing and relative roll signs;
- synthetic paired-frame nine-view end-to-end output;
- exact Cubemap5 rotations and cross-frame pair graph;
- native COLMAP rig round trip;
- shared BAE pose indices and fixed sensor transform buffers;
- rig pose update/writeback behavior.

Not covered yet:

- real INSV IMU decoding/fusion;
- verified common nine-camera extrinsics;
- real Fish9 pipeline loading;
- Fish9 pair/group selection;
- nine-camera/two-center BAE integration;
- full Fish9 reconstruction or ERP-vs-Fish9 A/B.

## 10. Operational notes

- Activate the environment with `source /opt/conda/bin/activate`.
- The user handles Git synchronization manually; do not assume Git credentials
  are configured or perform remote synchronization without a new request.
- Both dual-fisheye prepare scripts refuse to overwrite an existing output
  directory and use an atomic staging-directory rename.
- Preserve existing data/log artifacts under `/kiri/dataset`.
- Use small 3-5 frame outputs for geometry/IMU visual checks, 20 frames for
  pair/group diagnostics, and only then move to 150-frame reconstruction.
- Treat masks as first-class input metadata. Invalid black pixels must not be
  silently accepted as normal feature regions.

## 11. Short status checklist

```text
[done] ERP video -> Cubemap5 prepare
[done] Cubemap5 center-only Stage A
[done] Cubemap5 all-face SIFT + VGGSfM
[done] Cubemap5 native COLMAP rig
[done] Cubemap5 shared-pose BAE and audit
[done] 150-frame ERP baseline

[done] INSV dual-stream probe/decode
[done] INSV offset_v3 calibrated dewarp
[done] INSV single-lens-only nine-view output
[done] paired-frame approximate nine-view output
[done] paired-frame relative accelerometer roll diagnostic

[not done] verified lens_from_rig and physical baseline units
[not done] INSV raw IMU decode/time alignment/fusion
[not done] absolute gravity leveling from INSV
[not done] normalized manifest consumed by pipeline
[not done] generic Fish9 Stage B loader
[not done] two-center-aware pair/group selection
[not done] Fish9 native COLMAP rig integration
[not done] Fish9 BAE integration and audit
[not done] 150-frame ERP-vs-Fish9 A/B
```
