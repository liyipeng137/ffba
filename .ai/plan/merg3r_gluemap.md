# Merg3r -> Gluemap Refinement 验证计划与进展

## 目标

验证 **Merg3r coarse pose + Gluemap/COLMAP-style refinement** 是否能在速度可控的前提下提升长序列 pose 质量。

当前阶段仍然是验证型两阶段 pipeline，不构建完整项目管线：

- A 脚本：在 MERG3R 环境中跑到 `align_extrinsics()` 完成，导出 coarse pose、图像、SuperPoint/LightGlue artifacts。
- B 脚本：在 Gluemap 环境中读取 A artifacts，构建 `S + P` tracks，写 COLMAP DB，triangulate，SelectTrack/filter，BA。

当前主线仍是 `SP`：

- `S`: SuperPoint extraction + LightGlue matching，替代原版 Gluemap 的 SIFT extraction + SIFT matching。
- `P`: ALIKED query -> VGGSfM prior tracks -> snap to SuperPoint -> TrackEstablishment-like DB writer。
- 暂不接入 `V` virtual tracks。
- 暂不接入 Gluemap augmented BA。

## 当前实现状态

### A 脚本：Merg3r coarse + S artifacts

文件：

```text
MERG3R/export_merg3r_refine_inputs.py
```

功能：

- 复用 Merg3r 的图像加载、sequence 构建、模型推理、`align_extrinsics()`、`restore_predictions_order()`。
- 在 align 完成后停止，不执行 Merg3r 原有 gradient BA。
- 基于 aligned pose 构建 pairs。
- 提取 SuperPoint features，并用 LightGlue 对 pairs 做 matching。
- 导出 B 脚本需要的 artifacts 到：

```text
<output_dir>/gluemap_refine_inputs/
```

当前 artifacts：

- `metadata.json`
- `coarse_poses.npz`
- `pairs.npy`
- `features_lightglue/`
- `matches_lightglue.npz`
- `images/`

pair 构建当前使用 HLoc-style pose pair：

```text
camera center distance top-k
+ viewing/principal-axis angle threshold filtering
```

默认参数：

```bash
--pair_k_pose 5
--pair_pose_rotation_threshold 30.0
--pair_k_similarity 0
--pair_temporal_window 0
```

注意：A 脚本会把 artifact 图像重新保存为顺序名：

```text
frame_000000.png
frame_000001.png
...
```

`metadata.json` 中保留：

```json
"original_image_names": [...],
"artifact_image_names": [...]
```

当前 B 脚本和输出 COLMAP reconstruction 使用的是 `artifact_image_names`，不是原始输入图片名。如果后续 Gaussian 训练需要和原始文件名对齐，需要改为保留原始 basename 或在 B 脚本写 reconstruction 时恢复原始 image name。

### B 脚本：S/P tracks -> merged DB -> triangulation -> Select/filter -> BA

文件：

```text
gluemap/run_merg3r_pycolmap_refine.py
```

功能：

- 读取 A 脚本 artifacts。
- 使用 A 输出的 `pairs.npy` 构建 VGGSfM groups。
- 每个 frame 可作为 center，邻居来自 pair adjacency。
- group 形式：

```text
[center, neighbor1, neighbor2, ...]
```

- group 内邻居排序：

```text
camera center distance
then abs(frame_id difference)
then frame_id
```

当前 P 分支：

```text
ALIKED query
-> VGGSfM tracker
-> snap to same-image SuperPoint keypoints
-> 1e-3 KDTree near-duplicate merge
-> prior COLMAP database
```

相关参数：

```bash
--vggsfm_query_source aliked       # 默认，旧行为可设为 superpoint
--aliked_detection_threshold 0.005 # 对齐原版 Gluemap
--vggsfm_query_points 1024
--prior_snap_to_superpoint
--prior_snap_threshold 1.0
--prior_keep_unsnapped
--prior_keypoint_merge_threshold 1e-3
```

当前 S 分支：

```text
SuperPoint features + LightGlue matches
-> database_lightglue.db
```

当前 SP merge：

```text
database_lightglue.db + database_vggsfm_prior.db
-> database_merged.db
```

merge 时 `primary_features_first=True`，因此每张图前半部分 keypoints 是 S 分支 SuperPoint，后半部分是 P 分支 prior keypoints。这满足原版 `select_tracks_from_merged()` 依赖的 offset 语义。

当前主流程：

```text
load artifacts
run VGGSfM P tracks
count S/P frame observations
drop low-coverage frames
write LightGlue DB
write VGGSfM prior DB
merge DBs
write Merg3r coarse reconstruction
pycolmap triangulation
SelectTrack
angular reprojection filtering
standard pycolmap BA
write refined_pycolmap
write refine_stats.json
```

当前输出：

```text
<artifact_dir>/pycolmap_refine/refined_pycolmap/
<artifact_dir>/pycolmap_refine/refine_stats.json
```

## 低覆盖帧过滤

B 脚本已加入类似 Merg3r `drop_untracked_frames()` 的过滤逻辑：

```text
run VGGSfM P tracks
-> count S/P observations per frame
-> drop low-coverage frames
-> remap image_names / images / extrinsic / features / pairs / matches / prior_tracks
```

当前代码默认：

```bash
--drop_low_coverage_frames
--min_frame_observations 10
```

实验中也使用过：

```bash
--min_frame_observations 5
```

过滤统计写入 `refine_stats.json` 的 `frame_filtering` 字段。

## SelectTrack 与 reprojection filtering

已将原版 Gluemap 的两步接入 B 脚本：

```text
triangulation
-> select_tracks_from_merged()
-> filter_reconstruction_by_reprojection_error()
-> BA
```

默认参数：

```bash
--enable_select_tracks
--select_track_min_support 512
--enable_reprojection_filter
--filter_reproj_error_type angular
--filter_reproj_error_threshold 0.5
```

说明：

- `select_tracks_from_merged()` 复用原版 Gluemap 逻辑。
- 当前 SuperPoint/LightGlue 的 per-image keypoint count 被当作原版的 `sift_count`。
- `filter_reconstruction_by_reprojection_error()` 使用原版 angular filtering，默认阈值 `0.5 deg`。
- 当前只过滤 real reconstruction；还没有 virtual reconstruction。

## Debug 输出

B 脚本默认开启 debug，可通过 `--no-debug_print` 关闭。

当前会打印：

- 输入 artifacts 规模。
- shared intrinsics 均值。
- VGGSfM tracking 配置与结果。
- VGGSfM query source 与 ALIKED extraction time。
- 每帧 `S` / `P` / `S+P` observation 统计。
- dropped frames。
- LightGlue DB 写入规模。
- Prior snap 统计：
  - total snapped / unsnapped / dropped
  - mean/max snap distance
  - center snap rate
  - neighbor snap rate
- Prior DB raw/merged keypoint 数。
- DB merge 状态。
- triangulation 点数。
- SelectTrack 前后点数与 S/non-S 分类。
- reprojection filter 删除 observation/track 数。
- BA summary。

原版 Gluemap 的 `TrackSnapping` 也已加入 debug：

- target SIFT keypoint 数量。
- total/center/neighbor snap rate。
- nearest target keypoint distance 分位数。

## 关键实验与结论

### 001：无 prior keypoint merge

配置：

```text
SuperPoint query -> VGGSfM P
no prior keypoint merge
keep all P observations
```

结果摘要：

```text
VGGSfM tracks = 33935
P observations = 93763
triangulated points = 57103
BA final cost = 0.661 px
BA termination = NO_CONVERGENCE
```

结论：BA cost 低，但 P tracks 没做图内 near-duplicate 合并，约束较碎，不代表全局更稳。

### 002：raw P 直接 1px merge

配置：

```text
SuperPoint query -> VGGSfM P
raw P keypoints 直接 1px KDTree merge
```

结果摘要：

```text
raw_kp = 93763
merged_kp = 83088
reduced = 10675
triangulated points = 51905
BA final cost = 0.959 px
BA termination = CONVERGENCE
```

结论：直接对 raw VGGSfM points 做 1px merge 语义过强，可能把独立预测点硬合并，不建议作为主线。

### 003：SuperPoint query + snap to SuperPoint + 1e-3 merge

配置：

```text
SuperPoint query -> VGGSfM P
snap to SuperPoint
keep unsnapped
1e-3 merge
```

结果摘要：

```text
snapped = 45625 / 93763
unsnapped_kept = 48138
triangulated points = 53318
BA final cost = 0.971 px
```

结论：机制更接近 Gluemap 的 TrackSnapping + TrackEstablishment，但由于 center query 本身来自 SuperPoint，center-side snap rate 有天然 bias。结果没有明显优于 001/002。

### 004：SuperPoint query + strict snap

配置：

```text
SuperPoint query -> VGGSfM P
snap to SuperPoint
--no-prior_keep_unsnapped
```

结果摘要：

```text
snapped = 45625
dropped = 48138
prior tracks = 9412
raw_kp = 21102
triangulated points = 30668
BA final cost = 0.834 px
```

结论：strict snap 更干净但覆盖明显不足。可作为 ablation，不适合作为主线。

### 005：ALIKED query + snap to SuperPoint，保留 unsnapped

配置：

```text
ALIKED query -> VGGSfM P
snap to SuperPoint
keep unsnapped
no SelectTrack/filter
```

结果摘要：

```text
VGGSfM tracks = 63314
P observations = 171967
snapped = 7142 / 171967 = 4.15%
center_snap = 5.0%
neighbor_snap = 3.7%
triangulated points = 82809
BA final cost = 0.595 px
BA time = 147s
```

结论：

- ALIKED query 明显增加了 VGGSfM track 数和 coverage。
- snap rate 很低，但原版 Gluemap 的 ALIKED -> SIFT snap rate 更低，约 `0.82%`。
- 因此低 snap rate 不是红旗；Gluemap 本身也大量依赖 unsnapped continuous VGGSfM tracks。
- 005 的收益主要来自更多 raw P coverage，而不是 snap 成功率。

### 原版 Gluemap snap rate 对照

原版日志：

```text
target SIFT keypoints total = 129628
snapped = 4727 / 573390 = 0.82%
center = 752 / 184381 = 0.41%
neighbor = 3975 / 389009 = 1.02%
```

结论：

- 原版 Gluemap 也不是靠高 snap rate 工作。
- TrackSnapping 是少量 anchor / 去重辅助，不是 P 分支主来源。
- 保留 unsnapped P observations 是原版设计的一部分。

### 006：ALIKED query + SelectTrack + reprojection filter

配置：

```text
ALIKED query
keep unsnapped
SelectTrack enabled
angular reprojection filter threshold = 0.5 deg
```

结果摘要：

```text
triangulated points = 82809
SelectTrack:
  S = 24263
  non-S = 58546
  kept non-S = 35945 / 58546
  removed = 22601

after SelectTrack:
  points = 60208

reprojection filter:
  mean angular error = 0.4864 deg
  median angular error = 0.3910 deg
  < 0.5 deg = 61.9%
  removed observations = 77829
  removed tracks = 19449

after filter:
  points = 40759

BA:
  residuals = 202066
  parameters = 123310
  initial cost = 1.027 px
  final cost = 0.367 px
  BA time = 77s
```

对比 005：

```text
BA final cost: 0.595 -> 0.367
BA time: 147s -> 77s
residuals: 465622 -> 202066
parameters: 249466 -> 123310
```

结论：

- SelectTrack + reprojection filter 是当前最有效的改进。
- 它保留了 ALIKED query 带来的 P coverage，又显著清理了 raw P outliers。
- 当前方向比继续纠结 snap rate 更重要。
- BA 出现过 CHOLMOD non-positive-definite warning，但仍完成并显著降 cost；需结合 Gaussian 结果判断是否实际有害。

## 当前与原版 Gluemap refinement 的差异

当前仍不是完整 Gluemap refinement。

原版 Gluemap 默认：

```text
num_refinement_iterations = 2
每轮:
  triangulation
  select_tracks_from_merged
  select_virtual_tracks_from_merged
  reprojection filtering
  iterative augmented BA
```

当前 B 脚本：

```text
1x write merged DB
1x write coarse reconstruction
1x pycolmap triangulation
1x select_tracks_from_merged
1x reprojection filtering
1x standard pycolmap BA
```

仍未包含：

- `num_refinement_iterations=2` 外层循环。
- `V` virtual tracks。
- `select_virtual_tracks_from_merged()`。
- augmented BA。
- BA/filter 多轮迭代。

## 当前判断

当前最合理主线：

```text
Merg3r aligned coarse pose
-> HLoc-style pose pairs
-> SuperPoint/LightGlue S
-> ALIKED query + VGGSfM P
-> snap P to SuperPoint where possible
-> keep unsnapped P
-> 1e-3 P near-duplicate merge
-> merged DB
-> pycolmap triangulation
-> SelectTrack
-> angular reprojection filter
-> pycolmap BA
```

核心结论：

- `--no-prior_keep_unsnapped` 不应作为主线；原版 Gluemap 也保留大量 unsnapped P。
- ALIKED query 比 SuperPoint query 更像原版设计，也显著提升 P coverage。
- snap rate 低是正常现象，不是当前瓶颈。
- SelectTrack + reprojection filter 是当前最明确有效的增益点。
- 最终质量仍需以 Gaussian 训练结果判断，不应只看 BA reprojection cost。

## Depth 与 refined pose 的同源性

A 脚本现在会导出 Merg3r `align_extrinsics()` 后的全局尺度 depth：

```text
<artifact_dir>/single_frame_depth/
```

该 depth 与 A 阶段的 coarse pose 是同源的，因此用于 TSDF 初始 mesh 时，`coarse` pose + A 阶段 depth 通常会比 `refined_pycolmap` pose + A 阶段 depth 更一致。

当前观察：

- 使用 B 阶段 `refined_pycolmap` pose 搭配 A 阶段 depth 做 TSDF 时，局部出现分层。
- 已确认 refined reconstruction 的 intrinsics 可能发生变化。
- 即使固定全局 scale，也不足以保证 refined pose 与旧 depth 匹配；BA 会改变每帧 pose/intrinsics，并且 BA 优化目标只约束 sparse tracks，不约束 dense depth。

当前判断：

- 这是符合预期的 pose-depth mismatch，不应直接用 TSDF 分层判断 refined pose 一定更差。
- 现阶段先专注 pose 优化，不把 depth/TSDF consistency 纳入主优化目标。
- 若后续需要使用 refined pose + dense depth 生成 mesh，应做 BA 后的 per-frame depth correction。

候选 correction 方向：

```text
refined pose + refined sparse points
-> per-frame project sparse points
-> sample old Merg3r depth at sparse observations
-> fit per-frame depth correction
-> corrected depth + refined pose -> TSDF
```

优先考虑：

- per-frame scale correction：`z_refined ~= s_i * d_old`
- per-frame scale+bias：`z_refined ~= s_i * d_old + b_i`
- per-frame inverse-depth affine：`1 / z_refined ~= a_i * (1 / d_old) + b_i`

其中 inverse-depth affine 更可能适配 feed-forward depth 的尺度/偏置误差，但暂不实现。

## 使用示例

A 脚本：

```bash
python MERG3R/export_merg3r_refine_inputs.py \
  --dataset /path/to/images \
  --output_dir /path/to/output
```

B 脚本当前主线：

```bash
python gluemap/run_merg3r_pycolmap_refine.py \
  --artifact_dir /path/to/output/gluemap_refine_inputs \
  --track_mode SP \
  --vggsfm_query_source aliked \
  --enable_select_tracks \
  --enable_reprojection_filter
```

常用 ablation：

```bash
# 旧 P query 行为
--vggsfm_query_source superpoint

# strict snap，只保留 snapped P
--no-prior_keep_unsnapped

# 放宽 reprojection filtering
--filter_reproj_error_threshold 0.75
--filter_reproj_error_threshold 1.0

# 增加 VGGSfM group 邻居
--neighbors_per_center 16
--neighbors_per_center 24
```

## 已完成检查

本地静态检查已通过：

```bash
ruff check MERG3R/export_merg3r_refine_inputs.py
ruff check gluemap/run_merg3r_pycolmap_refine.py
ruff check gluemap/gluemap/estimators/track_snapping.py
python -m py_compile MERG3R/export_merg3r_refine_inputs.py
python -m py_compile gluemap/run_merg3r_pycolmap_refine.py
python -m py_compile gluemap/gluemap/estimators/track_snapping.py
```

实际运行环境在云端，本地不验证完整 AI/SfM 依赖。

## 后续候选方向

短期：

- 用 Gaussian 训练最终产物比较：
  - 005：ALIKED query，无 SelectTrack/filter。
  - 006：ALIKED query + SelectTrack + 0.5deg filter。
  - 006 + `--filter_reproj_error_threshold 0.75`。
  - 006 + `--filter_reproj_error_threshold 1.0`。
- 观察 0.5deg filter 是否导致局部覆盖不足或 pose connectivity 变弱。
- 对 BA 前后 track length、每帧 points coverage 做进一步 debug。
- 如果 Gaussian 训练需要原图文件名，修复 artifact image naming。
- 暂不处理 TSDF/depth-pose mismatch；当前只记录现象，主线仍专注 pose。

中期：

- 尝试增大 `neighbors_per_center` 到 16/24，评估 P coverage、filter 后剩余 tracks、BA 稳定性与下游质量。
- 评估 BA 后 per-frame depth correction，用 refined sparse geometry 校正 A 阶段 dense depth，再用于 TSDF/mesh。
- 给 B 脚本加入简化多轮 refinement：

```text
for iter in range(num_refinement_iterations):
  triangulate
  SelectTrack
  filter high reprojection-error observations
  BA
```

- 评估是否接入 `V` virtual tracks。

长期：

- 若验证效果稳定，再把 A/B 两阶段整理成正式 pipeline。
- 支持更完整的 camera/intrinsics mapping。
- 支持高分辨率图像路径与低分辨率坐标缩放。
- 评估是否迁移 Gluemap augmented BA。
