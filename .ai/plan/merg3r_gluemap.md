# Merg3r -> Gluemap Refinement 缝合进展

## 目标

验证 **Merg3r coarse pose + Gluemap/COLMAP-style refinement** 是否能在速度可控的前提下提升长序列 pose 质量。

## 阶段状态

截至 2026-06-15，Merg3r 与 GlueMap refinement 的两阶段脚本缝合已经推进到单脚本 pipeline。

当前主入口：

```text
MERG3R/run_merg3r_gluemap_pipeline.py
```

该脚本已经把原先两阶段流程串到同一条管线中：

- Stage 1：Merg3r 前馈推理、sequence 对齐、coarse pose/depth 构建。
- Stage 2：构建 `S + P + V`，执行 GlueMap-style SelectTrack、SelectVirtualTrack、reprojection filtering、多轮 augmented BA。
- 输出 `refined_gluemap_aba/`、`virtual_gluemap_aba/`、`refine_stats.json` 等 refinement 结果。

原先两个阶段脚本仍保留为历史对照和 debug 入口：

```text
MERG3R/export_merg3r_refine_inputs.py
MERG3R/gluemap/run_merg3r_pycolmap_refine.py
```

但当前工程主线不再依赖 `artifact_dir` 在两个独立脚本之间传递中间产物。

最新实验中，切到 `pi3x` 前馈模型并将 S 分支改为原版 GlueMap 风格 SIFT 后，关键数值已经接近原版 GlueMap：

```text
original GlueMap iter2 real angular:
  mean=0.2077 deg, median=0.1219 deg, <0.5deg=91.4%

MERG3R+GlueMap, pi3x + SIFT + SPV/star-like iter2:
  mean=0.2390 deg, median=0.1568 deg, <0.5deg=92.1%
  final real tracks=103500, original GlueMap ~=103631

MERG3R+GlueMap, pi3x + SIFT + SPV/pose iter2:
  mean=0.2352 deg, median=0.1587 deg, <0.5deg=92.4%
```

因此当前主线已经从“能否缝合并接近原版”切换为“整理成稳定可复现实验管线”。

当前固定流程为：

```text
pi3x
+ SIMPLE_PINHOLE
+ SIFT as S
+ ALIKED/VGGSfM prior as P
+ group_strategy = pose
+ SPV augmented BA
```

已移除或收敛掉的分支：

- 不再保留 LightGlue/SuperPoint 作为主线 `S` 分支。
- 不再保留 `SP` mode 作为 pipeline 主流程，固定走 `SPV`。
- `group_strategy` 固定为 `pose`，不再接入 star-like mode。
- `vggsfm_query_source` 固定为 `aliked`。
- `vggsfm_tracker_input` 固定为 `1024`。
- `s_database_mode` 固定为 `sift`。
- `prior_snap_to_sift`、`prior_keep_unsnapped`、`drop_low_coverage_frames`、`enable_select_tracks`、`enable_reprojection_filter` 均作为默认启用逻辑，不再作为主线命令行开关。

## 当前实现状态

### 整合 pipeline

文件：

```text
MERG3R/run_merg3r_gluemap_pipeline.py
```

当前职责：

- 在单脚本内完成 Merg3r coarse stage 与 GlueMap-style refinement stage。
- Stage 1 使用低分辨率图像运行 `pi3x` 前馈推理、sequence 构建、`align_extrinsics()`、`restore_predictions_order()`。
- Stage 1 结束后直接把 coarse pose、low-res depth/depth_conf、pairs、low/high intrinsics、low/high images 传入 Stage 2，不再通过 `gluemap_refine_inputs/` artifact 目录中转。
- Stage 2 使用高分辨率图像与高分辨率内参构建 SIFT database、VGGSfM prior、merged DB、virtual reconstruction，并执行 SPV augmented BA。
- pipeline metadata 会记录当前固定配置、image pyramid 配置、coarse/refine timing 和 refine stats。

当前固定主线：

```text
MERG3R low-res inference/align
-> high-res SIFT S database
-> ALIKED query + VGGSfM tracker input 1024
-> P tracks snap to SIFT where possible
-> keep unsnapped P
-> filter low-coverage frames
-> Gluemap intrinsics_averaging
-> virtual-track diagnostics/preparation
-> merged S+P database
-> virtual reconstruction
-> SelectTrack + SelectVirtualTrack
-> real/virtual reprojection filtering
-> iterative augmented BA
```

### 工具模块整理

新增/整理到 `MERG3R/utils/` 的主流程工具：

```text
MERG3R/utils/gluemap_spv_refine.py
MERG3R/utils/gluemap_refine_core.py
MERG3R/utils/image_pyramid.py
```

职责拆分：

- `gluemap_spv_refine.py`：Stage 2 顶层编排，固定 `SIMPLE_PINHOLE + SIFT + ALIKED + pose groups + SPV` 主线。
- `gluemap_refine_core.py`：从旧 B 脚本抽出的 GlueMap/COLMAP 工具函数，包括 SIFT database、VGGSfM prior、frame filtering、intrinsics averaging、virtual tracks、merged DB、triangulation、SelectTrack、SelectVirtualTrack、augmented BA 等。
- `image_pyramid.py`：整合 `scripts/crop_img.py` 的整数比例 crop/resize 思路，构建低/高分辨率两组图片，并提供 low/high image tensor 读取和 low/high K 缩放工具。

当前 `run_merg3r_pycolmap_refine.py` 不再作为 pipeline 的运行依赖，后续只作为历史参考和回归对照。

### 高低分辨率处理

当前 pipeline 默认启用 image pyramid：

```bash
--image_pyramid
--stage1_downscale_n 4
--stage1_multiple 14
--stage2_scale_factor 0
```

含义：

- 输入图片先经过整数比例 crop/resize，输出 low/high 两组图片。
- low-res 图片用于 Merg3r/pi3x 前馈推理、align、coarse depth、coarse pose。
- high-res 图片用于 SIFT、VGGSfM prior、COLMAP database、final reconstruction 和后续 Gaussian 训练对齐。
- `stage2_scale_factor=0` 表示 high-res 侧使用 `stage1_downscale_n` 对应的比例，通常接近原始输入分辨率。
- Stage 2 的全局内参由 Stage 1 low-res intrinsic 按 low->high 尺度线性缩放得到。

当前坐标域约定：

- `build_virtual_track_diagnostics` 的 CovisibilityExtraction / generation 阶段直接使用 Stage 1 low-res depth 与 low-res K，不额外 resize depth。
- `VirtualTrackPreparation` 阶段回到 high-res 图像域，使用 high-res global intrinsic 做 virtual track 更新、subsample 和 global update。
- SIFT 直接读取 high-res 图像。
- VGGSfM tracker 仍固定使用 `1024` input，并通过 tracker image changes 映射回 high-res 图像域。

最近一次 high/low 切换对比显示：

- high K 相对 low K 约为 `4x` 缩放，符合预期。
- high-res SIFT keypoints/matches 显著增加，final real points 增加。
- virtual track 数量基本稳定，未观察到明显坐标系错误。
- high-res 流程耗时明显增加，因此已继续加入 SIFT 提取限制。

### SIFT database 复用与限制

当前 refinement 会先构建一次 prefilter SIFT database，用于统计 `S + P` observation 并执行 low-coverage frame filtering。

过滤帧后不再重新跑完整 SIFT extract + match，而是通过：

```text
filter_sift_database_for_refine()
```

从第一遍 SIFT database 复制并过滤出 final SIFT database：

- 删除 dropped frames。
- 保留未 drop 图像的 keypoints/descriptors。
- 重写 images/cameras，并重新映射 image id。
- 过滤 matches/two-view geometries 中涉及 dropped frames 的 pair。

这样避免第二遍重复执行高分辨率 `pycolmap.extract_features()` 和 `pycolmap.match_image_pairs()`。

当前 high-res SIFT 速度瓶颈仍主要来自第一遍 extraction/matching，因此已在 `MERG3R/gluemap/gluemap/utils/colmap.py` 的 `prepare_sift_database()` 中限制 SIFT 提取规模：

```text
FeatureExtractionOptions.max_image_size = 1024
FeatureExtractionOptions.sift.max_num_features = 4096
```

判断：

- `max_image_size=1024` 限制 SIFT 提取阶段的输入尺度，优先降低像素处理成本。
- `max_num_features=4096` 限制每图 SIFT descriptor 数量，进一步降低 matching 成本。
- 暂未调整 `FeatureMatchingOptions.max_num_matches`，因为当前有效 matches/pair 远低于默认上限，主要瓶颈不是最终存储 match 数。

后续可保留两档配置：

```text
质量/速度平衡: max_image_size=1600, max_num_features=8192
速度优先:     max_image_size=1024, max_num_features=4096
```

当前代码先采用速度优先档，以验证 high-res pipeline 的整体吞吐。

### 历史 A 脚本：Merg3r coarse + S artifacts

文件：

```text
MERG3R/export_merg3r_refine_inputs.py
```

功能：

- 复用 Merg3r 的图像加载、sequence 构建、模型推理、`align_extrinsics()`、`restore_predictions_order()`。
- 在 align 完成后停止，不执行 Merg3r 原有 gradient BA。
- 基于 aligned pose 构建 pairs。
- 提取 SuperPoint features，并用 LightGlue 对 pairs 做 matching；该分支当前保留为 ablation，最新推荐主线使用 B 阶段 SIFT，不再依赖 A 阶段 LightGlue 数据库作为 S。
- 额外导出未 mask 的 raw `final_predictions["depth"]`，以及存在时的 `depth_conf`，供 B 阶段 virtual-track 构建使用。
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
- `raw_geometry.npz`
- `single_frame_depth/`

说明：

- `raw_geometry.npz` 中的 raw depth/depth_conf 用于 B 阶段复刻 Gluemap virtual-track 构建。
- `single_frame_depth/` 是 Merg3r aligned pose 同源 dense depth，当前与 BA refine 主线无关，暂不用于 B 阶段。
- `features_lightglue/` 与 `matches_lightglue.npz` 当前主要用于 `--s_database_mode lightglue` ablation；`--s_database_mode sift` 时 B 阶段会自行构建 SIFT database。

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

### 历史 B 脚本：S/P/V tracks -> merged DB / virtual reconstruction -> refinement

文件：

```text
gluemap/run_merg3r_pycolmap_refine.py
```

功能：

- 读取 A 脚本 artifacts。
- 使用 A 输出的 `pairs.npy` 构建 VGGSfM groups。
- 支持两种 group 构建：
  - `group_strategy=star`：复用 GlueMap `BaseStarDataset` 风格的 star-like pruning。
  - `group_strategy=pose`：按 camera center distance 从 pair adjacency 中取邻居。
- 在 low-coverage frame filtering 后，对保留帧的每帧初始内参执行 Gluemap `intrinsics_averaging()`，替代原先的简单全局均值。
- 保存每帧初始内参、averaged 后的 shared global intrinsic、intrinsics mapping。
- 构建 Gluemap-style `V` virtual tracks 诊断数据：
  - 生成阶段使用每帧初始 `K`。
  - BA 前预处理阶段使用 averaged global `K`。
  - 使用 filtering 后的 image order，depth/depth_conf 同步 drop 废弃帧。
  - `SP` 模式只写指标；`SPV` 模式会把结果转成 virtual reconstruction 并进入 augmented BA。
- `track_mode` 目前只保留两个测试入口：`SP` 和 `SPV`。
- 默认 camera model 已改为 `SIMPLE_PINHOLE`，不再默认沿用 A 阶段 metadata 中的 `PINHOLE`。这样 pycolmap/GlueMap 的 normalized reprojection filter 会使用单 focal 相机模型；在 `pi3x` 路径下该假设与原版 GlueMap/pi3x 更一致，是当前推荐默认路径。
- 每个 frame 可作为 center，邻居来自 pair adjacency；star-like 和 pose group 的 angular error 已验证差异较小，group strategy 不是当前主瓶颈。
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
-> snap to same-image SIFT/SuperPoint keypoints
-> 1e-3 KDTree near-duplicate merge
-> prior COLMAP database
```

相关参数：

```bash
--vggsfm_query_source aliked       # 默认，旧行为可设为 superpoint
--aliked_detection_threshold 0.005 # 对齐原版 Gluemap
--vggsfm_query_points 1024
--vggsfm_tracker_input 1024        # GlueMap-style 1024 tracker input，并映射回当前图像域
--prior_snap_to_sift               # s_database_mode=sift 时的推荐路径
--prior_snap_to_superpoint         # s_database_mode=lightglue 时的旧路径
--prior_snap_threshold 1.0
--prior_keep_unsnapped
--prior_keypoint_merge_threshold 1e-3
```

当前 S 分支：

```text
--s_database_mode lightglue:
  SuperPoint features + LightGlue matches
  -> database_lightglue.db

--s_database_mode sift:
  GlueMap/COLMAP SIFT extraction + matching
  -> database_sift.db
```

当前 SP merge：

```text
lightglue mode:
  database_lightglue.db + database_vggsfm_prior.db
  -> database_merged.db

sift mode:
  database_vggsfm_prior.db + database_sift.db
  -> database_merged.db
```

merge 后每张图前半部分 keypoints 都保持为 S 分支，后半部分是 P 分支 prior keypoints。这满足原版 `select_tracks_from_merged()` 依赖的 offset 语义：

- `lightglue` mode 通过 `primary_features_first=True` 保持 SuperPoint 在前。
- `sift` mode 通过原版风格 `primary_features_first=False`，让 secondary `database_sift.db` 排在前。

当前 `SP` 主流程：

```text
load artifacts
run VGGSfM P tracks
count S/P frame observations
drop low-coverage frames
intrinsics_averaging on kept frames
build virtual-track diagnostics
write S database (LightGlue or SIFT)
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

当前 `SPV` 主流程：

```text
load artifacts
run VGGSfM P tracks
count S/P frame observations
drop low-coverage frames
intrinsics_averaging on kept frames
build virtual tracks + diagnostics
write S database (LightGlue or SIFT)
write VGGSfM prior DB
merge DBs
write Merg3r coarse reconstruction
build virtual reconstruction from virtual tracks
for iter in range(num_refinement_iterations):
  triangulate real reconstruction from previous reconstruction
  SelectTrack on real reconstruction
  SelectVirtualTrack on virtual reconstruction
  angular reprojection filtering on real/virtual reconstructions
  iterative augmented BA
write refined_gluemap_aba
write virtual_gluemap_aba
write refine_stats.json
```

当前输出：

```text
<artifact_dir>/pycolmap_refine/refined_pycolmap/
<artifact_dir>/pycolmap_refine/refined_gluemap_aba/
<artifact_dir>/pycolmap_refine/virtual_gluemap_aba/
<artifact_dir>/pycolmap_refine/refine_stats.json
<artifact_dir>/pycolmap_refine/intrinsics_refine_inputs.npz
<artifact_dir>/pycolmap_refine/virtual_track_stats.json
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

## Intrinsics averaging 与 VirtualTrack 诊断

### Intrinsics averaging

当前 B 脚本已不再使用简单全局平均内参，而是在 frame filtering 后复用 Gluemap：

```text
gluemap.estimators.intrinsics_averaging.intrinsics_averaging()
```

当前实现仍是 shared-camera 简化版本：

```text
intrinsics_mapping = {image_id: 0}
communities = [all kept frames]
```

输出：

```text
<artifact_dir>/pycolmap_refine/intrinsics_refine_inputs.npz
```

其中包含：

- `initial_intrinsics`：过滤后每帧自己的初始内参。
- `averaged_intrinsics`：对每帧展开后的 averaged shared intrinsic。
- `shared_global_intrinsic`：最终 shared K。
- `intrinsics_mapping`：当前全为 0。
- `image_names`：过滤后的 artifact image names。

相关统计写入 `refine_stats.json["intrinsics"]`。

### VirtualTrack generation

当前目标已从“只构建诊断数据”推进到“在 `SPV` 模式下把 virtual tracks 喂进 augmented BA”。`SP` 模式仍可只构建并打印 diagnostics，不改变原来的标准 BA 主线。

已接入生成阶段三步：

```text
_convert_from_depth_to_world_points
_verify_by_reprojection_n2
_calculate_virtual_tracks
```

输入：

- `raw_geometry.npz["depth"]`
- `raw_geometry.npz["depth_conf"]`，如果 A 阶段提供
- filtering 后的 `extrinsic`
- filtering 后每帧初始 `K`
- filtering 后 image order

groups 构建仍复用当前 VGGSfM groups：

```text
build_vggsfm_groups(pairs, num_images, neighbors_per_center=8, centers=camera_centers)
```

这会让当前实验中的 group 数等于过滤后帧数；每个 group 是：

```text
[center, nearest neighbor 1, ...]
```

center-local extrinsics 当前由全局 w2c 构造：

```text
T_local_j = T_global_j @ inv(T_global_center)
```

### VirtualTrack preparation

已复用原版 Gluemap `VirtualTrackPreparation` 的三个私有步骤：

```text
_update_virtual_tracks
_subsample_virtual_tracks
_update_virtual_tracks_global
```

其中：

- generation 阶段使用每帧初始 `K`。
- `_update_virtual_tracks` 使用 averaged global `K` 重投影。
- `_subsample_virtual_tracks` 默认每组保留 `100` 个 virtual points。
- `_update_virtual_tracks_global` 使用 global pose/global K 再更新部分 virtual tracks。

输出：

```text
<artifact_dir>/pycolmap_refine/virtual_track_stats.json
```

并同步写入：

```text
refine_stats.json["virtual_tracks"]
```

可选保存完整 debug tensor：

```bash
--save_virtual_tracks_debug
```

### VirtualTrack -> augmented BA

`SPV` 模式下，B 脚本现在会复用当前构建出的 `virtual_predictions_dict`：

```text
establish_tracks_from_predictions_dict(add_tracks=False, add_virtual_points=True)
-> initialize_world_points()
-> build_reconstruction_for_ba()
-> select_virtual_tracks_from_merged()
-> iterative_bundle_adjustment(real_reconstruction, virtual_reconstruction)
```

为支持 virtual-only TrackEstablishment，`track_establishment.py` 已补齐：

- `add_tracks=False` 时从 `tracks_virtual` 推断 shape。
- 不再无条件访问 `predictions_dict["tracks"]` / `predictions_dict["scores"]`。
- 只在 `add_tracks=True` 时维护真实 track 的 `pts2d_idx_all`。

当前 `SPV` 里的 virtual points 不写入 COLMAP database，而是单独构建 `virtual_reconstruction`，augmented BA 时把 virtual residual 加到同一个 Ceres problem 中，共享 real reconstruction 的 pose/intrinsic 参数块。

### Generation validity 统计修正

已修正当前诊断代码中的 generation validity 统计方式。

原因：

- `CovisibilityExtraction.main()` 中 `valid_virtual` 的返回存在命名/顺序歧义。
- `project_tracks()` 的真实返回顺序是：

```text
tracks_virtual
points3d_virtual
valid_mask
is_negative
```

当前 B 脚本不再通过 `CovisibilityExtraction.main()` 读取 generation 阶段的 `valid_virtual`，而是直接按三阶段调用，并按 `project_tracks()` 的真实返回顺序解析：

```text
tracks_virtual, points3d_virtual, valid_virtual, isnegative_virtual
```

因此重新运行后，`virtual_track_stats.json["generation"]` 中的：

- `valid_observations`
- `center_valid_observations`
- `neighbor_valid_observations`
- `pair_coverage`
- `negative_observations`

才是 generation 阶段真实 validity 统计。

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
- `SP` 模式只过滤 real reconstruction。
- `SPV` 模式先对 real reconstruction 做 `select_tracks_from_merged()`，再用 real pair coverage 对 virtual reconstruction 做 `select_virtual_tracks_from_merged()`。
- `SPV` 模式在进入 augmented BA 前会分别对 real / virtual reconstruction 做 reprojection filtering。

## Debug 输出

B 脚本默认开启 debug，可通过 `--no-debug_print` 关闭。

当前会打印：

- 输入 artifacts 规模。
- intrinsics averaging 后的 shared global K。
- VGGSfM tracking 配置与结果。
- VGGSfM query source 与 ALIKED extraction time。
- 每帧 `S` / `P` / `S+P` observation 统计。
- dropped frames。
- virtual-track diagnostics 总览：
  - group 数。
  - final valid virtual observations。
  - subsample 后 points/group。
  - virtual-track 构建耗时。
- S database 写入规模：
  - LightGlue mode: SuperPoint/LightGlue keypoints/matches。
  - SIFT mode: SIFT keypoints/matches。
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
- `SPV` augmented refinement：
  - virtual TrackEstablishment 结果。
  - virtual point initialization inlier/outlier 数。
  - SelectVirtualTrack 删除量。
  - augmented BA real/virtual residual 数。
  - iterative BA/filter 每轮统计。

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

### 007：Intrinsics averaging + VirtualTrack 诊断阶段快照

配置：

```text
ALIKED query
keep unsnapped
SelectTrack enabled
angular reprojection filter threshold = 0.5 deg
intrinsics_averaging enabled
virtual-track diagnostics enabled
virtual tracks not added to BA  # 当时 SPV augmented BA 尚未接入
```

该阶段 `181` 帧运行摘要：

```text
input images = 181
frame filtering dropped = [24, 25, 58]
kept images = 178

intrinsics_averaging:
  fx = 374.06
  fy = 373.37
  cx = 176
  cy = 240

virtual-track diagnostics:
  groups = 178
  skipped_centers = 0
  generation points/group = 850
  after subsample points/group = 100
  final valid observations = 125750
  center valid observations = 17800
  frames without virtual observations = 0
  global update displacement median ~= 0 px
  virtual diagnostics time ~= 28s

triangulated points = 82730
SelectTrack:
  points = 82730 -> 59902
reprojection filter:
  mean angular error = 0.4653 deg
  median angular error = 0.3720 deg
  < 0.5 deg = 64.1%
  removed observations = 72817
  removed tracks = 17901
after filter:
  points = 42001
```

结论：

- `groups == kept frames`，说明当前 star/group 结构已贴近 Gluemap 的 per-frame center 设计。
- BA 前的 virtual-track preparation 数据健康：subsample 后每组 `100` 点、无空帧、pair coverage 基本完整。
- `_update_virtual_tracks_global` 位移接近 0，说明当前由 global w2c 构造 center-local extrinsics 的方式与 global update 一致。
- 这次运行中 generation 阶段的 validity 指标曾受 `CovisibilityExtraction.main()` 返回命名/顺序歧义影响；代码已改为按 `project_tracks()` 真实返回顺序统计，需要重跑后再看新的 generation validity 数据。

### 008：SPV + augmented BA 首次接通

日志：

```text
spv_log/logs_181_spv_all.log
```

配置摘要：

```text
track_mode = SPV
camera_model = PINHOLE  # 本次日志仍使用旧默认
num_refinement_iterations = 2
augmented_ba_max_filter_iterations = 3
augmented_ba_normalized_reproj_threshold = 0.01
ba_max_num_iterations = 100
```

运行摘要：

```text
input images = 181
frame filtering:
  dropped = [24, 25, 58, 59]
  kept images = 177

intrinsics_averaging:
  fx = 374.06
  fy = 373.37
  cx = 176
  cy = 240

virtual tracks:
  groups = 177
  established tracks = 17700
  virtual observations = 125253
  average observations / virtual track = 7.08
  initialized virtual tracks = 17700
  initialization outliers = 0

SelectTrack:
  SIFT/SuperPoint-class tracks = 18373
  non-S tracks = 58407
  kept non-S = 25450 / 58407
  removed = 32957

SelectVirtualTrack:
  kept = 17496 / 17700
  removed = 204

pre-BA reprojection filtering:
  real mean angular error = 0.5443 deg
  real median angular error = 0.4436 deg
  real < 0.5 deg = 55.5%
  real removed observations = 73994
  real removed tracks = 14925
  virtual mean angular error = 0.0000 deg
  virtual removed observations = 0
  virtual removed tracks = 0

iterative augmented BA iteration 1:
  start real tracks = 28898
  start virtual tracks = 17496
  real observations = 76162
  virtual observations = 123973
  residual blocks before virtual = 76162
  residual blocks after virtual = 200135
  virtual negative-depth constraints = 395
  Ceres cost = 1.026006e5 -> 6.844887e4
  termination = NO_CONVERGENCE at 100 iterations
```

结论：

- `SPV` 从 virtual track generation、virtual-only TrackEstablishment、virtual reconstruction、SelectVirtualTrack 到 augmented BA 的链路已经接通。
- virtual 侧统计符合预期：`177 groups * 100 points/group = 17700 tracks`，平均 track length `7.08`，没有初始化 outlier。
- real 侧 0.5deg angular filter 删除量偏大，说明 triangulated real tracks 与当前 MERG3R coarse pose 的一致性一般；这不是程序错误，但需要作为质量指标继续观察。
- Ceres `NO_CONVERGENCE` 不是崩溃原因；100 次迭代内 cost 明显下降，只是达到迭代上限。
- 最后崩溃发生在 augmented BA 后的 normalized reprojection filter fallback：

```text
camera.focal_length
ValueError: Check failed: idxs.size() == 1 (2 vs. 1)
```

原因是本次使用 `PINHOLE` camera model，pycolmap 的 `camera.focal_length` 只支持单 focal 模型；`PINHOLE` 有 `fx/fy` 两个 focal 参数。为快速跑通当前 SPV+BA 主线，B 脚本默认 camera model 已切到 `SIMPLE_PINHOLE`。后续如果要继续支持 `PINHOLE`，应把 `reprojection_error.py` 中 normalized error 的 focal 归一化改成 `mean(camera.params[camera.focal_length_idxs()])`。

### 009：pi3x + SIFT as S 后接近原版 GlueMap

背景：

- 早期测试实际使用 `vggt_omega`，其当前导出路径没有约束 `fx=fy`。
- 当前 B 阶段默认 `SIMPLE_PINHOLE`，并使用 GlueMap-style shared intrinsic。
- 这会让 `vggt_omega` 的非 `fx=fy` K 与 B 阶段相机假设不一致，导致 real angular error 长期停在 `mean ~= 0.58-0.61 deg`。
- 切换到 `pi3x` 后，mogo 内参恢复隐式约束 `fx=fy`，与原版 GlueMap/pi3x 和当前 `SIMPLE_PINHOLE` 更一致。
- 再把 S 分支从 LightGlue 改为原版风格 SIFT 后，S-only 和 P-only angular error 都明显改善。

关键日志：

```text
spv_log/logs_181_spv_all_09_sift_debug.log
spv_log/logs_181_spv_all_10_pose_debug.log
```

star-like + SIFT：

```text
SIFT DB ready: keypoints=115553, pairs=442, matches=25067
Prior DB written: input_tracks=156119, pairs=2508, raw_kp=727558

iter 2:
  S-only mean=0.1819, median=0.1193, <0.5deg=94.4%
  P-only mean=0.2415, median=0.1586, <0.5deg=92.0%
  real mean=0.2390, median=0.1568, <0.5deg=92.1%
  Final: 103500 real tracks, 20782 virtual tracks
```

pose group + SIFT：

```text
SIFT DB ready: keypoints=122104, pairs=355, matches=21763
Prior DB written: input_tracks=156529, pairs=2801, raw_kp=800540

iter 2:
  S-only mean=0.1675, median=0.1075, <0.5deg=95.4%
  P-only mean=0.2377, median=0.1608, <0.5deg=92.2%
  real mean=0.2352, median=0.1587, <0.5deg=92.4%
  Final: 99437 real tracks, 20851 virtual tracks
```

原版 GlueMap 对照：

```text
iter 2:
  real mean=0.2077, median=0.1219, <0.5deg=91.4%
  final real tracks ~= 103631
```

结论：

- Merg3r + GlueMap SPV/ABA 缝合链路已经完成，数值已接近原版 GlueMap。
- 此前 high angular error 的主因不是 virtual tracks、star group 或 pair density，而是 `vggt_omega` 的 K/camera 假设与 `SIMPLE_PINHOLE` 不一致。
- 在 `pi3x + SIFT as S` 下，S-only/P-only 都进入合理范围，说明当前剩余差异更多是工程 parity 和实现细节，不是核心流程缺失。
- star-like 与 pose group 的 angular error 差别很小；若目标是贴近原版，仍建议保留 star-like 作为正式对齐方向，pose 可作为 sanity check。

## 当前与原版 Gluemap refinement 的差异

当前核心 refinement 链路已经贴近原版 Gluemap，但仍不是完全同构实现。

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

当前 B 脚本 `SP`：

```text
1x write merged DB
1x write coarse reconstruction
1x build virtual-track diagnostics before BA
1x pycolmap triangulation
1x select_tracks_from_merged
1x reprojection filtering
1x standard pycolmap BA
```

当前 B 脚本 `SPV` 已包含：

```text
1x write merged DB
1x write coarse reconstruction
1x build virtual tracks + diagnostics before BA
1x build virtual reconstruction
num_refinement_iterations outer loop
per iteration:
  pycolmap triangulation
  select_tracks_from_merged
  select_virtual_tracks_from_merged
  real/virtual reprojection filtering
  iterative augmented BA
```

仍和原版存在差异：

- 没有跑原版 `GlobalGluer`；当前 global pose 来自 MERG3R aligned pose。
- 没有原版 `pose_inconsistent`。当前 center-local extrinsics 直接由 global w2c 构造，因此 local/global pose discrepancy 近似为 0；若后续引入独立 local star pose 或 global pose averaging，需要补 pose-inconsistent diagnostics/pruning。
- `predictions_dict["tracks"]` 的真实 P tracks 没有复刻成原版 tensor 结构；real tracks 仍通过 COLMAP database + pycolmap triangulation 产生。
- `SPV` 的 augmented BA 是 Merg3r-adapted GlueMap 流程，不是直接调用原版 `run_refinement_pipeline()` 的完整输入。
- 默认 camera model 已改为 `SIMPLE_PINHOLE`，与当前 `pi3x` 主线更一致；若需要保留 `PINHOLE`，需要修 normalized reprojection focal 归一化。
- 当前没有直接复刻原版 GlueMap 的 resize/restore 实现，而是采用 image pyramid：low-res 供 Stage 1 推理和 depth，high-res 供 Stage 2 refinement 和最终输出。

## 当前判断

缝合阶段已经结束。当前最合理主线是：

```text
Merg3r aligned coarse pose
-> dense/pose pairs
-> SIFT S database
-> ALIKED query + VGGSfM P on 1024 tracker input
-> snap P to SIFT where possible
-> keep unsnapped P
-> 1e-3 P near-duplicate merge
-> drop low-coverage frames
-> Gluemap intrinsics_averaging
-> build VirtualTracks
-> merged DB
-> build virtual reconstruction
-> repeat num_refinement_iterations:
   triangulate real reconstruction
   SelectTrack
   SelectVirtualTrack
   real/virtual reprojection filter
   augmented BA
```

核心结论：

- `--no-prior_keep_unsnapped` 不应作为主线；原版 Gluemap 也保留大量 unsnapped P。
- ALIKED query 比 SuperPoint query 更像原版设计，也显著提升 P coverage。
- snap rate 低是正常现象，不是当前瓶颈。
- SelectTrack + reprojection filter 是当前最明确有效的增益点。
- `SPV` augmented BA 的链路已接通，并已完成两轮 refinement 输出。
- `pi3x + SIMPLE_PINHOLE + SIFT as S` 是当前最接近原版 GlueMap 的组合。
- `vggt_omega` 路径如果继续支持，需要单独处理 camera model / `fx=fy` 假设，不应直接混入当前主线结论。
- 当前没有 `pose_inconsistent` 的直接风险较低，因为 local group extrinsics 和 global pose 同源；后续若引入原版 GlobalGluer/pose averaging，需要补 diagnostics。
- 现在的重点不再是验证能否接近原版，而是稳定 high/low 坐标域、参数配置化、运行耗时和下游 Gaussian 质量验证。

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

当前推荐主线使用整合脚本：

```bash
python MERG3R/run_merg3r_gluemap_pipeline.py \
  --dataset /path/to/images \
  --output_dir /path/to/output \
  --path_tracker /path/to/vggsfm_tracker
```

默认配置已经固定为：

```text
pi3x + SIMPLE_PINHOLE + SIFT + ALIKED + pose groups + SPV
image_pyramid enabled
stage1_downscale_n = 4
stage1_multiple = 14
stage2_scale_factor = 0
pair_k_pose = 25
num_refinement_iterations = 2
```

常用可调参数：

```bash
# 输入采样
--num_images 181
--subsample 1

# image pyramid
--stage1_downscale_n 4
--stage1_multiple 14
--stage2_scale_factor 0

# pair 构建
--pair_k_pose 25
--pair_pose_rotation_threshold 30.0

# VGGSfM group 与 BA
--neighbors_per_center 25
--num_refinement_iterations 2
--ba_max_num_iterations 100

# debug
--save_virtual_tracks_debug
--no-debug_print
```

旧两阶段脚本仍可用于对照：

```text
MERG3R/export_merg3r_refine_inputs.py
MERG3R/gluemap/run_merg3r_pycolmap_refine.py
```

但它们不再是当前工程主入口。

## 已完成检查

最近一次代码整理后，本地静态检查已通过：

```bash
black MERG3R/utils/image_pyramid.py \
      MERG3R/utils/gluemap_refine_core.py \
      MERG3R/utils/gluemap_spv_refine.py \
      MERG3R/run_merg3r_gluemap_pipeline.py

ruff check MERG3R/utils/image_pyramid.py \
           MERG3R/utils/gluemap_refine_core.py \
           MERG3R/utils/gluemap_spv_refine.py \
           MERG3R/run_merg3r_gluemap_pipeline.py

python -m py_compile MERG3R/utils/image_pyramid.py \
                     MERG3R/utils/gluemap_refine_core.py \
                     MERG3R/utils/gluemap_spv_refine.py \
                     MERG3R/run_merg3r_gluemap_pipeline.py
```

完整 AI/SfM 运行仍以云端/目标 conda 环境为准，本地 `pycodex` 环境不验证 torch/pycolmap/vggsfm 全链路。

## 已完成工程化事项

- 两阶段脚本已缝合为 `run_merg3r_gluemap_pipeline.py`。
- Stage 2 逻辑已从旧 B 脚本拆入 `MERG3R/utils/`。
- 主线分支已收敛为 `pi3x + SIMPLE_PINHOLE + SIFT + ALIKED + pose + SPV`。
- 高/低分辨率图片预处理已整合为 image pyramid。
- Stage 1 low-res 与 Stage 2 high-res 的 image/K/depth 坐标域已经分离。
- SIFT database 已支持 prefilter 后过滤复用，避免第二遍重复 extract + match。
- 高分辨率 SIFT 已增加 `max_image_size=1024` 与 `max_num_features=4096` 限制，降低第一遍 SIFT 成本。

## 下一阶段优化方向

当前不再优先大改核心算法，后续重点是稳定性、可配置化和性能观测。

- 把 SIFT 限制参数从硬编码整理为 pipeline 配置，支持 `1024/4096` 与 `1600/8192` 两档。
- 缓存 VGGSfM prior tracks：group、neighbors、tracker input、query source 不变时复用。
- 缓存 virtual-track diagnostics：image/depth/extrinsic/K/groups 不变时复用。
- 减少重复 pycolmap triangulation 输出目录 IO。
- 对 debug 统计增加轻量/完整两级：默认只打印核心数值，完整模式再计算昂贵分桶。
- 对 SIFT/BA/virtual-track 阶段分别记录 wall time，优先优化耗时最高环节。
- 将“场景外远点”相关诊断加入 stats，而不是直接做硬删除：`xyz_norm`、depth 分位数、track length、triangulation angle、S-only/P-only/mixed 来源。

### 保留观察项

- 继续用 Gaussian 训练最终产物做质量判断，不只看 BA reprojection cost。
- 保留 `SP` vs `SPV` 对照。
- 保留 `star` vs `pose` 严格对照，但当前不作为主瓶颈。
- 暂不优先处理 TSDF/depth-pose mismatch；若后续需要 refined pose + dense depth，再做 BA 后 per-frame depth correction。
- 如果后续继续支持 `vggt_omega`，单独评估 `PINHOLE` 或 A 阶段强制/恢复 `fx=fy`，避免复现非 `fx=fy` K 强行进入 `SIMPLE_PINHOLE` 的问题。
