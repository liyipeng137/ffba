# MERG3R + GlueMap Debug: real angular reprojection errors 偏高

更新时间：2026-06-10

## 当前问题

在当前 MERG3R -> GlueMap SPV/augmented BA 融合流程中，`real: angular reprojection errors` 显著高于原版 GlueMap。

原版 GlueMap 在同一组 181 帧输入上的 real angular error：

| 来源 | 迭代 | real angular mean | median | `<0.5 deg` | angular filter removed |
| --- | ---: | ---: | ---: | ---: | --- |
| `spv_log/logs_181_gluemap_ori.log` | iter 1 | 0.2926 | 0.2138 | 85.7% | 58,339 obs / 901 tracks |
| `spv_log/logs_181_gluemap_ori.log` | iter 2 | 0.2077 | 0.1219 | 91.4% | 35,268 obs / 643 tracks |

当前融合流程的典型结果：

| 来源 | 配置 | 迭代 | real angular mean | median | `<0.5 deg` | angular filter removed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `logs_181_spv_all_04.log` | dense pairs, native tracker/group 旧逻辑 | iter 1 | 0.6057 | 0.4294 | 57.1% | 172,076 obs / 7,646 tracks |
| `logs_181_spv_all_04.log` | dense pairs, native tracker/group 旧逻辑 | iter 2 | 0.5995 | 0.4160 | 58.3% | 168,182 obs / 7,418 tracks |
| `logs_181_spv_all_05_1024.log` | tracker input=1024 | iter 1 | 0.5947 | 0.4420 | 56.1% | 216,432 obs / 8,406 tracks |
| `logs_181_spv_all_05_1024.log` | tracker input=1024 | iter 2 | 0.5827 | 0.4260 | 57.7% | 211,230 obs / 8,067 tracks |
| `logs_181_spv_all_06_1024_star.log` | tracker input=1024, star-like group | iter 1 | 0.5956 | 0.4314 | 57.0% | 201,483 obs / 7,918 tracks |
| `logs_181_spv_all_08_1024_star_debug.log` | pi3x, tracker input=1024, star-like group | iter 1 | 0.3501 | 0.2419 | 82.6% | 79,078 obs / 1,732 tracks |
| `logs_181_spv_all_08_1024_star_debug.log` | pi3x, tracker input=1024, star-like group | iter 2 | 0.3230 | 0.2113 | 84.6% | 69,737 obs / 1,395 tracks |

阶段性结论：早期测试使用的前馈模型是 `vggt_omega`，其当前实现没有约束 `fx=fy`。在 B 阶段使用 `SIMPLE_PINHOLE` 和 GlueMap 风格 shared intrinsic 时，这会导致相机模型约束与导出 K 的假设不一致，表现为 real angular error 长期停在 `mean ~= 0.58-0.61 deg, median ~= 0.42-0.44 deg`。

切换到 `pi3x` 后，pi3x 使用 mogo 恢复内参，隐式约束 `fx=fy`，与原版 GlueMap 的 pi3x 逻辑及当前 `SIMPLE_PINHOLE` 更一致。重新跑后 real angular error 改善到 `mean=0.3501 deg, median=0.2419 deg, <0.5deg=82.6%`，已经明显接近原版 GlueMap iter 1 的 `mean=0.2926 deg, median=0.2138 deg, <0.5deg=85.7%`。

## 已做实验

### 1. 提高 A 阶段 pair 密度

A 阶段输出：

```text
[EXPORT] images=181, pairs=2644
[EXPORT] pair degree: min=25, median=28.0, mean=29.2, p90=35.0, max=42, zero=0
```

B 阶段 prior tracking 上涨：

```text
VGGSfM prior done: groups=181, tracks=154722, observations=684962
```

效果：

- real track 数量有提升；
- final `refined_gluemap_aba` 的 real points 有提升；
- 但 `real: angular reprojection errors` 仍维持在 `mean ~= 0.60`，没有接近原版。

判断：pair 密度不是主要瓶颈，至少不是 angular error 偏高的直接原因。

### 2. tracker input 改为 GlueMap-style 1024

改动：VGGSfM tracker 在 1024 padded image 上跑，再把坐标映射回当前 image/domain。

结果：

```text
VGGSfM prior done: tracks=153767, observations=791459, tracker_input=1024
Prior DB written: input_tracks=153767, pairs=2850, raw_kp=791459, merged_kp=777084
SelectTrack: points=151990 -> 79087, observations=793145 -> 482295
real: angular reprojection errors: mean=0.5947 deg, median=0.4420 deg, < 0.5 deg: 56.1%
```

第二轮：

```text
SelectTrack: points=151922 -> 79939, observations=793939 -> 488638
real: angular reprojection errors: mean=0.5827 deg, median=0.4260 deg, < 0.5 deg: 57.7%
```

效果：

- P/prior observations 明显增加；
- final real points 从旧配置的约 61k 提升到约 67.8k；
- 但 angular error 基本没有改善。

判断：1024 tracker input 对数量有帮助，但不是当前 angular error 的主因。

### 3. group 构建切到 star-like

改动：B 阶段使用 GlueMap `BaseStarDataset` 风格构建 star-like group，`groups=181`，`group_neighbors_median=25.0`。

结果：

```text
VGGSfM prior done: groups=181, group_strategy=star, group_neighbors_median=25.0,
tracks=153042, observations=726323, tracker_input=1024

Prior DB written: input_tracks=153042, tracks=153042, pairs=2551,
raw_kp=726323, merged_kp=713650

SelectTrack: points=150798 -> 81261, observations=729662 -> 456956
SelectVirtualTrack: points=21094 -> 21092, observations=524243 -> 524200

real: angular reprojection errors: mean=0.5956 deg, median=0.4314 deg,
< 0.5 deg: 57.0%
real: removed 201483 observations, 7918 tracks
```

效果：

- group 结构更贴近原版；
- SelectTrack 后保留的 real tracks/observations 有一定变化；
- angular error 仍基本不变。

判断：star-like group 不是当前 angular error 高的主要原因。

### 4. 按 track source 分桶统计 angular error

新增 debug：在 SelectTrack 后、angular filter 前，按 point3D track source 分桶：

- `S-only`: 所有 observation 都来自 S/SuperPoint-LightGlue 区间；
- `P-only`: 所有 observation 都来自 P/VGGSfM prior 区间；
- `mixed`: 同一 3D track 同时包含 S 和 P observation。

当前 star-like + 1024 结果：

```text
S-only: tracks=15039, track_obs=75908, mean=0.6329, median=0.4587, <0.5deg=53.3%
P-only: tracks=66222, track_obs=381048, mean=0.5882, median=0.4272, <0.5deg=57.7%
mixed: tracks=0, track_obs=0

real: angular reprojection errors: mean=0.5956 deg, median=0.4314 deg, < 0.5 deg: 57.0%
```

观察：

- `mixed=0`，当前 merged DB 三角化出的 real tracks 基本没有 S/P 跨源融合；
- P-only 占 observation 主体，但 P-only 并不比 S-only 更差；
- S-only 本身也明显偏高，甚至 mean/median 略高于 P-only。

判断：

- 高 angular error 不是单纯由 P/prior track 拉坏；
- S-only 也高，说明需要优先检查 camera/K/坐标域一致性；
- `mixed=0` 不一定是 bug。原版 `merge_colmap_databases` 也是拼接 keypoints 和 remap matches，不会自动跨数据库按坐标 dedup 出 mixed tracks。

### 5. 前馈模型从 vggt_omega 切换到 pi3x

背景：

- 之前全部测试实际使用的是 `vggt_omega`；
- `vggt_omega` 当前导出实现没有约束 `fx=fy`；
- 当前 B 脚本使用 `SIMPLE_PINHOLE`，并且 `intrinsics_averaging` 后得到 shared K；
- `pi3x` 使用 mogo 恢复内参，隐式约束 `fx=fy`，这与原版 GlueMap pi3x 模型逻辑一致。

新日志：`spv_log/logs_181_spv_all_08_1024_star_debug.log`

配置：

```text
Loaded artifacts: images=181, pairs=2662, image_size_hw=(476, 350),
camera_model=SIMPLE_PINHOLE, track_mode=SPV
VGGSfM prior done: groups=181, group_strategy=star, group_neighbors_median=25.0,
tracks=156123, observations=727566, tracker_input=1024
Intrinsics averaged: fx=361.19, fy=361.19, cx=175.00, cy=238.00
Virtual-track diagnostics done: valid_obs=523806, points/group median=117.0
```

iter 1：

```text
SelectTrack: points=155801 -> 75260, observations=772165 -> 448902
S-only: tracks=14988, track_obs=82793, mean=0.5025, median=0.3500, <0.5deg=63.6%
P-only: tracks=60272, track_obs=366109, mean=0.3156, median=0.2286, <0.5deg=86.9%
mixed: tracks=0
real: angular reprojection errors: mean=0.3501 deg, median=0.2419 deg, < 0.5 deg: 82.6%
real: removed 79078 observations, 1732 tracks
Initial observations: real=369824, virtual=522887
```

iter 2：

```text
SelectTrack: points=155795 -> 75044, observations=772821 -> 448125
S-only: tracks=14958, track_obs=82933, mean=0.4907, median=0.3346, <0.5deg=64.7%
P-only: tracks=60086, track_obs=365192, mean=0.2849, median=0.1960, <0.5deg=89.2%
real: angular reprojection errors: mean=0.3230 deg, median=0.2113 deg, < 0.5 deg: 84.6%
real: removed 69737 observations, 1395 tracks
Final: 72022 real tracks (366057 obs), 21005 virtual tracks (520852 obs)
```

观察：

- real angular error 从 vggt_omega 阶段的 `mean ~= 0.60` 明显下降到 `mean=0.35/0.32`；
- `<0.5 deg` 从约 57% 提升到 82.6%/84.6%，已经接近原版 GlueMap；
- angular filter 删除量从约 20 万 obs 降到 7-8 万 obs，删除 track 数也从约 8k 降到约 1.4k-1.7k；
- final real points 提升到 72,022，高于 vggt_omega + 1024/star 阶段的约 67k；
- P-only error 已经非常接近原版整体水平，iter 2 P-only median=0.1960，`<0.5deg=89.2%`；
- S-only 仍明显偏高，iter 2 mean=0.4907，median=0.3346，`<0.5deg=64.7%`。

判断：

- 此前 angular error 偏高的首要原因，很可能是 `vggt_omega` 导出的非 `fx=fy` 相机内参与当前 `SIMPLE_PINHOLE`/GlueMap-style shared K 假设不一致；
- 切到 pi3x 后，相机内参约束与原版 GlueMap/pi3x 更一致，因此 real angular error 大幅改善；
- 当前剩余差距主要集中在 S-only，而不是 P-only。

## 与原版 GlueMap 流程的关键差异

### 数据库 merge 本身基本一致

原版 GlueMap 在 `global_refinement.py` 中调用：

```python
merge_colmap_databases(
    db_path_primary=tracks_db,
    db_path_secondary=database_sift,
    output_path=database_merged,
    primary_features_first=False,
)
```

这会让 SIFT keypoints 在 merged DB 中排在前面，tracks/prior keypoints 排在后面。

当前 B 脚本调用：

```python
merge_colmap_databases(
    database_lightglue,
    database_vggsfm_prior,
    database_merged,
    primary_features_first=True,
)
```

这同样让 S/LightGlue keypoints 在前，P/prior keypoints 在后。

因此，当前 `sift_count`/`s_keypoint_count` 分区与原版 SelectTrack 的前后区间逻辑是一致的。

### prior database 构建方式仍有差异

原版 prior database：

- 由 `prepare_glomap_prior(...)` 构建；
- 内部调用 `TrackEstablishment.establish_keypoints_and_correspondences(...)`；
- 对每个 group 主要构造 center-neighbor star correspondences。

当前 prior database：

- 由 B 脚本的 `write_tracks_database(...)` 构建；
- 内部通过 `tracks_to_keypoints_and_matches(...)` 把 VGGSfM tracks 转成 keypoints/matches；
- 当前实现会把同一条 prior track 的所有 observations 做 all-pairs clique correspondences。

这个差异可能影响三角化出来的 P-only track 结构，但由于 S-only 也偏高，它未必是 angular error 偏高的首要原因。

## 当前判断

截至目前，几次实验都说明：

1. 提高 pair 密度主要提升数量，不显著改善 angular error。
2. tracker input 改为 1024 主要提升 P observations，不显著改善 angular error。
3. group 改为 star-like 后，angular error 仍基本不变。
4. 使用 `vggt_omega` 时，S-only 与 P-only angular error 都高，且 `mixed=0`。
5. 切换到 `pi3x` 后，P-only angular error 大幅改善，整体 real angular error 接近原版；说明 camera/K 约束一致性是此前最大问题。

因此当前最可疑方向是：

1. 对 `vggt_omega` 路径，需要明确是否应该使用 `PINHOLE` 或在 A 阶段强制/恢复 `fx=fy` 后再进 B；
2. 对当前 `pi3x` 路径，剩余差距主要集中在 S-only，可能来自 SuperPoint/LightGlue 与原版 SIFT 的匹配差异、S keypoint 坐标域、或 LightGlue DB 写入/过滤策略；
3. 当前 prior DB 的 all-pairs clique 构建方式仍不同于原版 `TrackEstablishment` 的 star correspondence，但它已不是解释 pi3x 后整体 angular error 的首要问题。

## 建议下一步 debug

### A. 先隔离 S-only

只跑当前 MERG3R coarse pose + global K + `database_lightglue.db`：

- 不引入 P/prior；
- 不引入 virtual tracks；
- triangulate 后直接统计 angular error。

目的：

- 如果 S-only 仍是 `mean ~= 0.6`，问题基本在 camera/K/坐标域或 S match 本身；
- 如果只跑 S 明显变好，则问题可能来自 merged DB、SelectTrack 或 P/S 并存后的三角化行为。

### B. 比较 shared K vs per-frame K

用同一套 S 数据库做两次 coarse reconstruction/triangulation：

1. `intrinsics_averaging` 后的 shared K；
2. 每帧 MERG3R 初始 K。

目的：

- 判断 global K 是否把 angular error 放大；
- 特别检查 `SIMPLE_PINHOLE` 是否过强约束了 `fx/fy`。

### C. 打印 keypoint 坐标域诊断

对 S keypoints 和 P keypoints 分别打印：

- 每图 `x/y min/max`；
- 超出 image width/height 的 keypoints 数量；
- `K cx/cy` 与 image center 的偏差；
- image size 是否与 database camera width/height 一致。

目的：

- 排除坐标映射、width/height、1024 -> artifact image 域回映射污染等问题。

### D. 直接评估 pose/K 的 epipolar consistency

抽样 LightGlue matches，用当前 MERG3R `R,t,K` 直接计算 pair-level epipolar/angular residual，不经过三角化。

目的：

- 如果 epipolar residual 已经高，说明 camera/K 与 2D matches 不一致；
- 如果 epipolar residual 低但 triangulation angular 高，再查数据库写入、triangulation options 和 track 构建。

### E. 对齐 prior DB 构建方式

尝试把当前 VGGSfM prior tracks 组织成 `predictions_dict["tracks"] / ["scores"] / ["indexes"]`，复用原版 `prepare_glomap_prior(...)` 或 `TrackEstablishment.establish_keypoints_and_correspondences(...)` 生成 prior database。

目的：

- 排除 `tracks_to_keypoints_and_matches` 的 all-pairs clique 与原版 star correspondence 差异；
- 让 P/prior database 更贴近原版 GlueMap。
