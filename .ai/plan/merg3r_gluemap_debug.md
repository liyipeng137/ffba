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
| `logs_181_spv_all_08_1024_star_debug.log` | pi3x, LightGlue as S, tracker input=1024, star-like group | iter 1 | 0.3501 | 0.2419 | 82.6% | 79,078 obs / 1,732 tracks |
| `logs_181_spv_all_08_1024_star_debug.log` | pi3x, LightGlue as S, tracker input=1024, star-like group | iter 2 | 0.3230 | 0.2113 | 84.6% | 69,737 obs / 1,395 tracks |
| `logs_181_spv_all_09_sift_debug.log` | pi3x, SIFT as S, tracker input=1024, star-like group | iter 1 | 0.2835 | 0.2040 | 89.7% | 57,260 obs / 1,554 tracks |
| `logs_181_spv_all_09_sift_debug.log` | pi3x, SIFT as S, tracker input=1024, star-like group | iter 2 | 0.2390 | 0.1568 | 92.1% | 43,590 obs / 1,065 tracks |
| `logs_181_spv_all_10_pose_debug.log` | pi3x, SIFT as S, tracker input=1024, pose group | iter 1 | 0.2758 | 0.2057 | 90.4% | 未完整记录 |
| `logs_181_spv_all_10_pose_debug.log` | pi3x, SIFT as S, tracker input=1024, pose group | iter 2 | 0.2352 | 0.1587 | 92.4% | 45,091 obs / 958 tracks |

阶段性结论：早期测试使用的前馈模型是 `vggt_omega`，其当前实现没有约束 `fx=fy`。在 B 阶段使用 `SIMPLE_PINHOLE` 和 GlueMap 风格 shared intrinsic 时，这会导致相机模型约束与导出 K 的假设不一致，表现为 real angular error 长期停在 `mean ~= 0.58-0.61 deg, median ~= 0.42-0.44 deg`。

切换到 `pi3x` 后，pi3x 使用 mogo 恢复内参，隐式约束 `fx=fy`，与原版 GlueMap 的 pi3x 逻辑及当前 `SIMPLE_PINHOLE` 更一致。重新跑后 real angular error 改善到 `mean=0.3501 deg, median=0.2419 deg, <0.5deg=82.6%`，已经明显接近原版 GlueMap iter 1 的 `mean=0.2926 deg, median=0.2138 deg, <0.5deg=85.7%`。

进一步把 S 分支从 A 阶段导出的 LightGlue database 改为原版 GlueMap 风格的 SIFT extract + match 后，real angular error 已经达到或接近原版量级：star-like group 下 iter 2 为 `mean=0.2390 deg, median=0.1568 deg, <0.5deg=92.1%`，pose group 下 iter 2 为 `mean=0.2352 deg, median=0.1587 deg, <0.5deg=92.4%`。

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

### 6. S 分支从 LightGlue 改为原版 GlueMap 风格 SIFT

背景：

- 原版 GlueMap 的 S 分支是 `database_sift.db`；
- 当前融合脚本此前使用 A 阶段导出的 LightGlue database 作为 S；
- 为贴近原版，B 脚本新增 `s_database_mode=sift`，不再依赖 A 阶段导出的 LightGlue 特征和匹配；
- prior track snap 逻辑对应增加 `prior_snap_to_sift`，让 P/prior keypoint 尝试 snap 到当前 SIFT keypoint。

新日志：`spv_log/logs_181_spv_all_09_sift_debug.log`

配置：

```text
SIFT DB ready: keypoints=115553, pairs=442, matches=25067
Prior snap to sift: snapped=7259, unsnapped_kept=720299, mean_dist=0.688,
center_snap=0.5%, neighbor_snap=1.1%
Prior DB written: input_tracks=156119, tracks=156119, pairs=2508,
raw_kp=727558, merged_kp=725690
```

iter 1：

```text
SelectTrack: kept 7781 SIFT + 98097/148094 non-SIFT, removed 49997
SelectTrack: points=155875 -> 105878, observations=722586 -> 545479
S-only: tracks=7781, track_obs=22889, mean=0.2225, median=0.1549, <0.5deg=92.3%
P-only: tracks=98097, track_obs=522590, mean=0.2862, median=0.2063, <0.5deg=89.6%
real: angular reprojection errors: mean=0.2835 deg, median=0.2040 deg, < 0.5 deg: 89.7%
```

iter 2：

```text
SelectTrack: kept 7797 SIFT + 97969/148131 non-SIFT, removed 50162
SelectTrack: points=155928 -> 105766, observations=723339 -> 545390
S-only: tracks=7797, track_obs=22976, mean=0.1819, median=0.1193, <0.5deg=94.4%
P-only: tracks=97969, track_obs=522414, mean=0.2415, median=0.1586, <0.5deg=92.0%
real: angular reprojection errors: mean=0.2390 deg, median=0.1568 deg, < 0.5 deg: 92.1%
Final: 103500 real tracks (490627 obs), 20782 virtual tracks (512471 obs)
```

观察：

- SIFT DB 的 match 数量不比 LightGlue 多，甚至更少，但 S-only angular error 明显更好；
- LightGlue as S 的 iter 2 S-only 为 `mean=0.4907, median=0.3346, <0.5deg=64.7%`；
- SIFT as S 的 iter 2 S-only 降到 `mean=0.1819, median=0.1193, <0.5deg=94.4%`；
- P-only 也随之从 LightGlue as S 的 `mean=0.2849, median=0.1960, <0.5deg=89.2%` 改善到 `mean=0.2415, median=0.1586, <0.5deg=92.0%`；
- final real points 达到 103,500，已经非常接近原版 GlueMap 的 103,631。

判断：

- 在 pi3x 修正 K/camera 假设后，剩余主要差距确实来自 S 分支；
- SIFT 的数量不一定更多，但几何质量更贴近原版 GlueMap 的三角化/SelectTrack 假设；
- SIFT as S 后，当前融合流程的 angular error 和 final real point 数量都已接近原版。

### 7. pose group 与 star-like group 对照

新日志：`spv_log/logs_181_spv_all_10_pose_debug.log`

配置：

- `s_database_mode=sift`
- `vggsfm_tracker_input=1024`
- `group_strategy=pose`
- 当前日志从 `SIFT DB ready` 后开始截取，没有包含完整 header。

关键结果：

```text
SIFT DB ready: keypoints=122104, pairs=355, matches=21763
Prior snap to sift: snapped=7883, unsnapped_kept=792657, mean_dist=0.690,
center_snap=0.5%, neighbor_snap=1.1%
Prior DB written: input_tracks=156529, tracks=156529, pairs=2801,
raw_kp=800540, merged_kp=798478
```

iter 1：

```text
SelectTrack: kept 7534 SIFT + 93857/148625 non-SIFT, removed 54768
SelectTrack: points=156159 -> 101391, observations=794965 -> 581328
S-only: tracks=7534, track_obs=21075, mean=0.2077, median=0.1467, <0.5deg=93.7%
P-only: tracks=93857, track_obs=560253, mean=0.2784, median=0.2080, <0.5deg=90.2%
real: angular reprojection errors: mean=0.2758 deg, median=0.2057 deg, < 0.5 deg: 90.4%
```

iter 2：

```text
SelectTrack: kept 7539 SIFT + 94026/148676 non-SIFT, removed 54650
SelectTrack: points=156215 -> 101565, observations=795708 -> 583137
S-only: tracks=7539, track_obs=21162, mean=0.1675, median=0.1075, <0.5deg=95.4%
P-only: tracks=94026, track_obs=561975, mean=0.2377, median=0.1608, <0.5deg=92.2%
real: angular reprojection errors: mean=0.2352 deg, median=0.1587 deg, < 0.5 deg: 92.4%
Final: 99437 real tracks (525608 obs), 20851 virtual tracks (520596 obs)
```

与 star-like + SIFT 对比：

| 指标 | star-like + SIFT | pose + SIFT | 判断 |
| --- | ---: | ---: | --- |
| iter 1 real mean | 0.2835 | 0.2758 | 基本一致，pose 略低 |
| iter 2 real mean | 0.2390 | 0.2352 | 基本一致 |
| iter 2 `<0.5deg` | 92.1% | 92.4% | 基本一致 |
| final real tracks | 103,500 | 99,437 | pose 少约 4k |
| final virtual tracks | 20,782 | 20,851 | 基本一致 |

观察：

- pose group 的 angular error 与 star-like 差别很小；
- pose group 产生的 prior DB 更密，`raw_kp=800540` 高于 star-like 的 `727558`；
- 但 pose 的 SelectTrack 剪枝更重，final real tracks 比 star-like 少约 4k；
- 这次对比里 SIFT DB 本身也有差异，`pairs/matches/keypoints` 不完全一致，因此不是严格只改 `group_strategy` 的 A/B 对照。

判断：

- `group_strategy` 不是当前 angular error 的主要影响项；
- 如果目标是贴近原版 GlueMap，star-like 仍然是更合理默认；
- pose 可作为 sanity check，但目前没有证据说明它优于 star-like。

### 8. star 图结构确认

GlueMap 的 star 图构建没有 camera-center radius 或 spatial radius 配置。相关控制主要是：

- `valid_dg_threshold`：Doppelgangers score 过滤；
- `max_neighbors` / 当前 B 脚本的 `neighbors_per_center`：每个 center 最多保留多少邻居；
- `sequential_edges` / 当前 B 脚本的 `star_sequential_window`：可选保留顺序边；
- connectivity repair：如果图断开，会补回部分边保证连通。

star group 内部结构是：

```text
center_i -> [neighbor_a, neighbor_b, neighbor_c, ...]
```

单个 group 不会递归展开 neighbor-of-neighbor。全局图当然可以有多跳路径，例如 `frame_0 -- frame_1 -- frame_2`，但每个 star inference/refinement group 只使用 center 的直接邻居。

在当前 `skip_doppelgangers=True` 时，如果所有 pair score 都设为 `1.0`，star-like group 的 top-K 邻居选择没有真实 score 区分，更多依赖输入 pair 顺序和 connectivity repair；pose group 则按 camera-center distance 排 direct neighbors。这解释了两者结果相近但不完全一致。

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

当前 B 脚本在 `s_database_mode=lightglue` 时调用：

```python
merge_colmap_databases(
    database_lightglue,
    database_vggsfm_prior,
    database_merged,
    primary_features_first=True,
)
```

这同样让 S/LightGlue keypoints 在前，P/prior keypoints 在后。

当前 B 脚本在 `s_database_mode=sift` 时已改为更贴近原版：

```python
merge_colmap_databases(
    database_vggsfm_prior,
    database_sift,
    database_merged,
    primary_features_first=False,
)
```

因此，当前 `sift_count`/`s_keypoint_count` 分区与原版 SelectTrack 的前后区间逻辑是一致的。SIFT mode 下，数据库 merge 顺序也已经贴近原版 GlueMap。

### prior database 构建方式仍有差异

原版 prior database：

- 由 `prepare_glomap_prior(...)` 构建；
- 内部调用 `TrackEstablishment.establish_keypoints_and_correspondences(...)`；
- 对每个 group 主要构造 center-neighbor star correspondences。

当前 prior database：

- 由 B 脚本的 `write_tracks_database(...)` 构建；
- 内部通过 `tracks_to_keypoints_and_matches(...)` 把 VGGSfM tracks 转成 keypoints/matches；
- 当前实现会把同一条 prior track 的所有 observations 做 all-pairs clique correspondences。

这个差异可能影响三角化出来的 P-only track 结构。当前 SIFT mode 已经让 angular error 和 final real points 接近原版，因此它更像是后续做 strict parity 时需要继续收敛的差异，而不是此前 `mean ~= 0.60 deg` 的首要原因。

## 当前判断

截至目前，几次实验都说明：

1. 提高 pair 密度主要提升数量，不显著改善 angular error。
2. tracker input 改为 1024 主要提升 P observations，不显著改善 angular error。
3. group 改为 star-like 后，angular error 仍基本不变。
4. 使用 `vggt_omega` 时，S-only 与 P-only angular error 都高，且 `mixed=0`。
5. 切换到 `pi3x` 后，P-only angular error 大幅改善，整体 real angular error 接近原版；说明 camera/K 约束一致性是此前最大问题。
6. S 分支从 LightGlue 改为原版风格 SIFT 后，S-only、P-only 和整体 angular error 都进一步接近原版。
7. star-like 与 pose group 的 angular error 差别很小，group strategy 不是当前主影响项。

因此当前判断是：

1. 对 `vggt_omega` 路径，需要明确是否应该使用 `PINHOLE` 或在 A 阶段强制/恢复 `fx=fy` 后再进 B；
2. 对当前 `pi3x + SIFT` 路径，angular error 已接近原版，主要关注点应转向 strict parity：final real observations、pair graph、frame filtering、prior DB 构建方式是否与原版一致；
3. 当前 prior DB 的 all-pairs clique 构建方式仍不同于原版 `TrackEstablishment` 的 star correspondence，但它已不是解释此前高 angular error 的首要问题。

## 建议下一步 debug

### A. 做严格 star vs pose 对照

用完全相同的 A 输出、frame filtering 结果和 SIFT DB 输入，只切换：

- `group_strategy=star`
- `group_strategy=pose`

目的：

- 排除当前 `logs_181_spv_all_09_sift_debug.log` 与 `logs_181_spv_all_10_pose_debug.log` 中 SIFT DB 自身不一致带来的干扰；
- 确认 group strategy 对 final real tracks 少约 4k 的影响是否稳定。

### B. 对齐原版 prior DB 构建方式

尝试把当前 VGGSfM prior tracks 组织成 `predictions_dict["tracks"] / ["scores"] / ["indexes"]`，复用原版 `prepare_glomap_prior(...)` 或 `TrackEstablishment.establish_keypoints_and_correspondences(...)` 生成 prior database。

目的：

- 排除 `tracks_to_keypoints_and_matches` 的 all-pairs clique 与原版 star correspondence 差异；
- 让 P/prior database 更贴近原版 GlueMap；
- 观察 final real observations 和 SelectTrack 剪枝行为是否更接近原版。

### C. 固定 181 帧与 frame filtering

目的：

- 当前部分日志显示 merged DB 为 179 images，说明 frame filtering 可能 drop 了帧；
- 若要与原版 181 帧严格比较，需要固定 drop 策略，或先关闭低覆盖帧过滤。

### D. 继续保留 keypoint 坐标域诊断

对 S keypoints 和 P keypoints 分别打印：

- 每图 `x/y min/max`；
- 超出 image width/height 的 keypoints 数量；
- `K cx/cy` 与 image center 的偏差；
- image size 是否与 database camera width/height 一致。

目的：

- 排除坐标映射、width/height、1024 -> artifact image 域回映射污染等问题。

### E. 对 vggt_omega 路径单独决策 camera model

如果后续仍要支持 `vggt_omega`：

- 评估是否应使用 `PINHOLE` 而不是 `SIMPLE_PINHOLE`；
- 或在 A 阶段对导出内参显式恢复/约束 `fx=fy`。

目的：

- 避免再次把非 `fx=fy` 的 K 强行喂给 `SIMPLE_PINHOLE`，复现 `mean ~= 0.60 deg` 的错误模式。
