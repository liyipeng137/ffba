# 两视图忽略与全来源 SelectTrack 对照

日期：2026-09-08。状态：已实现，远程三组 BAE 回放及质量诊断完成，退出码均为 0。

## 变更

- `utils/gluemap_refine_core.py` 三角化使用 `ignore_two_view_tracks=True`。
- `run_select_tracks` 向原 GlueMap 选择器传递空的受保护 SIFT 数量表，使全部
  tracks 进入同一个候选集合；真实来源表仅用于前后统计。无需改动或重编译 C++。
- 固定种子 42 的 shuffle、pair 支持阈值、几何过滤顺序、BAE 参数保持原样。
- 选择范围记录为 `selection_scope=all_sources`。下层旧日志中的 SIFT/non-SIFT
  是“受保护/待选择”分组，在此模式下会显示 0 SIFT；实际来源看上层统计。
- 两项默认作用于共享后端，两种 prior 使用相同规则。没有新增主流程 CLI 参数。

`ignore_two_view_tracks` 影响三角化，不等同于最终 min_track_length=3。
SelectTrack 是按 pair 覆盖去冗余，不是几何质量排序，也不保证空间均匀覆盖。

## 冻结输入

远程主机 `Runpod-4090-lyp-UUU`，Ubuntu/RTX 4090，conda `gluemap-merg3r`。
源目录 `/kiri/tmp/ffba_789_loma_355_batch2_t8`。
复用 `database_merged.db`、SIFT feature ID、`coarse` 相机与
`intrinsics_refine_inputs.npz`，不重复特征提取或匹配，不覆盖源产物。

实验目录 `/kiri/tmp/ffba_789_track_selection_ablation`：

| 子目录 | ignore_two_view_tracks | SelectTrack |
| --- | --- | --- |
| baseline | False | SIFT 优先 |
| no_two_view | True | SIFT 优先 |
| all_sources | True | 全来源共同筛选 |

脚本 `scripts/replay_track_selection.py` 在同一后端中回放三个版本。对照只恢复源快照
中的原三角化和 SelectTrack 函数，每组写入有效策略、输入哈希和完整 refinement 统计。
coarse 相机经历存取，三角化存在运行差异；原先全流程日志仅作参考，新的 baseline
才是这次比较基线。耗时仅代表后端回放，不能报告为端到端提速。

## 质量检查

`scripts/evaluate_track_selection.py`：

- 所有帧是否有观测、最弱帧观测数、8×8 网格覆盖（每格至少 2 个观测）。
- 全部 tracks 与长度 ≥3 tracks 的覆盖，避免将主动删除两帧点误读为长轨质量下降。
- track 长度、来源与残差分布、采样最大三角化角度。
- 相机中心经过 Sim(3) 对齐后的变化、旋转变化与焦距变化。变化不是误差真值。
- 从原始 SIFT/LoMa DB 的每个已验证 pair 固定采样最多 64 个匹配，种子 20260908；
  通过图像名映射到 merged DB，必要时交换匹配列，并校验源关键点坐标与 merged DB
  对应块完全一致。避开此次发现的旧合并逻辑的列顺序问题。
  在三组优化相机上计算同一批对应的对称极线角度残差，分别报告 SIFT/prior。
  不依赖最终是否保留该点，减少选择偏差；这些匹配参与过上游建图，不是独立留出集。
- 没有独立 GT 或下游渲染对照时，只能判定这些检查是否出现退化信号，不能证明绝对质量持平。

## 验证结果

本地核心/LoMa/评估测试 66 通过，真实 pygluemap 测试本地跳过；远程核心与评估 28 项全部通过。
新增真实 C++ 测试验证冗余 SIFT 可以删除，交换来源标签不影响保留结果，来源统计正确。
固定匹配几何误差、相机全局相似变换对齐及源图像 ID 反序的三项数值测试通过。

## 实测结果

| 指标 | baseline | no_two_view | all_sources |
| --- | ---: | ---: | ---: |
| 后端墙钟 | 527.35 s | 417.29 s | 380.33 s |
| 最终点数 | 495,254 | 311,352 | 250,208 |
| 最终观测数 | 2,253,527 | 1,887,422 | 1,651,957 |
| SIFT-only 点数 | 351,432 | 175,950 | 100,883 |
| Prior-only 点数 | 143,820 | 135,401 | 149,324 |
| 最终两帧 track 数 | 208,196 | 23,218 | 24,643 |
| 最少每帧观测 | 444 | 373 | 374 |
| 长度 ≥3 track 平均网格覆盖 | 90.801% | 90.847% | 90.954% |
| 全部 track 平均网格覆盖 | 92.744% | 91.655% | 91.906% |
| 固定 SIFT 匹配极线误差中位数 | 0.042340° | 0.042384° | 0.042455° |
| 固定 SIFT 匹配极线误差 P90 | 0.150679° | 0.150784° | 0.150758° |
| 固定 prior 匹配极线误差中位数 | 0.072205° | 0.072190° | 0.072119° |
| 对齐后相机旋转变化 P90 | — | 0.02448° | 0.03062° |
| 焦距相对基线变化 | — | +0.00378% | +0.00940% |

联合方案后端提速 27.88%，点数减少 49.48%，观测减少 26.69%。仅忽略两视图已获得
20.87% 的后端提速；共同筛选在该基础上再减少 8.86% 耗时。所有组均有 789 帧，
无零观测帧、无少于 64 观测的帧，track 图始终为一个 789 帧连通分量。
no_two_view 和 all_sources 首轮三角化均为 444,575 点 / 2,113,777 观测，区别从
SelectTrack 开始。没有为速度重新匹配、减少 LoMa pairs 或调整 BAE 参数。

固定匹配评估共 757,805 个样本：SIFT 316,648，prior 441,157。上述误差是
对称极线角度误差，不是最终三维点的角度重投影误差。源匹配包含未通过多视图过滤的
对应，prior 的绝对 P90 约 3.4°，不能将其直接解释为最终重建误差。

最终点集的 SIFT-only 角度均值从 0.03999° 升至 0.05157°，prior 从 0.11967°
降至 0.11818°。点集及长度分布变化很大；固定匹配指标基本稳定，不据此把 SIFT
拟合残差升高单独归因为相机精度下降。

## 局部质量与结论边界

- 最弱帧 `frame_000261.png`：观测 444 → 374，长度 ≥3 track 网格覆盖保持
  59.375%。此帧源 SIFT 验证匹配为 0；prior 固定匹配中位数 0.16918° → 0.17185°，
  P90 1.15590° → 1.17294°，存在小幅变差，但没有失去多视图空间覆盖。
- 最大旋转变化在 `frame_000217.png`，约 0.5477°；`frame_000003.png` 为
  0.4270°，`frame_000223.png` 为 0.3085°。这些是相对基线变化，不是 GT 误差。
- 217 帧 SIFT 固定匹配中位数 0.06422° → 0.07438°，P90 却从 0.19088°
  降至 0.18224°；prior P90 4.97571° → 5.43790°。局部指标并非全面改善，不能
  宣称完全无损。003/223 帧的 SIFT P90 则有所下降。
- 联合方案长度 ≥3 track 的平均网格覆盖略升，最差单帧覆盖变化为下降两个网格
  （3.125 个百分点）。全部 tracks 的平均网格覆盖下降约 0.84 个百分点，保留了
  小幅覆盖代价这个事实。网格二值覆盖也不能保证每格深度分布或所有表面都保持。
- 暂未发现系统性相机/多视图覆盖退化，支持继续试用这版减量策略；没有独立 GT、
  点云表面误差或下游渲染 A/B，因此尚未证明真实重建质量持平。优先复查上述弱帧。

## 输入审计中发现的既有问题

1. 首轮回放脚本直接用 PyCOLMAP 打开源 DB。它即使只 open/close 也会改变 SQLite
   文件字节哈希。独立副本实验确认所有 13 张表的行数与内容哈希相同，而物理哈希改变。
   所以本轮不声称源 DB 字节完全未变；匹配没有重新生成，相机初值文件哈希相同。
   原执行脚本保存在远程 `replay_track_selection_used.py`；当前脚本已修订为每组
   使用 SQLite backup 副本，防止将 PyCOLMAP 写入行为作用于源 DB。
2. 既有 `merge_colmap_databases` 将输出图像 ID 排序后，没有在 ID 顺序反转时交换
   matches 的两列。本数据有 14 个 SIFT pairs / 8,404 个已验证对应受影响；源对应
   未交换的版本全部存在于 merged DB。183 个对应在 2 个 pairs 中发生索引越界，
   其余错误未必能靠索引范围发现。位置：`third_party/GlueMap/gluemap/utils/colmap.py`
   的普通匹配与 two-view geometry 合并分支。
   本次保留该上游输入不变，以隔离用户要求的两项筛选变更；质量评估改用原始源 DB
   按名字正确对齐的匹配。这个合并问题已在后续独立修复（见下节），不能归因于本次策略；
   上述三组 BAE 结果仍使用修复前数据库。

## 后续：数据库匹配列顺序修复

2026-09-08 按用户要求修复 `merge_colmap_databases`。新增共用的
`_remap_matches_to_output_pair`：先按源图像顺序加各自的关键点偏移，输出 ID 反序
时再同时交换匹配两列。primary/secondary 的普通 matches 和几何 inlier_matches
四个分支统一使用该逻辑，覆盖零偏移及不相等偏移，不修改源匹配数组。

本地与远程真实 PyCOLMAP 数据库回归测试均 4 项通过，覆盖正序/反序 ID 和
`primary_features_first=True/False`；旧实现的两种反序用例均能复现失败。

远程从原始 SIFT/LoMa DB 的独立 SQLite backup 副本重建：
`/kiri/tmp/ffba_789_merge_pair_order_fix/database_merged.db`。
所有关键点坐标块与源 DB 一致；全部匹配逐行与独立按名称、偏移和列顺序映射的结果一致。

| 表 | 总 pairs | 总对应数 | 变化 pairs | 变化行数 | 越界修复前 → 后 |
| --- | ---: | ---: | ---: | ---: | ---: |
| matches | 12,175 | 4,738,790 | 14 | 8,799 | 191 → 0 |
| two_view_geometries | 10,481 | 4,546,506 | 14 | 8,390 | 183 → 0 |

此前记录 8,404 个受影响 SIFT 内点，其中 14 行两列索引相等，交换前后数值不变，
所以实际变化行为 8,390。源数据库和旧三组模型保持作为历史对照；此次仅重建及审计
合并数据库，未重新运行 BAE，不推断修复后的重建质量变化。

[完整合并审计](../../logs/ffba_789_merge_pair_order_fix/audit.json)。

## 产物

- 远程全部模型：`/kiri/tmp/ffba_789_track_selection_ablation/{baseline,no_two_view,all_sources}/refined_gluemap_aba`
- [本地质量指标](../../logs/ffba_789_track_selection_ablation/quality_comparison.json)
- [诊断图](../../logs/ffba_789_track_selection_ablation/quality_diagnostics.png)
- [匹配列顺序审计](../../logs/ffba_789_track_selection_ablation/pair_order_audit.json)
- [数据库打开行为审计](../../logs/ffba_789_track_selection_ablation/database_open_audit.json)
- 各组 `refine_stats.json`、`replay_manifest.json`、运行日志同目录归档。
