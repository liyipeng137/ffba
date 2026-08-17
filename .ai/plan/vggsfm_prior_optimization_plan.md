# MERG3R / GlueMap：VGGSfM Prior 性能优化与实验计划

更新时间：2026-08-17  
适用入口：`MERG3R/run_merg3r_gluemap_pipeline.py`  
主要实现：`MERG3R/utils/gluemap_spv_refine.py`、`MERG3R/utils/gluemap_refine_core.py`

## 1. 目标与边界

目标是在不显著降低最终 SfM 质量、弱帧覆盖和管道稳定性的前提下，降低 Stage B 中 VGGSfM prior tracking 的耗时与显存传输开销。

本计划覆盖五条工作线：

1. feature map 以 FP16/BF16 常驻 tracker GPU；
2. 降低 `neighbors_per_center`，寻找边际收益拐点；
3. 避免同一无向 pair 的双向无条件 tracking；
4. 先利用 SIFT 统计 coverage deficit，只让 P 补充不足的 frame/pair；
5. 评估其他 tracker / matcher 是否能替代 VGGSfM tracker。

本阶段不调整 `pair_k_pose`。实验中固定完整 SIFT pair graph，避免把“SIFT 图变稀”和“VGGSfM group 变小”混为一个变量。

## 2. 当前实现事实

当前顺序大致为：

```text
ALIKED query
-> 对每张图建立一个 VGGSfM center group
-> 每个 center 从全局 pair graph 中取最近 K 个邻居
-> center query 在整个 group 中 tracking
-> 再运行 SIFT database / matching
-> 用 S + P coverage 过滤弱帧
-> P snap 到 SIFT，合并数据库
-> triangulation / selection / reprojection filter / BA
```

这里有三个直接后果：

- `neighbors_per_center=25` 只限制 VGGSfM group，不限制 SIFT 使用的全局 pair graph；
- 一个无向 pair `{i, j}` 通常同时出现在 `i -> j` 和 `j -> i` 两个 center group 中；两次 query 集不同，因此不是严格重复，但存在较高的计算与约束冗余概率；
- VGGSfM 先于 SIFT 执行，因此当前无法根据实际 SIFT coverage 决定哪些 frame/pair 需要 P。

## 3. 总体原则

### 3.1 优化顺序

```text
P0 建立统一日志和基线
   -> P1 验证 fmap GPU 常驻
   -> P2 扫描 neighbors K，确定第一版拐点
   -> P3 去除无条件双向 tracking
   -> P4 SIFT deficit-aware 调度
   -> P5 替代模型评估
```

P2、P3、P4 必须依次做。后一个阶段以上一阶段选出的配置为基线，但保留原始 K=25/bidirectional/full-P 作为长期质量对照。

### 3.2 单变量与可复现

每组实验固定：

- 数据集与图像顺序；
- Stage A coarse pose / intrinsics / pair graph；
- `pair_k_pose` 及其他 pair graph 参数；
- ALIKED、SIFT、VGGSfM 阈值与 query points；
- BA backend、迭代数、reprojection threshold；
- GPU 型号、软件环境和并发设置。

建议复用同一份 Stage A artifact，避免粗位姿的运行波动进入 Stage B 对比。性能测试区分 cold run 和 warm run；结论以至少两次 warm run 的中位数为准。

### 3.3 三类状态不能混用

| 状态 | 含义 |
|---|---|
| implemented | 代码与 CLI 已具备，静态检查通过 |
| runtime-verified | 已在目标 CUDA 环境完整运行并确认日志、显存和输出 |
| quality-accepted | 已通过固定数据集的质量门槛，可成为新默认值 |

## 4. 统一观测指标

### 4.1 性能与工作量

每次运行记录：

- `timing.vggsfm_prior_tracks`；
- `vggsfm.precompute_fmaps.seconds`；
- `vggsfm.group_tracking_time`；
- fmap cache 的 dtype、device、bytes、是否 fallback；
- peak allocated / reserved CUDA memory；
- group 数、平均/最大 views per group；
- directed pair memberships：`sum(len(group) - 1)`；
- reciprocal pair 数及比例；
- attempted query-view 数：`sum(query_count(center) * neighbor_count(center))`；
- 每秒产生的 raw observations、每秒产生的最终有效 P points。

`group_tracking_time` 是主要性能指标；整个 `vggsfm_prior_tracks` 还包含固定的 fmap 与 query 提取成本，应同时报告，避免 K 降低后固定成本掩盖收益。

### 4.2 Raw tracker 产出

- `vggsfm.num_tracks`；
- `vggsfm.num_observations`；
- track length mean / median / p90；
- visibility、score、in-bounds 各级通过率；
- 每个 center 和每个 pair 的有效 observation 数；
- 每个 neighbor rank 的有效 observation 数。

Raw track 数只用于解释计算过程，不能独立作为质量结论。

### 4.3 Coverage 与互补性

- `frame_filtering.s_observations`；
- `frame_filtering.p_observations`；
- S+P 后 dropped frame 数；
- SIFT 最弱 10%/25% 帧的 P coverage；
- SIFT 已充足帧上产生的 P observations 占比；
- P 对原本不足帧的 deficit fill ratio；
- P snap 到 SIFT 的比例，以及 unsnapped P 的保留量。

需要重点看 per-frame 尾部，而不是只看均值。

### 4.4 几何有效性

- triangulation 后的 `p_only` / `mixed` points 和 observations；
- select-track 前后的 P 存活率；
- reprojection filter 对 P observations/tracks 的删除率；
- `p_only`、`mixed` 的 angular error median / mean / p90；
- threshold 内 observation 比例；
- final real points / observations；
- BA normalized final cost 或单 observation cost，不能直接比较残差数量不同的 total cost。

建议派生：

```text
P geometry yield       = final (p_only + mixed) points / group_tracking_time
P triangulation yield  = triangulated (p_only + mixed) points / prior input tracks
P selection survival   = selected P points / triangulated P points
P useful coverage rate = P observations 落在 SIFT deficit frames 的比例
```

### 4.5 最终任务质量

有 GT 时优先：

- Sim(3) 对齐后的 camera-center ATE；
- rotation error；
- RPE；
- 保留/注册图像数。

无 GT 时使用：

- dropped frames；
- final angular/reprojection error；
- final points 和 observations；
- BA 前后 pose 改变量与收敛状态；
- 下游 mesh / Gaussian 的固定评测。

## 5. P0：先补可观测性

### 5.1 必加日志

在 `refine_stats.json` 中增加稳定 schema：

```text
vggsfm.schedule
  mode
  candidate_undirected_pairs
  scheduled_centers
  scheduled_directed_pairs
  reciprocal_pairs
  reciprocal_ratio
  attempted_query_views
  groups / neighbors summary

vggsfm.neighbor_rank_buckets
  rank range
  attempted
  visibility_pass
  score_pass
  in_bounds_pass
  accepted_observations
  pairs_with_observations

vggsfm.frame_coverage
  query_count
  output_observations
  tracks_as_center
  tracks_as_neighbor
```

推荐 rank buckets：`1-4`、`5-8`、`9-12`、`13-16`、`17-25`。

### 5.2 双向冗余日志

对同时执行 `i -> j`、`j -> i` 的 pair，记录：

- 两个方向各自产生的 raw matches；
- snap/merge 后各方向新增的 unique keypoints/matches；
- reverse direction 相对 first direction 的增量 novelty；
- 两个方向各自的几何过滤存活量（若 provenance 能保留到数据库/track）。

最关键的量是：

```text
reverse novelty = reverse 新增且未被 forward 覆盖的有效约束 / reverse 总约束
```

如果完整 provenance 第一版改动过大，可以先做 tracking 后、写 DB 前的近似 novelty；后续再追踪到 triangulation。

### 5.3 Deficit 日志预留

即使尚未启用 deficit scheduler，也先输出：

- 每帧 SIFT matched observations；
- 有效 SIFT neighbor pair 数；
- SIFT keypoint spatial coverage；
- 按候选阈值计算出的 deficit score；
- 若启用调度，该帧被选中/跳过的原因。

### 5.4 P0 验收

- 同一运行能从 stats 还原总 group workload；
- rank bucket 总 accepted observations 与 raw tracker observations 口径一致，差异有说明；
- directed/reciprocal 统计能解释当前双向开销；
- stats 写入不显著增加 tracking 时间和 GPU 内存。

## 6. P1：fmap FP16/BF16 GPU 常驻

### 6.1 当前状态

| 项目 | 状态 |
|---|---|
| GPU 预分配与 BF16/FP16 存储 | implemented |
| 显存预算与 OOM/容量不足 fallback | implemented |
| group 使用缓存时避免重复 H2D | implemented |
| 静态检查 | passed |
| 目标 VGGSfM CUDA 环境完整运行 | pending |
| peak memory / speed 实测 | pending |

### 6.2 验证项

这不是质量消融项，但需要一次数值与稳定性 sanity check：

- `resident_on_tracker_device=true`；
- storage dtype 为 BF16 或 FP16；
- 无逐 group CPU -> GPU fmap copy；
- GPU peak memory 在预算内；
- tracker 输出量、最终结果没有异常跃迁；
- 显存不足时能自动回退 CPU cache 并完成运行。

## 7. P2：扫描 neighbors，寻找边际效应拐点

### 7.0 扫描前置排序修正

已实现 VGGSfM group 的分层排序：先选择满足
`pair_pose_rotation_threshold` 的邻居，同层再按 camera-center 距离排序，
最后使用 image index distance / index 做稳定 tie-break。只有 rotation-valid
邻居不足 `neighbors_per_center` 时，才会消费 unfiltered/fill 邻居。

`refine_stats.json -> vggsfm.group_stats` 会记录排序策略、阈值，以及最终入组的
rotation-valid / unfiltered 邻居数量。目标 CUDA 环境验证仍待完成。

K=25 与 K=12 对照所需的第一批日志已经实现：逐 neighbor rank 的
visibility/score/in-bounds/accepted funnel、实际 attempted query-views、query
成轨率与 track length 分布、最终 real track source 分类，以及最终按
S-only/P-only/mixed 分桶的 angular error（含 p90）。目标 CUDA 环境完整运行
验证仍待完成。

### 7.1 实验矩阵

固定 `pair_k_pose=25`，第一轮扫描：

```text
K = 4, 8, 12, 16, 25
schedule = current bidirectional/full-center
query_points = 1024
```

若拐点落在 8～12，再补 `K=6`、`K=10`；不要一开始做过密扫描。

数据至少覆盖：

- 正常连续序列；
- 重复纹理或低纹理序列；
- 大视角/稀疏共视序列；
- 当前已知 SIFT coverage 最弱的序列。

### 7.2 边际曲线

对每个 K 画/记录：

```text
x = group_tracking_time 或 attempted_query_views
y1 = final p_only + mixed points
y2 = SIFT 弱帧的 P coverage p10
y3 = dropped frames
y4 = P angular error median/p90
y5 = 最终 ATE/RPE（若有）
```

同时直接观察 rank buckets。如果 `17-25` 的 accepted rate、reverse novelty 和几何存活率都显著较低，说明 K=25 的尾部邻居价值有限。

### 7.3 第一版决策门槛

选择满足以下条件的最小 K：

- 相比 K=25，VGGSfM group tracking 节省至少 30%；
- 不新增 dropped frame；
- SIFT 最弱 10% 帧的 P coverage p10 下降不超过 10%；
- final `p_only + mixed` 有效点下降不超过 5%；
- P angular median/p90 不明显恶化；
- 有 GT 时 ATE/RPE 落在运行波动范围内。

这些百分比是第一版工程门槛，应根据基线方差修订。初始候选预计在 K=8～12，但在实验完成前不修改默认值。

## 8. P3：避免双向无条件重复

### 8.1 问题定义

当前无向 pair `{i, j}` 常被执行两次：

```text
i 作为 center：ALIKED(i) -> j
j 作为 center：ALIKED(j) -> i
```

两次不是完全等价，因为 query 来源不同；因此不能直接假设删掉一半没有质量损失。需要用 reverse novelty 和最终几何存活量判断。

### 8.2 调度模式

建议实现统一 `prior_schedule_mode`：

1. `bidirectional`：当前行为，质量基线；
2. `oneway`：每个无向 pair 只分配一个 query center；
3. `conditional_reverse`：先跑主方向，只有覆盖/匹配不足时再跑反方向。

`oneway` 的第一版方向选择应确定且可复现。可先使用：

```text
优先让 SIFT coverage 更差的帧作为 query center；
若相同，优先 ALIKED query 更丰富的帧；
仍相同则按 image index 决定。
```

在 P4 尚未完成时，可先用 frame index 或 ALIKED query 数做纯调度验证；最终策略应与 SIFT deficit 合并。

`conditional_reverse` 的候选触发条件：

- 主方向 accepted observations 小于 pair target；
- query center 运行后仍有 coverage deficit；
- 主方向 visibility/score pass rate 过低；
- 两帧都属于弱覆盖帧。

### 8.3 实验矩阵

使用 P2 选出的 K：

```text
D0 bidirectional
D1 oneway
D2 conditional_reverse
```

重点比较：

- scheduled directed pairs 和 reciprocal ratio；
- attempted query-views 与 tracking time；
- reverse novelty；
- 弱帧 coverage；
- `p_only + mixed` 几何存活量；
- 最终质量。

### 8.4 验收目标

- directed pair workload 接近减少 40%～50%，或 conditional 模式显著低于基线；
- 不新增 dropped frame；
- 有效 P geometry 和最终质量满足 P2 的相同 guardrail；
- 日志能解释哪些 pair 触发了 reverse 以及原因。

## 9. P4：SIFT deficit-aware VGGSfM

### 9.1 必要的流程重排

调整为：

```text
SIFT extraction/matching
-> 统计 per-frame / per-pair S coverage
-> 计算 deficit 和 P tracking schedule
-> ALIKED query + VGGSfM，仅处理选中的 centers/pairs
-> S + P frame filtering
-> 后续数据库合并与 refinement
```

SIFT 本来就必须执行，提前它主要改变调度信息可用性，不额外增加一次 SIFT。

### 9.2 不应只用一个 raw match count

单纯 `s_observations >= 10` 太弱，可能被同一局部区域的大量重复匹配“刷满”。建议 deficit 至少包含：

- `obs_deficit`：matched/inlier observations 是否不足；
- `pair_deficit`：具有足够 SIFT 支持的邻居 pair 是否不足；
- `spatial_deficit`：关键点是否只集中在小范围；
- 可选 `geometry_deficit`：可三角化基线/视角是否不足。

第一版可以先使用简单、易解释的两项：

```text
frame deficit = max(0, target_obs - sift_obs)
             + lambda * max(0, target_pairs - supported_sift_pairs)
```

spatial/geometry 指标先记录，确认确有必要后再进入调度决策。

### 9.3 调度策略分级

建议逐步实现，避免一次上复杂闭环：

- `full`：所有 center、所有 K 邻居，当前基线；
- `deficit_centers`：只让 deficit frame 作为 center，邻居仍从原 pair graph 选；
- `deficit_pairs`：只保留至少一端 deficit 或 SIFT pair 支持不足的 pair；
- `budgeted_deficit`：按 deficit score 排序，在总 query-view 预算内分配；
- `adaptive_deficit`：分批 tracking，达到 target 后提前停止。

邻居选择不应只按 camera-center 距离。对 deficit center，优先选择：

- SIFT/几何支持更可靠的 anchor frame；
- 与 center 有合理共视且具有足够视差的 frame；
- 尚未被已选邻居覆盖的方向/时间段；
- 避免多个几乎等价的近邻。

### 9.4 Deficit 专属指标

```text
deficit frames before P
deficit frames after P
total deficit amount before/after
deficit fill ratio
P observations on deficit frames
P observations wasted on already-sufficient frames
tracking cost per rescued frame
tracking cost per unit deficit filled
```

### 9.5 实验矩阵

以上一阶段最佳方向策略与 K 为基线：

```text
C0 full
C1 deficit_centers
C2 deficit_pairs
C3 budgeted_deficit（预算为 full workload 的 25% / 50% / 75%）
C4 adaptive_deficit（如静态策略仍有明显浪费再做）
```

### 9.6 验收目标

- VGGSfM tracking workload/时间相对 P3 基线进一步下降；
- dropped frames 不增加；
- deficit fill ratio 接近 full-P；
- 大多数 P 约束落在真实不足的 frame/pair 上；
- 最终 geometry/pose 通过同一质量 guardrail。

## 10. P5：替代模型评估

### 10.1 先定义替换接口

不要先按模型名字改 pipeline。先定义统一 prior provider 输出：

```text
input:
  images
  scheduled centers/pairs/groups
  optional query points
  coarse pose/intrinsics

output:
  tracks: [(image_idx, xy), ...]
  confidence / visibility
  per-pair and per-frame provenance
  timing / peak memory / workload stats
```

所有替代方案最终都转换成当前 `prior_tracks` 语义，再复用 snap、database merge、triangulation 和 BA。这样模型评估不会同时改掉下游逻辑。

### 10.2 候选类别

候选应按能力分组，而不是直接混排：

- pairwise sparse matcher：容易接入，适合 deficit pair，需做跨 pair track union；
- pairwise dense matcher：低纹理覆盖可能更好，但输出量和显存需控制；
- temporal point tracker：适合有真实时间连续性的序列，不适合任意 pose-neighbor group；
- multi-view tracker：最接近当前 VGGSfM 输出语义，替换成本最低。

具体模型 shortlist 在 P2～P4 完成后再根据瓶颈选择，并在评估时查阅对应模型的官方代码、权重许可和输入限制。

### 10.3 两级评测

Level 1，小规模 provider benchmark：

- 相同 scheduled pairs/groups；
- 相同分辨率和预算；
- pair coverage、cycle consistency、几何验证通过率；
- runtime、peak memory、模型加载时间。

Level 2，完整 pipeline：

- weak-frame rescue；
- `p_only + mixed` 存活量；
- reprojection/angular error；
- ATE/RPE 或下游质量；
- 总 Stage B 时间。

只有 Level 1 进入 Pareto 前沿的模型才跑完整 Level 2。

### 10.4 替代门槛

替代模型至少满足其一：

- 在相同质量下显著更快；
- 在相同耗时下明显改善弱帧 coverage/几何质量；
- 支持更好的 deficit/conditional 调度，整体计算更少。

模型本身单次 forward 更快但需要更多 pairs、复杂 track union 或更高分辨率时，必须按完整 Stage B 成本比较。

## 11. 实验编号与记录模板

### 11.1 编号

```text
Bxx  baseline / observability
Kxx  neighbors sweep
Dxx  direction scheduling
Cxx  coverage-deficit scheduling
Mxx  alternative model
```

示例：

```text
K08_seq390_bae_r1
D02_condrev_seq390_bae_r1
C03_budget50_seq390_bae_r1
```

### 11.2 每次运行记录

```markdown
#### Experiment ID

- date:
- git commit / dirty diff note:
- dataset / frame count:
- cached Stage A artifact:
- GPU / software environment:
- command:
- changed variable:
- baseline experiment:
- output directory:
- refine_stats.json:

Performance:
- fmap precompute:
- group tracking:
- VGGSfM total:
- Stage B total:
- peak CUDA memory:
- attempted query-views:

Coverage:
- S obs p10 / median:
- P obs p10 / median:
- deficit before / after:
- dropped frames:

Geometry:
- triangulated p_only / mixed:
- selection survival:
- reprojection removal rate:
- P angular median / p90:
- final points / observations:

Final quality:
- ATE / RPE / rotation error:
- downstream metric:

Conclusion:
- pass/fail guardrail:
- observed tradeoff:
- next action:
```

### 11.3 汇总表

| ID | fmap | K | schedule | deficit mode | model | directed pairs | query-views | track time | dropped | P geom | P error | final quality | decision |
|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| B00 | GPU FP16/BF16 | 25 | bidirectional | full | VGGSfM | TBD | TBD | TBD | TBD | TBD | TBD | TBD | pending |

## 12. 实施追踪

| 阶段 | 工作项 | 实现 | CUDA 验证 | 质量结论 | 依赖 |
|---|---|---|---|---|---|
| P0 | 统一 schedule / rank / direction / deficit 日志 | pending | pending | n/a | — |
| P0a | K sweep：rank funnel / workload / final P geometry 日志 | done | pending | n/a | — |
| P1 | fmap FP16/BF16 GPU 常驻 | done | pending | sanity pending | — |
| P2 | K=4/8/12/16/25 sweep | pending | pending | pending | P0, P1 |
| P3 | oneway / conditional reverse | pending | pending | pending | P0, P2 |
| P4 | SIFT-first + deficit scheduler | pending | pending | pending | P0, P3 |
| P5 | prior provider 接口与替代模型 | pending | pending | pending | P2～P4 结论 |

## 13. 推荐的近期执行顺序

1. 先补 P0 日志，不改变现有 tracking 语义；
2. 在目标 CUDA 环境跑 B00，验证 fmap 常驻并建立 K=25 基线；
3. 跑 K=4/8/12/16/25，选第一版 K；
4. 基于 direction novelty 实现 `oneway` 和 `conditional_reverse`；
5. 把 SIFT 移到 prior tracking 前面，先做静态 `deficit_centers`；
6. 只有静态 deficit 仍存在明显浪费时，再做动态 early-stop；
7. 用已经稳定的调度与指标接口评估替代模型。

## 14. 预期最终形态

如果各阶段假设成立，目标流程应收敛为：

```text
Stage A coarse pose + full candidate pair graph
-> SIFT extraction/matching
-> frame/pair coverage diagnosis
-> deficit-aware directed P schedule
-> compact VGGSfM groups with conditional reverse
-> S + P merge
-> triangulation / filtering / BA
```

其中：

- `pair_k_pose` 表示全局候选图密度；
- `neighbors_per_center` 只作为单 center 的最大 P 预算，而不是固定必须用满；
- P 的目标从“对所有图统一生成更多 tracks”转为“用最少计算补齐 SIFT 的结构性缺口”。
