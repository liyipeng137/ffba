# SIFT-first VGGSfM Center / Group V1 设计

状态：待确认后实施
更新时间：2026-09-04
目标管线：MERG3R Stage A + SIFT + ALIKED/VGGSfM prior + GlueMap/BAE
主要实施位置：`run_merg3r_gluemap_pipeline.py`、`utils/gluemap_spv_refine.py`、
`utils/gluemap_refine_core.py`

## 1. 文档目的

本文定义第一版 SIFT-first VGGSfM 调度改造，用于后续实施、A/B 测试和任务交接。
第一版只实施以下四项：

1. 在 VGGSfM tracking 前构建并验证完整 SIFT database；
2. 保持旧 pose candidates，同时为 SIFT pair graph 增加局部时序召回；
3. 根据真实 SIFT 几何支持减少 VGGSfM center；
4. 使用“owned frames / adjacent-center bridges / projected-overlap fill”三层结构
   构造 selected center 的 VGGSfM group。

第一版不实施 query 空间基础配额。该方向保留为后续独立实验，避免在第一轮同时改变
center 数、group 邻居和 query 分布。

## 2. 目标与验收立场

更新后的目标是：

> 缩短完整管线运行时间，同时保持或提升最终重建质量。

其中：

- 端到端 wall time 下降是优化目标；
- 质量持平或提升是硬门槛；
- center 越少、track 越多或最终 points3D 越多，都不能单独作为成功标准。

主要加速来源应是：只让 selected centers 发出 VGGSfM query，减少
`attempted_query_views` 和 VGGSfM group forward 数量。

SIFT extraction/matching 本来就需要执行。SIFT-first 主要改变执行顺序；新增的局部
时序 pairs 会增加少量 matching 工作，但完整管线的净时间仍必须低于基线。

## 3. 当前基线与问题

### 3.1 当前 SIFT pair 逻辑

Stage A 当前使用 coarse pose 构建 pairs：

```text
对每个 frame i：
  1. 排除视轴夹角 >= pair_pose_rotation_threshold 的 frame；
  2. 对剩余 frame 按 camera-center distance 排序；
  3. 主动选择前 pair_k_pose 个；
  4. 转为无向 pair 并去重。
```

当前默认参数是：

```text
pair_k_pose = 25
pair_pose_rotation_threshold = 30 degrees
```

这套逻辑不是只看角度：角度决定候选资格，camera-center distance 决定排序。
它简单、覆盖较宽且已有质量基线，因此第一版保持不变，并明确命名为
`legacy_pose_pairs`。

它的不足是：

- Stage A coarse pose 误差可能通过 30 度硬门槛漏掉真实局部共视；
- camera-center 最近的 pairs 可能偏向小基线；
- 没有显式保证时间相邻帧进入 SIFT matching；
- 未进入候选图的 pair 不可能再被真实 SIFT 证据恢复。

第一版不重新设计 pose candidate ranking，只通过局部 temporal pairs 补充召回。

### 3.2 当前执行顺序

当前主流程大致为：

```text
Stage A
-> 对所有 frames 构建 VGGSfM groups
-> 对所有 frames 作为 center 运行 VGGSfM
-> 构建 SIFT DB
-> frame filtering / merge / triangulation / BA
```

因此当前无法使用实际 SIFT 内点和空间覆盖决定：

- 哪些 frames 可以不再作为 center；
- skipped frame 应归属于哪个 owner；
- 哪些 selected centers 需要显式 bridge。

### 3.3 当前 projected-overlap group 逻辑

当前 projected-overlap 对每一个 frame 构建 group：

```text
candidates
  = rotation-valid legacy pose candidates
  UNION DINO top-k candidates

ranking
  = directed depth projected-overlap
  -> projected grid coverage
  -> projected visible ratio
  -> DINO similarity
  -> camera-center distance

selection
  = top neighbors_per_center
```

这套逻辑能够为 SIFT 弱纹理区域召回仍然具有图像共视的 VGGSfM 邻居。第一版保留
其候选来源、depth projected-overlap 计算和普通邻居排序，不改造成纯 SIFT 排序。

需要修改的是其调度入口和 group 成员优先级：

- 只为 selected centers 计算 group；
- owned frames 必须进入 group；
- valid adjacent selected centers 优先于普通 fill；
- 剩余 slots 才使用原 projected-overlap 排序填充。

## 4. 设计原则

### 4.1 分离不同 graph 的语义

第一版必须显式区分：

```text
legacy_pose_pairs
  Stage A 根据 pose 产生的旧候选图

sift_candidate_pairs
  legacy_pose_pairs + local temporal pairs

verified_sift_edges
  SIFT matching 后通过几何和 coverage 条件的调度边

projected_overlap_candidates
  旧 rotation-valid pose candidates + DINO top-k

vggsfm_groups
  selected centers 对应的最终有向 group
```

禁止继续让同一个 `pairs` 变量同时隐式承担以上全部含义。

### 4.2 SIFT 决定 center 安全性

SIFT 的职责是：

- 判断最近 selected center 是否足以代表当前 frame；
- 建立 owner 关系；
- 判断相邻 selected centers 是否具有真实几何 bridge。

只有在 SIFT 支持充分且没有达到最大时序间隔时，当前 frame 才能被跳过为 center。

### 4.3 Projected-overlap 负责普通 VGGSfM 邻居

VGGSfM 不应只追随 SIFT 强匹配区域。普通邻居继续使用已有
pose + DINO + depth projected-overlap 逻辑，以保留其对弱纹理区域的补充价值。

第一版不把 SIFT inlier、DINO similarity、pose、depth 和时间再融合成新的加权分数。

### 4.4 Center 抽稀必须保守并可退化

`max_center_gap` 提供硬上界。如果 SIFT 失败，算法自动退化为更密集的 centers，
最坏情况下恢复每帧都是 center，而不是删除弱帧。

### 4.5 第一版只改变调度，不改变 tracker 和后端

第一版不改变：

- VGGSfM tracker 权重和内部推理；
- ALIKED query 数量与选择方式；
- prior track 建立、snap 和 merge 规则；
- triangulation、track filtering 和 BA 参数；
- projected-overlap 的 depth scoring 公式；
- DINO retrieval 的现有生成方式。

## 5. V1 总体管线

```text
ordered images
  |
  +-- Stage A
  |     coarse pose / intrinsics / depth
  |     legacy_pose_pairs
  |     DINO retrieval matrix
  |
  +-- SIFT extraction on every frame
  |
  +-- sift_candidate_pairs
  |     legacy_pose_pairs
  |     UNION temporal pairs with |i-j| <= local_sift_window
  |
  +-- SIFT matching + geometric verification
  |     inlier matches
  |     source/target grid coverage
  |     verified_sift_edges
  |
  +-- coverage-gated center selection
  |     selected_centers
  |     owner mapping
  |
  +-- three-layer VGGSfM groups
  |     L1 mandatory owned frames
  |     L2 valid adjacent selected-center bridges
  |     L3 existing projected-overlap ranked fill
  |
  +-- VGGSfM tracking for selected centers only
  |
  +-- reuse the same SIFT DB
        frame filtering / merge / triangulation / BA
```

## 6. SIFT-first Database 与 Pair 补充

### 6.1 Pair 集合

定义：

```python
legacy_pose_pairs = build_pose_pairs(
    extrinsic,
    pair_k_pose,
    pair_pose_rotation_threshold,
)

temporal_pairs = {
    canonical_pair(i, j)
    for i in range(num_images)
    for j in range(i + 1, min(num_images, i + local_sift_window + 1))
}

sift_candidate_pairs = legacy_pose_pairs | temporal_pairs
```

第一版建议：

```text
local_sift_window = 2
```

时序 pairs 不受 30 度 pose rotation threshold 限制。它们进入 SIFT matching 后仍需
经过 COLMAP 几何验证，不会被直接视为有效边。

必须分别审计：

- legacy pose pair 数；
- temporal 新增 pair 数；
- 与 legacy 重合的 temporal pair 数；
- 最终 SIFT candidate pair 数；
- 每一来源的 verified pair 成功率。

### 6.2 SIFT DB 只构建一次

新的执行顺序必须保证：

1. 对所有原始 frames 提取 SIFT；
2. 对 `sift_candidate_pairs` 匹配；
3. 在 center selection 前读取 pair 几何统计；
4. VGGSfM 完成后继续复用同一个 SIFT DB；
5. frame filtering 后沿用现有 database remap/filter 逻辑；
6. 不重新运行第二次 SIFT extraction 或 matching。

### 6.3 必须使用 verified inliers

调度统计应读取 COLMAP `two_view_geometry.inlier_matches`，不能仅使用 raw
`matches` 表的数量。

对每条 verified pair 保存：

```text
inlier_count
source_grid_coverage
target_grid_coverage
```

coverage 使用 8x8 网格。一个 cell 至少有两个 verified inliers 才算 occupied：

```text
grid_coverage = occupied_cells / 64
```

source 和 target 必须分别计算。

### 6.4 有效调度边

第一版初始工程值：

```text
sift_schedule_grid_size = 8
sift_schedule_min_inliers_per_cell = 2
sift_schedule_min_pair_inliers = 128
sift_schedule_min_grid_coverage = 0.20
```

定义：

```python
pair_is_valid(i, j) = (
    inlier_count(i, j) >= min_pair_inliers
    and source_grid_coverage(i, j) >= min_grid_coverage
    and target_grid_coverage(i, j) >= min_grid_coverage
)
```

这些值必须配置化并输出完整分布。第一版不实现按场景动态阈值，但在完整实验前可根据
P0 audit 调整一次默认值。

## 7. Coverage-gated Center Selection

### 7.1 基本算法

```python
centers = [0]
owner = {0: 0}
last_center = 0

for image_idx in range(1, num_images):
    gap = image_idx - last_center
    supported = pair_is_valid(last_center, image_idx)

    if gap >= max_center_gap or not supported:
        centers.append(image_idx)
        last_center = image_idx
        owner[image_idx] = image_idx
    else:
        owner[image_idx] = last_center
```

第一版建议：

```text
vggsfm_max_center_gap = 2
```

所有局部 SIFT 边有效时，181 帧理论上约产生 91 个 centers。困难区间会自动插入
额外 centers，因此实际数量在约 91 到 181 之间。

### 7.2 Owner 语义

非 center frame 只有在 `owner -> frame` 是有效 SIFT edge 时才允许存在。

owner 的含义是：

- 该 frame 不再作为 VGGSfM query center；
- 它必须作为 owner center group 的 target view；
- owner center 发出的 query 应有机会在该 frame 形成 observations。

owner 并不表示该 frame 从最终 reconstruction 删除。

### 7.3 安全检查

center selection 后执行确定性检查：

1. 每个 frame 都有 owner；
2. center 的 owner 是自身；
3. 非 center 的 `owner -> frame` 是 valid SIFT edge；
4. 相邻 selected centers 的时序间隔不超过 `max_center_gap`；
5. 任何不满足条件的 frame 立即提升为 center；
6. 提升后重新计算受影响的局部 owner。

如果一段 SIFT 全部失败，该段退化为每帧一个 center。

### 7.4 输入顺序假设

本算法假设输入图片索引具有真实时间顺序。对无序图片集合必须关闭 sparse scheduling
并使用 legacy/full-center 模式。

## 8. 三层 VGGSfM Group

### 8.1 总体结构

对每个 selected center 构造：

```text
group =
  center
  + Layer 1: mandatory owned frames
  + Layer 2: valid adjacent selected-center bridges
  + Layer 3: projected-overlap ranked fill
```

`neighbors_per_center` 表示 center 之外的目标邻居容量。V1 已确认固定使用
`neighbors_per_center=16`，不与 center 抽稀同时扫描 K。

### 8.2 Layer 1：Mandatory Owned Frames

```python
owned_frames(center) = [
    frame
    for frame in all_frames
    if frame != center and owner[frame] == center
]
```

规则：

- owned frame 必须进入 group；
- 不参与 projected-overlap top-k 竞争；
- 不能因普通 K 截断而被删除；
- 按 frame index 排序，保证确定性。

这是 center 抽稀正确性的核心约束。

### 8.3 Layer 2：Adjacent Selected-center Bridges

对 selected center 序列中的前一个和后一个 center：

```python
if pair_is_valid(center, adjacent_center):
    add_bridge(adjacent_center)
```

规则：

- 只加入具有 valid SIFT edge 的相邻 selected center；
- 前后各最多一个；
- 去除与 owned frames 的重复；
- 不为 SIFT 无效边强造 bridge；
- 无效相邻 center 仍可能在 Layer 3 被 projected-overlap 选中。

bridge 用于让相邻 groups 共享视图和 track 支撑。

### 8.4 Layer 3：Projected-overlap Fill

剩余容量继续使用当前已有逻辑：

```text
candidate pool
  = rotation-valid legacy pose candidates
  UNION DINO top-k

rank by
  projected_overlap descending
  projected_grid_coverage descending
  projected_visible_ratio descending
  DINO similarity descending
  camera-center distance ascending
  image index ascending
```

第一版不把所有 verified SIFT neighbors 加入普通 fill pool，避免改变已验证的
projected-overlap 候选语义。SIFT 只通过 owned 和 adjacent bridge 影响强制/优先成员。

从 ranked candidates 中：

1. 删除 center；
2. 删除已在 Layer 1/2 中的成员；
3. 按原顺序填充剩余 slots。

### 8.5 容量与异常

伪代码：

```python
members = unique(owned_frames + adjacent_bridges)
remaining = neighbors_per_center - len(members)

if remaining > 0:
    members += projected_overlap_fill[:remaining]

group = [center, *members]
```

如果强制成员数量超过 K：

- 保留所有 owned frames；
- 保留有效 adjacent bridges；
- 允许 group 临时超出 K；
- 输出 warning 和 audit；
- 第一版不引入二次拆组。

在 `max_center_gap=2` 下，通常只有一个 owned frame，预期不会频繁超限。

### 8.6 只处理 Selected Centers

当前 projected-overlap builder 遍历所有 frames。新接口必须显式接收
`selected_centers`，只准备这些 source centers 的 depth samples、候选打分和 groups：

```python
for center in selected_centers:
    build_three_layer_group(center)
```

skipped frames 可以作为 target neighbor，但不能再产生自己的 VGGSfM query group。

## 9. 运行模式与回退

为了获得可信 A/B 和快速故障回退，建议提供三个显式模式：

### 9.1 legacy

```text
旧执行语义
full centers
旧 projected-overlap groups
旧 query selection
```

该模式用于复现当前认可基线。

### 9.2 sift_first_full

```text
先构建 SIFT DB
使用 sift_candidate_pairs
所有 frames 仍是 center
旧 projected-overlap groups
旧 query selection
```

该模式用于验证 SIFT-first 重排、DB 复用和 temporal pair 增量，不启用 center 抽稀。

### 9.3 sift_first_sparse

```text
先构建 SIFT DB
使用 sift_candidate_pairs
coverage-gated selected centers
三层 groups
旧 query selection
```

这是本设计的目标模式。

不能只把 `max_center_gap=1` 当作严格 legacy 回退，因为三层 group 中的 adjacent
bridges 仍可能改变旧 group 成员。严格回退必须使用显式 `legacy` 模式。

## 10. 建议配置

| 参数 | V1 建议值 | 说明 |
| --- | ---: | --- |
| `sift_temporal_window` | 2 | 强制加入的局部时序 pair 范围 |
| `sift_schedule_grid_size` | 8 | SIFT coverage 网格边数 |
| `sift_schedule_min_inliers_per_cell` | 2 | occupied cell 最少 verified inliers |
| `sift_schedule_min_pair_inliers` | 128 | valid schedule edge 最少 inliers |
| `sift_schedule_min_grid_coverage` | 0.20 | pair 两端最小 coverage |
| `vggsfm_max_center_gap` | 2 | selected centers 最大时序间隔 |
| `neighbors_per_center` | 16 | 已确认的质量基线 K；V1 不扫描 |
| `vggsfm_query_points` | 保持基线 | V1 不修改 query 数量 |
| `projected_overlap_dino_candidates` | 保持基线 | V1 不修改 |
| `projected_overlap_samples` | 保持基线 | V1 不修改 |

## 11. 审计输出

建议输出：

```text
output_dir/vggsfm_schedule.json
```

建议 schema：

```text
mode
config
  temporal window
  SIFT thresholds
  max center gap
  group K
  projected-overlap settings

pair_graphs
  legacy_pose_pair_count
  temporal_pair_count
  temporal_new_pair_count
  sift_candidate_pair_count
  verified_pair_count
  valid_schedule_pair_count
  pair-source verification rates

sift_pair_stats
  inlier distribution
  source coverage distribution
  target coverage distribution
  per-pair records

center_selection
  selected_centers
  selected_count / skipped_count
  owner per frame
  per-frame reason:
    first_frame
    max_gap
    insufficient_sift_support
    safety_promotion
    skipped_supported

groups
  center
  owned_frames
  adjacent_bridges
  projected_overlap_fill
  final_members
  group_size
  overflow

timing
  SIFT extraction
  SIFT matching
  SIFT schedule analysis
  projected-overlap group build
  VGGSfM fmap preparation
  VGGSfM tracking
  refinement
  total wall time
```

日志必须能够回答：

- temporal pairs 实际补回了多少 verified edges；
- 为什么某一帧被跳过或保留为 center；
- skipped frame 是否出现在 owner group；
- 每个 group member 来自哪一层；
- center 减少后 projected-overlap 和 VGGSfM 分别节省多少时间；
- SIFT matching 增量是否抵消了 tracking 节省。

## 12. 实施规划

### P0：纯统计与数据结构

目标：不改变当前输出。

- 引入不同 pair/graph 的明确命名或数据结构；
- 从 SIFT DB 读取 two-view verified inliers；
- 计算双向 8x8 grid coverage；
- 输出阈值分布；
- 实现 center selector 和 three-layer assembler 的纯 Python/NumPy 单测。

验收：

- 不改变 reconstruction；
- pair 统计与 COLMAP database 一致；
- audit 能复现 `pair_is_valid`。

### P1：SIFT-first DB 与复用

目标：建立 `sift_first_full` 模式。

- 将 SIFT extraction/matching 移到 VGGSfM 前；
- 继续复用同一个 DB；
- 保持 full centers 和旧 projected-overlap groups；
- 验证 frame filtering 后 DB remap；
- 保留 `legacy` 模式。

验收：

- `legacy` 能复现旧基线；
- `sift_first_full` 在关闭 temporal 增量时与 legacy 质量一致；
- SIFT 只构建一次。

### P2：Temporal Pair 召回

目标：构建 `legacy_pose_pairs UNION temporal_pairs`。

- 保持旧 pose candidate builder 不变；
- 增加 `|i-j| <= 2` 的 temporal pairs；
- temporal pairs 绕过 pose rotation gate；
- 统计新增 pairs 的 matching/verification 成功率；
- 仍保持 full centers 和旧 groups。

验收：

- 无 zero-degree frame；
- temporal 增量没有导致错误 database id；
- matching 增量耗时被记录；
- 最终 SIFT observations/coverage 不低于 legacy pair graph。

### P3：Center 抽稀与三层 Group

目标：建立 `sift_first_sparse` 模式。

- 实现 coverage-gated center selector；
- 生成 owner mapping；
- projected-overlap builder 只处理 selected centers；
- 强制 Layer 1 owned frames；
- 优先 Layer 2 valid adjacent-center bridges；
- Layer 3 复用现有 projected-overlap fill；
- 记录 group provenance 和 workload。

验收：

- 每个 skipped frame 都在 owner group；
- 181 帧实验相对基线新增 dropped frames 不超过 2；
- 所有 dropped frame 必须记录 id 和原因，且不能造成主干断裂；
- group 无重复成员；
- selected centers 和 group audit 可确定性复现；
- `attempted_query_views` 和 VGGSfM tracking time 明显下降。

### P4：完整 A/B 与默认值决定

固定同一输入、模型、BA 参数、K、query 数和 projected-overlap 参数：

```text
A. legacy
   old order + old pose pairs + full centers + old groups

B. sift_first_full / temporal_window=0
   isolate SIFT-first reorder and DB reuse

C. sift_first_full / temporal_window=2
   isolate temporal SIFT pair addition

D. sift_first_sparse / temporal_window=2 / max_center_gap=2
   target V1
```

只有 B、C 通过回归后，才解释 D 的 center/group 效果。

完成 181 帧和 300 帧两组指定数据的测试后，再决定是否把
`sift_first_sparse` 设为默认。

### P4.1 固定验收数据

| 数据集 | 已核对图片数 | 用途 |
| --- | ---: | --- |
| `/kiri/codex_use_data/181_room` | 181 | P0 阈值分布、完整回归和点云检查 |
| `/kiri/codex_use_data/300_room` | 300 | 更长序列的时间收益、调度稳定性和质量回归 |

两组数据的文件名均可按字典序恢复时间顺序。P0 应同时输出两组数据的 SIFT
inlier/coverage 分布；正式 A/B 使用同一套冻结阈值，不针对单个场景分别调参。

### P4.2 上线基线参数

以下是当前最常用、测试最充分的 legacy 参数组合，作为 A/B 的正式质量和时间基线：

```bash
source /opt/conda/bin/activate
conda activate gluemap-merg3r

python run_merg3r_gluemap_pipeline.py \
  --dataset <DATASET> \
  --output_dir <OUTPUT_DIR> \
  --prior_match_topology star \
  --ba_backend bae \
  --bae_max_num_iterations 20 \
  --num_refinement_iterations 3 \
  --bae_optimize_intrinsics \
  --bae_robust_loss huber \
  --bae_huber_delta 1.0 \
  --final_bae_huber_delta 2.0 \
  --filter_reproj_error_threshold 1.0 \
  --neighbors_per_center 16 \
  --vggsfm_group_strategy projected_overlap \
  --vggsfm_group_batch_size 3
```

除新增 schedule mode 和 schedule 参数外，B/C/D 必须保持以上参数不变。

## 13. 测试要求

### 13.1 纯逻辑单测

- temporal window=2 正确生成 i+1/i+2 pairs；
- temporal 与 pose pair canonicalize 后正确去重；
- source/target grid coverage 分别计算；
- raw matches 不会被误当作 verified inliers；
- 所有局部边有效时 gap=2 选择 0,2,4,...；
- 某个 owner edge 无效时当前 frame 被提升；
- gap=1 选择全部 frames；
- 任意 frame 都有合法 owner；
- owned frame 不会被 K 截断；
- adjacent bridge 只在 valid SIFT edge 时加入；
- Layer 3 不重复添加 Layer 1/2 成员；
- group 成员顺序稳定；
- overflow 会保留强制成员并记录。

### 13.2 小序列集成测试

使用 8 到 16 帧检查：

- SIFT DB 只建立一次；
- three modes 均能运行；
- SIFT pair audit 与 database 一致；
- schedule audit 与实际 tracker groups 一致；
- skipped frame 仍存在于最终 image 集合；
- database image id remap 正确；
- triangulation 和 BA 完成。

### 13.3 完整场景测试

至少记录：

- legacy/temporal SIFT pair 数和 verified 成功率；
- selected center 数、比例和时序分布；
- group size histogram 和 batching buckets；
- attempted query views；
- VGGSfM tracking 与完整 wall time；
- prior tracks、observations 和 track length；
- 每帧 S/P observations 的 min/p10/median；
- dropped/registered frames；
- final S-only/P-only/mixed points；
- final angular error median/p90；
- BA 收敛与 pose drift；
- 固定视角点云的空间连续性和密度。

## 14. 质量与性能验收

### 14.1 质量硬门槛

分别相对 181_room 和 300_room 使用上线参数得到的 legacy baseline：

1. 每个场景相对基线新增 dropped frames 不超过 2；
2. registered image 数相应最多减少 2，且不得造成时序主干断裂或独立 component；
3. angular error median 和 p90 均不得超过各自 legacy baseline 的 1.03 倍；
4. 每帧 P observations 的尾部不出现结构性下降；
5. 最终稀疏点云保持相当或更好的空间连续性；
6. 弱纹理、快速运动区间不能因 center 抽稀出现明显空洞；
7. 最终 points3D 数量只作为辅助指标。

### 14.2 性能目标

1. D 的完整端到端 wall time 必须低于 A；
2. VGGSfM tracking time 和 attempted query views 必须明显下降；
3. projected-overlap group build 应因 selected centers 减少而下降；
4. temporal SIFT matching 增量必须被总节省覆盖；
5. tracking 时间下降 25% 可作为第一轮参考目标，不作为牺牲质量的理由。

在没有 GT 的场景中，是否“质量持平”必须同时结合注册帧、角度误差、每帧 coverage、
track length 和固定视角点云判断。

## 15. V1 明确非目标

第一版不处理：

- query 空间基础配额或 ALIKED 4096 候选池；
- RoMaV2 或其他新 tracker；
- 重新设计 DINO retrieval；
- 修改 projected-overlap depth scoring 公式；
- 用 SIFT support 取代 projected-overlap 普通邻居排序；
- 动态 SIFT 阈值；
- motion utility、多跨度 bucket 或复杂加权；
- GeoCalib / view-graph calibration；
- covariance 加权 BA；
- sequential/loop-closure 来源隔离；
- 3D voxel thinning；
- 最小 set cover、ILP 或全局最优 center selection；
- 修改 triangulation、track selection 或 BA 参数。

这些方向只有在 V1 audit 明确指出问题后再单独立项。

## 16. 风险与失败回退

### 16.1 SIFT 阈值过严

表现：多数 frames 因 SIFT support 不足被提升，几乎没有时间收益。

处理：

- 查看 inlier/coverage 分布；
- 只调整 schedule threshold，不改变 SIFT database；
- 不通过降低质量门槛强行追求更少 center。

### 16.2 SIFT 阈值过松

表现：center 显著减少，但弱区域出现 P observation 或点云覆盖下降。

处理：

- 回退阈值或 `max_center_gap`；
- 检查 owned frame 的双向 grid coverage；
- 不立即加入新的复杂评分。

### 16.3 Projected-overlap 普通 fill 无法补弱 center

表现：SIFT weak 导致的新 center 仍获得低质量 VGGSfM tracks。

处理：

- 先审计其 DINO/pose candidates 和 depth overlap；
- 再决定是否把 verified SIFT neighbors 加入普通候选池；
- 该改变不进入 V1 初始实现。

### 16.4 SIFT-first 重排引入数据库问题

表现：full-center 模式也出现 frame id、camera id、matches 或结果变化。

处理：

- 立即停止 center 实验；
- 使用 `legacy` 模式回退；
- 先修复 DB 复用和 remap，直到 B 通过。

### 16.5 严格回退

任何时候均可使用：

```text
vggsfm_schedule_mode = legacy
```

恢复旧执行顺序、full centers、旧 projected-overlap groups 和旧 query selection。

## 17. 实施前决定与冻结项

### 17.1 基线 Group K（已确认）

V1 固定使用：

```text
neighbors_per_center = 16
```

这是当前测试较充分的参数。第一轮不同时扫描 K，所有 legacy/full/sparse A/B 均使用
相同的 K=16。

### 17.2 Schedule 阈值冻结方式（按 P0 执行）

建议先使用：

```text
min_pair_inliers = 128
min_grid_coverage = 0.20
min_inliers_per_cell = 2
```

但在 P0 输出两组验收数据的分布后，允许调整一次，再冻结用于正式 A/B。

P0 先在 181_room 和 300_room 上只统计、不启用 center 抽稀，并离线模拟多组阈值。
根据两组数据共同分布调整一次后冻结；正式 A/B 不再按场景分别调参。

### 17.3 质量数值容差（已确认）

建议硬条件：

- 每个验收场景新增 dropped frames 不超过 2；
- dropped frames 必须逐帧审计，不能造成时序主干断裂；
- registered images 相应最多减少 2；
- angular median 和 p90 分别最多相对对应 legacy baseline 恶化 3%；
- 目标仍是质量持平或提升，3% 只是自动回归的最大容差；
- 固定视角点云不能出现明显退化。

### 17.4 默认模式切换时机（已确认）

实现完成后仍默认 `legacy`。只有 181_room 和 300_room 均通过质量与性能验收后，
再单独决定是否把默认切换到 `sift_first_sparse`。

## 18. 交接清单

接手实施时按以下顺序检查：

1. 确认目标分支仍包含完整 SIFT + VGGSfM + projected-overlap 路径；
2. 记录认可基线的完整命令、K、query 和 BA 参数；
3. 先实现 P0/P1，不在同一阶段启用 center 抽稀；
4. 所有调度逻辑使用纯 Python/NumPy，避免依赖 CUDA；
5. 保持 projected-overlap 普通排序实现不变，只扩展 selected-center 和三层装配接口；
6. 每阶段保留对应运行模式和 audit；
7. 小序列通过后再运行完整 181 帧；
8. 未通过质量硬门槛时不继续 query、K 或 DINO 优化。

与 `.ai/plan/vggsfm_prior_optimization_plan.md` 的关系：旧文档覆盖更广泛的 VGGSfM
性能优化和实验方向；本文档是当前第一版实现的权威范围。若两者冲突，以本文档为准。

## 19. V1 实施与首轮验收结果（2026-09-04）

V1 已实现三种运行模式；默认仍为 `legacy`。两组验收均使用第 12.1 节的上线参数，
只切换 `--vggsfm_schedule_mode`。以下时间是单次端到端实测，不把 Stage A 的运行波动
解释为调度收益。

| 数据 | 模式 | centers | attempted query-views | dropped | final points | refine | end-to-end |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 181_room | legacy | 181 | 2,965,504 | 5 | 103,797 | 251.6s | 334.0s |
| 181_room | sparse | 147 | 2,408,448 | 6 | 90,570 | 219.9s | 280.7s |
| 300_room | legacy | 300 | 4,915,200 | 0 | 282,543 | 626.6s | 696.1s |
| 300_room | sparse | 162 | 2,654,208 | 0 | 227,235 | 474.5s | 565.3s |

角度误差结果：

- `181_room`：P-only median/p90 相对恶化约 0.27%/0.92%；S-only
  median/p90 相对恶化约 3.28%/1.48%。掉帧多 1 帧；
- `300_room`：P-only median/p90 改善约 2.70%/3.11%；S-only
  median/p90 相对恶化约 0.72%/0.31%。没有新增掉帧。

结论：center 抽稀、三层 group 和 SIFT-first DB 复用均已按设计工作，角度误差和掉帧
基本守住门槛，refinement 时间分别下降约 12.6% 和 24.3%。但 final points 分别下降
约 12.7% 和 19.6%，尚不能声称“质量持平”。因此：

- 保持 `legacy` 为默认和严格回退路径；
- 将 `sift_first_sparse` 保持为实验模式；
- 下一轮应优先调整 center 保留策略或补偿 skipped center 的 prior 覆盖，而不是继续
  放松 SIFT 强边阈值；
- 两次完整运行的 audit 位于
  `/kiri/tmp/ffba_v1_eval_{181,300}_{legacy,sparse}`（181 legacy 实际目录后缀为
  `legacy_retry`）。
