# SIFT-first VGGSfM Center / Group / Query 设计

状态：待实施设计（design only）  
更新时间：2026-09-04  
目标管线：MERG3R Stage A + SIFT + ALIKED/VGGSfM prior + GlueMap/BAE  
主要实施位置：`run_merg3r_gluemap_pipeline.py`、`utils/gluemap_spv_refine.py`、
`utils/gluemap_refine_core.py`

## 1. 文档目的

本文档用于后续实现与任务交接，定义一版尽量简单、可解释的 VGGSfM prior
调度方案。核心改动是：

1. 先完成独立 SIFT 图，再决定 VGGSfM center 和 group；
2. 不再默认每一帧都是 VGGSfM center；
3. VGGSfM query 使用空间基础配额，但不压制强纹理区域继续获得更多 query。

本文档只定义方案，不表示当前分支已经实现。实施时应以仍保留 VGGSfM 路径的
基线分支为起点；当前 RoMaV2 实验分支可以提供空间采样和审计代码参考，但本方案
不接入 RoMaV2。

## 2. 背景与问题

原始 SIFT + VGGSfM 管线在部分室内和中等范围室外数据上已经能够产生均匀、
中等密集且质量较好的稀疏点云。已有 181 帧实验也显示，经过 group batching 和
`projected_overlap` 优化后，VGGSfM 能够产生大量长度较好的多视图 track，性能并不
一定弱于以 pairwise matcher 重新构造 track 的方案。

当前 VGGSfM 调度仍有两个可以简化和优化的地方：

- 默认每帧都作为 center，导致相邻高重叠帧重复发出相似 query；
- query 候选如果在最终选择前已做全局 top-k，画面中候选较少但仍有有效纹理的
  grid cell 可能完全没有 query。

本设计不以“中心帧越少越好”或“query 越均匀越好”为目标。真正目标是：

- 删除已经被相邻 center 充分代表的冗余 center；
- 新内容、快速运动、遮挡和弱匹配位置仍自动保留更多 center；
- 每个存在合格候选的空间 cell 获得最低 query 保障；
- 强纹理区域仍可通过全局质量竞争取得大部分剩余 query；
- 在质量不明显退化的条件下减少 VGGSfM tracking 工作量。

## 3. 设计原则

### 3.1 SIFT 是独立观测源和调度依据

SIFT pair graph 必须在 VGGSfM group 之前建立，并且不能由最终 VGGSfM group
反向决定。这样可以：

- 用实际匹配结果判断帧间是否具有可用共视；
- 在切换 center/group 策略时保持 SIFT 约束不变；
- 避免把“SIFT 图变稀”和“VGGSfM center 减少”混成同一个实验变量。

### 3.2 Center 抽稀是 coverage-gated，而不是固定删帧

一帧只有在能被最近的 selected center 通过 SIFT 充分覆盖时才允许不作为 center。
如果 SIFT 支撑不足，它立即成为新的 center。算法必须允许在困难区间退化回“每帧
都是 center”。

### 3.3 使用硬上界保证保守性

`max_center_gap` 是相邻 selected center 的最大时序间隔。即使 SIFT 支撑一直良好，
达到该间隔也必须选择新 center。第一版使用 `max_center_gap=2`，最多只减少约一半
center，不直接追求最小集合。

### 3.4 Query 配额只提供下限，不提供上限

空间配额解决的是“有合格候选但一个 query 都没拿到”，不是强制所有 cell 数量
相等。每个 cell 完成基础选择后，剩余 query 仍由所有候选按原始质量全局竞争。

### 3.5 第一版保持单一、可解释规则

第一版不加入 DINO、RoMaV2、累计 motion target、动态阈值、ILP/set-cover、
多来源加权或复杂的 sequential/loop-closure 状态机。若简单方案无法满足质量门槛，
先通过日志定位失败原因，再决定是否增加机制。

## 4. 总体管线

```text
ordered images
  │
  ├─ Stage A: coarse pose / intrinsics / depth / pose pairs
  │
  ├─ SIFT extraction on every frame
  │
  ├─ SIFT candidate pairs
  │    ├─ Stage A pose pairs
  │    └─ mandatory local temporal pairs
  │
  ├─ SIFT matching + geometric verification
  │    └─ inliers + bidirectional grid coverage
  │
  ├─ coverage-gated center selection
  │    ├─ SIFT weak -> promote current frame
  │    └─ max gap reached -> promote current frame
  │
  ├─ VGGSfM group construction
  │    ├─ mandatory owned frames
  │    ├─ adjacent selected centers when valid
  │    └─ fill remaining neighbors by SIFT support
  │
  ├─ ALIKED query candidate pool
  │    ├─ per-cell minimum quota
  │    └─ remaining budget by global detector score
  │
  └─ VGGSfM tracking -> merge with fixed SIFT DB -> triangulation / BA
```

## 5. 阶段 A：建立独立 SIFT 图

### 5.1 候选 pair

SIFT candidate pairs 定义为：

```text
Stage A pose pair graph
UNION
all local temporal pairs with |i - j| <= local_sift_window
```

第一版：

```text
local_sift_window = max_center_gap = 2
```

局部 pair 强制加入的原因是 center selector 必须能够检查最近 center 与当前帧之间的
SIFT 支撑，不能因为 Stage A 粗 pose 漏边而错误地把帧提升为 center。

SIFT extraction、matching 和 geometric verification 只执行一次。后续 VGGSfM
调度和最终 SIFT database 复用同一结果。

### 5.2 Pair 统计

对每条通过几何验证的 SIFT pair `{i, j}` 保存：

```text
inlier_count
source_grid_coverage
target_grid_coverage
```

grid 默认使用 `8 x 8`。一个 cell 至少包含 `2` 个 verified inliers 才算 occupied：

```text
grid_coverage = occupied_cells / 64
```

source/target coverage 必须分别计算。一个 pair 在 source 上分布良好，不代表它在
target 上也具有完整覆盖。

### 5.3 有效 SIFT 边

第一版建议：

```text
min_pair_inliers       = 128
min_pair_grid_coverage = 0.20
```

定义：

```python
pair_is_valid(i, j) = (
    inlier_count(i, j) >= min_pair_inliers
    and source_grid_coverage(i, j) >= min_pair_grid_coverage
    and target_grid_coverage(i, j) >= min_pair_grid_coverage
)
```

阈值是初始工程值，不应被视为固定真值。实现后先输出数据分布，再决定是否调整。
第一版不使用按场景动态阈值。

## 6. 阶段 B：Coverage-gated Center Selection

### 6.1 基本规则

从第 0 帧开始按时间顺序扫描：

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

语义：

- 当前帧能被最近 center 通过 SIFT 有效覆盖，并且尚未达到最大间隔，则作为该
  center 的 owned frame，不再重复发出一组中心 query；
- SIFT 支撑不足，说明当前视图没有被最近 center 充分代表，立即提升为 center；
- 即使一直有良好支撑，达到 `max_center_gap` 后也必须建立新 center。

### 6.2 第一版的保守参数

```text
max_center_gap = 2
```

在连续高重叠的 181 帧序列中，理论最少约 91 个 center。困难区域会插入额外
center，因此实际数量位于约 91～181 之间。

如果质量稳定，再单独测试 `max_center_gap=3`。第一版不要直接使用更大间隔。

### 6.3 示例

若所有局部 SIFT edge 都有效：

```text
centers: 0, 2, 4, 6, 8, ...
owner:   0->0, 1->0, 2->2, 3->2, 4->4, 5->4, ...
```

若 `4 -> 5` 的 SIFT 支撑不足：

```text
centers: ..., 4, 5, 7, 9, ...
owner:        4->4, 5->5, 6->5, 7->7, ...
```

因此快速运动或新内容区域会自然恢复更密的 center。

### 6.4 安全检查与退化

center selection 完成后执行一次确定性检查：

1. 每个非中心帧必须有一个 owner；
2. `owner -> frame` 必须是有效 SIFT edge；
3. 任意相邻 selected centers 的时序间隔不得超过 `max_center_gap`；
4. 不满足 1～3 的帧直接提升为 center，然后重新计算局部 owner。

如果某一段 SIFT 全部失败，算法在该段退化为每帧一个 center，而不是删除弱帧。

## 7. 阶段 C：VGGSfM Group Construction

### 7.1 Group 的强制成员

每个 selected center 的 group 按以下优先级构造：

1. center 自身；
2. 该 center 的 owned frames；
3. 时间上前一个和后一个 selected center，但仅在对应 SIFT edge 有效时加入；
4. 从其余有效 SIFT neighbors 中填充到 `neighbors_per_center`。

这里 `neighbors_per_center=16` 表示最多 16 个邻居，不含 center 自身。

owned frames 是调度正确性的硬约束，不能被后续 top-k 排名移除。相邻 selected
centers 的优先加入用于增强 group 间连续性，但不会为 SIFT 无效 pair 强造边。

### 7.2 剩余邻居排序

第一版使用一个简单、封顶的 SIFT support score：

```text
pair_coverage = min(source_grid_coverage, target_grid_coverage)
pair_score = pair_coverage * min(inlier_count, 512) / 512
```

按 `pair_score` 降序填充剩余 group slots。inlier count 在 512 封顶，避免局部纹理
特别强、但空间覆盖一般的 pair 仅凭匹配数量占满 group。

稳定 tie-break：

```text
更小的 image index distance
-> 更小的 image index
```

第一版不额外加入 pose、depth overlap、DINO、时间软权重或多跨度 bucket。

### 7.3 Group 容量异常

在 `max_center_gap=2` 下，一个 center 的 owned frames 通常不超过 1，强制成员数量
远低于 16。如果未来参数导致强制邻居超过 `neighbors_per_center`：

- 强制成员优先，允许该 group 暂时超出 K；
- 输出 warning 和 audit；
- 不静默删除 owned frame。

第一版不为这个低概率情况引入复杂的二次拆组逻辑。

## 8. 阶段 D：带基础配额的 VGGSfM Query Selection

### 8.1 目的

空间配额的目的不是把强纹理区域的 query 强制减弱，也不是让所有 cell 数量相等。
它只保证：某个 cell 如果存在达到正常检测阈值的候选点，就有机会获得少量基础
query。

### 8.2 候选池必须大于最终 query 数

若最终需要 1024 个 query，ALIKED 不能先全局截断到 1024 后再做 grid selection，
否则外围低排名候选已经不可恢复。

第一版建议：

```text
ALIKED candidate pool = 4096
final VGGSfM queries  = 1024
query grid            = 8 x 8
min queries per occupied cell = 4
```

候选点仍必须满足正常 ALIKED detection threshold。禁止为了填满 cell 配额而降低
检测阈值或生成无效点。

### 8.3 两阶段选择

第一阶段——基础配额：

```python
selected = []
for cell in grid_cells:
    candidates = valid_candidates_in(cell)
    selected += top_by_detector_score(candidates, limit=4)
```

最多使用：

```text
64 cells * 4 queries = 256 queries
```

没有合格候选的 cell 不占用配额，未使用预算自动回到第二阶段。

第二阶段——全局竞争：

```python
remaining = all_valid_candidates - selected
selected += global_top_by_detector_score(
    remaining,
    limit=1024 - len(selected),
)
```

最终效果：

- 每个有合格候选的 cell 最多先获得 4 个基础 query；
- 强纹理区域仍可在剩余约 768 个或更多 slots 中获得大量 query；
- 空 cell 不会被伪造特征填充；
- detector score 仍是主要排序依据。

如果需要 NMS，沿用 ALIKED 自身或现有 query NMS；第一版不额外叠加新的密度惩罚。

## 9. 建议默认参数

| 参数 | 第一版默认值 | 作用 |
| --- | ---: | --- |
| `sift_schedule_grid_size` | `8` | SIFT pair coverage 网格边数 |
| `sift_schedule_min_inliers_per_cell` | `2` | cell 被视为 occupied 的最少 inliers |
| `sift_schedule_min_pair_inliers` | `128` | 有效 SIFT edge 最少 inliers |
| `sift_schedule_min_grid_coverage` | `0.20` | pair 两端最小 grid coverage |
| `vggsfm_max_center_gap` | `2` | selected centers 最大时序间隔 |
| `neighbors_per_center` | `16` | VGGSfM 每组最大普通邻居数 |
| `vggsfm_query_candidate_points` | `4096` | ALIKED 空间选择前候选池 |
| `vggsfm_query_points` | `1024` | 最终 VGGSfM query 数 |
| `vggsfm_query_grid_size` | `8` | query 配额网格边数 |
| `vggsfm_min_queries_per_cell` | `4` | occupied cell 基础 query 配额 |

第一轮实验只主动扫描：

```text
vggsfm_max_center_gap = 1, 2, 3
vggsfm_min_queries_per_cell = 0, 4, 8
neighbors_per_center = 12, 16
```

不要一次性做全笛卡尔积。推荐实施顺序：先固定 query 配额为 0，验证 center 调度；
center 方案稳定后，再固定 selected centers 比较 query 配额。

## 10. 审计与输出

实现必须输出足以复现 group 调度的 audit，建议写入：

```text
output_dir/vggsfm_schedule.json
```

建议 schema：

```text
config
  thresholds / max_center_gap / K / query settings

sift_graph
  candidate_pairs
  verified_pairs
  valid_schedule_pairs
  inliers summary
  source/target grid coverage summary

center_selection
  selected_centers
  num_selected / num_skipped
  per-frame owner
  per-frame selection reason:
    first_frame
    max_gap
    insufficient_sift_support
    safety_promotion
    skipped_supported

groups
  center
  mandatory_owned_frames
  adjacent_selected_centers
  ranked_fill_neighbors
  final_members

queries
  center
  candidate_count
  occupied_cells
  floor_selected
  global_fill_selected
  final_count
  per-cell final counts
```

必须能够从 audit 回答：

- 为什么某一帧是或不是 center；
- 每个 skipped frame 被哪个 center 负责；
- 某个 neighbor 是强制成员还是排序补入；
- 基础 query 配额实际使用了多少；
- 强纹理 cell 是否仍获得了超过基础配额的 query。

## 11. 实施拆分

### P0：只建立 SIFT 调度统计

- 保持旧 VGGSfM groups 完全不变；
- 在现有 SIFT DB 上计算 pair inliers 和双向 grid coverage；
- 输出统计分布与 `pair_is_valid` 结果；
- 验证统计与 COLMAP two-view geometry 一致。

验收：不改变最终 reconstruction，新增 audit 可复现。

### P1：SIFT-first 顺序与 DB 复用

- 将 SIFT extraction/matching 移到 VGGSfM group 构造之前；
- VGGSfM 之后继续复用同一 SIFT DB；
- 确保 frame filtering 后的 DB remap 仍正确；
- 固定 Stage A、SIFT pairs 和 BA 参数做回归。

验收：使用旧 full-center groups 时，结果在正常运行波动范围内。

### P2：Center selector 与 group builder

- 实现 coverage-gated 时序扫描；
- 实现 owner、安全提升和强制 group members；
- 写入 `vggsfm_schedule.json`；
- `max_center_gap=1` 应等价于 full-center 基线。

验收：`max_center_gap=1` 回归通过，再测试 2；不要直接修改默认值为 3。

### P3：Query 基础配额

- 将 ALIKED 候选池扩大到 4096；
- 实现 per-cell top-4 + global fill；
- 保留 detector threshold 和原始 score；
- 输出每个 center 的 cell 分布。

验收：`min_queries_per_cell=0` 等价于全局 top-k 基线；设置为 4 后，强纹理 cell
仍可超过 4，且总 query 不超过 1024。

### P4：完整质量/性能 A/B

- 固定同一 Stage A artifact；
- 固定同一 SIFT DB/pair graph；
- 固定 BA backend、迭代数与过滤阈值；
- 分别测试 full-center、gap=2、gap=2+query-floor；
- 至少包含正常、快速运动和弱纹理序列。

## 12. 测试要求

### 12.1 纯逻辑单测

- 所有 pair 有效时，`gap=2` 选择 `0, 2, 4, ...`；
- 某个局部 pair 无效时，当前帧立即被提升；
- `gap=1` 选择全部帧；
- 任意帧最终都有合法 owner；
- owned frame 不会被 group top-k 删除；
- group member 去重且顺序稳定；
- query floor 先覆盖多个 cell，再全局补满；
- 空 cell 不会生成 query；
- 强纹理 cell 可获得超过基础配额的 query；
- 候选不足时返回所有有效候选且不报错。

### 12.2 小序列集成检查

使用 8～16 帧检查：

- SIFT DB 只构建一次；
- schedule audit 与实际 VGGSfM groups 一致；
- database image id remap 正确；
- triangulation 和 BA 能完成；
- `gap=1, floor=0` 与旧调度输出规模接近。

### 12.3 完整数据验证

在 181 帧和其他目标场景记录：

- selected center 数和减少比例；
- VGGSfM group tracking 时间；
- attempted query-view 数；
- prior tracks/observations 和长度分布；
- 每帧 P observations 的 min/p10/median；
- SIFT + P 后 dropped frames；
- final S-only/P-only/mixed points；
- final angular error median/p90；
- query grid occupied cells 与 per-cell 分布；
- 固定视角的稀疏点云人工检查。

## 13. 质量与性能验收原则

本方案质量优先。初始验收原则：

1. 不允许相对 full-center 基线新增 dropped frame；
2. final angular error 和 BA 收敛不能出现明显恶化；
3. per-frame P observation p10 和 query grid coverage 尾部不能显著下降；
4. 最终点云仍应保持用户认可的中等密度和空间连续性；
5. tracking 时间或 attempted query-view 应有可解释的下降；
6. raw track 数下降本身不是失败，但不能以明显场景覆盖损失换取速度。

在没有 GT 的情况下，最终点数不能单独作为质量标准。必须同时查看注册帧、每帧
尾部覆盖、track length、几何误差和固定视角点云。

第一轮可将“VGGSfM tracking 时间下降至少 25%”作为性能参考目标，但它不是覆盖
质量门槛的理由。如果 `gap=2` 导致点云明显变稀，应先回退 `max_center_gap` 或增加
query，而不是立即引入更复杂的调度器。

## 14. 明确非目标

本阶段不处理：

- RoMaV2 tracking 或 RoMa/VGGSfM hybrid；
- DINO retrieval 或 loop closure；
- GeoCalib / view-graph calibration；
- covariance 加权 BA；
- 3D voxel thinning；
- sequential/loop-closure 来源状态机；
- 动态场景语义 mask；
- 最小 set cover、ILP 或全局最优 center selection；
- 多套 pose/SIFT/depth/时间软权重融合。

这些方向只有在第一版简单方案的审计明确指出缺口时再单独立项。

## 15. 回退策略

实现必须保留清晰回退：

```text
max_center_gap = 1
min_queries_per_cell = 0
```

该配置应恢复为：

- 每帧都是 center；
- query 按全局 detector score 选择；
- SIFT-first 执行顺序保留，但不改变 VGGSfM 调度语义。

如果连该配置都不能复现基线，说明问题在 SIFT-first 重排、数据库复用或 ID remap，
不应继续调 center 或 query 参数。

## 16. 后续交接建议

接手实施任务时，先确认目标分支仍包含完整 VGGSfM 路径，并完成以下检查：

1. 找到当前 SIFT DB 构建、VGGSfM group 构造和 ALIKED query 提取入口；
2. 确认现有 `neighbors_per_center=16`、`vggsfm_group_batch_size=3` 基线可运行；
3. 先实现 P0/P1，不在同一个提交中同时改 center 和 query；
4. 为 schedule 使用纯 Python/NumPy 数据结构，避免调度逻辑依赖 CUDA；
5. 每一阶段保留 `gap=1/floor=0` 回归配置；
6. 在改变默认参数前完成完整 181 帧和至少一个弱纹理场景验证。

与已有 `.ai/plan/vggsfm_prior_optimization_plan.md` 的关系：旧文档覆盖更广泛的
VGGSfM 性能优化和实验观测；本文档只收敛“SIFT-first、center 抽稀、query 基础
配额”三项具体设计。后续实现以本文档的简单规则为第一版，不自动继承旧文档中
尚未验证的复杂调度设想。
