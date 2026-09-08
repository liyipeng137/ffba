# SIFT + LoMa 轻量 Prior V1 设计

2026-09-08 后端筛选更新：按用户要求忽略两视图 tracks，并让 SIFT/prior 共同进入
SelectTrack；此前保留 SIFT、允许两视图的描述作为历史基线。当前规则及冻结输入质量
对照见 [track 筛选实验](track_selection_ablation.md)。

状态：全候选、3/5/5 与执行加速已完成 789 图 GPU 实测；固定输入的 GPU 数值一致性仍待验收
更新时间：2026-09-08
实验分支：`codex/loma-prior-lite`  
分支起点：`codex/prior-pose-input` / `fedf938`  
主入口：[`run_merg3r_gluemap_pipeline.py`](../../run_merg3r_gluemap_pipeline.py)  
LoMa 源码：[`third_party/LoMa`](../../third_party/LoMa)

## 新增设计：批处理与执行流水加速

详见 [LoMa 执行加速设计](loma_execution_acceleration_design.md)。已增加
`loma_match_batch_size`、`loma_extract_batch_size`、`loma_preprocess_workers`、
`loma_geometry_workers`、`loma_feature_cache`。对照默认分别为 `1 / 1 / 0 / 1 / cpu`，
首轮组合试验为 `8 / 2 / 4 / 4 / cuda`，本轮保持 LoMa-B 与现有 3/5/5。
参数已贯通主入口与 prior 执行器；提取预处理、稳定索引、有限在途队列、缓存释放、
计时口径与分项验证顺序见新文档。下文早期 V1 的 CPU-only cache 描述是历史实现状态。

## 当前修订：SIFT 引导的 3/5/5 pair 筛选

本节记录首轮 GPU 结果后的已授权修订，优先于下文历史 V1 中“只标注、全部执行、
不设来源配额”的表述。全候选 V1 保留为 `--loma_pair_selection all` 对照；
默认改为 `sift_guided`。默认 provider 仍是 VGGSfM。

- SIFT pair 图与 LoMa 的 pose ∪ DINO top-30 ∪ temporal 候选池保持原样。
- 全保留 temporal ±2（实际窗口沿用 `sift_temporal_window`）；从剩余候选中，
  每图主动选择 sufficient 最多 3、insufficient 最多 5、untried 最多 5 个邻居。
- 三类互斥，指 SIFT 支持类别，不是 pose/DINO 来源。Temporal 不占名额，
  某类不足不跨类补齐；每图独立选择，任一端选中即保留，无向去重后只匹配一次。
  入边不消耗该帧主动选择名额，因此最终 degree 可以超过 13。
- Sufficient 的基础排序为 `min(source_coverage, target_coverage)`、SIFT 内点数、
  当前图到候选图的 DINO 相似度降序；insufficient/untried 为 DINO 相似度、
  最小覆盖率、内点数降序。未尝试边不填伪造 SIFT 数据；非有限 DINO 值排最后。
  这层分类和排序不采用“内点越少越优先”的反向筛选。
- 逐次选择时，优先考虑与该帧已选邻居的序号间隔均大于时序窗口的候选；
  无此候选时从剩余基础排序首位选择，不因此少选。已选邻居以 temporal 初始化，
  然后按 sufficient → insufficient → untried 更新。该软偏好只是时序去冗余，
  不声称已实现基于真实视角差异或空间共视的选择。
- 基础排序最终以图像 ID 打破并列；无向边保留所有选择方向、类内次序及去冗余原因。
  不运行动态增配、匹配后提前停止或弱帧自适应配额。
- 789 张图按窗口 2，执行数量上界为 `1575 + 789 × 13 = 11832`；实际数量必须
  由候选池计算，不能将此上界当作已测 pair 数。相对首轮 22973 对，上界减少 48.5%。

参数：`--loma_pair_selection {sift_guided,all}`；
`--loma_sufficient_neighbors 3 --loma_insufficient_neighbors 5 --loma_untried_neighbors 5`。
每类允许 0，禁止负值。全候选模式忽略这些选择名额。
关键点数、检测分辨率、匹配/几何阈值、track 建立与 BAE 保持原设置，cap 继续关闭。

`prior_loma_pairs.json` 保留完整候选池，新增 `selected`、`executed`、`selection`。
未执行的 pair 不写伪造的零匹配/零内点结果。`pair_selection` 统计区分候选/选中/跳过数量、
每帧主动选择数、选中图连通分量与 degree；验证图统计仅针对实际执行的结果。

验证：新增测试覆盖 3/5/5 上限、时序保留、不跨类补齐、入边保留、排序与时序软偏好、
确定性、全候选回退，以及未选中边不进入模型或 DB 的编排衔接。
云端 3/5/5 已实测：选中 9331/22973 对，prior 644.41 s，端到端 1442.05 s；
相对全候选端到端减少 32.8%，最终 P-only 点减少约 4.1%。验证图保持一个连通分量，
第 261 帧验证后节点从 858 降到 554，仍需检查最终弱帧覆盖。两轮 SIFT DB 不同，
不能将全部质量变化归因于 pair 筛选。证据见
[运行数据](../../logs/ffba_789_loma_355/)。本机现有统计文件未包含逐 pair 审计和
DINO 矩阵，不能据此复原精确筛选名单。

## 1. 目标与结论

保留当前相机初值、SIFT、三角化和 BAE 主流程，用 LoMa 自然产出的 learned feature
matches 补充 SIFT 覆盖不足的约束，实验性替代 VGGSfM prior。

目标是端到端更快，同时保持或改善相机和重建质量。核心不是增加 raw matches，
而是先验证 LoMa 能否自然产生比 VGGSfM 更少、同时足以补充 SIFT 的有效多视图 tracks。

用户已明确 V1 不添加额外控制逻辑：保留 `bae_optimize_intrinsics` 开关，不新增
全局 pair 预算、query/track/observation 配额、抽稀或按缺口调度；LoMa 首版运行不启用
`bae_max_observations`。是否优化内参由该开关决定，不在设计中擅自固定开启或关闭。
模型自带的匹配规则、必要的坐标/索引正确性及现有三角化和几何过滤继续沿用，
不再叠加新规则。本文中 789 的 cap 数据只描述旧 VGGSfM 运行，不是 LoMa 的配置要求。

Pair 设计已更正：SIFT 保持现有 pose rotation threshold + 时序逻辑；LoMa 必须
通过独立路径选择 pairs，保留 prior 对 SIFT 候选图之外连接的补充能力。不能将
“不额外控制 track 数量”理解为“两者必须使用同一个 pair 集合”。第 6 节将 V1
具体规划为继承 prior 候选池并全部执行，SIFT 结果先用于支持状态标注；真正按
支持不足程度调整选邻，放在取得首版结果后的独立实验中，不隐式增加运行控制。

LoMa 是首个候选 provider，不预设其在当前室内序列上的速度或定位精度必然优于
VGGSfM。V1 保持 VGGSfM 为默认与回退路径，不改现有运行命令的默认行为。

## 2. 已核实的当前状态

### 2.1 当前调用链

```text
run_merg3r_gluemap_pipeline.py
  -> Pi3X / MERG3R coarse stage
     或 Nerfstudio prior-pose stage
  -> utils/gluemap_spv_refine.py::run_gluemap_spv_refinement
  -> SIFT-first schedule（实验模式）
  -> utils/gluemap_refine_core.py::run_vggsfm_prior_tracks
  -> prior database + SIFT database -> merged database
  -> 每轮 triangulation / SelectTrack / geometry filter / observation cap / BAE
```

默认 BAE 路径只使用真实 S/P tracks，不使用 GlueMap virtual tracks。
新 provider 必须支持有 depth 的前馈输入和无 depth 的 prior-pose 输入。

最新 SIFT-first V1 已实现 center 抽稀与三层 group，但保留原 query、prior 成轨和
后端规则。LoMa 路线改变 learned correspondence 的生成方式，是新的实验轴。

### 2.2 789 测试证据

来源：[`logs/789_test.log`](../../logs/789_test.log)。日志从 SIFT 提取中途开始，
没有完整启动命令和全部配置；它记录的是 `sift_first_sparse + sift_pose_dino`
无深度分组，不能直接作为 `projected_overlap` 路径的性能证据。

- 789 帧，433 centers，group size 17，query 数 1024。
- VGGSfM 输出 408,022 tracks / 2,878,567 observations；track length median=6、p90=12。
- Prior 总耗时 273.31s，其中 group tracking 126.94s、fmap precompute 3.21s。
- 余下约 143.16s 不能仅凭日志归因，需要 `refine_stats.json` 拆分。
- 最终未因低覆盖过滤掉帧；输出 202,376 points / 1,998,602 observations。

| 外层轮次 | 三角化 observations | SelectTrack 后 | 几何过滤后 | Cap 后 | Cap 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3,889,283 | 2,994,271 | 2,941,768 | 1,999,999 | 127.45s |
| 2 | 3,899,235 | 3,066,306 | 3,061,018 | 2,000,000 | 122.49s |
| 3 | 3,898,903 | 3,065,373 | 3,060,623 | 1,999,995 | 122.00s |

每轮总减少约 49%，其中大部分来自覆盖选择和预算裁剪。几何过滤分别只减少
SelectTrack 后观测的 1.75%、0.17%、0.15%；不能把总减少量解释为误匹配比例。
Huber downweighted 数量也不是被删除的 observations。

三轮 cap 合计 371.94s，占 1,587.66s 端到端时间的 23.4%。Augmented refinement
共 954.88s。BAE 日志中的三次求解耗时合计 28.02s，不包含完整输入准备等成本。

### 2.3 影响设计的实现事实

1. `prune_reconstruction_for_bae_observation_budget` 按短 track 优先删除，同长度
   再比较角度误差和三角化角度；来源仅用于统计，没有 SIFT 保留配额或空间覆盖约束。
2. 每轮 `triangulate_from_seed_reconstruction` 都使用完整 DB 和 `clear_points=True`，
   上轮因 cap 删除的约束可能再次成轨。当前裁剪结果不约束下一轮候选池。
3. 当前 snap 修改 prior 坐标；`merge_colmap_databases` 仍拼接两套 keypoints 和索引，
   不在该步骤统一 SIFT/prior feature ID。坐标重合不等于已延长同一条 SIFT track。
4. 789 日志仅 23,872 个 prior observations 被 snap，约 0.83%；最终 mixed tracks
   为 2。P-only 占最终 observations 约 84.2%，不能仅用 mixed 数量评价 prior 价值。
5. 当前 ALIKED query 提取遍历所有图像，未限制为 selected centers。

上述是独立优化线索，不在 V1 同时修改，否则无法区分 matcher、调度与后端的收益。

## 3. V1 范围

### 包含

- 从本仓库 `third_party/LoMa` 加载模型；记录实际导入路径、架构和权重身份。
- SIFT-first；每帧一次 learned 特征提取与缓存。
- SIFT/prior 使用独立 pair 路径；LoMa 继承 VGGSfM group 的 pose + DINO 候选
  召回，并保留时序边。SIFT 提供支持状态标注，具体规则见第 6 节。
- 将验证后的 matches 写入 COLMAP prior DB，由现有三角化链路成轨。
- 提供 `vggsfm / loma` provider 选择，记录自然产出规模与阶段耗时。
- LoMa 首版使用 `bae_max_observations=0`，保留 `bae_optimize_intrinsics` 开关。

### 不包含

- 替换 SIFT、Stage A、相机模型或 BAE；调整 BA 轮数、鲁棒损失和过滤参数。
- 自动切换默认 provider，或隐式退回 VGGSfM 掩盖 LoMa 失败。
- 改动 VGGSfM 原有 query/group/snap 路径。
- 新增全局 pair 预算、query/track/observation 配额、center 抽稀、deficit/anchor
  调度、动态提前停止。V1 不沿用原 group 的末端 K=16 截取。
- 自定义并查集成轨和 score/cycle/track-length 附加筛选。
- 全局重写 cap、跨轮冻结候选、重做 SelectTrack；只在 LoMa 首版配置中关闭 cap。
- 跨来源 SIFT/LoMa 节点融合、强制坐标 snap、特征定位优化。
- 稠密匹配、虚拟 tracks、depth 驱动的强制过滤及复杂自适应补匹配闭环。
- Ceres 的 LoMa 验收；V1 LoMa 明确限定 BAE，VGGSfM 原支持范围保持不变。

## 4. 数据流与接口

```text
相同 images / initial cameras
  -> SIFT: existing pose + temporal pairs -> SIFT DB（一次）
  -> SIFT verified 内点与双向 grid coverage
  -> LoMa: pose UNION DINO top-30 UNION temporal
  -> 标注 SIFT 支持不足 / 支持充分 / 未尝试（不据此删边）
  -> 每图固定 LoMa features（一次）
  -> cached-feature pair matching
  -> 模型自带匹配过滤 + 标准两视图几何验证
  -> database_loma_prior.db
  -> existing SIFT DB merge
  -> existing triangulation / SelectTrack / geometry filter / BAE
     （bae_max_observations=0；内参优化由 bae_optimize_intrinsics 决定）
```

建议主入口新增 `--prior_provider {vggsfm,loma}`，默认 `vggsfm`。
LoMa 路径先准备 SIFT 并分析支持状态，但 prior pairs 的准入不依赖 SIFT 匹配是否成功；不要求人为
启用 `vggsfm_schedule_mode`，也不构建 VGGSfM tracker 或 fmaps。选邻可以复用
纯几何/retrieval helper，无需复用 tracker 调度。现有 VGGSfM 专属参数只影响 VGGSfM，配置审计
必须列出当前 provider 的实际生效参数，避免把未使用的参数当作实验配置。

不新增预算控制参数或隐式截断。LoMa 首版运行明确记录 `bae_max_observations=0`；
实现时不改写 VGGSfM 的 cap 行为，也不偷偷给 LoMa 加上自动 cap。出现资源问题时
记录实际规模和失败阶段，不能自动减少 tracks 后仍把结果标为无额外控制的 V1。

LoMa provider 的内部输出建议为：

```text
features[image_id]:
  keypoints_work_px[M,2]       # 稳定顺序，一张图一套坐标
  normalized_keypoints[M,2]
  descriptors[M,D]
  detector_scores[M]          # 模型提供时记录
verified_matches[(i,j)]:
  feature_indices[K,2]
  match_scores[K]
  geometric_inlier_mask / provenance
stats:
  timings / memory / pair counts / unique observations
```

LoMa DB writer 使用稳定 feature indices；可紧凑重编号，但必须保存唯一映射。
不要把每条 track 再展开成重复 keypoints 后靠坐标聚类恢复身份。
只导出几何验证后的真实匹配边，不把连通分量扩成未经验证的 clique/star 匹配。
这样无需改动 VGGSfM 的现有输出接口，也避免第一版进行大范围 provider 重构。

## 5. 特征提取和模型选择

- 首版先使用 LoMa-B 的标准配置，B128 等变体留作后续对照，不把模型 sweep
  作为接入前置要求。B 使用 DeDoDe-G，单看 matcher 大小不能预测总耗时。
- 使用本地 LoMa 模型自带的特征数默认值（当前 `num_keypoints=2048`），不为满足
  BAE 预算额外压到 1024，也不增加提取后的 top-k、网格配额或降采样。
  这是沿用模型原生配置，不代表 detector 输出无限关键点；日志记录实际生效值。
- 每张图只提取一次。禁止循环调用会为每对图重复提取特征的 `model.match(pathA,pathB)`。
- 工作图坐标不随 detector/descriptor resize 改变；记录原图、工作图、网络输入的映射，
  验证 normalize、resize/pad、半像素 convention 和逆变换。
- 不直接照搬 `algos/loma_tracking.py` 的 `-0.5`、clamp 或固定 resize：先核对本地
  LoMa 与 COLMAP 坐标约定，再确定转换。测试应覆盖非方图与边缘点。
- 默认可先使用 CPU feature cache + 按需上 GPU；GPU cache 作为独立性能项。
  记录 cache 实际 dtype、位置和峰值显存。
- 模型初始化、权重加载、compile/warmup 与 steady-state 分开计时；正式端到端
  仍计入实际发生的启动成本，不以热缓存 matcher-only 吞吐代替整段性能。

## 6. SIFT 与 LoMa 的 Pair 规划

### 6.1 V1 决策与后续边界

SIFT 保持当前 pose + temporal 图；LoMa 独立增加 DINO 召回，继承旧 VGGSfM 的
候选池而非末端 selected groups。先匹配全部 prior candidates，并用已完成的
SIFT 结果标注支持状态。此轮不按弱边筛选、不恢复 group top-K、不添加 observation cap。

SIFT 在 V1 的作用是暴露补充需求和提供可复用评估数据，暂不减少 LoMa 的执行集合。
单纯改变执行顺序不会减少总计算；不能将支持状态标注宣传成已实现的调度加速。
后续根据实测，才决定是否让 SIFT 信息进入有限邻居选择。

### 6.2 SIFT Pair：保持已有主干

```text
P_pose = 当前 build_pose_pairs 的结果
         视轴夹角 < pair_pose_rotation_threshold
         -> 每帧按相机中心距离取 pair_k_pose 个候选
         -> 无向去重
P_time = 0 < |i-j| <= sift_temporal_window
P_sift = P_pose UNION P_time
```

沿用当前配置，默认起点为 rotation threshold=30°、pair_k_pose=25、temporal window=2。
若对照基线显式覆盖这些值，则两路实验共用该基线，不因接入 LoMa 重设 SIFT 参数。
时序边独立加入，不再套用 rotation threshold；在转弯等情况下仍会尝试匹配。
帧顺序沿用主入口的实际输入顺序，prior-pose 模式对应 transforms.json 的 frames 顺序。

这里指当前 SIFT-first 的 pair 语义，不能因 VGGSfM legacy 默认值而在 LoMa 路径
丢失 temporal pairs。VGGSfM 旧模式保持原语义，不在同一改动中迁移其默认值。

### 6.3 LoMa Pair：独立视觉召回 + 既有几何/时序候选

```text
P_dino = 每帧在全序列中取 DINO similarity top-30（排除自身）
         -> 任意一个方向选中即保留
         -> 无向去重
P_loma = P_pose UNION P_dino UNION P_time
```

V1 具体规则：

1. 每帧参与 DINO 召回，不抽稀 centers，不使用 owner/owned/adjacent-center 结构。
2. DINO 候选不全局套用 SIFT rotation threshold，也不要求 SIFT 已匹配成功。
3. 三个来源取并集，一对图只匹配一次；不要求 mutual-top-K，不将 group 展开为 clique。
4. 不再从候选中取 K=16，不设 global pair cap、来源配额或按支持状态提前停止。
5. 每图 LoMa features 只提取一次，所有 pairs 复用稳定 feature ID。
6. 有 depth、无 depth 模式采用同一 pair 规则。V1 不调用 overlap 排序来删减执行集合。

这里两图不是相同集合；LoMa 图由 prior 的额外视觉召回扩展。本规划中
`P_sift` 是 `P_loma` 的子集，交集边也正常执行，保留同 pair 内的特征互补和 learned
长轨连接机会。不同算法生成两图不要求两图互斥；也不能保证 DINO 一定贡献新边。

Top-30 是复用旧 prior 候选召回规模，不是“没有任何 pair 参数”，也不是全量
all-pairs。单帧主动召回 30 个不代表无向 degree 最大为 30。LoMa 自然观测量较小
不代表匹配任务较少，实际 pair 数、耗时和显存必须记录。

### 6.4 SIFT 支持标注：不足、充分、未尝试

先完成 P_sift 的匹配与标准几何验证，再对每个 P_loma pair 附加标签。复用
`analyze_sift_schedule_graph` 的内点数、双向 grid coverage 和参数，不重跑 SIFT。

| 状态 | 定义 | V1 行为 |
| --- | --- | --- |
| `insufficient` | 已尝试 SIFT，但未达到当前内点数或双向覆盖标准 | LoMa 正常执行，重点观察补充收益 |
| `sufficient` | 已尝试 SIFT，达到当前标准 | LoMa 正常执行，观察区域补充和长轨支持 |
| `untried` | 不在已执行的 SIFT pair 集合中 | LoMa 正常执行，观察独立召回的收益 |

沿用当前标注默认值：8×8 grid、每 cell 至少 2 内点、每 pair 至少 128 内点、
两张图各自 coverage >=0.20。实际生效值写入输出；这些是旧 schedule 的支持标准，
只用作初始分类，不是 LoMa 的匹配/成轨门槛，也不证明三角化质量。

`insufficient` 另记录 `low_inlier_count`、`low_source_coverage`、
`low_target_coverage` 原因，可同时出现，并保留原始连续数值。零内点是已尝试但
未建立有效匹配；未尝试不填成零分。任务错误/数据库损坏不能伪装成 SIFT 弱边，
需要根据实际执行结果单独报告。全任务失败不能继续用“全部弱边”做有效实验。

当前 grid coverage 相对于完整图像，不是相对于真实重叠区；低 coverage 可能由
小重叠而非特征失败造成。因此旧 `valid_schedule_edge=False` 只提供不足信号，
不能直接解释为 LoMa 应优先执行或能够补救。

### 6.5 几何共视信息的用途

有 depth 时，可复用当前 directed round-trip projected overlap 评分，辅助解释
“共视较好但 SIFT 弱”和“共视本身不足”。优先复用已有结果；若需重新计算大量
pairs，先作为离线分析，不为匹配所有候选而额外支付无作用的在线排序成本。

保留 i->j 和 j->i 方向，缺少某方向时记作缺失，不能当成 0。低 depth confidence
或 coarse 几何偏差也可能导致低分，不将该分数视作独立真值。

无 depth 时，DINO 相似度、时序间隔、姿态差和基线先只作连续诊断数据；DINO
相似不等于真实共视。不复用当前 `sift_pose_dino` 中 SIFT 强边优先的 lexicographic
排序，也不简单反转为 SIFT 最弱边优先。

### 6.6 执行与输出接口

```text
build_sift_candidate_pairs(...) -> P_sift
prepare_sift_database(...)      -> sift DB
analyze_sift_schedule_graph(...) -> records for executed SIFT pairs
build_loma_candidate_pairs(P_pose, P_time, retrieval_matrix, dino_topk=30)
                                -> P_loma + source provenance
annotate_loma_pairs_with_sift(P_loma, sift_records, executed_sift_pairs)
                                -> sufficient / insufficient / untried
LoMa cached-feature matching on ALL P_loma
                                -> indexed matches + standard verification
existing triangulation          -> tracks
```

函数名是建议接口，尚未实现。pair 使用排序后的 image indices，保存全部来源标签，
例如同一 pair 可同时来自 pose、temporal 和 DINO。DINO 保存召回方向、rank 和
similarity，稳定排序处理相同分数；只从有限有效 similarity 中召回，不用 NaN
候选凑足数量。复用 Stage A 已有 retrieval matrix，不另提取一套 DINO 特征。
模型/配置输出应有独立 `loma_dino_candidates=30` 字段，避免依赖未使用的
`vggsfm_group_strategy`；该值继承旧候选规模，不增加额外二次截断。

### 6.7 V1 如何评价“补上 SIFT 不足”

按上述三种状态分别报告 candidate/executed/verified pair 数、有效匹配数、matched
unique feature nodes，以及关联到 2/3/4+ view tracks 的情况。frame/grid 覆盖和相机
质量仍作为全局指标，不能只看每 pair 的匹配数量。

本图构造下 `P_loma - P_sift` 对应未尝试的新增候选；`P_sift - P_loma` 应为空，
可作为图装配检查。SIFT/prior 实际验证成功的图不保证这种包含关系，也不保证连通。

同一 learned node/track 可能由多类 pair 支持，不能把各类关联 track 数直接相加。
保留 pair provenance 和 feature ID，分别报告每类唯一节点数与全局去重结果，明确
观察到的关联不等于删除该类边后的因果增益。若需证明某类边贡献，后续再做删边消融。

首轮重点回答：

- 共视迹象较好但 SIFT 支持不足的 pairs，是否确实有更高的补充收益？
- `untried` 的 DINO 新边能否验证成功，并增加跨段或弱帧连接？
- `sufficient` 边是否仍贡献新区域观测或 learned 多视图 tracks？
- 三类的计算成本如何，完整流程是否仍满足轻量方向？

### 6.8 后续 SIFT-aware 选邻：有证据后再进入执行决策

若 V1 表明某些候选计算收益明显较低，再单独实验让 SIFT 信息影响执行集合：
首先考虑独立共视迹象，在共视程度可比的候选间优先支持不足的 pairs；未尝试
候选保留独立召回机会，支持充分边仍允许帮助 learned tracks 延长。

此阶段才讨论近似共视分层、top-K 或其他选邻规则。不能直接用 `1/inlier_count`
排序、把未尝试当最弱、只匹配失败边或全删 SIFT 强边。若两帧本来就无有效共视，
再弱的 SIFT 结果也不代表有补充价值。

首版不实现这套筛选，不预设收益，也不通过调整执行顺序宣称减少工作量。

## 7. 几何验证与成轨

### 7.1 Pair 验证

先保留 LoMa mutual matches 和原始 match scores，再运行独立的两视图几何验证。
优先复用当前 pycolmap 几何验证能力，基于图像对应关系估计几何；粗相机只作辅助
审计，不新增 coarse pose/depth 检查来删除匹配。

阈值必须显式注明单位及其对应分辨率。LoMa 初始 match threshold=0.1 是模型默认
值，并非已校准的内点概率。不额外添加 score、cycle、coverage 或视差门槛。
后续三角化与现有几何过滤沿用基线设置。

### 7.2 稳定节点与现有成轨

节点唯一键为 `(image_id, feature_id)`。匹配结果直接写成 keypoints 和 indexed
two-view matches，交给当前 pycolmap 三角化链路建立 tracks、处理关联与合并。
V1 不另写基于并查集的 greedy track builder，不额外引入冲突拒绝或闭环过滤层。

保证索引、坐标和 DB 映射正确属于接入正确性，不是观测配额。不要复用旧
`algos/loma_tracking.py` 中先 union 再任取同图节点的实现。

### 7.3 Track 使用

所有通过现有链路的 tracks 正常参与后端，不添加最小 3 views、最大 track length、
top-k tracks 或来源配额。2/3/4+ views 分布仅用于观察自然成轨结果。

## 8. 观测规模：只测量，不控制

固定每帧 learned keypoints 后，在一个节点最多属于一条 track 的条件下：

```text
learned unique observations <= sum_i keypoints_i <= N * Q
```

同一 feature 与多个邻居匹配只增加边数，不增加独立观测节点。该上限依赖稳定 ID
和 writer 不复制节点，不适用于当前每个 group 独立预测坐标的 VGGSfM 产出。

该关系用于理解数据结构，不用于推导首版 Q 或目标 observation cap。实际成轨
规模有待实测，不能预设 LoMa 必然少于 VGGSfM 或低于 2M。

V1 记录检测节点数、matched unique nodes、三角化后和现有过滤后的 observations。
不能把 pair match endpoints 总和或 SIFT DB 全部检测点数当作 BAE observations。

LoMa 首版明确使用 `bae_max_observations=0`，不把现有 cap 留作自动兜底，
不新增任何 track 数量裁剪。先观察 LoMa 的自然规模、耗时和质量；只有实际结果
表明需要控制，后续才单独讨论相关方案。`bae_optimize_intrinsics` 仍是保留的
内参优化开关，其余 BAE 求解设置沿用当前配置，不添加 LoMa 专属限制。

## 9. 接入位置与现有代码复用

| 位置 | 计划职责 |
| --- | --- |
| `run_merg3r_gluemap_pipeline.py` | provider 参数、校验、生效配置与 run summary |
| `utils/gluemap_spv_refine.py` | SIFT-first 编排、provider 分支、LoMa DB 接入 |
| 新的 `utils/loma_prior.py` | 模型加载、feature cache、pair matching、统计 |
| Prior pair builder 与 indexed DB writer/helper | 独立选邻、无向去重、稳定 ID 和坐标/索引转换 |
| `utils/gluemap_refine_core.py` | 复用 SIFT/三角化/BAE，仅补必要 writer 或统计接口 |
| `third_party/LoMa` | 已复制的模型源码，V1 不改模型实现 |

`algos/loma_tracking.py` 接在另一个入口 `run_merg3r_bae_pipeline.py`，不是目标主线。
其中模型初始化和特征匹配可参考；depth/3D 过滤、加权点初始化及冲突处理不直接移植。
先使用现有数据库合并规则；LoMa 与 SIFT 坐标接近也保持独立来源，后续节点融合
作为单独实验，不声称 V1 已直接延长 SIFT tracks。

## 10. 输出与可观测性

建议新增 `prior_loma_stats.json`、`database_loma_prior.db`；导出独立 prior pair 列表
及与 SIFT 图的集合差异，不为此额外实现动态 schedule 系统。
可选缓存放在 `loma_features/`，必须校验图片身份、模型、分辨率和提取参数后才能复用。
`refine_stats.json` 中提供通用 provider 字段，保留旧 VGGSfM 字段的兼容性。

至少记录：

- model/source/weights、输入尺寸、cache、device、软件版本和完整生效配置。
- extraction、matching、verification、track assembly、DB、triangulation、SelectTrack、
  geometry filter、cap、BAE preparation/solve、总 Stage B 和端到端耗时。
- candidate/executed/verified pair 数、来源、SIFT/prior 交集与差集、degree、连通分量、
  孤立帧、弱帧救回；区分候选图连通与验证后连通。
- raw matches、unique nodes、2/3/4+ view tracks、track length 分布。
- 每帧 8×8 grid 的至少 1/2 observations 覆盖率，尤其 p10 和最弱帧。
- S-only/P-only/mixed 的现有过滤前后规模、角度误差分位数和三角化角度。
- 相机/focal 变化、固定视角点云和可用的下游渲染指标。

S/P/mixed 当前依据数据库索引来源分类，不能将它等同于完整匹配生成 provenance。
cap 字段在 LoMa V1 中应明确显示 disabled，不因输出格式兼容而实际启用裁剪。
额外网格/轨迹分析可离线进行，不为完成大量诊断而阻塞最小接入。

## 11. 分阶段实施与验收

### P0：记录基线

保留原命令、代码版本、图片和相机初值、SIFT DB。补齐本次 789 日志缺少的配置与
`refine_stats.json`，无法取得时重新记录基线，不猜测缺失参数。

首版使用 LoMa-B 原生配置，不做预算或阈值校准。记录预处理与 feature ID 稳定性。

### P1：最小 provider 接入

实现可复用缓存、标准 pair 验证、indexed DB writer 和日志。先验证：

- 坐标映射和 feature index remap，写库/读库后匹配一致。
- 重复边、空匹配、短序列及跨 pair feature ID 复用。
- LoMa 首版 cap 关闭；未增加观测抽稀、配额或自定义成轨过滤。
- 无 depth 输入不触碰 depth；VGGSfM 默认路径没有被 provider 改动改变语义。
- 小规模真实 CUDA smoke，包含模型权重、几何验证和一次完整 refinement。

CPU 单元测试不能代替 CUDA/pycolmap/BAE 端到端验证。

### P2：完整对照

| 实验 | 用途 |
| --- | --- |
| A：当前 SIFT + VGGSfM sparse | 保留原始质量与耗时基线，注明已有 cap |
| B：SIFT + LoMa V1，无 observation cap | 观察自然产出的规模、耗时与质量 |

冻结图片、初始相机、SIFT pair/DB、相机模型、BA 轮次、现有过滤阈值以及
`bae_optimize_intrinsics` 的开关状态。A 若因规模仍使用 cap，应明确与 B 存在该
配置差异，不把全部收益归因于 matcher；LoMa 的 pair 输入和 VGGSfM groups 也
不是完全相同的任务。资源允许时可在较小序列让两者均关闭 cap 进一步对照。
SIFT-only、B128 或减少 VGGSfM query 的消融留待初版结果后，不作为首版前置要求。

先验证 181/300 规模，再测 789；789 的 prior-pose 结果与前馈结果分表呈现。
正式性能结论至少重复两次、交替运行顺序，报告波动和缓存状态。

### P3：首轮结果评估

首版以跑通最小接入、获得未额外控量的真实结果为目标，不预设 track 上限、
20% 加速硬门槛或新的运行限制。记录掉帧、空间覆盖、角度误差、相机轨迹和固定
视角结果，讨论 LoMa 是否实现预期。LoMa 改变了点集，来源误差统计不再是逐点
对应比较，不能仅凭更少点或更低残差宣称质量改善。

覆盖不足帧或关键桥接帧的退化不能被整体平均值掩盖。相机轨迹在必要时做 Sim(3)
对齐；有独立参考时报告 ATE/RPE，没有参考时只报告差异与稳定性，不能声称绝对精度。

这一步先回答：LoMa 是否自然产生更少但有效的 tracks、完整流程是否更快、
关闭 cap 后的 BAE 是否可运行、相机和重建质量如何。V1 仍是实验 provider；
是否切换默认以及是否需要额外优化，均依据结果另行决定。

## 12. 参考与状态边界

- [当前 README](../../README.md)
- [SIFT-first V1 设计及实测结果](sift_first_vggsfm_group_design.md)
- [旧 prior 优化计划](vggsfm_prior_optimization_plan.md)：其中 deficit-aware 和 provider
  评估可作参考，不能把旧计划中的条目当成已实现功能。
- [本地 LoMa 概述](../summary/loma_summary.md)
- [LoMa 官方代码](https://github.com/davnords/LoMa)
- [LoMa 官方结果](https://www.davnords.com/loma)

### 2026-09-07 实现记录

- 已接入 `--prior_provider loma` 和 `--loma_dino_candidates`，VGGSfM 仍为默认。
  `--bae_optimize_intrinsics` 保留开启默认值，新增对应的
  `--no-bae_optimize_intrinsics`；LoMa V1 拒绝正的 observation cap，提示使用 0。
- `utils/loma_prior.py` 实现独立 pair 图、SIFT 支持标注、LoMa-B 原生提取与
  CPU feature cache、native mutual match、pycolmap 几何验证和 indexed DB writer。
  LoMa 坐标按 `(normalized + 1) * [W,H] / 2` 回到工作图；保留固定行号，不 snap。
- `utils/gluemap_spv_refine.py` 编排 provider 分支，复用原 SIFT、删帧重映射、DB
  合并和后端。LoMa 不创建 VGGSfM group/center，不进入旧的 depth-dependent LoMa 路径。
  `utils/gluemap_refine_core.py`、上游模型源码与现有 BAE 实现均未修改。
- 实际输出为 `prior_loma_pairs.json`、`prior_loma_stats.json`、
  `database_loma_prior.db`，以及共用的 `refine_stats.json`/COLMAP 模型。
  pair JSON 使用过滤前 image index；删帧映射另记于 stats，DB 使用保留帧的新 ID。
  特征仅本次运行缓存，不实现磁盘缓存复用。验证后匹配及 geometry 写入 DB；
  JSON 保存 raw/inlier 数与原始 match score 均值，未持久化逐匹配原始分数。
- 已记录候选/验证图连通性、孤立帧、每类 pair 的唯一观测和最终 learned track
  长度分布。类别可以重叠，不能相加或解释为各类 pair 的因果收益。
  更细的逐帧 learned 网格覆盖、救回帧归因及相机/渲染对照仍属于离线评估。
  model load、feature extraction、matching、verification 分开计时；首次 compile/warmup
  包含在对应阶段中，尚未单列 steady-state 吞吐。
- 本机 pycolmap 4.1.1 已验证真实两视图几何、DB 读写/合并、删帧索引及 100 条
  三视图 track 的原生三角化。编排测试覆盖有/无深度模式，使用合成 matcher 和
  替代 BAE 求解器，因此不能视为模型推理或 BAE 数值验收。
- 本机 `/Users/lyp/pycodex` 未安装 PyTorch，尚未运行 LoMa 权重推理或 BAE GPU
  实验；一个既有 prior 图片加载测试也因此无法通过。GPU smoke 与 P2/P3 对照仍待执行。
- 检查结果：11 项新增 LoMa 测试通过；与 `test_gluemap_refine_core.py`、
  `test_nerfstudio_prior.py` 合并执行为 37 passed / 1 failed，失败项是上述缺少
  PyTorch 的既有图片加载测试。Ruff、语法编译和 `git diff --check` 通过。

LoMa track 量会显著少于 VGGSfM 是待验证预期，不是实测结论。
