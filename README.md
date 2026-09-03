# FeedForwardWithBA：MERG3R + RoMaV2 ordered tracking + GlueMap

本分支是面向质量实验的 FFBA 管线。Stage A 仍由 Pi3X/MERG3R 提供粗 pose、
intrinsics 和 depth；Stage B 已将活动的 VGGSfM prior 路径替换为直接使用
RoMaV2 的有序长轨迹前端。SIFT 保留为独立的补充观测源，后续三角化、过滤和
GlueMap augmented BA/BAE 维持原流程。

当前目标场景是室内和中等范围室外序列，输入必须按文件名排序后具有正确时序。

## 管线概览

```text
ordered RGB frames
  │
  ├─ Stage A: Pi3X/MERG3R
  │    ├─ coarse poses
  │    ├─ per-frame intrinsics
  │    └─ depth / depth confidence
  │
  ├─ RoMaV2 low-resolution adjacent pass
  │    └─ measured image motion
  │
  ├─ tracking plan
  │    ├─ hard adjacent backbone: (i-1) -> i
  │    ├─ continuity direct anchor
  │    └─ geometry direct anchor
  │
  ├─ RoMaV2 high-resolution tracking
  │    ├─ certainty + ALIKED spatial births
  │    ├─ capped births + per-cell candidate prefilter
  │    ├─ accumulated covariance filtering
  │    ├─ per-frame spatial thinning
  │    ├─ minimum track length + weak-frame rescue
  │    └─ direct/sequential consistency refinement
  │
  ├─ independent SIFT database on the selected edge graph
  │
  └─ GlueMap triangulation/filtering + Ceres or BAE
       └─ refined COLMAP reconstruction
```

## 设计要点

### Stage A 仍然被复用

Stage A 不再决定 prior tracker 的固定邻居组，但它不是废弃计算。粗 pose、
intrinsics 和 depth 会被用于非相邻 direct anchor 的预筛选：

- 有向 projected overlap；
- source-grid coverage；
- depth consistency；
- 粗 parallax；
- 后续 BA 的初始化。

DINO retrieval 不再被此路径强制计算。相邻主干始终存在，因此粗 pose 较差时也
不会切断时序连续性；Stage A 几何只决定额外 direct edge 是否值得匹配。

### 实际图像运动，而不是时序软权重

RoMaV2 首先在相邻帧上执行低分辨率 batch matching，测量归一化 median flow。
tracking plan 使用累计实际运动选择跨度：

- `continuity_direct` 偏向较短的累计运动目标，修复逐帧传播漂移；
- `geometry_direct` 偏向较大的累计运动目标，并结合 Stage A overlap/parallax；
- 每个 target 最多各选一个，且两个 anchor 不重复；
- `max_anchor_gap` 限制目标场景下的最大搜索跨度。

旧 `ordered_motion` 中 pose/DINO/时序/depth-overlap 的候选并集和 `1.05x` 时序软
加权已经退出活动路径。

### 长 track 的质量控制

每一步 RoMaV2 输出 overlap certainty 和 2x2 precision。precision 被转换到当前
工作图像的像素 covariance 后沿 track 累加：

- 累计 certainty 使用路径上的最小值；
- 累计 covariance 使用逐步相加；
- 总 sigma 超过 `roma_max_track_sigma_px` 的 track 会被停止；
- direct path 只有在与 sequential path 几何一致且 covariance 更低时才替换；
- 不一致的高置信 direct 证据会提高 observation covariance，使漂移在后续传播
  时更容易被过滤。

covariance 当前用于前端筛选、direct path 选择和审计，不直接作为 BA 权重。

### 空间密度与 thinning

新点由 dense certainty 候选和 ALIKED salient points 合并产生。选择器按规则网格
round-robin，并执行像素半径抑制；候选预筛选也按 cell 分配预算，避免全局
certainty top-k 在均衡器运行前就删除低纹理区域。传播到 target 后再次按
certainty、covariance 和 track length 做空间 thinning。

为抑制短轨 birth cohort 在场景表面形成块状聚集：

- 初始帧可以建立完整主干，后续每帧新生默认最多 `384` 条；
- birth 使用独立的 `6 px` NMS，而传播 thinning 仍使用 `3 px`；
- 默认只保留长度至少为 `3` 的 RoMa track；
- 若严格长度门限会让某帧断开，只按空间均衡方式救回少量长度 `2` 的轨迹，
  将该帧补到 `16` 个 RoMa observations，不恢复全局短轨洪水。

上述上限控制的是 RoMa prior；SIFT 仍独立提供纹理角点，不受该门限影响。

RoMa observations 不再强制 snap 到 SIFT keypoints。两个数据库独立写入后再由
GlueMap 合并，避免 prior 精度被最近邻 SIFT 位置覆盖。

GlueMap 原有 `SelectTrack` 会在 SIFT pair coverage 足够时无条件删除 non-SIFT
track；本路径将其关闭，否则通过上述质量门限的 RoMa prior 仍会在 BA 前被全部
清空。RoMa 的数量控制由每帧空间上限和 covariance 筛选负责。

### 匹配拓扑

RoMa prior 默认以 `chain` 拓扑写入 COLMAP：长 track 只产生相邻 observation
之间的 pair match，不通过传递闭包凭空创建远跨度 matches。`all_pairs` 和 `star`
仍保留为实验选项。

### Epipolar verification 的当前作用

每条 RoMa edge 会从 dense field 采样高置信 correspondences，并用 MAGSAC 估计
fundamental matrix，记录 inlier ratio。它目前只做诊断，不参与 hard rejection，
因为粗内参、低纹理或近平面场景可能让单一阈值误杀有效 edge。审计结果可用于
下一轮确定是否按 edge 类型设置门限。

## 环境

```bash
source /opt/conda/bin/activate
conda activate gluemap-merg3r
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`expandable_segments` 可减少 181 帧 Stage A 中大块临时张量造成的 CUDA 显存碎片。

RoMaV2 源码直接从 `third_party/vidmap/third_party/RoMaV2/src` 导入。首次运行会
自动下载 `romav2.0.1.pt` 到 Torch Hub checkpoint cache。

## 运行

质量优先的默认运行：

```bash
python run_merg3r_gluemap_pipeline.py \
  --dataset /kiri/ff_data/181/input/181_img_data \
  --output_dir /kiri/ff_data/181/output/ffba_roma_antispot_v2 \
  --ba_backend bae
```

先用少量帧做集成 smoke test：

```bash
python run_merg3r_gluemap_pipeline.py \
  --dataset /kiri/ff_data/181/input/181_img_data \
  --output_dir /tmp/ffba_roma_smoke \
  --num_images 8 \
  --subset_size 8 \
  --ba_backend bae
```

重要的 RoMa 参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--roma_lowres_size` | `560` | 相邻运动测量的方形分辨率 |
| `--roma_lowres_batch_size` | `4` | 低分辨率相邻 pair batch |
| `--roma_highres_max_size` | `1200` | high-res matching 的最大边 |
| `--roma_max_keypoints` | `1500` | 每帧传播后的空间 thinning 上限 |
| `--roma_max_births_per_frame` | `384` | 非初始帧的 RoMa 新生轨迹上限 |
| `--roma_min_track_length` | `3` | 常规 RoMa track 最短长度 |
| `--roma_short_track_rescue_min_observations` | `16` | 弱帧短轨安全补点目标；`0` 关闭 |
| `--roma_aliked_keypoints` | `750` | 每帧加入候选池的 ALIKED 点数 |
| `--roma_min_confidence` | `0.05` | RoMa overlap certainty 下限 |
| `--roma_nms_radius` | `3.0` | target thinning 空间抑制半径，像素 |
| `--roma_birth_nms_radius` | `6.0` | 新生点空间抑制半径，像素 |
| `--roma_max_track_sigma_px` | `8.0` | 累计 covariance 的 track sigma 上限 |
| `--roma_max_anchor_gap` | `12` | direct anchor 最大帧跨度 |
| `--roma_continuity_motion_target` | `0.06` | continuity anchor 累计 flow 目标 |
| `--roma_geometry_motion_target` | `0.12` | geometry anchor 累计 flow 目标 |
| `--roma_min_stage_a_overlap` | `0.10` | Stage A direct 候选 overlap 下限 |
| `--roma_min_stage_a_grid_coverage` | `0.15` | Stage A source-grid coverage 下限 |
| `--roma_direct_consistency_px` | `4.0` | direct/sequential 最小一致性容差 |
| `--roma_spatial_grid_size` | `8` | 空间均匀选择的网格边数 |
| `--roma_spatial_prefilter_oversample` | `8` | 每 cell 候选预筛选倍率 |
| `--roma_epipolar_diagnostics` | on | 记录 MAGSAC epipolar 统计 |
| `--prior_match_topology` | `chain` | RoMa prior 写库拓扑 |

`--roma_compile` 默认关闭，便于实验迭代；确认输入形状稳定后可显式开启。

## 主要输出

```text
output_dir/
  pipeline_config.json
  pipeline_stage_a_summary.json
  images/
  pred_depth/
  roma_ordered_tracking/
    tracking_plan.json
    track_observations.npz
  database_roma_prior.db
  database_sift.db
  database_merged.db
  coarse/
  refined_gluemap_aba/
  refine_stats.json
```

`tracking_plan.json` 包含低分辨率相邻运动、Stage A 几何候选、实际 edge plan、
sequential/direct/epipolar 统计。`track_observations.npz` 保存每条最终 track 的图像
索引、像素位置、certainty、covariance 和 observation provenance。

## 验证

项目自身测试只收集顶层 `tests/`，避免误收 vendored 项目的独立测试套件：

```bash
PYTHONPATH=. pytest -q tests
```

当前实现覆盖 planning、空间 round-robin、Stage A geometry 和 chain topology 的
纯数值测试。完整质量评估应在相同输入上切换回旧分支做 A/B，并至少比较：

- 注册帧数、稀疏点数和 observation 数；
- track length 与累计 sigma 分布；
- 每帧/每网格覆盖率；
- triangulation angle 和 reprojection error；
- 远景与画面边缘的人工检查；
- sequential/direct edge 的 epipolar inlier ratio。

## 当前范围

本版有意暂缓 DINO loop closure、sequential/loop-closure 独立来源图、GeoCalib 和
view-graph calibration。相机模型仍为 `SIMPLE_PINHOLE`。这些方向适合在当前
ordered backbone 的质量稳定后分阶段加入，避免同时改变过多误差来源。
