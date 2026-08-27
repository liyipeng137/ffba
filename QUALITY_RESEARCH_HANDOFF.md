# FeedForwardWithBA 质量提升研究交接

> 更新时间：2026-08-26  
> 当前分支：`dev/dynamic_reproh_threshold`  
> 创建本文前的提交：`1f919d7 Tune final refinement with a separate BAE Huber delta`  
> 创建本文前工作树干净，分支与 `origin/dev/dynamic_reproh_threshold` 同步；本文自身尚未提交  
> 下一阶段目标：暂停以“加速/跑通”为主要目标的调参，转为用可复现的对照实验定位并提升最终重建质量。

## 1. 交接结论

当前管线已经具备完整的粗重建、VGGSfM prior、SIFT/prior 合并、GlueMap track 筛选和 BAE 精修能力，也已完成若干性能与显存可用性改动。但现有实验还不能证明最终质量达到预期：

- 在 300 帧数据上，把 augmented refinement 从 2 轮增加到 3 轮，并在最后一轮把 pre-BA angular filter 从 `1.0°` 收紧到 `0.5°`，最终结果没有观察到明显收益；该实验随后被撤回。
- 这两份 300 帧日志的上游输入并不完全一致，因此它们只能说明“未看到明显改善”，不能作为严格 A/B 定量结论。
- 目前已经改为一个更温和的新实验：最后一轮把 BAE Huber delta 从 `1.0 px` 提高到 `2.0 px`，让中等残差获得更大权重，同时保留对大残差的 Huber 保护。代码已经完成，但还没有完整 CUDA 质量结果。
- 现有迹象表明，质量瓶颈未必只是 BAE 迭代次数或 loss。VGGSfM group/pair 的共视正确性、prior track 质量、SIFT/prior 组成、相机/焦距稳定性，以及 SelectTrack 的取舍都可能更重要。

下一位 agent 不应继续同时叠加多个参数变化。应先固定全部上游输入，建立严格 matched baseline，再逐项验证“多一轮 refinement”和“final Huber delta”分别带来的效果。

## 2. 当前代码与目录

仓库已经移除了旧的外层 `MERG3R/` 目录，正式逻辑位于仓库根目录。历史讨论中出现的 `MERG3R/...` 路径已过时。

主要入口与模块：

```text
run_merg3r_gluemap_pipeline.py       # 当前唯一正式入口、Stage A 和 CLI
utils/gluemap_spv_refine.py          # Stage B 编排、配置和统计输出
utils/gluemap_refine_core.py         # prior、数据库、筛选、剪枝、精修核心逻辑
third_party/gluemap/                 # 仓库内 GlueMap Python 代码与 VGGSfM tracker
third_party/bae/                     # BAE 后端
tests/test_gluemap_refine_core.py    # 关键行为保持测试
logs/                                # 已有实验日志
```

当前分支最近的关键提交：

```text
1f919d7  Tune final refinement with a separate BAE Huber delta
63054cf  Enhance ... final filter reprojection error threshold（历史尝试，后续已撤回）
4ef3a71  Add BAE observation budget pruning
5b087a4  Add CUDA memory cleanup diagnostics before augmented refinement
3b221e3  Batch feature snapping queries by image
75e5b46  Pair 逻辑简化：只保留 pose pair
0734f93  将原 MERG3R 目录内容迁移到仓库根目录
```

### 2.1 GlueMap/BAE import 边界

当前实现是有意的“本地 Python 源码 + 环境内编译包”组合：

- `utils/gluemap_refine_core.py::_ensure_gluemap_imports()` 会把仓库内 `third_party/gluemap` 插到 `sys.path` 首位，因此 `gluemap` 和 `thirdparty` Python 逻辑优先使用仓库版本，方便直接修改。
- 该函数会检查仓库内 `third_party/gluemap/gluemap` 是否存在，所以不能把仓库内整份 GlueMap 完全删除。
- `pygluemap` 是独立的编译扩展；只要已经安装在当前 Python 环境可搜索的位置，就可以被仓库内 GlueMap Python 代码导入，不要求其安装目录物理位于 `third_party` 下。
- BAE solver 通过普通 `import bae` 从当前 Python 环境加载，并打印实际 runtime 路径；它不会强制从仓库内 `third_party/bae` 加载。

因此推荐开发方式是：保留并修改仓库内 GlueMap Python 源码，同时在环境中安装兼容的 `pygluemap` 扩展；BAE 也以当前环境实际 import 到的版本为准。运行质量实验时应记录日志中的 `pygluemap/bae` 真实来源和版本，避免环境漂移。

## 3. 当前管线

### 3.1 总体数据流

```text
输入图片
  -> 两级图片金字塔
     -> 低分辨率 Stage A：Pi3X/MERG3R 粗几何
     -> 高分辨率 Stage B：SIFT + ALIKED/VGGSfM prior
  -> 合并 COLMAP database
  -> coarse triangulation
  -> augmented refinement 循环
     -> 重新三角化
     -> SelectTrack
     -> pre-BA reprojection filter
     -> 可选 BAE observation budget 剪枝
     -> BAE
     -> 仅最终轮做 post-BA normalized filter
  -> refined_gluemap_aba
```

### 3.2 Stage A：Pi3X/MERG3R 粗几何

1. 读取图片并构建内存中的 low/high 两级图片。
2. low image 用于 Pi3X feed-forward 粗推理；high image 用于 SIFT、VGGSfM tracking 和最终 refinement。
3. 根据 `sequence_type`、`subset_size`、`overlap` 等参数切分子序列。
4. 对各子序列推理，并用 `weighted_iterative` 等方式对齐、恢复原始帧序。
5. 得到 coarse extrinsics、intrinsics、depth、depth confidence。
6. 当 group strategy 为 `projected_overlap` 时，还会提前计算全图 DINO retrieval similarity matrix。
7. 用 `build_pose_pairs()` 构建 pose pair graph。

当前 pair 逻辑已经简化为：

- `--pair_k_pose 25`
- `--pair_pose_rotation_threshold 30`
- 对每个相机，先排除 viewing-axis 夹角大于等于阈值的候选，再按 camera-center distance 取前 K。
- 不再使用 `pair_k_similarity`、`pair_temporal_window`、`pair_pose_fill_unfiltered`。
- 不再对 pair 做补齐，也不保证所有 center 都有 pair。
- 原 `build_mixed_pairs()` 已更名并简化为 `build_pose_pairs()`。

需要持续关注 `pair graph` 的 `zero_degree_images`、degree 分布和覆盖率；“不强制每个 center 有 pair”是明确设计选择，不等于零度帧对质量无影响。

### 3.3 Stage B：SIFT + VGGSfM prior + GlueMap

1. 保存 high-resolution work images。
2. 构建 VGGSfM groups。
3. ALIKED 为每个 center 提供 query points，VGGSfM tracker 预测跨视图 prior tracks。
4. 构建 SIFT database。
5. 过滤观测覆盖过低的帧。
6. 将 prior observation snap 到每图最近的 SIFT keypoint；不能 snap 的 prior keypoint 仍作为 prior keypoint 保存。
7. 用 `star` 或 `all_pairs` topology 把 prior track 写成 COLMAP matches；当前实验固定使用 `star`。
8. 合并 SIFT/prior database，生成 coarse reconstruction。
9. 进入 augmented refinement。

Prior track 的作用不是假设它一定比 SIFT 准，而是补充 SIFT pair matching 在弱纹理、长距离、跨视图覆盖上的不足。当前普通 SelectTrack 阶段仍优先保留 SIFT tracks；只有启用 BAE observation budget 时，才会把 SIFT/Prior tracks 统一参与额外剪枝。

### 3.4 VGGSfM group

当前支持：

- `pose`：默认策略。按 rotation-valid 与 camera-center distance 选择邻居。
- `projected_overlap`：质量实验主要使用的策略。候选集合是 rotation-valid pose candidates 与 DINO top-k 的并集，再使用 coarse depth 的 projected overlap、depth consistency、grid coverage、visible ratio 等信号排序。

当前 parser 默认：

```text
neighbors_per_center=25
vggsfm_group_strategy=pose
vggsfm_group_batch_size=2
```

用户此前明确的实验配置通常为：

```text
vggsfm_group_strategy=projected_overlap
neighbors_per_center=16
vggsfm_group_batch_size=3
```

README 和部分历史日志也出现过 `neighbors_per_center=12`。后续实验必须以实际命令和 `pipeline_config.json/refine_stats.json` 为准，不能仅凭“默认参数”推断。

### 3.5 Augmented refinement

每一轮在 `run_merg3r_augmented_refinement_loop()` 中依次执行：

1. 使用上一轮优化后的 poses/intrinsics 作为 seed，以 `clear_points=True` 重新三角化；上一轮 points3D 不直接沿用。
2. `SelectTrack` 根据 SIFT/prior 支持关系选择 real tracks。
3. 如果 filter type 是 `angular`，记录按 S-only、P-only、mixed 分类的 angular error 统计。
4. 执行 pre-BA reprojection filter。
5. 如果启用 `bae_max_observations`，在 BAE 前删除完整 tracks，使 observation 数量不超过预算。
6. 执行 BAE 或 Ceres。
7. BAE 路径仅在最终 outer iteration 执行 post-BA normalized filter；中间轮的 points 会在下一轮被重新三角化，提前 post-filter 没有最终保留价值。

当前 parser 默认 `num_refinement_iterations=3`、`filter_reproj_error_type=angular`、`filter_reproj_error_threshold=0.5`。但用户先前声明的固定实验基线是 2 轮、angular `1.0°`。质量对照时必须显式传参，避免 parser 默认变化污染实验。

### 3.6 BAE 当前语义

BAE 路径的核心设计：

- 只优化 real sparse tracks，不使用 GlueMap virtual tracks。
- 相机模型为 `SIMPLE_PINHOLE`。
- `--bae_optimize_intrinsics` 优化 focal，principal point 固定。
- 默认 gauge 为 `two_cams`。
- `--bae_robust_loss none` 是普通 squared reprojection loss。
- `--bae_robust_loss huber` 是 IRLS Huber；delta 单位是 pixel。

Huber 权重近似为：

```text
s <= delta: weight = 1
s >  delta: weight = delta / s
实际 residual 乘 sqrt(weight)
```

因此：

- 降低 delta 会更早、更强地降权，是“更鲁棒”，不是“更强拟合”。
- 提高 delta 会更接近 L2，使中等残差对优化产生更大影响，但也提高被错误 observation 拉偏的风险。
- 提高 delta 不会增加 BAE optimizer iteration 数量。
- `final_bae_huber_delta` 仅在最后一个 outer refinement iteration 生效；不传时完全保持每轮使用 `bae_huber_delta` 的旧行为。
- 当 `bae_robust_loss=none` 时，Huber delta 参数没有实际作用。

BAE 后的 normalized filter 由 GlueMap 内部执行，阈值从较松到较严使用 `0.03 / 0.02 / 0.01`；BAE 路径过滤后不会再次执行 BAE。它主要用于最终输出清理，不应被理解为新的强 BA。

## 4. 当前建议复现的基线配置

用户此前声明的固定基线为：

```bash
python run_merg3r_gluemap_pipeline.py \
  --dataset DATASET \
  --output_dir OUTPUT \
  --prior_match_topology star \
  --virtual_verify_mode center \
  --ba_backend bae \
  --bae_max_num_iterations 20 \
  --num_refinement_iterations 2 \
  --bae_optimize_intrinsics \
  --bae_robust_loss huber \
  --bae_huber_delta 1.0 \
  --filter_reproj_error_type angular \
  --filter_reproj_error_threshold 1.0 \
  --neighbors_per_center 16 \
  --vggsfm_group_strategy projected_overlap \
  --vggsfm_group_batch_size 3
```

注意：旧参数 `--no-pair_pose_fill_unfiltered` 已被删除，不能再使用。

当前待验证的新实验是在完全相同上游条件下改为：

```bash
--num_refinement_iterations 3 \
--bae_huber_delta 1.0 \
--final_bae_huber_delta 2.0 \
--filter_reproj_error_type angular \
--filter_reproj_error_threshold 1.0
```

delta schedule 为 `1.0 / 1.0 / 2.0`。最后一轮仍使用基础 angular `1.0°` pre-filter；不存在 final filter threshold 特例。

## 5. 最近完成的工程改动

### 5.1 Pair 逻辑简化

已删除：

- `pair_k_similarity`
- `pair_temporal_window`
- `pair_pose_fill_unfiltered`
- `build_mixed_pairs`

当前只保留 pose K 和 rotation threshold。该改动减少了隐式补 pair，符合“只使用当前明确 pose 候选、不强制所有 center 有 pair”的决策。

### 5.2 VGGSfM group 与性能优化

已实现：

- `projected_overlap` group 策略。
- 多 group batch forward。
- 根据显存预算预计算并缓存 feature maps，优先以 BF16/FP16 常驻 GPU。
- position embedding cache。
- 移除 tracker forward 中重复的 `torch.cuda.empty_cache()`。

旧版 181 帧 K 对比的历史结果：

| 项目 | K=25 | K=12 |
|---|---:|---:|
| attempted query views | 4,633,600 | 2,224,128 |
| group tracking | 608.2 s | 598.6 s |
| final points | 117,167 | 约下降 20% |
| P-only points | 91,308 | 约下降 25% |
| dropped frames | `[24,25,58]` | 额外丢失 `152` |

结论仅可作为质量风险提示：K=12 显著减少理论 workload，但旧实现下 tracking 只快约 1.6%，并损失 coverage/P geometry。由于之后已经修改 group batching 和 fmap cache，这些时间数据不能代表当前实现，必须重新做 CUDA 对照。

### 5.3 Prior observation snapping 加速

`snap_prior_tracks_to_features()` 已从逐 observation 标量 KDTree query 改为按 image 批量 query：

- 保留原 observation 顺序。
- 保留 center-first/star topology 语义。
- `workers` 固定为 1，不需要手动调 workers 数。
- 单元测试对比旧标量参考实现，覆盖重复点、等距点和随机数据。

目标是与旧逻辑等价，不应改变最终重建质量。

### 5.4 进入 augmented refinement 前清理显存

已增加 Python GC、`torch.cuda.empty_cache()` 以及清理前后 allocated/reserved/driver free 显存日志，避免 VGGSfM tracker/fmap 阶段无用缓存继续占用 BAE 显存。

这属于阶段边界资源清理，不会改变重建数据本身。

### 5.5 BAE observation budget

新增 `--bae_max_observations`：

- `<= 0` 时禁用，保持原逻辑不变。
- 只在进入 BAE 前、且实际 observation 超限时生效。
- 直接在当前 reconstruction 中删除完整 track。
- SIFT/Prior 全部统一评估，但只在这个显存预算剪枝阶段如此；普通 SelectTrack 的 SIFT 优先逻辑不变。
- 短 track 优先删除；同长度时先删除高 angular error，再删除小 max triangulation angle；最后用 track ID 保证确定性。
- 每图至少保留 64 observations，该值为代码内常量，不暴露 CLI。
- 如果受每图 64 保护限制无法达到预算，直接报错并打印理论最低数量。
- 删除完整 track 可能使最终 observation 数略低于预算，这是接受的行为。

1000 帧 OOM 日志 `logs/bae_oom.log` 的关键数据：

```text
SelectTrack 后 observations: 3,586,022
angular filter 删除:          457,098
进入 BAE real observations:  3,128,924
进入 BAE real tracks:          623,932
GPU: RTX 4090 24 GB
结果: optimizer 第一步附近发生 cuSPARSE/OOM/illegal access
```

用户估计安全上限约 230 万 observations，README 示例使用 200 万。这个机制首先用于让 1000 帧流程可跑通，不应把“成功压到预算”自动解释为质量最优。

## 6. 最近的质量尝试

### 6.1 三轮 refinement + 最后一轮 angular 0.5°

历史尝试：

```text
num_refinement_iterations=3
基础 angular threshold=1.0°
最后一轮 angular threshold=0.5°
bae_huber_delta=1.0
```

日志：`logs/300_minitest_it3.log`

第三轮关键数据：

```text
SelectTrack 后 observations: 1,407,300
pre-BA filter 删除:            55,107 observations, 74 tracks
进入 BAE:                    1,352,193 observations
raw MSE:                     8.654 -> 3.185 px^2
Huber downweighted:          66.1% -> 41.2%
focal:                       1017.72 -> 1020.77
rotation delta mean:         2.073°
translation delta mean:      0.0886
post-BA filter 总删除:         2,728 observations
最终 points/observations:     265,063 / 1,349,465
最终 S-only angular p90:      0.0937°
最终 P-only angular p90:      0.1363°
augmented refinement time:   336.09 s
```

对照日志 `logs/300_minitest.log` 的最终摘要：

```text
2 rounds
最终 points/observations:     249,610 / 1,319,733
最终 S-only angular p90:      0.0912°
最终 P-only angular p90:      0.1251°
augmented refinement time:   160.47 s
```

用户从最终结果上没有看到明显差别。更重要的是，这两次 run 的 upstream 并非完全一致，包括 intrinsics/orientation、SIFT pairs、prior tracks 和输出路径等，所以不能把点数、角误差或耗时差异全部归因于第三轮/0.5°。

该参数 `final_filter_reproj_error_threshold` 已在提交 `1f919d7` 中移除，不要恢复它，除非先提出新的、严格隔离变量的证据。

### 6.2 最后一轮 Huber delta=2.0

当前实现：

```text
round 1: huber delta=1.0 px
round 2: huber delta=1.0 px
round 3: huber delta=2.0 px
```

动机：经过前两轮 SelectTrack、pre-filter 和 BAE 后，最后一轮可能已经更干净。将 delta 从 1 提高到 2，会让 `1–2 px` residual 恢复完整权重，并让更大 residual 的权重相对 delta=1 约提高一倍，同时仍保留 Huber 保护。

风险：第三轮日志在 delta=1 下，初始仍有 66.1% observations 被降权，优化结束仍有 41.2% 被降权。这说明数据并没有干净到可以放心切换 plain squared loss；delta=2 也仍可能把错误 observation 的影响放大。

状态：

- CLI、配置传递、每轮选择、统计和单元测试已完成。
- 未设置参数时保持旧行为。
- 已通过 Ruff、目标测试、`py_compile` 和 diff check；历史验证结果为 `15 passed`。
- 尚未跑完整 target Linux/CUDA matched quality test。

### 6.3 暂不建议直接切换 `bae_robust_loss=none`

`none` 不是“更强的 Huber”，而是取消鲁棒降权。考虑到当前最终轮仍有较大比例 residual 被 Huber 降权，直接使用 `none` 很可能让错误 correspondence、错误 track 或局部错误几何拉动相机和焦距。

若将来验证 `none`，至少需要：

- 固定完全相同的 BAE 输入 reconstruction。
- 检查 pixel residual 的 median/p90/p99/max，而不仅是 mean。
- 检查 pose drift、scale ratio、focal drift 和注册帧。
- 有明确的下游质量指标。
- 最好先验证 `delta=1.5/2/4` 的连续趋势，而不是直接跳到 `none`。

## 7. 三种 pre-BA filter 的理解

当前 `filter_reproj_error_type` 支持：

- `angular`：使用 observation ray 与重投影 ray 的夹角，单位 degree；对焦距/分辨率更易解释，是当前推荐基线。
- `pixel`：图像平面 pixel residual，与 BAE Huber delta 单位一致，但固定 `0.5/1 px` 往往远比 angular `0.5/1°` 严格，不能直接用相同数值替换。
- `normalized`：pixel residual 除以 focal，跨分辨率/焦距更稳定；当前 post-BA filter 使用 normalized threshold。

不要因为 pixel 与 BAE residual 单位一致，就假设最后一轮切换 pixel 会更收敛。filter 是删 observation，BAE 是优化参数，两者不是同一种“力度”。如果未来测试 final pixel filter，应新增明确的独立实验，并先根据 focal 把 angular threshold 换算到合理 pixel 量级，而不是直接使用 `0.5/1 px`。

## 8. 下一阶段建议：严格质量研究

### 8.1 先冻结上游

每次质量对照必须固定：

- 图片集合、顺序、EXIF/orientation 处理和 low/high 尺寸。
- Stage A subset 切分、coarse poses/intrinsics/depth/confidence。
- DINO retrieval matrix。
- pose pairs 与 VGGSfM groups。
- SIFT database、VGGSfM prior tracks、snapped prior database。
- random seed、软件版本和 GPU 类型。

理想方式是保存并复用相同的 coarse state/database，只对 augmented refinement 做离线重放。若当前代码还不能直接重放，这是优先级较高的实验基础设施，而不是新的质量算法。

### 8.2 第一组 matched ablation

保持所有其他参数一致，仅比较：

| 实验 | outer rounds | delta schedule | angular pre-filter | 目的 |
|---|---:|---|---:|---|
| A | 2 | `1 / 1` | `1.0°` | 原固定基线 |
| B | 3 | `1 / 1 / 1` | `1.0°` | 隔离“多一轮 refinement” |
| C | 3 | `1 / 1 / 2` | `1.0°` | 隔离 final delta=2 |

在 A/B/C 完成前，不要同时改变 group K、filter type、filter threshold、SelectTrack support 或 observation budget。

### 8.3 必须记录的指标

覆盖与组成：

- registered/dropped images。
- pair graph zero-degree images 和 degree 分布。
- VGGSfM group size、attempted query views、forming track rate、track length。
- 每图 observations 的 min/p10/median/zero-count。
- 最终 S-only、P-only、mixed track/observation 数量。
- 如果 observation budget 生效，记录按来源删除的 tracks/observations。

优化与稳定性：

- 每轮 BAE 输入 tracks/observations。
- raw pixel residual 的 mean/MSE、median、p90、p99、max。
- Huber 初始/最终 downweighted fraction 和最小权重。
- rotation/translation drift 的 mean/p90/max。
- scale ratio。
- focal 每轮变化。
- pre-filter/post-filter 删除的 observations 和完整 tracks。

最终质量：

- 不要只看训练 observation reprojection error。
- 优先建立与真实目标一致的下游指标，例如 hold-out view reprojection、camera trajectory continuity、mesh/GS 渲染质量、结构完整性和人工盲评。
- 残差降低但同时删除大量 tracks/frames，不应直接判为质量提升。

### 8.4 当前优先假设

建议按以下优先级研究：

1. **先判断问题是否真的在 BAE。** 第三轮仍明显移动 poses/focal，但用户没有看到最终质量收益，说明下游瓶颈可能来自错误/不足的约束，而不是 optimizer 没跑够。
2. **审计 projected-overlap groups。** 对失败帧、零度帧、低 coverage 帧输出可视化 contact sheet，检查候选是否真有共视、视差和表面覆盖。
3. **审计 SIFT/Prior track 组成。** 不预设 SIFT 一定正确或 Prior 一定错误；比较两类 track 的长度、角误差、视差、空间覆盖和对 pose 的影响。
4. **关注相机和焦距稳定性。** 每轮 focal、scale、pose drift 如果持续明显变化，可能表示 gauge、intrinsics 初始化或错误 tracks 仍在驱动系统。
5. **最后再扩展 loss/filter 搜索。** 只有确认输入约束质量足够后，再考虑 delta sweep、pixel final filter 或 plain loss。

### 8.5 1000 帧数据的额外约束

- `bae_max_observations` 是可运行性保护，不是质量目标。
- 300 帧 A/B/C 应把 cap 设为 0 或足够高，确保不触发。
- 1000 帧实验中应先找“刚好不 OOM”的最高预算，再评估不同剪枝量对 coverage 和下游质量的影响。
- 不建议通过降低 `select_track_min_support` 来直接控制 observation 数量；降低它可能保留更多 prior tracks，反而增加 BAE 输入。显存控制应优先使用明确的 `bae_max_observations`。

## 9. 明确不要做的事情

- 不要使用旧 `MERG3R/...` 路径或旧外层目录结构。
- 不要把不同 upstream 的两份日志称为严格 A/B。
- 不要一次同时改变 rounds、Huber delta、filter threshold、filter type 和 group K。
- 不要仅凭较低 reprojection error 判断质量，尤其当同时删除了更多 observations/tracks。
- 不要把 `delta` 提高解释为增加 optimizer iterations。
- 不要把 `delta` 降低解释为更强拟合。
- 不要把 `bae_robust_loss=none` 当作无风险的“最终强 BA”。
- 不要假设所有 SIFT tracks 都可靠，也不要假设 Prior tracks 天生更差。
- 不要把单元测试、静态检查或 macOS 本地检查描述为完整 Linux/CUDA 质量验证。
- 不要直接复用旧 K=25/K=12 的时间结论；当前 batching/fmap-cache 已不同。

## 10. 下一位 agent 的启动清单

1. 运行 `git status --short --branch` 和 `git log -5 --oneline`，确认仍在预期分支/提交。
2. 阅读本文件、`README.md`、`run_merg3r_gluemap_pipeline.py` 和 augmented refinement 主循环。
3. 从本次实验的实际 `pipeline_config.json`、`refine_stats.json` 与完整命令恢复参数，不根据 README 默认值猜测。
4. 先设计并执行 A/B/C matched ablation，不先新增更多参数。
5. 若用户给出新日志，先检查 upstream fingerprints 是否一致，再比较 refinement 指标。
6. 优先输出“质量瓶颈证据”，而不是继续无边界调参。

## 11. 关键文件和日志

```text
run_merg3r_gluemap_pipeline.py
utils/gluemap_spv_refine.py
utils/gluemap_refine_core.py
third_party/gluemap/gluemap/estimators/bae_solver.py
tests/test_gluemap_refine_core.py
README.md
logs/300_minitest.log
logs/300_minitest_it3.log
logs/bae_oom.log
```

最终交接原则：**先确保对照可比，再讨论质量是否提升；先定位约束质量问题，再扩大 BAE 参数搜索。**
