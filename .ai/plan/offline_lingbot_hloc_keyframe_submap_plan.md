# 离线全局关键帧与 Shared-Anchor Submap 方案

更新时间：2026-05-06

## 1. 背景与目标

当前 `VGGT-SLAM -> Pi3X` 路径已经跑通：

- Pi3X submap 推理
- overlap Sim3 对齐
- multi-overlap 联合估计
- LingBot prior pose 回环候选检测
- Pi3 小窗口 loop edge 注入
- dense 点云导出与 submap 调试导出

但当前主要瓶颈仍是：

- 连续 submap 链式拼接容易积累 drift
- overlap 只靠相邻窗口局部桥接，上限较低
- loop edge 已起作用，但质量和权重还不够稳定
- dense 重影主要来自 cross-submap global consistency

本方案面向**离线场景**：一开始已经拥有全部图片，并且用户会提供 LingBot prior pose，即当前 `VGGT-SLAM` 已支持加载的：

- `--lingbot_transforms_json`

目标是从当前“按图片顺序每满 N 张构建一个 submap”的在线式流程，逐步切到更适合离线全图的：

```text
LingBot prior pose + HLoc/LoMa 几何观测质量
    -> 全局 KeyFrame 选择
    -> shared-anchor local batch/submap 构建
    -> 任意 shared keyframes 跨 submap 对齐
    -> keyframe-first refinement
    -> non-keyframe refinement
```

核心思路参考 AMB3R，但不直接照搬 AMB3R-SfM：

- `LingBot pose` 用于全局几何覆盖、候选关系、初始 submap 划分
- `HLoc/LoMa triangulation observations` 用于判断哪些帧适合作为稳定 anchor
- `Pi3/VGGT confidence` 用于推理后融合、拒绝、权重调整


## 2. 设计原则

### 2.1 不再让 submap 只由连续窗口决定

当前连续窗口形式：

```text
submap_0 = frames[0:32]
submap_1 = frames[29:61]
submap_2 = frames[58:90]
```

主要问题是约束图天然接近链式结构。

新方案中，submap/local batch 应该由：

```text
若干 shared/global keyframes + 若干待注册 non-keyframes
```

构成，允许 keyframes 分布在 batch 中间，也允许两个 submap 共享多个非尾部关键帧。


### 2.2 关键帧不是“特征点最多”的帧

单帧 LoMa keypoint 数量只能作为弱指标，因为 `loma_aachen` 默认 `max_keypoints=4096`，很多帧可能都会接近上限。

更可靠的关键帧质量指标是：

```text
跑完 HLoc triangulation 后，每张图最终关联到 3D point 的观测数量
```

这个指标比 raw keypoint count 更接近“该帧是否能作为稳定几何 anchor”。


### 2.3 LingBot pose 负责覆盖，HLoc observations 负责可靠性

关键帧选择不应只看 pose novelty，也不应只看几何观测数量。

推荐评分逻辑：

```text
score(i) =
  w_pose * pose_novelty_to_selected(i)
+ w_obs  * normalized_triangulated_observations(i)
+ w_conn * normalized_match_connectivity(i)
- w_iso  * isolation_penalty(i)
```

其中：

- `pose_novelty_to_selected(i)`：该帧到已选关键帧的最小 LingBot pose distance
- `triangulated_observations(i)`：HLoc 三角化后该图像参与的 3D 点观测数量
- `match_connectivity(i)`：该帧与 pose-near / loop candidate 帧的有效匹配支持
- `isolation_penalty(i)`：pose 上很新但和其他帧几何连接很弱的惩罚


## 3. 数据输入与中间产物

### 3.1 输入

必需输入：

- 图片目录
- `lingbot_transforms_json`
- Pi3X checkpoint 或 pretrained Pi3X
- 手工共享 intrinsics：`fx/fy/cx/cy`

可选输入：

- 现有 `poseinit_hloc` 输出目录
- 现有 HLoc `features.h5 / matches.h5 / sparse/0`


### 3.2 新增中间产物

建议在运行目录下生成：

```text
offline_keyframes/
  image_index.json
  lingbot_prior_poses.npz
  hloc_features.h5
  hloc_matches.h5
  hloc_sparse/0/
  image_observation_stats.json
  keyframes.json
  local_batches.json
```

关键文件说明：

- `image_index.json`
  - 记录全局 frame id、image path、basename 的一一对应
- `lingbot_prior_poses.npz`
  - OpenCV c2w poses，按全局 frame id 对齐
- `image_observation_stats.json`
  - 每张图的 HLoc triangulated observation count
  - 每张图的 total matches / median matches / connected neighbor count
- `keyframes.json`
  - 全局关键帧列表和选择理由
- `local_batches.json`
  - 每个 Pi3/VGGT local batch 的 frame ids
  - 标注 anchor keyframes、new keyframes、non-keyframes、shared keyframes


## 4. 阶段计划

## 阶段 A：整理全局帧索引与 LingBot prior pose

目标：

- 统一 `VGGT-SLAM` 图片列表与 `lingbot_transforms_json`
- 生成稳定的 `global_frame_id`
- 输出 OpenCV c2w prior poses

任务：

1. 复用或抽出 `Pi3Solver._load_lingbot_prior_poses()` 中的 pose 加载逻辑。
2. 按图片 basename 对齐 VGGT-SLAM 输入图片和 LingBot transforms。
3. 生成：

```text
global_frame_id -> image_path
global_frame_id -> basename
global_frame_id -> lingbot_opencv_c2w
```

验收：

- 所有输入图片均能在 `lingbot_transforms_json` 中找到对应 pose
- 输出每帧 prior translation / rotation 的基本统计
- 若 basename 缺失或重复，直接报错


## 阶段 B：运行 HLoc/LoMa 并统计 triangulated observations

目标：

- 利用 `poseinit_hloc` / HLoc LoMa 流程生成稀疏重建
- 统计每张图最终关联到 3D point 的观测数量

现有可复用代码：

- `poseinit_hloc/pipeline.py`
- `Hierarchical-Localization/hloc/extract_features.py`
- `Hierarchical-Localization/hloc/match_features.py`

当前 `poseinit_hloc` 已经支持：

```text
LingBot/Nerfstudio transforms
  -> pairs_from_poses()
  -> loma_aachen feature extraction
  -> loma matching
  -> HLoc triangulation
  -> sparse/0 + points3D.ply
```

任务：

1. 复用 `pairs_from_poses()` 的思路，基于 LingBot prior pose 生成匹配 pairs。
2. 用 LoMa 提取局部特征。
3. 用 LoMa matcher 匹配 pairs。
4. 跑 HLoc triangulation。
5. 读取 COLMAP `images.bin / points3D.bin`，统计：

```text
per_image_observation_count
per_image_registered_point_count
per_image_pair_match_count
per_image_connected_neighbors
```

注意：

- 初版可沿用 `skip_geometric_verification=True`，便于快速打通。
- 后续建议增加几何验证或基于 LingBot prior pose 的匹配过滤，否则动态物体和重复纹理可能污染 keyframe 质量评分。

验收：

- 生成 `image_observation_stats.json`
- 能按 observation count 排序输出 top/bottom frames
- 能识别弱纹理、模糊、连接孤立的低质量候选帧


## 阶段 C：全局 KeyFrame 选择

目标：

- 用 LingBot pose 保证空间覆盖
- 用 HLoc observations 保证 anchor 稳定性
- 得到全局 `keyframes.json`

推荐算法：

1. 构建候选池：
   - 丢弃 observation count 太低的帧
   - 丢弃与其他帧连接过少的孤立帧
   - 保留必要的首尾帧或用户指定帧
2. 使用 LingBot pose distance 做 FPS / greedy coverage：
   - 每次选择离已选 keyframes 最远的高质量候选
   - 对 pose novelty 和 observation count 做联合打分
3. 加 temporal bridge keyframes：
   - 避免关键帧图在时间上断裂
   - 避免只选回环附近或纹理丰富区域
4. 加 loop-support keyframes：
   - 对 LingBot pose 显示闭环接近的区域，保留至少若干可匹配 anchor

初版配置建议：

```text
target_keyframe_ratio: 0.15 ~ 0.30
min_observations: 使用全局 observation count 的 25% 分位数
min_pose_distance: 按场景尺度归一化后设置
max_temporal_gap_without_kf: 20 ~ 50 帧
loop_anchor_per_region: 1 ~ 3
```

输出 `keyframes.json` 示例：

```json
{
  "keyframes": [
    {
      "frame_id": 12,
      "image_name": "000012.png",
      "reason": ["pose_coverage", "high_observations"],
      "observation_count": 842,
      "pose_novelty": 1.37
    }
  ]
}
```

验收：

- 关键帧覆盖完整轨迹
- 关键帧 observation count 分布明显优于全体帧均值
- 相邻 keyframes 不过密，也不存在长时间/长距离空洞
- 回环区域存在 shared keyframe 候选


## 阶段 D：构建 shared-anchor local batches/submaps

目标：

- 生成适合 Pi3/VGGT 前馈模型推理的 local batch
- 每个 batch 有多个 shared keyframe anchors
- non-keyframes 依附到附近 anchors

推荐 batch 结构：

```text
local_batch_k =
  anchor_keyframes: 4 ~ 8
  target_keyframes: 1 ~ 4
  non_keyframes: 16 ~ 28
  total_frames <= submap_size
```

构建原则：

1. 每个 non-keyframe 分配给若干最近 keyframes：
   - LingBot pose near
   - HLoc match connected
   - 时间邻近
2. 每个 batch 与已有 batch 至少共享多个 keyframes：
   - 不再限制 overlap 是上一 batch 尾部
   - shared keyframes 可以位于 batch 任意位置
3. 控制 batch 内共视：
   - 不追求一个 batch 覆盖完整大场景
   - 避免把视觉关联很弱的帧强行塞进同一次前馈推理

输出 `local_batches.json` 示例：

```json
{
  "batches": [
    {
      "batch_id": 0,
      "frame_ids": [1, 3, 7, 12, 14, 15, 17, 39],
      "anchor_keyframes": [3, 14],
      "target_keyframes": [12, 39],
      "non_keyframes": [1, 7, 15, 17],
      "shared_with": []
    },
    {
      "batch_id": 1,
      "frame_ids": [2, 3, 5, 6, 10, 14, 39, 42],
      "anchor_keyframes": [3, 14, 39],
      "target_keyframes": [42],
      "non_keyframes": [2, 5, 6, 10],
      "shared_with": [0]
    }
  ]
}
```

验收：

- 每个 batch 内 frame 数量不超过模型限制
- 每个 batch 至少有 2 个 anchor keyframes，首个 batch 除外
- 每个 non-keyframe 至少被一个 batch 覆盖
- 相邻或相关 batch 有 shared keyframes


## 阶段 E：改造 VGGT-SLAM/Pi3Solver 的 submap 数据结构

目标：

- 支持任意全局 frame id 的 submap
- 支持非连续 frame ids
- 支持 shared keyframes 的跨 submap 对齐

当前需要打破的假设：

- submap local index 与时间顺序强绑定
- overlap 只能是当前 submap 前缀与上一 submap 后缀
- graph node id 依赖 `submap_id + local_frame_index`

建议新增结构：

```text
Submap.global_frame_ids: list[int]
Submap.local_to_global_frame: dict[int, int]
Submap.global_to_local_frame: dict[int, int]
Submap.anchor_keyframe_ids: list[int]
Submap.non_keyframe_ids: list[int]
```

图节点建议从：

```text
node_id = submap_id + local_index
```

逐步改为显式映射：

```text
(submap_id, local_index) -> graph_node_id
global_frame_id -> one_or_more graph_node_ids
```

初版可以继续允许 shared keyframe 在多个 submap 中有多个 graph node，但必须为这些重复节点添加 shared-KF alignment constraints。

验收：

- 非连续 frame ids 的 submap 能正常推理、存储、导出
- 任意 shared keyframe 能被识别
- dense 导出能保留原始 global frame id


## 阶段 F：shared keyframes 跨 submap 对齐

目标：

- 替代当前“prefix/suffix overlap”逻辑
- 用所有 shared keyframes 联合估计跨 submap Sim3 / SL(4) 桥接

输入：

```text
current_submap.shared_global_frame_ids
prior_submap.shared_global_frame_ids
```

流程：

1. 找出两个 submap 共享的 global frame ids。
2. 对每个 shared keyframe：
   - 在当前 submap 找 local index
   - 在 prior submap 找 local index
   - 用对应 local_points + camera_poses 变到各自 submap 坐标
3. 聚合所有 shared keyframes 的高置信 3D-3D 对应。
4. 估计 `current_submap -> prior/global` Sim3。
5. 根据 residual / scale / rotation / translation 做 reject 或降权。
6. 添加 graph between factor / alignment factor。

验收：

- 不依赖前缀/后缀 overlap
- shared keyframes 数量、有效点数量、Sim3 residual 都有日志
- shared keyframes 不足时可 fallback 到当前连续 overlap 逻辑


## 阶段 G：KeyFrame-first refinement

目标：

- 先稳定关键帧骨架，再处理 non-keyframes

第一版 refinement：

1. 只对 keyframe batches 做 Pi3 小窗口复推理。
2. 每次 batch 输入：

```text
current_kf + nearby_kfs + loop_support_kfs
```

3. 用 shared anchors 对齐后，只在质量更好时更新 keyframe pose / points。

质量判断：

- Pi3 conf 是否提升
- shared-KF Sim3 residual 是否下降
- 与 LingBot prior 的相对位姿是否离谱
- 与 HLoc connectivity 是否一致

验收：

- keyframe graph 稳定性优于连续 submap baseline
- loop 区域不会因为单条坏边明显拉坏全局


## 阶段 H：Non-KeyFrame refinement

目标：

- 普通帧不作为强全局骨架
- 普通帧依附最近 keyframe anchors 做局部 refinement

参考 AMB3R 的模式：

```text
views = nearby_keyframe_anchors + nearby_non_keyframe_window
```

流程：

1. 对每个 non-keyframe 找：
   - LingBot pose 最近 keyframes
   - HLoc match connected keyframes
   - 时间邻近 non-keyframes
2. 组成 local refine batch。
3. Pi3 推理。
4. 用 keyframe anchors 对齐。
5. 仅当 local conf 或几何一致性更好时融合。

验收：

- dense 点云覆盖不下降
- non-keyframes 不会向 graph 注入强不稳定约束
- 低置信 non-keyframes 可延后、跳过或重试


## 5. 推荐实施顺序

### 第一阶段：不动 Pi3 推理主链路，只做离线分析产物

1. 读取 `lingbot_transforms_json`，生成全局 pose index。
2. 跑 HLoc/LoMa triangulation。
3. 统计 per-image observation count。
4. 生成并可视化/打印 keyframe candidates。

验收目标：

- 能输出合理的 `keyframes.json`
- 能确认高 observation frames 是否符合直觉


### 第二阶段：支持 shared-keyframe overlap 对齐

1. 保持 submap 仍由外部列表传入。
2. 手工或由 `local_batches.json` 构造非连续 submap。
3. 修改 `Submap` 保存 global frame ids。
4. 修改 overlap 检测为 shared global frame ids。
5. 联合 shared keyframes 做 Sim3。

验收目标：

- 同一组图片下，对比：
  - 连续 submap baseline
  - shared-keyframe submap
- dense 重影应减少或至少日志能解释失败原因


### 第三阶段：全局 keyframe-driven batch 构建

1. 根据 `keyframes.json` 自动生成 `local_batches.json`。
2. main 流程支持从 batch plan 读取 submap。
3. 添加 batch 质量日志。

验收目标：

- 不再依赖固定 `submap_size + overlapping_window_size` 顺序窗口
- 所有 non-keyframes 至少被覆盖一次
- 每个 batch 的 anchor 质量可追踪


### 第四阶段：KeyFrame-first / Non-KeyFrame refinement

1. 先做 keyframes 的局部复推理和图约束更新。
2. 再做 non-keyframes 的 anchor-based refinement。
3. 加入更保守的 reject / weighting。

验收目标：

- keyframe-only 点云和 dense 全量点云均可导出
- keyframe graph 比连续窗口更稳定
- non-keyframe refinement 不破坏 keyframe 骨架


## 6. 风险与注意事项

### 6.1 LingBot prior pose 不是最终真值

LingBot pose 可用于：

- 候选生成
- pose coverage
- loop 粗筛
- batch 构建

但不应无条件作为最终强约束。


### 6.2 HLoc observations 可能偏向纹理丰富区域

高 observation count 往往偏向：

- 纹理多的墙面/物体
- 光照好、清晰的图像

低 observation count 不一定代表 pose 覆盖不重要。

因此需要 temporal bridge / pose coverage 兜底，避免关键帧全集只覆盖纹理丰富区域。


### 6.3 Batch 内跨度过大可能让前馈模型退化

目标不是“每个 submap 尽量覆盖完整场景”，而是：

```text
每个 submap 有稳定 shared anchors，且待注册帧与 anchors 有视觉/几何关联
```


### 6.4 graph node id 需要谨慎改造

当前 `submap_id + local_index` 的隐式 node id 方案不适合非连续/共享 keyframe submap。

这部分改造应单独做，避免和 keyframe 选择、HLoc 统计混在一起。


## 7. 初版最小可行方案

MVP 推荐范围：

1. 提供 `--offline_keyframe_plan_dir`
2. 输入：
   - image folder
   - `--lingbot_transforms_json`
3. 跑或复用 HLoc/LoMa triangulation
4. 输出：
   - `image_observation_stats.json`
   - `keyframes.json`
   - `local_batches.json`
5. 暂不改变 Pi3Solver 主流程

MVP 完成后，再进入第二步：

1. 新增 `--submap_plan_json local_batches.json`
2. main 按 plan 中的 frame ids 构造 submap
3. Pi3Solver 支持 shared global frame ids 对齐


## 8. 当前结论

在离线全图已知，并且 LingBot prior pose 已被验证对回环候选有效的前提下，推荐路线是：

```text
LingBot pose 负责全局覆盖
HLoc/LoMa triangulation observation count 负责 anchor 质量
Pi3/VGGT confidence 负责推理后融合和 reject
```

这比继续沿用顺序连续 submap 更适合大场景和全局一致性问题，也比单纯按 LoMa keypoint 数量选关键帧更稳。

第一步应先实现离线 keyframe/submap plan 生成，不急于重写 Pi3Solver。等 `keyframes.json` 与 `local_batches.json` 的质量确认后，再改 shared-anchor submap 对齐。
