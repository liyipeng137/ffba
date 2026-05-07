# 离线 KeyFrame / Shared-Anchor Submap 当前进展

更新时间：2026-05-07

## 1. 当前目标

当前阶段目标是验证：

- 离线场景下，是否可以先全局选择 KeyFrames
- 再构建 shared-anchor local batches
- 用 shared keyframes 替代原始连续窗口的 prefix/suffix overlap
- 从而改善 Pi3X 版 VGGT-SLAM 的跨 submap dense 重影问题

暂不做：

- KeyFrame refinement
- Non-KeyFrame refinement
- BA / full global refinement


## 2. 已完成实现

### 2.1 离线 planner

已新增：

- `VGGT-SLAM/vggt_slam/offline_planner.py`

能力：

- 读取 `lingbot_transforms_json`
- 读取 HLoc/COLMAP `sparse/0/images.bin` 和 `points3D.bin`
- 统计每张图最终关联到 3D point 的 observation count
- 基于 LingBot pose coverage + HLoc observation quality 选择全局 KeyFrames
- 生成：
  - `image_index.json`
  - `image_observation_stats.json`
  - `keyframes.json`
  - `local_batches.json`


### 2.2 main.py 支持离线 submap plan

已新增参数：

- `--generate_offline_submap_plan`
- `--offline_plan_dir`
- `--hloc_sparse_dir`
- `--hloc_features_h5`
- `--submap_plan_json`
- `--offline_keyframe_ratio`
- `--offline_min_observation_quantile`
- `--offline_max_temporal_gap`
- `--offline_anchor_count`
- `--offline_min_shared_anchors`

在 `--submap_plan_json` 模式下：

- 不再使用原始光流筛选
- 不再使用原始连续窗口切分
- `overlapping_window_size` 基本无效
- submap 完全由 `local_batches.json` 的 `frame_ids` 决定


### 2.3 Submap 支持全局 frame id

`Submap` 已新增：

- `global_frame_ids`
- `anchor_keyframe_ids`
- `target_keyframe_ids`
- `non_keyframe_ids`
- shared global frame 查询接口

当前仍保持原有 graph node 设计：

```text
node_id = submap_id + local_index
```

shared keyframe 在不同 submap 中仍是不同 graph node，通过 shared-anchor between factors 连接。


### 2.4 Pi3Solver 支持 shared-anchor 对齐

当前对齐逻辑：

1. 当前 submap 与历史 submaps 比较 shared global frame ids
2. 若 shared 数量达到 `offline_min_shared_anchors`，使用 shared anchors 联合估计 Sim3
3. 若不足，则 fallback 到原始 prefix/suffix overlap

已添加 debug：

- `Pi3 per-anchor alignment residuals`
- `Pi3 per-anchor transform disagreement`

用于判断 shared anchors 是否能由同一个整体 Sim3 解释。


### 2.5 临时 keyframe/anchor 点云导出

已新增临时导出：

- `offline/keyframe_anchor_points.ply`

内容：

- 只导出 `anchor_keyframes + target_keyframes`
- 按 `global_frame_id` 去重
- 用于对比 full dense 点云和 keyframe-only 点云质量差异


## 3. 当前 local batch 策略

当前较合理的 batch 形式是：

```text
历史 shared anchors + 当前连续 target window
```

例如当前 `local_batches_v3.json` 中：

```text
batch0 target: 0-25
batch1 target: 26-51, shared: 12/18/20
batch2 target: 52-77, shared: 26/32/50
batch3 target: 78-103, shared: 56/68/70
...
```

这一版已经避免了早期问题：

- 不再固定共享 `0/5`
- 不再在 batch0 中突兀加入未来 keyframe，例如 `87`
- target window 连续覆盖全序列
- shared anchors 随时间向后滚动


## 4. 实验现象

### 4.1 scale 固定为 1 没有明显改善

曾临时强制跨 submap 对齐 scale 为 `1.0`。

观察：

- dense 点云效果变化不明显

结论：

- 当前主问题大概率不是统一 scale drift


### 4.2 加强点过滤改善 residual，但 dense 没明显改善

尝试：

- `depth_edge(local_depth, rtol=0.01)`
- 对 shared-anchor 对齐点使用更严格 `good_mask`

结果：

- shared-anchor residual 明显下降
- 但 dense 点云视觉效果没有明显改善

结论：

- 低质量/边缘点确实会污染 residual
- 但不是当前 full dense 重影的主因


### 4.3 per-anchor transform disagreement 不算特别大

debug 显示：

- 多数 shared anchor 单独估计的 transform 与 joint transform 差异不大
- rotation delta 多数在小角度范围
- scale 差异通常也较小

结论：

- 当前没有强证据说明必须使用 affine/projective 形变
- 不建议优先引入复杂形变


### 4.4 keyframe/anchor-only 点云明显好于 full dense

导出 `offline/keyframe_anchor_points.ply` 后观察：

- keyframe/anchor 点云仍有少量错位
- 但明显没有 full dense 点云那么严重

阶段性结论：

```text
shared-anchor / keyframe 骨架本身相对更稳定；
full dense 重影主要来自 non-keyframes / 普通帧点云贡献。
```


## 5. 当前判断

目前更可能的问题链路是：

```text
shared anchors 能把 submap 大体放到正确位置
    ↓
non-keyframes 依赖 Pi3 batch 内部相对 pose / 局部点图传播
    ↓
普通帧局部误差在 full dense 中大量叠加
    ↓
dense 点云表现为明显重影或一侧对齐、一侧错位
```

因此，当前问题不太像：

- 单纯 scale 错
- shared-anchor Sim3 完全失败
- 必须马上上任意形变

更像：

- non-keyframe dense 贡献过强
- batch 内远离 shared anchors 的普通帧缺少约束
- 普通帧点云需要更严格筛选、降权或后续 refinement


## 6. 当前梳理出的两套后续方案

### 方案 1：继续基于当前 VGGT-SLAM/Pi3Solver 框架

目标：

- 尽量保留当前 VGGT-SLAM 的速度优势
- 继续使用 submap + graph optimize 主流程
- 只在 batch 构建、shared-anchor 对齐、dense 导出策略上做轻量改造

核心 batch 形式从：

```text
prev_shared + target_window
```

扩展为：

```text
prev_shared + target_window + next_shared
```

其中：

- `prev_shared`：来自上一 batch 的 shared keyframes，用于当前 batch 与上一 batch 对齐
- `target_window`：当前实际推进/消费的连续帧窗口
- `next_shared`：下一个窗口中的 keyframes，提前放入当前 batch，用于约束当前 batch 右侧

建议第一版严格按时间顺序对齐：

```text
batch_i 对齐 batch_{i-1}
batch_{i+1} 对齐 batch_i
```

不默认从所有历史 submaps 中找 shared 最多的 prior。跨历史 batch 的对齐后续可作为 loop/cross-link 单独处理。

此方案仍保留：

- Pi3Solver
- Submap
- PoseGraph / SL(4)
- shared-anchor Sim3
- graph optimize

但需要加强：

- shared-anchor residual-based reject
- keyframe / non-keyframe 分层 dense 导出
- non-keyframe 更严格过滤或降采样

优点：

- 推理次数基本不增加
- 工程改动较小
- 便于和当前 baseline 对比

风险：

- non-keyframes 仍在 submap 内依赖 Pi3 相对 pose 传播
- 若 non-keyframe 局部漂移明显，full dense 仍可能重影


### 方案 2：分阶段 KeyFrame skeleton + BAE + Non-KeyFrame 补点

目标：

- 先得到更稳定的全局 KeyFrame-only skeleton
- 再把普通帧作为补充，而不是让普通帧参与污染全局结构

阶段 1：

```text
全局 KeyFrame-only reconstruction
    -> 复用当前 VGGT-SLAM/Pi3Solver 的 shared-anchor submap 思路
    -> 只处理 keyframes
    -> 输出 keyframe poses + keyframe dense points
```

随后：

```text
BAE 优化 keyframe poses / dense skeleton
```

阶段 2：

```text
固定或强约束 optimized keyframe skeleton
    -> 对 non-keyframes 做局部补点
```

non-KF 补点不建议继续走当前 VGGT-SLAM 的 PoseGraph/submap 入图逻辑，而更适合做一个新模块：

```text
anchor keyframes + non-keyframe window
    -> Pi3 前向
    -> 用 Pi3 输出的 anchor pointmaps 和 optimized KF dense points 做 3D-3D 对齐
    -> 只把 non-KF points/poses 变换到 fixed skeleton
    -> 高置信才导出
```

这一阶段可以借鉴 AMB3R 的思想：

- anchors 负责对齐
- non-KF 不反向修改 keyframe skeleton
- 多个 anchor 组合尝试时，保留置信度/对齐 residual 最好的结果
- 低质量 non-KF 跳过、延后或严格过滤

此方案中，SL(4) 不再是第二阶段的核心必需项。SL(4) 可保留用于：

- 第一阶段 KeyFrame-only skeleton baseline
- 可选 pose graph refinement
- 与现有 VGGT-SLAM 路线对比

优点：

- 更符合当前实验结论：keyframe/anchor 点云明显稳定于 full dense
- non-KF 不会污染全局骨架
- BAE 优化规模更小，仅作用在 keyframes

风险：

- 工程改造较大
- 需要新增 fixed-skeleton non-KF filler
- non-KF 补点策略需要单独设计和评估


## 7. 当前阶段判断

目前已有实验更支持以下判断：

```text
keyframe/shared-anchor skeleton 相对可靠；
full dense 的主要问题来自 non-keyframes 的大量点云贡献。
```

因此短期可以优先走方案 1，快速验证：

- `prev_shared + target_window + next_shared`
- non-KF dense 分层过滤
- shared-anchor reject

如果方案 1 仍无法明显改善 full dense，则更应转向方案 2：

```text
KeyFrame-only reconstruction
    -> BAE 优化 skeleton
    -> fixed skeleton 下补 non-KF
```


## 8. 下一步建议

优先级建议：

1. 先按方案 1 改 batch planner：加入 `next_shared`
2. 第一版对齐改为严格相邻 batch 顺序对齐
3. 尝试 `keyframes 全量保留 + non-keyframes 更严格过滤`
4. 尝试只导出靠近 anchors 的 non-keyframes
5. 若方案 1 仍不够，再进入方案 2 的 KeyFrame-only skeleton + BAE 路线
6. 暂不优先考虑 affine/projective 形变

当前最有价值的验证方向是：

```text
降低 non-keyframe 对 dense 的破坏，同时保持 keyframe/anchor 骨架稳定。
```
