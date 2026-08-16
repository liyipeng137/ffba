# MERG3R Dense Point Correction Plan

更新时间：2026-06-21（实测进展见下；原始设计 2026-05-12 见第 1 节起）

---

## 实测进展（2026-06-21）：GlueMap+BAE pipeline 上的诊断与 per-frame 尺度矫正

> 这是对本计划"诊断阶段 + 方案 A"的**实际执行与结论**。关联：[`merg3r_gluemap_bae_optimizations.md`](merg3r_gluemap_bae_optimizations.md)、[`lingbot_bae_depth_pipeline_plan.md`](lingbot_bae_depth_pipeline_plan.md)。
>
> **上下文差异**：本次工具走的是 **GlueMap+BAE pipeline 的落盘产物**（`refined_gluemap_aba/`、`pred_depth/`、`coarse/`），不是本计划原文里 main.py 的 in-memory `final_predictions` / `local_points`。问题本质相同（sparse BA anchors vs dense FF depth 的逐帧尺度），结论可迁移。

### 工具（已落地）

- `scripts/diagnose_depth_pose_consistency.py`：读 refined COLMAP + `pred_depth` + 可选 coarse，自包含解析 COLMAP（.bin/.txt，不依赖 pycolmap）。对每个 BA 点观测算 `r = z_ba / d_ff`，输出逐帧 ratio 统计、帧内 CV、空间平面 R²、错层 world spread、Sim3(coarse→refined) 形变，并给出 `global_scalar / per_frame_scalar / per_frame_smooth_field / model_based` 建议。
- `scripts/apply_per_frame_scale.py`：`--mode {scalar,affine,inv_affine}`。每帧从锚点鲁棒拟合（scalar=median(z/d)；affine=z~a·d+b；inv_affine=1/z~a·(1/d)+b），带 clamp / 少锚点回退 / affine 病态（深度跨度不足）回退 scalar / 正性保护；打印 affine-vs-scalar 残差对比；输出三格式 depth（对齐 `MERG3R/algos/utils.py::export_prediction_depth_maps`：`depth_npy/ depth_u16/ depth_vis/`）。

### 诊断结果（原始 pred_depth；390 帧，refined = bae huber δ=1.0/gate=1.0）

| 指标 | 值 | 含义 |
|---|---|---|
| global median ratio | 1.044 | FF depth 需 ×1.044 对上 refined |
| frame-median ratio CV | 2.43% | 帧间尺度变异（>2% → 非纯全局） |
| median in-frame CV | 2.87% | 帧内尺度很紧 → 单标量够 |
| median plane R² | 0.025 | 无空间结构场 |
| 错层 spread (rel depth) | median 2.3% / p90 7.0% | 实测分层量级 |
| Sim3 scale / 形变残差 | 1.0487 / median 3.2%、p90 7.5% | 全局尺度 + 非刚性形变 |

交叉校验：global median ratio 1.044 ≈ Sim3 scale 1.0487 → 诊断自洽。推荐 **per_frame_scalar**。

### 实验与结论

1. **per-frame scalar：成功，保留。** 对 scaled depth 重跑 diagnose：错层 median **2.3% → 1.28%（−44%）**，frame-median ratio CV **2.43% → ~0**，in-frame CV 不变（2.87%，scalar 本就不动帧内），p90 7.0% → 6.7%（几乎不变）。
2. **affine：无效。** anchor 残差 rel CV `2.87% → 2.77%`（仅 9% 帧 >20% 改善）→ 帧内残差**不是随深度变化的 affine 偏置**，是结构化不了的噪声。
3. **inv_affine：报告里的 17% 是脚本 residual bug**（`pred` 用了 `1/(a+b·d)`，应为 `d/(a+b·d)`，已修；`apply_model` 一直是对的，所以**生成的 depth 有效**，实际 ≈ scalar）。

### 修订原计划的结论

- **per-frame scalar 是正确且充分的"尺度"修法。** 本计划"方案 A：per-frame inverse-depth affine"经实测**不需要 affine/shift**——锚点处无深度结构，affine/inv-affine 都不优于纯 scalar。尺度这条线**收口**。
- **残余错层（median 1.28% / p90 6.7%）= 帧内逐像素深度误差地板**（无结构噪声）。且诊断指标基于**稀疏锚点**，所以 TSDF 里"部分视角更糟"几乎都在**无纹理 / 无锚点区**：per-frame 尺度由纹理区锚点定标，被外推到锚点覆盖不到的深度时会更偏。
- **剩下是 depth 质量问题，不是 pose/尺度问题。** 因此原计划"方案 B：sparse residual field""方案 C：cross-view"优先级下降——问题不在"传播 sparse residual"，而在锚点**根本覆盖不到**的区域。真正的下一杠杆是**提升无锚点区的 depth 质量**：用 RGB + 现已逐帧尺度对齐的 dense depth 作为 lingbot 的 `depth_in` 做精化（见 `lingbot_bae_depth_pipeline_plan.md`）。

### 下一步（三选一）

1. **接受现状**：median 错层 1.28%（2m 处 ~2.6cm）对多数用途够好。
2. **定位坏帧（便宜）**：把变糟视角对上 `per_frame_scale.csv` 的 `status / n_anchors`；疑点是 `fallback_clamp`（frame_000038/043）、低锚点帧、及诊断 `worst_frames`（38/34/33/7/31…）。若是少数帧 → 单独修（如 38/43 不一刀切回全局，用平滑后的邻居尺度）。
3. **治本（治尾巴）= lingbot 稠密精化**：把尺度对齐后的 dense depth 喂 lingbot，用 RGB 修无纹理区逐像素误差——这才是能动 p90 / worse-view 的杠杆。

---

## 1. 背景与目标

当前 MERG3R + Pi3X 主流程已经具备：

```text
Pi3X local_points
+ MERG3R alignment / global BA / optional BAE 后的 final poses
-> export_dense_local_point_map_ply()
-> dense_model_points.ply
```

这一步已经是基于优化后 pose 对 `local_points` 重新放置到世界坐标，但它仍然只是 direct merge：

- 不移动或修正每帧 dense local geometry。
- 不利用 BA/BAE sparse points 反向约束 dense。
- 不做多视角一致性更新。
- 不做 voxel / TSDF-like 聚合，只做 confidence 过滤、采样和 PLY 写出。

后续 dense correction 的目标是：

```text
使用 BA/BAE 后更可靠的 sparse points / tracks / poses，
对 Pi3X dense local_points 做可解释的几何校正，
再用 final poses 重新融合输出 dense 点云。
```

明确暂不采用：

- 训练或微调 GGPT / Point Transformer / learned refinement。
- Keyframe skeleton + non-KF filler 路线。

Voxel / TSDF-like fusion 可以作为最终聚合阶段，但不作为 dense 点“收敛”的核心手段。

## 2. 当前可用输入

MERG3R global BA / BAE 后可用：

- `final_predictions['extrinsic']`
  - final world-to-camera pose。
- `final_predictions['intrinsic']`
  - shared PINHOLE 或 per-frame intrinsics。
- `final_predictions['local_points']`
  - Pi3X camera-local dense point map，shape 约为 `[N,H,W,3]`。
- `final_predictions['depth']`
  - 当前等价于 `local_points[..., 2]`。
- `final_predictions['depth_conf']`
  - dense confidence。
- `final_predictions['points']`
  - global BA / BAE 后 sparse 3D points。
- `track`
  - per-frame sparse 2D observations。
- `points_id`
  - 每个 2D observation 对应的 sparse point id。
- `valid_track_mask`
  - global BA reprojection filtering 后的有效 observation mask。

这些数据足够构建每帧 sparse depth anchors：

```text
for each valid observation (frame i, pixel u/v, point id p):
    X_sparse_world = final_predictions['points'][p]
    X_sparse_cam_i = w2c_i @ X_sparse_world
    z_sparse_i(u,v) = X_sparse_cam_i.z

    X_dense_cam_i = local_points[i, v, u]
    z_dense_i(u,v) = X_dense_cam_i.z
```

核心约束来自：

```text
z_sparse / X_sparse_cam_i
should agree with
z_dense / X_dense_cam_i
```

## 3. 总体实施顺序

建议按以下顺序推进：

1. **诊断导出**
   - 先确认 sparse anchors 与 dense local_points 的 residual 分布。
2. **方案 A：Per-frame inverse-depth affine**
   - 每帧统一 depth scale/bias 校正。
   - 最稳的第一版 baseline。
3. **方案 B：Affine + sparse residual field**
   - 每个 dense 点有不同 correction strength。
   - 通过 sparse residual 平滑传播到 dense。
4. **方案 C：Cross-view consistency refinement**
   - 利用多视角投影一致性做过滤或迭代更新。
5. **方案 D：Final voxel / TSDF-like fusion**
   - 在 dense correction 后做最终聚合、去重和降噪。

## 4. 诊断阶段

### 4.1 构建 sparse-dense residual

对每个 valid track observation 计算：

```text
z_dense = local_points[i, v, u, 2]
X_sparse_cam = w2c_i @ points[p]
z_sparse = X_sparse_cam[2]

res_z = z_sparse - z_dense
res_invz = 1 / z_sparse - 1 / z_dense
res_xyz = X_sparse_cam - local_points[i, v, u]
```

建议统计：

- 每帧 valid anchor 数量。
- 每帧 `res_z / res_invz / res_xyz` 的 median、MAD、p90、p95。
- residual 与 `depth_conf` 的关系。
- residual 与 depth edge 的关系。
- residual 在图像上的热力图。

### 4.2 推荐导出

输出到 `output_dir/dense_debug/`：

- `sparse_anchor_points_cam_stats.json`
- `sparse_dense_residual_stats.json`
- `sparse_dense_residual_heatmap/*.png`
- `dense_before_correction.ply`
- `sparse_ba_points.ply`
- `sparse_anchor_overlay.ply`

目的：

```text
先判断 sparse BA/BAE anchors 是否真的能解释 dense 偏差。
如果 residual 噪声极大或空间分布无规律，后续 correction 容易拉坏 dense。
```

## 5. 方案 A：Per-frame inverse-depth affine

### 5.1 思路

对每帧拟合一个 robust inverse-depth affine：

```text
inv_z_sparse = a_i * inv_z_dense + b_i
```

然后应用到整帧 dense：

```text
inv_z_corr(u,v) = a_i * inv_z_dense(u,v) + b_i
z_corr(u,v) = 1 / inv_z_corr(u,v)
```

保持 Pi3X ray 方向不变：

```text
ray_xy = local_points[..., :2] / local_points[..., 2:3]
local_points_corr = concat(ray_xy * z_corr, z_corr)
```

### 5.2 为什么用 inverse depth

相比直接拟合：

```text
z_sparse = a * z_dense + b
```

inverse depth 对近处结构更敏感，通常也更适合 monocular / feed-forward depth 的尺度偏差校正。

### 5.3 拟合策略

建议第一版：

```text
1. 使用 valid_track_mask 过滤 anchors。
2. 过滤 z_dense <= eps 或 z_sparse <= eps。
3. 使用 robust quantile / MAD 去除 extreme residual。
4. 最小二乘拟合 inv_z_sparse = a * inv_z_dense + b。
5. 对 a/b 做范围限制。
```

保护条件：

- anchor 数小于 `min_anchors_per_frame` 时跳过该帧。
- `a` 超出合理范围时跳过或 clamp。
- 修正后 depth 非正或非 finite 的点回退到原值。
- 可用 `alpha_frame` 做 blend：

```text
inv_z_final = (1-alpha) * inv_z_dense + alpha * inv_z_corr
```

### 5.4 优点

- 实现简单。
- 可解释。
- 不需要训练。
- 不会改变每像素 ray 拓扑。
- 适合作为 dense correction baseline。

### 5.5 局限

- 只能修 per-frame global depth scale/bias。
- 不能修局部形变。
- 如果 sparse anchors 分布集中，可能对整帧泛化不稳。

## 6. 方案 B：Affine + sparse residual field

### 6.1 思路

在方案 A 后，计算 sparse residual：

```text
r_k = inv_z_sparse_k - (a_i * inv_z_dense_k + b_i)
```

将 sparse residual 传播到整张图：

```text
r_i(u,v) = sum_k w_k(u,v) * r_k / sum_k w_k(u,v)
```

最终：

```text
inv_z_corr(u,v) = a_i * inv_z_dense(u,v) + b_i + r_i(u,v)
```

这相当于非学习版 GGPT：

```text
不是用 Point Transformer 学 residual propagation，
而是用明确的几何/图像权重传播 sparse correction。
```

### 6.2 权重设计

第一版可以只用 2D RBF：

```text
w_k = exp(-||pixel - pixel_k||^2 / sigma_2d^2)
```

增强版加入：

- depth proximity：

```text
exp(-|inv_z - inv_z_k| / sigma_depth)
```

- RGB / feature similarity：

```text
exp(-||rgb - rgb_k||^2 / sigma_rgb^2)
```

- anchor confidence：

```text
w_k *= sparse_conf_k
```

- depth edge gate：

```text
edge 附近降低传播，避免跨物体边界扩散。
```

### 6.3 每点不同的收敛强度

最终建议使用 per-pixel gate：

```text
inv_z_final(u,v) =
    (1 - alpha(u,v)) * inv_z_dense(u,v)
  + alpha(u,v)       * inv_z_corr(u,v)
```

`alpha(u,v)` 可由以下因素决定：

- 到最近 sparse anchor 的距离。
- 周围 anchor 数量。
- anchor residual 一致性。
- Pi3X `depth_conf`。
- 是否处在 depth edge。
- 多视角一致性支持。

### 6.4 优点

- 可以修局部偏差。
- 每个 dense 点 correction strength 不同。
- 仍然不需要训练。
- 行为可解释，便于调参。

### 6.5 风险

- sparse anchors 太稀疏时可能过拟合局部。
- BA/BAE outlier 会污染 residual field。
- 传播半径过大容易跨物体边界拉坏 geometry。

必须保留：

- residual clip。
- min local anchors。
- max correction ratio。
- depth edge / confidence gate。

## 7. 方案 C：Cross-view consistency refinement

### 7.1 思路

用 corrected local_points 和 final poses 做多视角一致性：

```text
X_i_world = c2w_i @ local_points_corr_i(u,v)
project X_i_world into frame j
compare z_projected with z_corr_j(projected pixel)
```

如果一致：

- 增加该点 confidence。
- 可参与 final fusion。

如果不一致：

- 降权或过滤。
- 或用一致视角 depth 反向更新当前点。

### 7.2 邻接帧选择

建议先用已有 tracking graph / temporal neighbors：

- video：`steps=[1,2,3,5]`
- graph：LightGlue top-k neighbor pairs

避免全帧两两投影。

### 7.3 用途

第一阶段建议只用于 filtering / scoring：

```text
correction 后不一致的 dense 点不进入最终 PLY。
```

第二阶段再考虑迭代更新：

```text
sparse anchor correction
-> cross-view consistency update
-> sparse anchor correction
```

### 7.4 风险
 
- 需要处理遮挡。
- 需要 z-buffer。
- 需要投影边界和采样插值。
- 计算成本较高。

建议放在方案 A/B 验证有效后再做。

## 8. 方案 D：Final voxel / TSDF-like fusion

### 8.1 定位

Voxel / TSDF-like fusion 不作为 dense correction 的核心，而作为 correction 后的最终聚合：

```text
corrected local_points
+ final poses
-> world dense points
-> voxel aggregation
-> final dense_fused.ply
```

### 8.2 Voxel aggregation baseline

每个 voxel 内：

- confidence weighted average xyz/rgb。
- 保留 support count。
- residual / variance 过大的 voxel 丢弃或拆分。
- 每 voxel 限制最大点数。

可选 trimmed mean：

```text
丢弃 voxel 内距离 median 太远的点，再平均。
```

### 8.3 输出

- `dense_corrected_raw.ply`
- `dense_corrected_voxel.ply`
- `voxel_stats.json`

## 9. 推荐第一版实现

建议先实现：

```text
--dense_refine none|sparse_invdepth_affine
```

新增模块建议：

```text
MERG3R/algos/dense_refine.py
```

核心函数：

```python
def refine_dense_local_points_with_sparse_anchors(
    final_predictions,
    track,
    points_id,
    valid_track_mask=None,
    method="sparse_invdepth_affine",
    min_anchors_per_frame=32,
    max_residual_mad=5.0,
    alpha=1.0,
):
    ...
    return refined_local_points, stats
```

在 `main.py` 中的位置：

```text
global BA / optional BAE
-> dense_refine local_points
-> export_dense_local_point_map_ply()
-> write_recon_to_colmap()
```

注意：

- correction 应作用在 `final_predictions['local_points']`。
- 不应直接修改 sparse BA/BAE points。
- `final_predictions['depth']` 应同步更新为 refined `local_points[..., 2]`。
- dense export 继续使用 `local_points + final_predictions['extrinsic']`。

## 10. 验证标准

每个 dense correction 方案都至少输出：

- correction 前后 sparse anchor residual：
  - median / MAD / p90 / p95。
- 每帧使用 anchor 数量。
- 每帧 affine 参数或 residual field 统计。
- correction 前后 dense PLY。
- correction 前后投影深度图，可选。

重点观察：

- sparse anchor residual 是否下降。
- dense 点云重影是否减少。
- 是否引入局部撕裂、空洞、过度拉伸。
- non-anchor 区域是否保持稳定。

## 11. 暂缓项

以下方向暂不作为近期主线：

- GGPT / Point Transformer learned refinement。
- Keyframe skeleton + non-KF filler。
- 全量 dense 3D 点直接优化。
- 对每个 dense point 自由 3D 位移优化。

原因：

- learned refinement 依赖训练权重和数据分布适配。
- keyframe skeleton 是上一阶段方向，当前 MERG3R 主线暂不恢复。
- 全量 dense 点优化计算量过大，且容易过拟合/拉坏局部结构。
- 沿 ray 修 depth 比直接 3D 位移更稳定、更可解释。
