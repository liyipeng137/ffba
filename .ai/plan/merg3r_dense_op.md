# MERG3R Dense Point Correction Plan

更新时间：2026-05-12

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
