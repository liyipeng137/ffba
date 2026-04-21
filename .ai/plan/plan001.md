# plan001: LingBot-Map Dense Point Pose-Delta Correction 验证计划

## 1. 目标

验证以下流程是否能跑通，并判断 BAE 优化后的 sparse pose correction 是否能有效矫正 LingBot-Map 的 dense world points：

```text
LingBot-Map
  -> 原始 pose + dense world points
  -> HLoc 导入 LingBot pose/K 作为 reference model
  -> LoMa 特征匹配
  -> triangulation 导出 COLMAP sparse model
  -> BAE 优化 COLMAP sparse poses/points
  -> 用优化前后 pose 对 LingBot dense world points 做 pose-delta 矫正
```

第一版只验证几何闭环，不引入 LingBot-Depth refine。LingBot-Depth 放到下一阶段，在确认 corrected dense points / projected depth 更合理后再接入。

## 2. 核心思路

BAE 不直接优化 LingBot-Map 的稠密点。BAE 只处理 HLoc/LoMa 三角化得到的 COLMAP sparse model：

- sparse 2D observations
- sparse 3D points
- camera poses
- intrinsics

LingBot-Map 的 dense world points 保留为稠密几何来源。BAE 优化后的 pose 只作为每帧 dense point 的全局位姿校正信号。

对第 `i` 帧：

```text
C_i       = LingBot 原始 c2w
W_i       = inverse(C_i)
C'_i      = BAE 优化后的 c2w
X_i(u,v)  = LingBot 原始 dense world point
X'_i(u,v) = pose-delta 矫正后的 dense world point
```

矫正公式：

```text
X'_i(u,v) = C'_i * W_i * X_i(u,v)
```

等价理解：

```text
X_cam_i(u,v)  = W_i * X_i(u,v)
X'_world_i(u,v) = C'_i * X_cam_i(u,v)
```

这一步会改变 dense point 在世界坐标中的位置，但不会改变该点相对于来源帧自身相机的深度。它的价值来自跨帧融合和从优化后相机视角重新投影。

## 3. 数据与坐标约定

### 3.1 LingBot-Map 输出

需要保存：

```text
lingbot/
  images/                  # 输入 RGB，顺序固定
  intrinsics.npy            # [N, 3, 3]
  w2c.npy                   # [N, 4, 4], OpenCV world-to-camera
  c2w.npy                   # [N, 4, 4], OpenCV camera-to-world
  world_points.npy          # [N, H, W, 3]
  world_points_conf.npy     # [N, H, W]
  depth.npy                 # optional, [N, H, W]
```

注意：LingBot-Map `demo.py::postprocess()` 里 `predictions["extrinsic"]` 被 inverse 后是 `c2w`。HLoc/COLMAP/BAE 通常需要 `w2c`，因此必须显式保存两份，避免继续使用含糊的 `extrinsic` 命名。

### 3.2 COLMAP reference model

用 LingBot-Map 的 `K + w2c` 写出 COLMAP reference model：

```text
colmap_reference/
  cameras.bin 或 cameras.txt
  images.bin 或 images.txt
  points3D.bin 或 points3D.txt
```

第一版 reference model 可以没有有效 points3D，重点是相机和图像 pose 要正确。

约束：

- 优先使用 `PINHOLE` camera model。
- 如果 LingBot 每帧 K 不完全相同，第一版先使用 shared K：取第一帧或中位数。
- 图像名称必须和 HLoc feature/match 文件中的 name 完全一致。

### 3.3 HLoc + LoMa triangulation 输出

HLoc 使用 LingBot reference model 和 LoMa matches 进行 triangulation：

```text
hloc_outputs/
  features.h5
  matches.h5
  pairs.txt
  sparse_model/
    cameras.bin
    images.bin
    points3D.bin
```

这里的 sparse model 应该和 LingBot reference pose 处于同一世界坐标系附近。后续 BAE 优化得到的 pose 可以直接和 LingBot 原始 pose 做 delta。

如果改成完整 COLMAP reconstruction，而不是 reference-pose triangulation，则输出 pose 会有任意 Sim(3) gauge，需要额外和 LingBot pose 对齐。plan001 暂不走这条路。

### 3.4 BAE 输出

BAE 读取 HLoc triangulation 产物，输出：

```text
bae/
  cameras_optimized.txt 或 .npy
  images_optimized.txt 或 optimized_w2c.npy
  points3D_optimized.txt
  report.json
```

需要记录：

- initial reprojection loss
- final reprojection loss
- 每帧 pose delta 范数
- 每点 update norm 统计
- 是否优化 intrinsics

第一版建议固定 intrinsics，只优化 pose + sparse 3D points。

## 4. 实施步骤

### Step 1: LingBot-Map 推理与导出

输入图片序列，运行 LingBot-Map，导出原始几何。

产物：

```text
outputs/plan001/<scene>/lingbot/
  images/
  intrinsics.npy
  w2c.npy
  c2w.npy
  world_points.npy
  world_points_conf.npy
  meta.json
```

验收：

- `world_points` shape 为 `[N, H, W, 3]`。
- `w2c @ c2w` 接近单位阵。
- 随机抽样 `world_points[i]` 投影回 `image_i`，像素误差应接近 0 或在可解释的半像素偏移内。

### Step 2: 写 COLMAP reference model

把 LingBot 的 `K + w2c` 写成 COLMAP 模型。

产物：

```text
outputs/plan001/<scene>/colmap_reference/
  cameras.txt/bin
  images.txt/bin
  points3D.txt/bin
```

验收：

- `pycolmap.Reconstruction(reference_dir)` 能正常读取。
- image 数量等于 LingBot 帧数。
- 每个 image 的 `cam_from_world` 和导出的 `w2c` 一致。

### Step 3: HLoc + LoMa 匹配

使用 HLoc 流程：

```text
extract_features
match_features with loma
```

pairs 生成策略第一版使用 LingBot pose：

- 邻近帧 pairs：`i <-> i+1...i+k`
- 可选 loop pairs：基于 pose 距离和视角阈值筛选

产物：

```text
outputs/plan001/<scene>/hloc/
  pairs.txt
  features.h5
  matches.h5
```

验收：

- 每个关键帧至少有足够 matches。
- matches 数量统计正常。
- 随机可视化若干 pair，确认 LoMa 匹配没有明显错位。

### Step 4: HLoc triangulation

调用 HLoc `triangulation.main()`：

```text
reference_model = colmap_reference/
features       = features.h5
matches        = matches.h5
pairs          = pairs.txt
```

产物：

```text
outputs/plan001/<scene>/hloc/sparse/
  cameras.bin
  images.bin
  points3D.bin
```

验收：

- sparse points 数量大于最低阈值。
- 平均 track length 合理。
- reprojection error 合理。
- 注册图像数量等于输入图像数量，或至少覆盖待验证片段。

### Step 5: BAE sparse BA

读取 HLoc sparse model，运行 BAE。

配置建议：

```text
optimize_intrinsics = false
iters = 10~30
solver = PCG
device = cuda
```

产物：

```text
outputs/plan001/<scene>/bae/
  optimized_w2c.npy
  optimized_c2w.npy
  optimized_points3d.npy
  report.json
```

验收：

- final reprojection loss < initial reprojection loss。
- pose delta 没有异常爆炸。
- 优化后的 sparse points 没有大量 NaN/Inf。
- 和 LingBot 原始 pose 对比，轨迹变化连续。

### Step 6: Pose-delta 矫正 LingBot dense world points

对每帧 dense point 做：

```text
X'_i = C'_i * W_i * X_i
```

其中：

- `W_i` 来自 LingBot 原始 `w2c.npy`
- `C'_i` 来自 BAE 优化后的 `optimized_c2w.npy`
- `X_i` 来自 LingBot 原始 `world_points.npy`

产物：

```text
outputs/plan001/<scene>/dense_corrected/
  corrected_world_points.npy
  corrected_world_points_conf.npy
  corrected_dense.ply
  report.json
```

验收：

- corrected points 无 NaN/Inf。
- 单帧自投影深度与原始 depth 基本一致，这是预期现象。
- 多帧融合后的点云厚度/重影应比原始 LingBot dense cloud 更好。

### Step 7: 投影对比

做两组投影：

```text
A. 原始 LingBot dense points + 原始 LingBot pose
B. corrected dense points + BAE optimized pose
```

每组都使用 z-buffer 投影到每一帧相机，输出 depth。

产物：

```text
outputs/plan001/<scene>/projected_depth/
  raw/
    depth_npy/
    depth_vis/
  corrected/
    depth_npy/
    depth_vis/
  metrics.json
```

对比指标：

- valid pixel ratio
- depth discontinuity / edge artifacts
- temporal consistency
- 同一表面投影厚度
- 可视化 overlay

## 5. 必须先验证的 sanity checks

### Check A: LingBot point self-projection

验证：

```text
project(K_i, W_i, X_i(u,v)) ~= (u,v)
```

目的：

- 确认 LingBot `world_points`、`w2c`、`K` 坐标一致。
- 发现 `W/2` vs `W/2 - 0.5` 的像素中心问题。

### Check B: COLMAP reference pose 一致性

读取写出的 reference model，确认：

```text
colmap_w2c_i ~= lingbot_w2c_i
```

目的：

- 避免 OpenCV/OpenGL 或 quaternion 顺序错误。

### Check C: BAE pose 和 LingBot pose 在同一坐标系

确认：

```text
Delta_i = C'_i * W_i
```

的平移/旋转量处于合理范围，而不是整体巨大 Sim(3) 漂移。

如果 delta 异常大，说明可能：

- HLoc 走了完整 reconstruction，而不是 reference-pose triangulation。
- COLMAP reference model 写错。
- BAE 输出 pose 解析错。

### Check D: Pose-delta 自投影不变性

验证：

```text
W'_i * (C'_i * W_i * X_i) == W_i * X_i
```

这说明 pose-delta 只改变世界坐标位置，不改变来源帧相机坐标下的 depth。该现象是预期，不是 bug。

## 6. 主要风险

### 风险 1: LoMa sparse SfM 覆盖不足

LoMa 是局部特征匹配，不是 RoMa dense flow。它适合 sparse pose correction，但不能直接提供 dense points。

缓解：

- 增加 pair 数量。
- 使用 keyframe + local temporal pairs。
- 放宽 triangulation 阈值，但保留 reprojection error 过滤。

### 风险 2: BAE 只支持 shared PINHOLE

第一版使用 shared K。若 LingBot 每帧 K 波动明显，可能引入误差。

缓解：

- 先固定第一帧或中位数 K。
- 记录 LingBot 每帧 K 的 variance。
- 后续再扩展 BAE per-frame intrinsics。

### 风险 3: Pose-delta 只能做每帧刚体矫正

它不能修复 LingBot dense point map 内部的局部形变或深度误差。

缓解：

- plan001 只验证刚体矫正 baseline。
- 后续增加 per-frame Sim(3)、sparse correction field、depth scale field。

### 风险 4: Corrected dense cloud 跨帧融合仍有厚度

即使 pose 更准，LingBot 每帧 dense 预测本身可能仍不一致。

缓解：

- 使用 `world_points_conf` 过滤。
- 加 depth edge filter。
- z-buffer 投影时做置信度优先或近深度优先。
- 后续考虑 TSDF / voxel fusion。

## 7. plan001 不做的事情

本阶段暂不做：

- 不把全量 dense points 传入 BAE。
- 不训练 GGPT Point Transformer。
- 不做完整 COLMAP reconstruction 的 Sim(3) 对齐路线。
- 不接 LingBot-Depth refine，除非 Step 7 结果已经明显合理。
- 不做在线 streaming 集成，只做离线短序列验证。

## 8. 验证成功标准

plan001 可认为成功，如果满足：

1. LingBot pose/K/dense points 能稳定导出并通过 self-projection。
2. HLoc + LoMa 能基于 LingBot reference pose 三角化出有效 COLMAP sparse model。
3. BAE 能降低 sparse reprojection loss，并输出合理 pose delta。
4. Pose-delta corrected dense cloud 相比原始 LingBot dense cloud，在跨帧融合或投影上有更好的一致性。
5. corrected projected depth 至少不劣于 raw projected depth，并为后续 LingBot-Depth refine 提供更合理输入。

## 9. 下一阶段候选

如果 plan001 验证通过，下一阶段做：

1. 接入 LingBot-Depth refine。
2. 对比 raw projected depth vs corrected projected depth 的 refine 结果。
3. 引入 sparse BA point displacement，拟合 per-frame Sim(3) 或 depth correction field。
4. 从短序列扩展到 sliding-window 长序列。
