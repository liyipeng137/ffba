# Merg3r + GlueMap: BAE Backend 接入设计

更新时间：2026-06-16

## 目标

在当前已经跑通的 Merg3r + GlueMap SPV pipeline 中，为 GlueMap augmented bundle adjustment 新增一个独立的 BAE 后端，用于替代 Ceres 求解耗时较长的问题。

当前不改动现有 Ceres solver。新增实现应作为并行后端存在，便于同一输入下做：

```text
ceres backend
vs
bae backend
```

对比指标包括运行时间、reprojection loss、real/virtual track 数、angular reprojection error、最终 Gaussian 训练质量。

## 当前接入点

现有 Ceres augmented BA 入口：

```text
MERG3R/gluemap/gluemap/estimators/augmented_bundle_adjustment.py
```

当前核心函数：

```python
bundle_adjustment(
    reconstruction,
    virtual_reconstruction,
    negative_depth_observations,
    max_num_iterations=200,
    loss_type_normal="huber",
    loss_type_virtual="arctan",
)
```

计划新增独立文件：

```text
MERG3R/gluemap/gluemap/estimators/bae_solver.py
```

计划新增函数：

```python
bundle_adjustment_bae(
    reconstruction,
    virtual_reconstruction,
    negative_depth_observations,
    max_num_iterations=20,
    device="cuda",
)
```

该函数返回：

```python
(reconstruction, virtual_reconstruction, summary)
```

其中 `summary` 不要求兼容 `pyceres.SolverSummary`，可以是轻量 dict 或 dataclass。当前 `iterative_bundle_adjustment()` 不依赖 Ceres summary 的内部字段，因此第一版可用自定义 summary。

## BAE 版本来源

使用新版 BAE：

```text
MERG3R/bae/
```

忽略旧版本：

```text
MERG3R/third_party/bae/
```

当前参考入口：

```text
MERG3R/bae/ba_colmap.py
```

需要注意 Python import 路径，避免误导入旧版或 repo 根目录其他同名 `bae`。

建议在 `bae_solver.py` 中使用 lazy import helper：

```text
1. 从当前文件向上定位 MERG3R/bae。
2. 将 MERG3R/bae 插入 sys.path[0]。
3. import bae / ba_colmap 后检查实际 __file__ 路径。
4. 如果不是 MERG3R/bae 下的模块，直接报错。
```

## 第一版明确约束

第一版 BAE backend 按以下约束实现：

- 默认直接拼接 real + virtual observations；可通过 `--bae_real_only`
  诊断性跳过 virtual residual/point 参数。
- 默认不优化 intrinsics；开启 `--bae_optimize_intrinsics` 时，仅对
  `SIMPLE_PINHOLE` 优化 `f`，固定 `cx/cy`。
- 新增 `--bae_fix_gauge`，默认 `two_cams`，语义对齐 Ceres 版
  `TWO_CAMS_FROM_WORLD`；可显式传 `none` 复现无 gauge fix 行为。
- 不实现 robust loss，先用 plain squared loss。
- camera model 假设所有 image 共享同一个 camera。
- 优先支持当前主线 `SIMPLE_PINHOLE`；如遇 `PINHOLE` 可显式转换支持。
- 不落盘走 COLMAP txt/bin loader，直接从内存中的 `pycolmap.Reconstruction` 构建 BAE 输入。

## 与现有 Ceres 版的对应关系

现有 Ceres 版分两步：

```text
1. pycolmap.create_default_ceres_bundle_adjuster()
   为 real reconstruction 创建标准 COLMAP BA residual。

2. _add_virtual_track_residuals()
   手动把 virtual residual 追加到同一个 Ceres problem。
```

virtual residual 使用 real reconstruction 的 pose/intrinsics 参数块：

```python
cam_pose = reference_reconstruction.frames[ref_id].rig_from_world.params
camera_params = reference_reconstruction.cameras[camera_id].params
```

因此 real 和 virtual residual 共享同一组相机参数。

BAE 版应复刻这个语义：

```text
real observations
+ virtual observations
-> 同一个 BAE residual graph
-> 共享同一组 camera pose
-> real points 和 virtual points 分别作为 point 参数优化
-> camera intrinsics 默认作为 fixed buffer；可选作为 BAE 参数优化
```

## BAE 输入结构

`MERG3R/bae/ba_colmap.py` 当前核心输入为：

```text
camera_params:  [N, 7]  = [tx, ty, tz, qx, qy, qz, qw]
points_3d:      [M, 3]
intrinsics:     [3|4]   = SIMPLE_PINHOLE [f, cx, cy]
                       or PINHOLE [fx, fy, cx, cy]
points_2d:      [K, 2]
camera_indices: [K]
point_indices:  [K]
```

SPV BAE backend 需要扩展 observation 输入：

```text
points_2d:      [K, 2]
camera_indices: [K]
point_indices:  [K]
is_negative:    [K] bool
is_virtual:     [K] bool
```

其中：

- real observations: `is_virtual=False`, `is_negative=False`
- normal virtual observations: `is_virtual=True`, `is_negative=False`
- negative-depth virtual observations: `is_virtual=True`, `is_negative=True`

`is_virtual` 第一版不参与 residual 计算，但用于 stats 和 debug。

## Reconstruction -> BAE 数据桥

### Camera

第一版要求所有 image 共享同一个 camera：

```text
len(reconstruction.cameras) == 1
```

或所有 registered image 的 `camera_id` 相同。否则直接报错，不做静默平均。

### Intrinsics

BAE backend 默认固定 intrinsics。开启 `--bae_optimize_intrinsics` 时，
只允许 `SIMPLE_PINHOLE`，并只优化 `f`，固定 `cx/cy`；`fx=fy=f`
是参数化天然保证的约束。

`SIMPLE_PINHOLE` 直接使用 COLMAP 原生三参数：

```text
COLMAP SIMPLE_PINHOLE params = [f, cx, cy]
BAE intrinsics              = [f, cx, cy]
```

`PINHOLE` 直接使用：

```text
COLMAP PINHOLE params = [fx, fy, cx, cy]
BAE intrinsics        = [fx, fy, cx, cy]
```

不开启 `--bae_optimize_intrinsics` 时不更新 camera params。开启后将
优化后的 `[f, fixed cx, fixed cy]` 写回 real reconstruction 的 shared
camera，并同步给 virtual reconstruction 的同 id camera。

### Pose

pycolmap / COLMAP `Rigid3d.params` 存储顺序为：

```text
[qx, qy, qz, qw, tx, ty, tz]
```

BAE `ba_colmap.py` 使用：

```text
[tx, ty, tz, qx, qy, qz, qw]
```

因此需要双向转换：

```text
pycolmap -> BAE:
  [qx, qy, qz, qw, tx, ty, tz]
  -> [tx, ty, tz, qx, qy, qz, qw]

BAE -> pycolmap:
  [tx, ty, tz, qx, qy, qz, qw]
  -> [qx, qy, qz, qw, tx, ty, tz]
```

写回前需要 normalize quaternion。

### Real observations

遍历 `reconstruction.points3D`：

```text
for point3D_id, point3D in reconstruction.points3D.items():
  for elem in point3D.track.elements:
    image_id = elem.image_id
    point2D_idx = elem.point2D_idx
    xy = reconstruction.images[image_id].points2D[point2D_idx].xy
```

构建：

```text
points_2d.append(xy)
camera_indices.append(image_id_to_compact_idx[image_id])
point_indices.append(real_point_id_to_compact_idx[point3D_id])
is_virtual.append(False)
is_negative.append(False)
```

### Virtual observations

virtual reconstruction 中的 image id 不应假设与 real reconstruction 完全一致，当前 Ceres 版按 image name 映射：

```python
name_to_ref_id = {
    img.name: img_id
    for img_id, img in reference_reconstruction.images.items()
}
```

BAE 版沿用该策略：

```text
virtual image name -> real image id -> compact camera index
```

遍历 `virtual_reconstruction.points3D`：

```text
for virtual_point3D_id, point3D in virtual_reconstruction.points3D.items():
  for elem in point3D.track.elements:
    virtual_image_id = elem.image_id
    point2D_idx = elem.point2D_idx
    virtual_image = virtual_reconstruction.images[virtual_image_id]
    real_image_id = name_to_ref_id[virtual_image.name]
    xy = virtual_image.points2D[point2D_idx].xy
```

构建：

```text
points_2d.append(xy)
camera_indices.append(image_id_to_compact_idx[real_image_id])
point_indices.append(global_virtual_point_compact_idx)
is_virtual.append(True)
is_negative.append(
  virtual_image_id in negative_depth_observations
  and point2D_idx in negative_depth_observations[virtual_image_id]
)
```

注意：`negative_depth_observations` 的 key 是 virtual reconstruction 的 image id / point2D index，不是 real reconstruction 的 id。

## Point 参数拼接

BAE 的 `points_3d` 建议按以下顺序拼接：

```text
[real_points, virtual_points]
```

需要保留映射：

```text
real_point_compact_idx -> real point3D_id
virtual_point_compact_idx -> virtual point3D_id
```

写回时：

```text
optimized_points[:num_real_points]
  -> reconstruction.points3D[real_point3D_id].xyz

optimized_points[num_real_points:]
  -> virtual_reconstruction.points3D[virtual_point3D_id].xyz
```

## 自定义 GlueMap Residual

第一版不直接裸用 `ColmapResidual`，而是在 `bae_solver.py` 中定义 GlueMap SPV 专用 residual。

原因：

- 需要支持 real + virtual observations。
- 需要支持 virtual negative-depth residual。
- 需要保留 fixed intrinsics。
- 后续可能加 robust loss 或 real/virtual 权重。

### Ceres normal residual 参考

COLMAP 标准 reprojection residual：

```text
point_cam = R_cam_from_world * point_world + t_cam_from_world
pixel = CameraModel::ImgFromCam(camera_params, point_cam)
residual = pixel - observed_xy
```

如果 `point_cam.z <= eps`，COLMAP `ImgFromCam()` 返回 false，residual 置零。

### Ceres negative-depth residual 参考

GlueMap 自定义 Ceres functor：

```text
point_cam = R_cam_from_world * point_world + t_cam_from_world
point_cam = -point_cam
pixel = CameraModel::ImgFromCam(camera_params, point_cam)
residual = pixel - observed_xy
```

即 negative-depth 不是新的相机模型，只是在投影前把 camera-space 3D 点整体取负。

### BAE residual 设计

伪代码：

```python
@psjac
def project_gluemap(points, camera_params, intrinsics, is_negative):
    points_cam = pp.SE3(camera_params[..., :7]).Act(points)

    points_cam = torch.where(
        is_negative[..., None],
        -points_cam,
        points_cam,
    )

    z = points_cam[..., 2:3]
    valid = z > eps

    fx = intrinsics[..., 0:1]
    fy = intrinsics[..., 1:2]
    cx = intrinsics[..., 2:3]
    cy = intrinsics[..., 3:4]

    x = fx * points_cam[..., 0:1] / z + cx
    y = fy * points_cam[..., 1:2] / z + cy
    pixel = torch.cat([x, y], dim=-1)

    return torch.where(valid, pixel, torch.zeros_like(pixel))
```

residual：

```python
residual = project_gluemap(...) - points_2d
```

为更贴近 Ceres，invalid projection 的 residual 应置零，而不是产生 NaN/Inf。实现时要避免 `z <= eps` 时先除法再 mask 导致 NaN，可用 clamped denominator 或 normal/invalid 分支。

## Loss / robust kernel

第一版：

```text
plain squared loss
```

不实现：

- real Huber loss
- virtual Arctan loss
- real/virtual 不同权重

当前 Ceres 默认：

```text
real residual:    Huber
virtual residual: Arctan
```

因此第一版 BAE 与 Ceres 不完全等价。该差异需要写入 summary 和实验记录。若 plain squared loss 质量不足，再补 robust residual weighting。

## Gauge fixing

当前 Ceres 版执行：

```python
ba_config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
```

BAE 当前实现了一个语义对齐版本，而不是逐字节复刻 Ceres manifold：

```text
--bae_fix_gauge two_cams       # 默认
--bae_fix_gauge three_points
--bae_fix_gauge two_cams_full  # debug 强约束
--bae_fix_gauge none           # 复现无 gauge fix 行为
```

`two_cams` 策略：

```text
1. 选第一个参与 BAE 的 image，固定其 6 维 pose tangent DOF。
2. 选择第二个 baseline 非退化 image。
3. 计算 relative baseline 的最大绝对值轴。
4. 固定第二个 image 对应 translation tangent DOF。
5. 若 two-cams 失败，fallback 到 three-points。
```

该实现通过 BAE optimizer 的 `fixed_dof_mask` 完成：

```text
1. normal equation 中固定 DOF 的行列置零。
2. 固定 DOF 保留单位对角，避免线性系统奇异。
3. 参数更新前再次清零固定 DOF 增量。
```

由于 BAE 使用 PyPose SE3 tangent update，第二个相机固定的是 translation
tangent DOF，与 Ceres `SubsetManifold` 固定 ambient translation 分量不完全
一致，但消除 gauge 的语义一致。

无 gauge fix 时，BAE 可能出现全局 Sim3 / gauge drift。由于 reprojection error 对 gauge 不敏感，loss 降低不等于世界坐标稳定。

因此 BAE summary 中需要记录基础 drift 诊断：

```text
camera center centroid shift
camera center scale ratio
mean / max pose translation delta
mean / max rotation delta
initial_loss
ending_loss
```

如果后续观察到明显漂移，再加 fixed camera / fixed first-two-cameras 版本。

## 写回策略

BAE solve 完成后：

```text
1. optimized poses 写回 real reconstruction.frames[image_id].rig_from_world
2. optimized real points 写回 reconstruction.points3D
3. optimized virtual points 写回 virtual_reconstruction.points3D
4. 将 real reconstruction 的 optimized poses 同步给 virtual_reconstruction
5. camera params 不变
```

第 4 步复用当前 Ceres 版语义：

```text
virtual_reconstruction 的相机姿态应与 real reconstruction 保持一致
```

可沿用当前 `_update_poses_from_reconstruction()` 的 name-based 同步方式。

## Summary / Stats

BAE summary 建议包含：

```json
{
  "backend": "bae",
  "device": "cuda",
  "num_iterations": 20,
  "num_cameras": 181,
  "num_points_real": 100000,
  "num_points_virtual": 20000,
  "num_observations_real": 500000,
  "num_observations_virtual": 500000,
  "num_observations_negative": 0,
  "initial_loss": 0.0,
  "ending_loss": 0.0,
  "seconds": 0.0,
  "optimize_intrinsics": true,
  "intrinsics_initial": [1000.0, 512.0, 384.0],
  "intrinsics_final": [1002.5, 512.0, 384.0],
  "fix_gauge": true,
  "gauge_fix": {
    "requested": "two_cams",
    "applied": "two_cams",
    "translation_fixed_dim": 0,
    "num_fixed_pose_dofs": 7,
    "num_fixed_point_dofs": 0
  },
  "loss": "plain_squared"
}
```

可选附加：

```text
skipped virtual observations
skipped points with no xyz
invalid projection count
peak GPU memory
pose drift diagnostics
```

## Controller 层接入

第一版建议不要直接替换 `iterative_bundle_adjustment()` 内部逻辑，而是增加 backend 参数：

```text
ba_backend = "ceres" | "bae"
```

在 `MERG3R/gluemap/gluemap/controllers/augmented_bundle_adjustment.py` 中：

```python
if options.ba_backend == "bae":
    reconstruction, virtual_reconstruction, summary = bundle_adjustment_bae(...)
else:
    reconstruction, virtual_reconstruction, summary = bundle_adjustment(...)
```

然后在 pipeline config/CLI 中暴露：

```bash
--ba_backend ceres
--ba_backend bae
--bae_iters 20
```

默认仍为 Ceres，保证现有结果可复现。

## 第一版验证路径

建议按以下顺序验证：

1. 单轮 SPV，`num_refinement_iterations=1`，BAE backend。
2. 确认 BAE 输入规模：
   - cameras
   - real points
   - virtual points
   - real observations
   - virtual observations
   - negative observations
3. 确认 loss 正常下降，无 NaN/Inf。
4. 确认写回后 reconstruction 可继续执行 normalized reprojection filtering。
5. 对比 Ceres backend：
   - BA seconds
   - final real/virtual points
   - real angular mean/median/<0.5deg
   - virtual angular mean/median/<0.5deg
6. 再跑 `num_refinement_iterations=2`。
7. 最后用 Gaussian 训练结果判断真实质量。

## 主要风险

### 1. Loss 不等价

第一版 BAE 使用 plain squared loss，Ceres 使用 Huber/Arctan robust loss。若 outlier 尚未过滤干净，BAE 可能更容易被残余 outlier 拉动。

缓解：

- 依赖现有 BA 前 reprojection filter。
- 必要时补 robust residual weighting。

### 2. Gauge drift

默认 `--bae_fix_gauge two_cams` 后，BAE 会固定 gauge；但由于 BAE 的
PyPose tangent update 和 Ceres manifold 不完全一致，仍需检查世界坐标整体漂移或尺度变化。

缓解：

- summary 中记录 pose drift。
- summary 中记录 `gauge_fix` 的实际 applied strategy。
- 若 `two_cams` 仍不稳定，可临时使用 `--bae_fix_gauge two_cams_full`
  验证漂移是否来自 gauge。

### 3. Invalid projection 处理差异

Ceres / COLMAP 对 `z <= eps` residual 置零。BAE 如果直接除以 z，可能产生 NaN/Inf。

缓解：

- residual 中显式处理 invalid projection。
- summary 中记录 invalid projection 数量。

### 4. Import 混乱

仓库中存在多个 BAE 路径。

缓解：

- `bae_solver.py` 做路径校验。
- summary 中记录实际导入的 BAE module path。

### 5. Camera model 超出范围

第一版只支持 shared `SIMPLE_PINHOLE` / `PINHOLE`。

缓解：

- 遇到其他 camera model 直接报错。
- 当前主线固定 `SIMPLE_PINHOLE`，不影响主流程。

## 当前结论

按当前约束，BAE backend 没有结构性 blocker。

最小可行实现是：

```text
新增 bae_solver.py
-> 从 real reconstruction 抽 camera/real points/real obs
-> 默认从 virtual reconstruction 抽 virtual points/virtual obs/negative mask
-> 可通过 --bae_real_only 跳过 virtual residual/point 参数
-> 拼成一个 BAE problem
-> fixed shared intrinsics by default
-> optional SIMPLE_PINHOLE f-only intrinsics optimization with fixed cx/cy
-> plain squared loss
-> default COLMAP-style two-cams gauge fix
-> solve
-> 写回 real + virtual reconstruction
-> controller 通过 ba_backend 切换
```

这能直接验证 BAE 对当前 GlueMap SPV augmented BA 的加速收益，同时保留现有 Ceres 后端作为质量和回归基线。
