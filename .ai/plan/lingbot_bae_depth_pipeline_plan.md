# LingBot-Map + GGPT Tracks + BAE BA + LingBot-Depth 初步计划

## 0. 当前设想复述

目标链路：

1. 用 `lingbot-map` 对长序列图片做前馈推理，得到初始相机、深度、每像素世界坐标点云。
2. 参考 `GGPT/sfm/run_sfm`，用密集匹配构建 2D track，并用前馈 3D 点作为 track 的 3D 初值。
3. 不使用 GGPT 的 `pycolmap` BA，改用 `bae` 的 PyTorch BA 优化相机位姿和 3D 点。
4. 用 BAE 优化后的位姿/点云投影成 dense depth，参考 `Pi3/example_mm.py` 的 z-buffer 投影实现。
5. 把投影深度作为传感器深度输入 `lingbot-depth`，得到 refined metric depth。

最终产物：

- BAE 矫正后的相机位姿。
- BAE / DLT / 稠密化后的 dense ply。
- LingBot-Depth refine 后的深度图。

## 1. 总体可行性判断

这个方向整体可行，但需要重构成“稀疏几何约束驱动 dense 输出”的流程，而不是把 LingBot-Map 的所有稠密点直接交给 BAE。

主要原因：

- `lingbot-map` 能提供 GGPT 所需的 feed-forward 数据源：`images_ff`、`extrinsics`、`intrinsics`、`points`、`points_conf`。
- GGPT 的 `run_sfm` 已经证明了一种实用路径：先用 dense matching 建 track，再从 `ff_outputs['points']` 取对应像素的 3D 点作为 BA 初值。
- `bae` 的当前 COLMAP BA 入口本质上需要稀疏观测表：`points_2d`、`camera_indices`、`point_indices`、`points_3d`、`camera_params`、shared `PINHOLE` intrinsics。这和 GGPT 选出来的 sparse tracks 能对上。
- 直接用全量 dense points 做 BA 在内存、Jacobian 规模、异常点比例上都不现实。BAE 支持稀疏 BA，不等于适合把 `N * H * W` 个点全部作为优化变量。

建议的第一版目标：

- 保留 LingBot-Map 的稠密点作为初始化和最终重建底图。
- 只从 GGPT dense matching 中选择高质量 sparse tracks 进入 BAE。
- BAE 只优化位姿 + 被选中 sparse 3D points + 可选 shared intrinsics。
- 用优化后的相机重新 DLT 三角化更多 track，或把 BAE 位姿用于投影/融合 LingBot-Map 稠密点。
- 再导出 z-buffer depth 给 LingBot-Depth refine。

## 2. 最关键的数据契约

### 2.1 LingBot-Map 到 GGPT `ff_outputs`

GGPT `run_sfm()` 期待：

```python
ff_outputs = {
    "images_ff":    Tensor[N, H, W, 3],
    "extrinsics":   Tensor[N, 4, 4],      # w2c, OpenCV
    "intrinsics":   Tensor[N, 3, 3],
    "points":       Tensor[N, H, W, 3],   # world coordinates
    "points_conf":  Tensor[N, H, W],
}
```

LingBot-Map 源码里 `pose_encoding_to_extri_intri()` 先得到 OpenCV `w2c`，但 `demo.py::postprocess()` 又做了一次 SE3 inverse 后把 `predictions["extrinsic"]` 存成 `c2w`。因此适配 GGPT/BAE 时必须明确：

- 给 GGPT/BAE：使用 `w2c`。
- 给 Pi3 风格 depth 投影：使用 `c2w` camera pose。
- 文件里不要只叫 `extrinsic`，建议显式命名为 `w2c` / `c2w`。

### 2.2 GGPT tracks 到 BAE input

从 GGPT 的 `run_sfm()` 复用：

- `match_results['pred_matches_lr']`: `[Ntgt, Nsrc, H*W, 2]`
- `pred_scores`
- `pred_cycle_error`
- `sp_scores`

BAE 需要展开成 observation table：

```python
points_2d:       [Nobs, 2]
camera_indices:  [Nobs]
point_indices:   [Nobs]
points_3d:       [Ntracks, 3]
camera_params:   [N, 7]  # [tx, ty, tz, qx, qy, qz, qw], w2c
intrinsics:      [4] or [1, 4]  # fx, fy, cx, cy
```

`points_3d` 的初值沿用 GGPT 逻辑：

```python
pts3d_ba = ff_outputs["points"].reshape(-1, 3)[selected]
```

其中 `selected` 是 `N * H * W` 平铺后的 track id。

### 2.3 BAE output 到 depth 投影

BAE 优化出的相机位姿应保持为 `w2c`。投影 depth 时需要：

- `w2c -> c2w` 后传给 Pi3 的 `project_world_points_to_depth()`。
- 或写一个直接吃 `w2c` 的投影函数，避免重复求逆和命名混乱。

投影深度是 z-buffer 结果：

- 每帧深度为相机坐标系 z。
- 无命中像素为 0。
- 需要保存 `.npy` float32，作为 LingBot-Depth 的 `depth_in`。

### 2.4 LingBot-Depth intrinsics

LingBot-Depth `model.infer()` 需要归一化内参：

```python
K_norm = [
  [fx / W, 0,      cx / W],
  [0,      fy / H, cy / H],
  [0,      0,      1],
]
```

注意它的 `run.py` 目前会把输入 depth resize 到固定 `(1400, 1904)`，这对本流程不一定合适。更稳的方式是新增一个 pipeline wrapper，保证 RGB、depth、K 三者在同一分辨率下进入 `model.infer()`。

## 3. 存在风险和不合逻辑点

### 3.1 “稠密点直接进 BAE”风险很高

BAE 是稀疏 BA 优化器，但每个 3D 点仍是优化变量。若直接对 `N * H * W` 个点建 BA：

- 变量数量极大，例如 `100 frames * 378 * 518 ~= 19.6M` 个 3D 点。
- 观测数更大，Jacobian/Hessian 即使稀疏也很难承受。
- 稠密点包含天空、反光、动态物体、低纹理区域，异常点比例高，会拖垮 BA。

结论：BAE 第一版只吃 sparse selected tracks；dense 点只用于初始化、后续融合和深度投影。

### 3.2 GGPT 的 all-pairs dense matching 不适合长序列

GGPT `match_results['pred_matches_lr']` 是 `[N, N, H*W, ...]` 级别，时间和显存近似 `O(N^2HW)`。这对短序列可以，对 LingBot-Map 目标的长序列不现实。

第一版建议：

- 先在 20-100 帧短序列验证。
- 中期改成 window/keyframe matching：
  - 邻近窗口：`i` 匹配 `i +/- k`。
  - 关键帧：每隔 `s` 帧抽一个 anchor。
  - 回环候选：后续再接图像检索或共视性判断。

### 3.3 坐标系和像素中心约定必须先写测试

现有代码里存在几个容易错的点：

- LingBot-Map demo 输出的 `extrinsic` 实际是 `c2w`，GGPT 需要 `w2c`。
- GGPT 对 track 和内参有 `+0.5` 像素中心处理，并在输出时把 principal point 固定回 `W/2 - 0.5` / `H/2 - 0.5`。
- LingBot-Map 的 intrinsics 是 `W/2` / `H/2` 中心约定。
- BAE 的 `project_colmap()` 使用 OpenCV `x/z, y/z` 投影。

如果这里没有单元测试，后续 BA loss 可能下降但 geometry 实际偏半像素或直接翻转。

### 3.4 BAE 当前只支持 shared PINHOLE intrinsics

`bae/datapipes/colmap_loader.py` 和 `ba_helpers.py` 当前 COLMAP 模式只处理单一 shared `PINHOLE`。而 LingBot-Map 每帧可能输出不同 FoV/焦距。

第一版建议：

- 固定 shared intrinsics，取所有帧 `fx/fy` 的中位数或第一帧。
- 或先不优化 intrinsics，只优化 pose/points。
- 若发现焦距漂移明显，再扩展 BAE 支持 per-frame intrinsics。

### 3.5 LingBot-Depth refine 的输入不是普通 dense prediction

LingBot-Depth 被设计为 RGB + sensor depth refinement/completion。BAE 投影得到的 depth 会有：

- 空洞。
- 遮挡边界噪声。
- 稀疏 BA 点投影时可能覆盖率不足。

因此 depth refine 前最好有一个输入质量控制：

- z-buffer 后保留有效 mask。
- 去掉极端深度和孤立点。
- 可选做轻量形态学补洞，但不要过度平滑。
- 记录 valid ratio，低于阈值时不要盲目 refine。

## 4. 建议的 MVP 实现路线

### Phase 1：离线短序列验证

目标：跑通 20-100 帧，确认坐标系、BAE 优化和 depth refine 都能闭环。

任务：

1. 写 `LingBotMapAdapter`
   - 输入图片目录。
   - 输出 GGPT-compatible `ff_outputs`。
   - 显式保存 `w2c.npy`、`c2w.npy`、`intrinsics.npy`、`points.npy`、`points_conf.npy`。

2. 拆 GGPT `run_sfm()`
   - 保留 dense matching。
   - 抽出 track selection。
   - 抽出 DLT triangulation。
   - 把 `pycolmap BA` 替换成 `run_bae_ba(tracks_ba, masks_ba, pts3d_ba, w2c, K)`。

3. 写 BAE 数据转换
   - `tracks_ba[N, P, 2] + mask[N, P] -> observation table`。
   - `w2c[N,4,4] -> camera_params[N,7]`。
   - `camera_params[N,7] -> w2c[N,4,4]`。

4. 跑 BAE BA
   - 先固定 intrinsics。
   - 使用 `PCG` solver。
   - 输出 initial / final reprojection loss。
   - 输出每帧 pose delta、每点 update norm，检查是否发散。

5. 用 BAE 相机做 DLT
   - 复用 GGPT 的 DLT 逻辑。
   - 产出 `sfm_outputs = {extrinsics, intrinsics, points, point_masks}`。

6. 投影 dense depth
   - 初版用 DLT points 或 LingBot dense points 经 BAE pose 投影。
   - 保存 `depth_npy`、`depth_u16`、`depth_vis`。

7. LingBot-Depth refine
   - 使用同分辨率 RGB/depth/K。
   - 输出 refined depth 和 camera-space points。

验收指标：

- BAE final reprojection loss 低于 initial。
- 优化后相机没有明显翻转/尺度爆炸。
- DLT 有效点覆盖率合理。
- depth 投影有效像素比例可解释。
- LingBot-Depth 输出保持 metric scale，没有整体尺度漂移。

### Phase 2：长序列可扩展版本

目标：从短序列扩展到几百/几千帧。

任务：

1. matching 改为图结构
   - local temporal edges。
   - anchor/keyframe edges。
   - 可选 loop closure edges。

2. BA 改为局部/分层
   - sliding window BA。
   - keyframe BA。
   - 固定旧窗口点，只优化当前窗口 pose/points。

3. 点云融合
   - BAE sparse points 用于相机校正。
   - LingBot dense world points 用优化后的相机/尺度做融合。
   - 用 `points_conf`、depth edge、sky/dynamic mask 过滤。

4. 深度 refine 批处理
   - 对每帧独立 refine。
   - 支持断点续跑。
   - 保存中间 valid mask 和投影统计。

### Phase 3：质量提升

目标：提高鲁棒性和最终 dense 质量。

方向：

- 增加 robust loss 或 residual clipping。
- 增加动态物体/天空 mask。
- 用 BAE pose 后重新筛 track，再二次 BA。
- 对 dense point projection 加 normal/depth consistency filter。
- 评估 `LingBot-Map depth`、`DLT depth`、`projected dense point depth` 三种输入给 LingBot-Depth 的效果差异。

## 5. 建议先做的实验

### Experiment A：坐标系 sanity check

不用 matching，直接验证：

- LingBot `world_points[i, y, x]` 用 `w2c[i] + K[i]` 投影回 `(x, y)`。
- 平均 reprojection error 应接近 0 或在半像素级。
- 若误差系统性为 0.5，需要统一 pixel center convention。

### Experiment B：GGPT track + BAE sparse BA

使用 20-50 帧：

- 复用 GGPT selected tracks。
- BAE 固定 K，只优化 pose + sparse points。
- 对比 `pycolmap BA` 和 `BAE BA` 的 final reprojection error、耗时、pose delta。

### Experiment C：BAE pose + DLT dense points

使用 BAE 优化后的相机跑 GGPT DLT：

- 观察 `point_masks` 覆盖率。
- 导出 DLT ply。
- 与 LingBot raw ply 对齐比较。

### Experiment D：投影 depth -> LingBot-Depth

比较三种输入：

1. BAE DLT points 投影 depth。
2. LingBot dense world points + BAE pose 投影 depth。
3. LingBot 原始 depth。

输出 refined depth 后比较：

- 有效区域完整度。
- 边界质量。
- 是否保持尺度。
- 是否出现大面积 hallucination。

## 6. 推荐目录结构

建议新增：

```text
.ai/plan/
  lingbot_bae_depth_pipeline_plan.md

pipeline/
  lingbot_adapter.py
  ggpt_track_builder.py
  bae_ba_runner.py
  dlt_triangulation.py
  depth_projection.py
  lingbot_depth_refine.py
  run_pipeline.py

outputs/
  <scene>/
    lingbot_map/
    tracks/
    bae/
    dlt/
    projected_depth/
    refined_depth/
```

也可以先不建 `pipeline/`，直接做一个 `scripts/prototype_lingbot_bae_depth.py`，但建议逻辑上按以上模块拆分，便于替换 GGPT/BAE/Depth 任一环节。

## 7. 第一版实现注意事项

- 不要复用含糊的 `extrinsics` 命名；内部统一 `w2c` / `c2w`。
- 所有中间结果保存 shape 和 convention 到 sidecar json。
- BAE 输入先用 `float64`，和现有 colmap loader 保持一致。
- 先固定 intrinsics，跑通后再尝试优化 intrinsics。
- BA tracks 数量先保守，比如每帧 512-2048。
- long sequence 不要 all-pairs matching，先从短序列闭环。
- depth refine 前保留 projected depth mask，避免把无效 0 深度当成可靠观测。

## 8. 当前结论

你的设想方向成立，但需要把“稠密点 BA”改成“稀疏 track BA + dense projection/refinement”。

最合理的落地路径是：

```text
LingBot-Map ff_outputs
  -> GGPT dense matching
  -> selected sparse tracks
  -> BAE sparse BA
  -> BAE pose + DLT / dense point fusion
  -> z-buffer projected depth
  -> LingBot-Depth refine
```

最先应该验证的是坐标系和 BAE sparse BA。如果这两步不稳定，后面的 dense depth/refine 会放大错误；如果这两步稳定，后续主要是工程规模化和质量过滤问题。
