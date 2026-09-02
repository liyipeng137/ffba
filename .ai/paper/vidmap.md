# VidMap 思想与实现细节：面向融合改造的 Agent 快速上手文档

> 目的：帮助后续 Agent 在较短时间内理解 VidMap 的方法主线、当前开源实现、关键数据结构和可融合边界，并据此规划与本工作区 `FeedForwardWithBA` 的融合改造。
>
> 本文基于：
>
> - 论文：**VidMap: Exploiting Temporal Structure for Video-Based Structure-from-Motion**，arXiv:2607.27194v1，2026-07-29。
> - 官方仓库：<https://github.com/cvg/vidmap>
> - 检查的仓库快照：`main@92cf522e3fcde1a215bd33978b6515a2784e8dec`，最后提交日期 2026-08-14。
> - 本工作区：`/kiri/FeedForwardWithBA`。
>
> 注意：官方 README 明确说明，当前仓库结果与论文表格存在少量差异，原因包括新版 COLMAP、回环 loss 权重、内参归一化关键帧选择和不确定性感知的轨迹终止。因此，本文将“论文方法”与“当前代码实现”同时说明；做复现时应固定 commit 和配置。

---

## 1. 一句话结论

VidMap 是一个**时间顺序感知、但非因果的离线全局 SfM 系统**：它沿视频时间轴用 RoMa v2 稠密匹配建立可信的稀疏长轨迹，将回环作为来源可追踪的软连接，再用度量单目深度、相机标定先验和 Ceres 全局优化联合求解相机轨迹、内参、深度尺度与稀疏三维结构。

它的核心不是一个新的端到端网络，而是一套系统级组合原则：

1. 学习模型负责困难但局部的感知任务：稠密匹配、图像检索、深度预测、焦距先验。
2. 显式几何负责全局一致性、精度和长序列扩展。
3. 时间相邻观测与回环观测具有不同可靠性，必须保留来源并使用不同鲁棒策略。
4. 单目深度应作为可校正的软标尺，而不是固定真值。

---

## 2. VidMap 试图填补的结构性空缺

### 2.1 SLAM 的优势与问题

SLAM 利用视频的时间顺序，相邻帧跟踪通常可靠，也能根据运动自适应选择关键帧。但经典 SLAM 是在线、因果、增量式系统：

- 当前帧只能利用过去信息；
- 初始化错误会向后传播；
- 前向运动、低视差和局部对称可能过早锁定错误几何；
- 跟踪短暂丢失后不一定能够恢复；
- 通常依赖已知或较准确的相机内参。

### 2.2 SfM 的优势与问题

离线 SfM 可以在看完全部图像后做全局初始化和全局优化，也可以联合细化相机内参。但传统 SfM通常把输入当成无序图片集：

- 不区分相邻视频帧和检索得到的远距离图像对；
- 无法利用时间连续性来消除重复结构歧义；
- 错误回环经传递闭包合并进 track 后，可能污染整条轨迹；
- 固定图片抽样或全局匹配不能很好适应视频运动速度。

### 2.3 VidMap 的设计定位

```text
SLAM：有时间意识，但在线、因果、会过早提交
SfM ：可全局优化，但通常丢弃时间来源信息

VidMap：有时间意识 + 离线全局优化 + 不过早提交
```

论文将这种范式称为 non-causal：前端可以沿时间顺序处理，但在看完整段视频之前不提交最终几何解。

---

## 3. 系统总览

```text
视频 / 有序图像序列
  │
  ├─ GeoCalib bootstrap：为关键帧运动评分提供初始共享内参
  │
  ├─ RoMa v2 低分辨率帧间匹配
  │    └─ 基于归一化图像运动选择关键帧
  │
  ├─ RoMa v2 高分辨率匹配
  │    ├─ 顺序传播稀疏 tracks
  │    ├─ multi-flow 跨 hop 漂移校正
  │    └─ 协方差、置信度、极线几何与密度过滤
  │
  ├─ MegaLoc 回环检索
  │    └─ RoMa v2 在既有关键点上查询对应 + 几何验证
  │
  ├─ Depth Anything 3：关键点深度、有效性和可选完整深度图
  │
  ├─ GeoCalib：共享焦距/内参先验
  │
  └─ mapper_inputs 边界
       │
       ├─ view-graph 过滤与标定
       ├─ 基于匹配和深度的相对位姿
       ├─ 来源感知 Rotation Averaging
       ├─ 深度一致性过滤
       ├─ 顺序骨架 + 回环软观测的 track 建立
       ├─ 两阶段、深度增强的 Global Positioning
       └─ 多轮、深度增强的 Bundle Adjustment
            │
            └─ COLMAP 稀疏模型 rec/
```

---

## 4. 当前实现依赖与技术栈

### 4.1 主要学习模块

| 任务 | 当前实现 | 作用 |
|---|---|---|
| 稠密匹配 | RoMa v2 v2.0.1，`precise` 模式 | 稠密 warp、置信度、每像素 precision/covariance |
| 回环检索 | MegaLoc | 给每个关键帧检索远距离候选 |
| 单目/视频深度 | Depth Anything 3 Video | 提供度量深度、有效性与可选置信度 |
| 相机先验 | GeoCalib | 共享内参/焦距初始化和不确定性 |
| 辅助显著点 | ALIKED | 补充 RoMa 置信度采样得到的稀疏点覆盖 |

仓库会在第一次前端运行时下载约 9 GB 模型权重。

### 4.2 几何与求解后端

- COLMAP 4.1 / PyCOLMAP：相机模型、重投影 cost、几何数据结构及输出格式。
- GLOMAP 风格全局建图：视图图、旋转平均、全局位置求解。
- Ceres Solver：Rotation Averaging、Global Positioning、BA 等主要非线性优化。
- 默认大规模线性求解器：CPU `SPARSE_SCHUR`。
- 可选：`DENSE_SCHUR`、`ITERATIVE_SCHUR`。
- 可选 GPU：Ceres 2.3+；稀疏 Schur 需要 CUDA + cuDSS。

重要判断：VidMap 的 BA 是 Ceres BA，但不是直接调用一行 `pycolmap.bundle_adjustment()`。项目提供 C++ native extension，自行构造 `ceres::Problem`，复用 COLMAP cost/manifold，并加入深度、尺度、内参先验和自定义鲁棒损失。

---

## 5. 前端实现细节

### 5.1 输入与总体入口

常用命令：

```bash
python -m vidmap.frontend INPUT_VIDEO_OR_IMAGE_DIR --output OUTPUT_DIR
python -m vidmap.map --mapper-inputs OUTPUT_DIR/mapper_inputs --output OUTPUT_DIR

# 或一次执行
python -m vidmap.run --input_data INPUT_VIDEO_OR_IMAGE_DIR --output OUTPUT_DIR
```

最终 COLMAP 模型位于：

```text
OUTPUT_DIR/rec/
```

前端主入口：

```text
vidmap/frontend/pipeline.py::Frontend
vidmap/frontend/tracking/composition.py::TrackingPipeline
```

当前 `TrackingPipeline.run()` 的顺序是：

1. 构建确定性的缓存/产物路径；
2. 关键帧低分辨率匹配和候选选择；
3. 高分辨率稀疏轨迹传播；
4. 顺序轨迹传递关系构建；
5. 回环检索、扩展匹配和 LC mask；
6. 深度预测并在关键点采样；
7. 可选 GeoCalib；
8. 过滤、几何验证并发布 `mapper_inputs`。

### 5.2 关键帧选择

VidMap 不按固定帧率抽帧，而是使用 RoMa v2 的低分辨率稠密 warp 估计帧间图像运动。

当前默认低分辨率匹配尺寸：

```text
560 × 560
```

每个候选关键帧上的稀疏点被传播到后续帧；当足够多的点满足下列情况时插入新关键帧：

- 归一化图像位移超过阈值；
- 因遮挡或低置信度失去跟踪；
- 触发其他强制/前视剪枝条件。

当前默认 `max_normalized_keypoint_drift` 约为 `0.11`；仓库 README 示例可覆盖为 `0.08` 等。关键帧判定现已基于相机内参归一化，这是当前仓库与论文最初实现之间的变化之一。

代码入口：

```text
vidmap/frontend/keyframes/processing.py
vidmap/frontend/keyframes/selection.py
vidmap/frontend/keyframes/selector.py
vidmap/frontend/options/keyframes.py
```

### 5.3 RoMa v2 稠密匹配输出

对图像对 `(Ia, Ib)`，RoMa v2 返回：

```text
warp_AB       : Ia 每个像素映射到 Ib 的坐标
overlap_AB    : 每像素匹配/重叠置信度
precision_AB  : 每像素 2×2 定位精度矩阵
```

VidMap 将 `precision_AB` 求逆得到定位协方差：

\[
\Sigma_{a\rightarrow b}(x)=P_{a\rightarrow b}(x)^{-1}.
\]

代码包装后的数据结构：

```python
RoMaMatch(
    matches,       # target-image pixel coordinates
    certainty,     # per-pixel certainty
    covariance,    # optional 2×2 covariance
)
```

关键代码：

```text
vidmap/frontend/models/romav2.py
vidmap/frontend/models/romav2_inference.py
```

当前模型固定信息：

```text
RoMa v2 version : v2.0.1
setting         : precise
bidirectional   : false
true high-res   : true
```

### 5.4 “稠密匹配”不等于“稠密 BA”

RoMa v2 输出的是稠密场，但 VidMap 最终仍建立稀疏 SfM tracks：

```text
稠密 warp / certainty / covariance
  → 从置信度图和显著点中采样稀疏源点
  → 在稠密 warp 上双线性查询目标坐标
  → 传播成带协方差的稀疏 tracks
  → 写入 COLMAP 风格关键点与匹配
  → Global Positioning / BA
```

当前轨迹默认参数中的重要值：

```text
max_kps                              = 1500
min_conf                            = 0.05
nms_radius                          = 3
max_sequential_track_sigma_roma_px  = 8.0
multiflow_hops                      = (8, 6, 4, 2, 1)
tvg_max_epipolar_error              = 4.0 px
```

这些数值是当前代码默认值，不应被当成论文不可改变的理论常数。

代码入口：

```text
vidmap/frontend/tracking/kernels.py
vidmap/frontend/tracking/state.py
vidmap/frontend/tracking/propagation.py
vidmap/frontend/options/tracking.py
```

### 5.5 稀疏点采样与覆盖

新轨迹点主要从 RoMa certainty map 中采样，并与 ALIKED 提出的显著点结合：

- 保留仍可传播的旧轨迹；
- 在旧轨迹附近做 NMS，避免重复；
- 丢弃低于置信度阈值的点；
- 使用密度感知的概率采样，使点分布相对均匀；
- 对纹理少但 RoMa 仍有可靠输出的区域保持覆盖。

这一步解释了为何 VidMap 不完全等同于 `SuperPoint/ALIKED + RoMa`：RoMa 的稠密场既负责找对应，也参与决定从哪里创建/延续稀疏轨迹。

### 5.6 顺序传播与 multi-flow 漂移校正

最基本的轨迹传播是：

\[
\hat{x}_i^{seq}=W_{i-1\rightarrow i}(x_{i-1}).
\]

如果假设每段位移误差独立，相邻传播的累计协方差近似为：

\[
\Sigma_i^{seq}=\Sigma_{i-1}^{seq}+\Sigma_{i-1\rightarrow i}.
\]

仅逐帧传播会累积漂移，因此 VidMap 还计算更早关键帧到当前帧的直接预测：

\[
\hat{x}_i^{(j)}=W_{j\rightarrow i}(x_j),
\qquad
\Sigma_i^{(j)}=\Sigma_j^{seq}+\Sigma_{j\rightarrow i}.
\]

先选择 trace 最小的候选：

\[
j^*=\arg\min_j \operatorname{tr}(\Sigma_i^{(j)}).
\]

但跨较远帧的直接匹配容易受到视觉混淆，因此只有在它与相邻传播预测一致时才接受：

\[
\|\hat{x}_i^{(j^*)}-\hat{x}_i^{seq}\|_2^2
< \tau\operatorname{tr}(\Sigma_i^{(j^*)}).
\]

否则保留相邻帧预测。当前代码的 hop 调度为 `(8, 6, 4, 2, 1)`，而论文使用一般窗口 `W` 描述。

关键代码：

```text
vidmap/frontend/tracking/multiflow.py
vidmap/frontend/tracking/long_track_refinement.py
vidmap/frontend/tracking/sparse_track_history.py
```

### 5.7 轨迹过滤

轨迹会在以下情况被过滤或终止：

- RoMa certainty 太低；
- 累计顺序定位标准差过大；
- 穿过遮挡/深度边缘而产生异常；
- 基础矩阵极线误差过大；
- 轨迹过度集中、破坏空间覆盖；
- 后端深度一致性判断为异常。

注意：协方差不仅用于过滤。进入 Global Positioning 时，2D 像素协方差会通过相机反投影和旋转 Jacobian 传播到 3D bearing covariance，从而对白化后的几何残差加权。

### 5.8 回环检索和匹配

回环流程：

1. 对每个关键帧提取 MegaLoc 全局描述子；
2. 检索 top-k 候选，排除已经由局部顺序窗口覆盖的 pair；
3. 使用 RoMa v2 对候选 pair 做高分辨率匹配；
4. 在现有稀疏关键点位置查询稠密 warp；
5. 将预测落到目标图既有关键点上；
6. 做几何验证和最少匹配数过滤；
7. 写出逐匹配的 loop-closure mask。

当前默认候选/过滤参数包括：

```text
nquery                 = 10
retrieval_min_score    = 0.1
tcorr_min_matches      = 200
lc_match_thresh        = 0.05
lc_pair_nms_radius     = 2
```

代码入口：

```text
vidmap/frontend/loop_closure/retrieval.py
vidmap/frontend/loop_closure/matching.py
vidmap/frontend/loop_closure/extended_matches.py
vidmap/frontend/models/megaloc.py
```

### 5.9 单目度量深度

论文将每个关键帧的单目度量深度记为：

\[
m_{ik}, \quad \sigma_{ik},
\]

即第 `i` 张图在第 `k` 个观测位置处的预测深度及不确定性。

当前仓库使用 Depth Anything 3 Video backend，默认：

```text
type              = da3_video
window_size       = 1
ref_view_strategy = middle
process_res       = 504
```

默认只持久化关键点位置的采样深度和有效性；若启用 `--cache-depth-maps`，会额外保留完整深度图。

代码入口：

```text
vidmap/frontend/depth.py
vidmap/frontend/depth_loading.py
vidmap/frontend/models/depth/da3_video.py
vidmap/frontend/options/depth.py
```

### 5.10 未标定视频

相机内参分两层估计：

1. GeoCalib 对多个高置信/均匀采样关键帧执行 shared-intrinsics 标定；
2. 后端 view-graph calibration 使用几何条件良好的图像对进一步细化；
3. BA 最终联合优化焦距、可选主点和额外畸变参数。

当前关键帧 bootstrap 会从完整序列均匀选择最多 30 帧，并要求这些帧具有一致图像尺寸。

代码入口：

```text
vidmap/frontend/geocalib.py
vidmap/frontend/preparation/camera_priors.py
vidmap/mapper/stages/view_graph_calibration.py
extensions/colmap/src/stages/view_graph_calibration.cc
```

---

## 6. “来源感知”是最关键的数据建模设计

### 6.1 为什么普通 track transitivity 会失败

传统 SfM 通常将匹配图做 union-find/传递闭包：

```text
A:a ↔ B:b
B:b ↔ C:c
=> A:a、B:b、C:c 属于同一个 track
```

如果 `B:b ↔ C:c` 是由重复走廊造成的错误回环，错误会把原本独立的顺序轨迹硬合并，后续优化难以知道污染来自哪里。

### 6.2 VidMap 的 track 语义

VidMap 先忽略所有 LC 匹配，仅使用顺序匹配建立 union-find track 骨架；随后将回环以额外观测附加：

```text
TrackRecord
  observations                 # 顺序骨架观测
  loop_closure_observations    # 回环软观测
  loop_closure_anchors         # 回环连接到的来源/锚点
```

这样错误回环不会把两条可靠顺序链永久焊接为一条 track。

关键代码：

```text
extensions/colmap/src/stages/track_establishment.cc
vidmap/mapper/stages/tracks.py
```

### 6.3 来源感知鲁棒损失

Rotation Averaging 中：

- 顺序边：Huber，较信任；
- 回环边：Cauchy，允许强降权。

Global Positioning 中，几何残差和深度残差都可以根据来源选择不同 loss。当前第一阶段默认大意为：

```text
sequential geometry : Huber
sequential depth    : Trivial
loop geometry       : Cauchy(scale=2, weight=0.4)
loop depth          : Cauchy(scale=2, weight=0.4)
```

第二阶段会放松/调整阈值并使用第一阶段结果 warm start。

重要实现事实：论文在 BA 阶段认为几何已接近正确，因而 BA 对观测统一使用鲁棒损失，不再按来源区分。来源感知最关键的作用阶段是 Rotation Averaging 和 Global Positioning，而不是最终 BA。

---

## 7. Mapper 的完整阶段顺序

当前代码的精确入口：

```text
vidmap/mapper/mapper.py::Mapper._solve()
```

顺序如下：

1. `ViewGraphFilter`
   - 移除不能进入标定与全局求解的边/观测。

2. `ViewGraphCalibrator`
   - 在未标定模式下根据 view graph 优化内参。

3. `GlobalPositioner.prepare_bearings`
   - 统一构建后续相对位姿和 GP 使用的 bearing 与 covariance。

4. `RelativePoseEstimator`
   - 估计两视图相对位姿；可利用已知深度提高退化运动稳定性。

5. `RotationAverager`
   - 进行两次 pass 的全局旋转平均；顺序边/回环边使用不同鲁棒损失。

6. `DepthConsistencyFilter`
   - 根据成对相对位姿传播深度，标记深度比值异常观测。

7. `TrackBuilder`
   - 顺序匹配建立骨架，回环作为软观测附加，再做 track problem filtering。

8. `GlobalPositioner`
   - 固定旋转，联合求解相机中心、3D 点、bearing 尺度和每图深度尺度。

9. `BundleAdjuster`
   - 联合细化位姿、点、内参和深度尺度；多轮过滤与鲁棒 loss annealing。

---

## 8. Rotation Averaging

对每条图像对边 `(i,j)`，从匹配和深度估计相对旋转 `R̃ij`，然后求全局旋转：

\[
\min_{\{R_i\}}
\sum_{(i,j)}
\rho_{ij}\left(
\left\|
\log(R_j^\top \tilde R_{ij}R_i)
\right\|
\right).
\]

其中 `ρij` 由边的来源决定。实现位于：

```text
vidmap/mapper/stages/rotation_averaging.py
extensions/colmap/src/stages/video_rotation_averaging.cc
```

这个阶段的意义是先稳住姿态，避免在位置和结构未知时一次性优化所有变量。

---

## 9. 深度增强的 Global Positioning

### 9.1 优化变量

固定 Rotation Averaging 得到的 `Ri`，GP 联合优化：

- 相机中心 `ci`；
- 稀疏三维点 `Xk`；
- bearing/观测尺度；
- 每张图的深度图尺度 `si`。

### 9.2 bearing 几何残差

令 `vik` 为观测对应的世界坐标系单位射线，`dik` 为沿射线的非负尺度：

\[
e^{GP}_{ik}=v_{ik}-d_{ik}(X_k-c_i).
\]

该残差使用从 2D 定位协方差传播得到的 bearing covariance 加权。

### 9.3 深度残差

相机坐标系中的几何深度：

\[
z_{ik}=e_3^\top R_i(X_k-c_i).
\]

正常情况下使用 log 比值：

\[
r^{depth}_{ik}=\log\frac{z_{ik}}{s_i m_{ik}}.
\]

log residual 的意义：

- 对相对比例误差对称；
- 远近尺度上的误差更均衡；
- 天然适合正值深度和乘性尺度。

### 9.4 深度尺度先验

网络提供近似度量尺度，但每张图允许校正：

\[
r_i^{scale}=\log s_i.
\]

因此系统不是：

```text
完全相信网络尺度
```

也不是：

```text
每图尺度完全自由
```

而是：

```text
以网络度量尺度为软锚点，同时允许多视图几何校正每图尺度
```

### 9.5 总体目标

可抽象为：

\[
E_{GP}=
\sum_{(i,k)}
\rho_{l_{ik}}^{geom}
\left(e_{ik}^{\top}\Sigma_{ik}^{-1}e_{ik}\right)
+
\sum_{(i,k)}
\rho_{l_{ik}}^{depth}
\left(\frac{r_{ik}^{depth}}{\sigma_{ik}}\right)
+
\sum_i
\rho^{scale}
\left(\frac{\log s_i}{\sigma_{s,i}}\right).
\]

`lik ∈ {seq, lc}` 表示来源标签。

### 9.6 当前代码是两阶段 GP

第一阶段：

- 随机/几何初始化位置；
- 强调顺序支持；
- 对回环使用较保守的 loss 和权重；
- 产生深度尺度与相机中心初值。

第二阶段：

- 使用第一阶段 warm start；
- 使用 log depth residual；
- 调整鲁棒尺度和误差阈值；
- 进一步筛选/稳定几何。

代码入口：

```text
vidmap/mapper/stages/global_positioning/positioner.py
vidmap/mapper/stages/global_positioning/native_options.py
extensions/colmap/src/stages/global_positioning.cc
vidmap/mapper/options/positioning.py
```

当前代码还包含一个默认权重为 0 的实验性 temporal-acceleration prior。它不属于论文的核心默认贡献，融合时不要误以为必须启用。

---

## 10. Bundle Adjustment 实现

### 10.1 结论

VidMap 使用传统 Ceres Solver 的非线性最小二乘框架，但问题定义由 VidMap 定制。

核心代码：

```text
vidmap/mapper/stages/bundle_adjustment/adjuster.py
vidmap/mapper/stages/bundle_adjustment/native_options.py
extensions/colmap/src/stages/bundle_adjustment.cc
extensions/colmap/src/stages/depth_prior.h
extensions/colmap/src/stages/intrinsics_prior.h
```

### 10.2 优化变量

\[
\{T_i\},\quad \{X_k\},\quad \{K_i\},\quad \{s_i\}.
\]

具体包括：

- 位姿：7 维 `quaternion xyzw + translation xyz`；
- 旋转：Eigen quaternion manifold；
- 点：3D world coordinates；
- 相机参数：由 COLMAP camera model 决定；
- 每图深度 shift/scale 参数块：当前常规路径固定 shift，优化 scale。

### 10.3 残差块

#### 重投影残差

\[
e^{BA}_{ik}=\pi(K_i,T_i,X_k)-x_{ik}.
\]

使用 COLMAP 的 `ReprojErrorCostFunctor`，并结合关键点定位不确定性/权重与鲁棒 loss。

#### 深度残差

\[
r^{depth}_{ik}=\log\frac{z_{ik}}{s_i m_{ik}}.
\]

实现为 `LogScaledDepthErrorCostFunctor`；异常、短 track 或小三角化角观测可以切换到更强的 Cauchy loss。

#### 深度尺度先验

约束每图尺度不要无依据地偏离 metric depth 的初始尺度。

#### 内参先验

对 GeoCalib/view-graph calibration 给出的内参使用标准差加权的 prior，最终允许 BA 修正焦距等参数。

### 10.4 gauge fixing

默认固定第一帧位姿以消除全局 SE(3) gauge；度量深度尺度先验进一步锚定全局尺度。代码还支持固定旋转、固定全部位姿和固定部分相机参数。

### 10.5 求解器

默认：

```text
linear_solver  = sparse_schur
preconditioner = schur_jacobi   # 仅 iterative_schur 真正使用
use_cuda       = false
```

GPU 可通过配置开启：

```bash
mapping.mapper.gp.solver_backend.use_cuda=true
mapping.mapper.ba.solver_backend.use_cuda=true
```

### 10.6 多轮 BA 和 graduated robustness

当前实现不是一次 BA 结束，而是：

- 常规 BA pass；
- 观测/点过滤；
- 更严格阈值的 refinement pass；
- 缩紧重投影和深度鲁棒 loss；
- 可选最终点 refinement。

当前默认 BA 主要配置：

```text
normal.iterations                    = 3
normal.kp_stddev                     = 2.0
normal.reproj_loss_name              = soft_l1
annealing.iterations                 = 3
annealing.kp_std                     = 0.5
annealing.reproj_loss_name           = soft_l1
depth.reg_loss_name                  = cauchy
depth.scale_reg_loss_name            = soft_l1
variable_point_track_length_threshold = 15
```

---

## 11. 关键数据与持久化边界

### 11.1 前端产物

`FrontendArtifacts` 主要包含：

```text
track_pairs        # 顺序传播使用的 pair
retrieval_pairs    # 回环检索 pair
sparse_features    # 稀疏关键点及不确定性
sparse_matches     # 顺序匹配
extended_matches  # 加入回环后的匹配
depth             # 关键点采样深度
full_depth         # 可选完整深度图
geocalib_batch     # 可选共享相机先验
```

前端最终发布 `mapper_inputs/`，而 Mapper 只依赖这一稳定边界。对于融合改造，这是比直接调用内部 Python 类更好的接口。

### 11.2 输出

```text
OUTPUT_DIR/
  mapper_inputs/
  rec/                         # 最终 COLMAP model
  frontend_config.yaml         # 解析后的完整前端配置
  mapping_config.yaml          # 解析后的完整后端配置
  playback_trace/              # 可选求解过程快照
  depth scales / caches        # 视运行选项而定
```

### 11.3 缓存契约

当前实现大量使用 artifact fingerprint、模型 commit/checkpoint hash、输入文件顺序 hash 和配置语义 hash。修改前端模型、图像分辨率、关键帧计划或匹配参数后，应主动使缓存 identity 变化，不能复用旧 H5/数据库产物。

---

## 12. 代码导航表

| 想理解/修改的模块 | 首要入口 |
|---|---|
| 一次运行 | `vidmap/run.py` |
| 前端总编排 | `vidmap/frontend/pipeline.py` |
| 学习前端阶段顺序 | `vidmap/frontend/tracking/composition.py` |
| RoMa v2 包装 | `vidmap/frontend/models/romav2.py` |
| 稠密场查询和稀疏采样 | `vidmap/frontend/tracking/kernels.py` |
| multi-flow | `vidmap/frontend/tracking/multiflow.py` |
| 轨迹状态机 | `vidmap/frontend/tracking/state.py` |
| 回环检索 | `vidmap/frontend/loop_closure/retrieval.py` |
| 回环匹配 | `vidmap/frontend/loop_closure/matching.py` |
| 深度 | `vidmap/frontend/depth.py` |
| GeoCalib | `vidmap/frontend/geocalib.py` |
| Mapper 总编排 | `vidmap/mapper/mapper.py` |
| 相对位姿 | `vidmap/mapper/stages/relative_pose/` |
| Rotation Averaging | `vidmap/mapper/stages/rotation_averaging.py` |
| track 来源建模 | `extensions/colmap/src/stages/track_establishment.cc` |
| Global Positioning | `vidmap/mapper/stages/global_positioning/` |
| GP Ceres 实现 | `extensions/colmap/src/stages/global_positioning.cc` |
| BA Python 策略 | `vidmap/mapper/stages/bundle_adjustment/adjuster.py` |
| BA Ceres 问题 | `extensions/colmap/src/stages/bundle_adjustment.cc` |
| 默认配置结构 | `vidmap/frontend/options/`, `vidmap/mapper/options/` |
| Ceres backend 选择 | `vidmap/mapper/options/solver.py` |

---

## 13. 论文实验告诉我们的工程优先级

LaMAR 消融中的 full W-AUC：

| 版本 | full W-AUC |
|---|---:|
| 完整 VidMap | 88.5 |
| GP 中移除 depth | 37.7 |
| 不优化每图 depth scale | 39.7 |
| 放弃 metric scale，只由多视图决定 scale | 74.1 |
| 移除 loop closure | 85.5 |
| 不区分 seq / LC loss | 79.9 |

工程含义：

1. 对长视频而言，**GP 初始化中的深度和尺度优化是第一优先级**；只在最终 BA 加深度不够。
2. 回环有用，但“保留来源 + 差异化鲁棒损失”比单纯增加回环边更重要。
3. 最终精度来自前端轨迹质量与全局优化的共同作用，不能只移植其中一个函数就期待完整收益。
4. 在短、标定准确、视差充分的数据上，深度先验收益较小，甚至可能带来噪声；必须保留消融开关。

---

## 14. 与本工作区 FeedForwardWithBA 的关系

### 14.1 当前工作区已有能力

根据 `/kiri/FeedForwardWithBA/README.md`，当前主流程是：

```text
Stage A：Pi3X + MERG3R
  → 分块前馈 pose / intrinsics / depth
  → subset 全局对齐
  → coarse pair graph

Stage B：GlueMap 风格稀疏精修
  → SIFT database
  → VGGSfM prior tracks（ALIKED query）
  → prior tracks snap 到 SIFT
  → 三角化 / track selection / filtering
  → BAE 或 Ceres BA
```

两套系统的共同点：

- 都是“学习前端 + 显式几何后端”；
- 都能得到 coarse pose、intrinsics、depth 和稀疏 tracks；
- 都使用 COLMAP 模型/数据库作为重要交换格式；
- 都有多轮三角化、过滤和 BA；
- 都关心长序列、尺度漂移和退化运动。

主要差异：

| 维度 | 当前 FeedForwardWithBA | VidMap |
|---|---|---|
| 初始全局几何 | Pi3X/MERG3R feed-forward + subset alignment | RA + depth-augmented GP |
| 轨迹前端 | SIFT + VGGSfM prior tracks | RoMa v2 稠密场采样和时间传播 |
| pair graph | 主要由 coarse pose 邻接构造 | 时间邻接 + MegaLoc 回环 |
| 来源建模 | 当前主要按数据库/track 合并 | 显式 seq / LC provenance |
| 深度参与位置 | 主要为 Stage A 和后处理/分析 | GP 和 BA 都有显式深度残差与每图 scale |
| BA 默认后端 | BAE（PyTorch/LM/PCG），可选 Ceres | 定制 Ceres |
| 视频结构 | 可使用序列/rig 信息，但主稀疏 refinement 不等同于 VidMap | 时间顺序是核心一等信息 |

### 14.2 最有价值的可移植思想

按预期收益和工程风险排序：

1. **seq / LC 来源保留与差异化鲁棒 loss**。
2. **每图深度尺度变量 + metric scale prior**。
3. **在 BA 之前加入 depth-augmented Global Positioning**。
4. **RoMa v2 时间轨迹替换或补充 VGGSfM/SIFT tracks**。
5. **运动感知关键帧和 multi-flow covariance**。

---

## 15. 推荐的融合路线

### 路线 A：先移植思想到现有管线，推荐

这是风险最低且便于归因的路线。

#### Phase 0：固定基线与交换格式

- 固定现有 Stage A/Stage B 基线 commit、命令和数据集。
- 保存每阶段 COLMAP model、数据库、轨迹数、重投影误差和轨迹 ATE/RPE。
- 定义统一的 `image_id ↔ 时间索引 ↔ rig/frame id ↔ image name` 映射。

#### Phase 1：给现有 pair/observation 增加 provenance

在当前 `utils/gluemap_refine_core.py` 的数据库构建和 track selection 边界增加：

```text
pair_type        ∈ {sequential, pose_neighbor, loop_closure, rig_same_time}
observation_type ∈ {sequential, loop_closure, prior_track, sift}
```

第一步不改变 BA，仅保证标签能贯穿数据库、track 构建和日志。

#### Phase 2：增加来源感知 loss

- 对 pose graph / global solve 中的回环边使用更重尾 loss；
- 对顺序边使用 Huber/Soft-L1；
- 不要让错误回环通过 union-find 直接硬合并两条顺序 track；
- 可先在 Ceres 路线验证，再移植到 BAE。

#### Phase 3：在 BAE 中加入深度 scale 变量

当前 BAE `ColmapResidual` 主要返回重投影误差。建议新增：

```text
per-image log scale α_i，令 s_i = exp(α_i)
log depth residual = log(z_ik) - log(s_i * m_ik)
scale prior        = α_i / σ_s
```

总损失：

\[
L=L_{reproj}+\lambda_d L_{depth}+\lambda_s L_{scale}.
\]

必须同时带入：

- depth validity；
- depth uncertainty/confidence；
- behind-camera 检查；
- 小三角化角/短 track 的更强鲁棒 loss；
- depth loss 的独立 annealing。

不要简单把原始 feed-forward depth 作为固定 `z` 真值。

#### Phase 4：加入独立 Global Positioning 初始化

固定 Stage A 或 RA 旋转，先优化 camera centers、points 和 depth scales，再进入现有三角化/BA 循环。

若直接复用 VidMap native GP，需要先满足它的数据契约；否则可在 BAE/PyTorch 中实现简化版本用于验证。

#### Phase 5：接入 RoMa v2 时间轨迹

建议先让 RoMa tracks 与现有 VGGSfM/SIFT tracks 并存：

- RoMa tracks 作为 sequential backbone；
- SIFT/VGGSfM 作为显著点补充或独立证据；
- MegaLoc 或 coarse pose pair graph 产生 loop candidates；
- 对重复/冲突观测不要直接 union；
- 逐步比较覆盖、轨迹长度、定位协方差和最终重建。

### 路线 B：直接调用 VidMap mapper

把当前 Stage A/Stage B 前端结果适配成 VidMap `mapper_inputs/`，然后调用 VidMap mapper。

优点：

- 快速获得 RA + GP + BA 完整能力；
- 直接复用来源感知和深度尺度优化；
- Ceres 实现成熟。

缺点：

- `mapper_inputs` 契约和 native extension 耦合较深；
- 当前本项目的多相机 rig 语义可能不满足 VidMap 假设；
- 需要统一 feature index、LC mask、深度采样、相机模型和 pose convention；
- 构建 COLMAP 4.1 + 定制 extension + Ceres 2.3 的环境成本较高。

### 路线 C：整体引入 VidMap，再把 MERG3R 作为初始化先验

可将 Pi3X/MERG3R 的全局姿态或中心作为 VidMap GP 的 warm start，保留 VidMap 前端和后端。该路线潜在上限高，但改动最大，不适合作为第一步。

---

## 16. 融合时必须处理的语义问题

### 16.1 坐标与位姿 convention

必须明确每个边界使用：

```text
w2c 还是 c2w
translation t 还是 camera center c
quaternion 顺序 xyzw 还是 wxyz
左乘还是右乘
相机 z 轴朝向
```

VidMap/COLMAP常见关系：

\[
x_c=R x_w+t,
\qquad
c=-R^\top t.
\]

任何融合适配器都必须加 round-trip test：`w2c → center/rotation → w2c`。

### 16.2 feature id 对齐

VidMap 的 LC mask 与 `all_matches` 行严格对齐；track 观测通过 `(image_id, point2D_idx)` 编码。融合时以下操作都会破坏对齐：

- 在写 mask 后重新排序 matches；
- snap 到 SIFT 后未更新 point2D index；
- 数据库 merge 时去重但未同步 provenance；
- filter 只过滤 matches，不过滤对应 mask/covariance/depth。

建议所有过滤函数返回同一个 row mask，并一次性应用到：

```text
matches
provenance
certainty
covariance
depth/depth_validity
```

### 16.3 深度单位和尺度

确认深度是：

- camera-z depth，还是 Euclidean ray distance；
- metric depth，还是 scale-ambiguous inverse depth；
- 原图像素坐标采样，还是 resize/crop 后坐标采样；
- 针孔投影视图、鱼眼视图还是 ERP 上的深度。

VidMap 的公式使用 camera-frame z depth。若输入是 ray distance `d`，需根据归一化 ray 的 z 分量转换。

### 16.4 相机模型

当前 FeedForwardWithBA 主路线多使用 shared `SIMPLE_PINHOLE`，而 pano/dual-fisheye preparation 会展开为多视图 rig。VidMap可复用 COLMAP camera models，但 GeoCalib bootstrap、关键帧运动归一化和深度采样都隐含特定成像模型/图像尺寸假设。

### 16.5 多相机 rig：当前最大的直接复用障碍

当前 VidMap GP 代码明确要求：

```text
global positioning requires one image per frame
```

而本工作区已有 Cubemap5、Overlap3、双鱼眼 Fish9 等“同一时间多个相机/投影视图”的 rig 数据。因此不能未经修改直接把所有 rig images 当作 VidMap 的普通视频帧送入 GP。

可选处理：

1. 第一阶段仅对 center 视图运行 VidMap，再将 rig 外参传播到其他视图；
2. 将 GP 的优化变量从 per-image center 改为 per-rig-frame body center/pose，同一 frame 的多个 camera 通过固定 rig extrinsics 关联；
3. 暂时只移植来源感知和 depth-scale BA，不复用原始 GP；
4. 把同时间 rig edge 单独标为 `rig_same_time`，其 loss 和 gauge 不能与 temporal/LC edge 混同。

对于本项目，建议先采用方案 1 或 3，再评估方案 2。

### 16.6 动态物体

VidMap 依赖稠密匹配和几何验证，但论文主线没有把动态物体建模为独立运动。普通公开视频中，大面积动态前景仍可能产生一致但错误的局部 tracks。融合时可复用现有 mask、语义过滤或基于多帧重投影的动态观测剔除。

---

## 17. 建议新增的统一中间数据模型

为了避免直接耦合 VidMap H5、COLMAP DB 与 BAE tensor，可先定义项目内中立结构：

```python
ImageRecord:
    image_id
    name
    time_index
    frame_id
    camera_id
    width, height
    intrinsics_prior
    pose_prior

ObservationRecord:
    image_id
    feature_id
    xy
    covariance_2x2
    certainty
    depth
    depth_stddev
    depth_valid

PairMatchRecord:
    image_id1, image_id2
    feature_ids1, feature_ids2
    provenance          # sequential / loop / pose-neighbor / rig
    geometric_inlier

TrackRecord:
    sequential_observations
    loop_observations
    loop_anchors
```

然后分别实现：

```text
VidMap mapper_inputs adapter
COLMAP database adapter
BAE tensor adapter
debug/audit serializer
```

这是融合中最值得先做的结构性工作。

---

## 18. 测试与诊断清单

### 18.1 前端轨迹

- 每帧关键点数分布；
- track 长度分布；
- 每个 hop 的使用比例；
- RoMa certainty 与实际极线误差关系；
- covariance trace 与重投影误差是否单调相关；
- seq/LC pair 数和 inlier ratio；
- 纹理少、运动模糊、重复结构区域的覆盖图。

### 18.2 来源感知

- 每条 match 的 provenance 是否贯穿数据库、track 和 GP；
- 禁用 LC 后顺序骨架是否保持不变；
- 注入错误 LC 后是否只降权 LC，不破坏 seq track；
- union-find 是否完全忽略 LC，LC 是否只作为软观测附加。

### 18.3 深度与尺度

- `geometric z / predicted depth` 每帧中位数；
- 优化前后 `s_i` 曲线；
- `log s_i` 是否出现异常跳变；
- 深度残差按 track length、三角化角、图像区域分桶；
- 关闭 metric prior、固定 scale、自由 scale 三种消融。

### 18.4 优化

- 初始/final cost；
- 各类 residual block 数量；
- behind-camera point 数；
- 每轮过滤的 observation/point 数；
- Ceres termination type；
- gauge 固定后 Hessian/PCG 是否仍病态；
- BA 前 GP 解是否已经在正确拓扑和尺度附近。

### 18.5 最终指标

- ATE / RPE；
- 按 10/25/50/100 m 窗口计算的局部轨迹误差；
- scale drift；
- 回环前后拓扑正确性；
- 注册帧比例；
- 重投影误差中位数/P90；
- 峰值显存、RAM、运行时间随关键帧数的增长。

---

## 19. 最小消融矩阵

融合后至少保留下列开关：

| ID | seq tracks | LC | provenance loss | depth in GP | depth in BA | scale optimize |
|---|---:|---:|---:|---:|---:|---:|
| A | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| B | ✓ | ✗ | - | ✓ | ✓ | ✓ |
| C | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| D | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| E | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ |
| F | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| G | 现有 VGGSfM/SIFT | 现有 pair graph | ✗ | ✗ | ✗ | ✗ |

这样可以区分收益来自：轨迹前端、回环、来源建模、GP 初始化、BA 深度约束还是尺度优化。

---

## 20. 后续 Agent 的推荐阅读顺序

如果只有 30 分钟：

1. 本文第 1～3 节：建立系统概念。
2. 第 6 节：理解 provenance-aware tracks。
3. 第 9～10 节：理解 GP 与 BA 的差别。
4. 第 14～16 节：理解与当前项目融合的障碍。

如果要改前端：

1. `vidmap/frontend/tracking/composition.py`
2. `vidmap/frontend/models/romav2.py`
3. `vidmap/frontend/tracking/state.py`
4. `vidmap/frontend/tracking/multiflow.py`
5. `vidmap/frontend/loop_closure/`

如果要改后端：

1. `vidmap/mapper/mapper.py`
2. `vidmap/mapper/options/positioning.py`
3. `vidmap/mapper/stages/global_positioning/native_options.py`
4. `extensions/colmap/src/stages/global_positioning.cc`
5. `vidmap/mapper/stages/bundle_adjustment/adjuster.py`
6. `extensions/colmap/src/stages/bundle_adjustment.cc`

如果要接入本工作区：

1. `/kiri/FeedForwardWithBA/README.md`
2. `/kiri/FeedForwardWithBA/run_merg3r_gluemap_pipeline.py`
3. `/kiri/FeedForwardWithBA/utils/gluemap_spv_refine.py`
4. `/kiri/FeedForwardWithBA/utils/gluemap_refine_core.py`
5. `/kiri/FeedForwardWithBA/bae_pipe.py`
6. `/kiri/FeedForwardWithBA/algos/bundle_adjustment.py`

---

## 21. 最终建议

对于当前 `FeedForwardWithBA`，不要第一步就整体替换成 VidMap。更稳健的融合顺序是：

```text
先保留现有 Pi3X/MERG3R coarse geometry
  → 为现有匹配和 track 增加 seq/LC/rig provenance
  → 在 BAE/Ceres 中实现差异化鲁棒 loss
  → 加入每图深度 scale 和 log-depth residual
  → 验证后再引入独立 GP
  → 最后评估 RoMa v2 时间轨迹是否替换 VGGSfM/SIFT
```

原因：论文消融显示，最大的长程收益来自“深度增强的 GP + 可优化深度尺度”，而错误回环的主要风险来自来源丢失。先移植这两层语义，能够在保留现有稳定前馈初始化和多相机 rig 支持的同时，逐步获得 VidMap 的核心能力，并让每一步都可以独立消融和回退。

---

## 22. 外部资料

- 论文 PDF：<https://arxiv.org/pdf/2607.27194>
- 官方代码：<https://github.com/cvg/vidmap>
- RoMa v2：<https://arxiv.org/abs/2511.15706>
- GLOMAP：论文参考文献 [42]，VidMap mapper 的主要全局 SfM 基础。
- COLMAP：<https://colmap.github.io/>
- Ceres Solver：<https://ceres-solver.org/>

