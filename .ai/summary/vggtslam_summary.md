# VGGT-SLAM 2.0 项目概述

**论文**：
- VGGT-SLAM 1.0: [arXiv:2505.12549](https://arxiv.org/abs/2505.12549)（NeurIPS 2025）
- VGGT-SLAM 2.0: [arXiv:2601.19887](https://arxiv.org/abs/2601.19887)

**机构**：MIT SPARK Lab（Dominic Maggio, Luca Carlone）

## 定位

VGGT-SLAM 是一个**基于前馈式模型 VGGT 的稠密 RGB SLAM 系统**。核心思路：用 VGGT 的单次前向推理替代传统 SLAM 中的深度估计+位姿跟踪，在 **SL(4) 流形**上做全局位姿图优化，支持回环检测和可选的开集 3D 语义检索。

---

## main.py 整体流程

```
输入：图片文件夹
    ↓  RAFT 光流 → 关键帧选取（disparity 过滤）
    ↓  每攒满 submap_size 帧（默认 16）
    ↓  Solver.run_predictions()
    │   ├─ VGGT 前向推理（整批帧）→ pose_enc, depth, depth_conf
    │   ├─ pose_enc → extrinsic[S,3,4] + intrinsic[S,3,3]
    │   └─ 回环检测（SALAD 图像检索）→ 若命中，再次 VGGT 推理两帧对
    ↓  Solver.add_points()
    │   ├─ 深度图反投影 → world_points[S,H,W,3]
    │   ├─ 构建 Submap（位姿 + 点云 + 颜色 + 置信度）
    │   └─ 添加图节点/边（普通 + 回环）
    ↓  PoseGraph.optimize()（GTSAM SL(4) 非线性优化）
    ↓  Viser 可视化更新
    ↓  下一批（保留 overlapping_window_size=1 帧重叠）
```

---

## 核心组件

### 1. VGGT 前馈推理（`solver.run_predictions`）

```python
images = load_and_preprocess_images(image_names)   # [S, 3, H, W]
predictions = model(images)                         # 一次前向，全批推理
extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], ...)
# extrinsic: [S, 3, 4]，w2c
# depth: [S, H, W, 1]，深度图
# depth_conf: [S, H, W]，置信度
```

VGGT 对每个 submap 的所有帧**整批**推理，一次得到该批所有帧的位姿和深度，无需逐帧跟踪。

---

### 2. Submap（子地图）

每个 Submap 包含：
- `poses`: `[S, 4, 4]` w2c 矩阵
- `pointclouds`: `[S, H, W, 3]` 世界坐标点云（深度图反投影得到）
- `colors`: `[S, H, W, 3]` RGB 颜色
- `conf` / `conf_threshold`: 置信度 + 百分位过滤阈值
- `proj_mats`: `K_4x4`，投影矩阵（用于 SL(4) 分解）
- `retrieval_vectors`: SALAD 图像嵌入（用于回环检索）
- `semantic_vectors`: CLIP 图像嵌入（可选，用于语义查询）

Submap 之间通过**重叠帧**（默认 1 帧）做尺度对齐和约束连接。

---

### 3. 位姿图优化（PoseGraph，基于 GTSAM SL(4)）

**SL(4) 流形**（特殊线性群，4×4 行列式=1 的矩阵）是 VGGT-SLAM 的核心创新：

```
传统 SLAM: SE(3) = R ∈ SO(3) + t ∈ R³  （刚体变换）
VGGT-SLAM: SL(4) = 4×4 投影变换矩阵    （包含任意尺度/剪切）
```

使用 SL(4) 而非 SE(3) 的原因：VGGT 的每个 submap 内部有任意射影变换关系（而非严格欧式变换），SL(4) 能更好地表达这种内部一致但跨 submap 尺度不一的结构。

约束类型：
- **Prior Factor**：第一帧固定在原点（锚定）
- **Inner Submap**：同一 submap 内相邻帧的相对变换
- **Intra Submap**：相邻 submap 重叠帧之间的约束（需估计尺度）
- **Loop Closure**：回环帧对的约束

**跨 submap 尺度估计**（`scale_solver.py`）：
```python
# 重叠帧的点云在两个 submap 坐标系下的坐标，通过最小二乘估计尺度因子
scale_factor = estimate_scale_pairwise(points_in_curr_frame, points_in_prev_frame)
```

---

### 4. 回环检测（`loop_closure.py`）

- **图像检索**：使用 **SALAD**（基于 DINOv2 的图像检索模型）提取全局描述子，通过余弦相似度找到历史相似帧
- **验证**：将候选帧对输入 VGGT 推理，检查 `image_match_ratio`（图像匹配比例，< 0.85 则拒绝）
- **回环 Submap**：通过验证的回环会创建独立的 LC Submap，插入到 PoseGraph 并添加约束边

---

### 5. 开集语义检索（可选 `--run_os`,我们不需要）

- **CLIP + Perception Encoder**：为每帧图像提取语义嵌入，存入 Submap
- **SAM3**：收到文本查询后，用 CLIP 检索最相关帧，再用 SAM3 分割目标
- **3D 包围盒**：将分割 mask 对应的点云计算 OBB，在 Viser 中可视化

---

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--submap_size` | 16 | 每个 submap 的新帧数 |
| `--overlapping_window_size` | 1 | 相邻 submap 重叠帧数 |
| `--min_disparity` | 50 | 关键帧选取的最小光流位移（像素） |
| `--conf_threshold` | 25.0 | 过滤低置信度点的百分位阈值 |
| `--lc_thres` | 0.95 | 回环检测相似度阈值（越高=越多回环） |
| `--max_loops` | 1 | 每 submap 最大回环数 |

---

## 与普通前馈模型的接入差异

VGGT-SLAM 的 VGGT 调用与标准 VGGT 用法完全一致（`model(images)`），区别在于：

1. **分批处理**：不是所有帧一次推理，而是每 `submap_size` 帧一批
2. **输出利用**：只使用 `pose_enc`、`depth`、`depth_conf`，不使用 `world_points`（自己用深度图反投影重算）
3. **回环时**：额外将两帧拼成 batch=2 再推理一次 VGGT，利用 `image_match_ratio` 验证回环
