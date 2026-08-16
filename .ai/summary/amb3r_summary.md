# AMB3R-SfM 项目概述

**论文**：[AMB3R: Accurate Feed-forward Metric-scale 3D Reconstruction with Backend (arXiv:2511.20343)](https://arxiv.org/abs/2511.20343)  
**项目页**：[hengyiwang.github.io/projects/amber](https://hengyiwang.github.io/projects/amber)

## 定位

AMB3R 是一个**前馈式 metric-scale 3D 重建**框架，有三个版本：

| 版本 | 入口 | 说明 |
|------|------|------|
| Base | `demo.py` | 单次前向推理，交互式可视化 |
| AMB3R-VO | `slam/run.py` | 流式 SLAM（Visual Odometry） |
| **AMB3R-SfM** | `sfm/run.py` | 本文重点，离线全图 SfM |

---

## sfm/run.py 整体流程

```
输入：图片文件夹（--data_path）
    ↓  Demo Dataset → DataLoader
    ↓  AMB3R 模型加载（base model）
    ↓  pipeline = AMB3R_SfM(model)
    ↓  pipeline.run(images)   ← 核心入口
    │   ├─ 阶段 0: 特征提取 + 图像聚类
    │   ├─ 阶段 1: 地图初始化（最佳 cluster）
    │   ├─ 阶段 2: 粗配准（Coarse Registration）
    │   └─ 阶段 3: 全局精化（Global Mapping）
    ↓  输出：poses[T,4,4] + pts[T,H,W,3] + conf[T,H,W]
    ↓  后处理：置信度过滤 + 边缘/天空 mask + 随机降采样
    ↓  保存：scene_*.ply + scene_*_results.npz
```

---

## pipeline.run() 四个阶段详解

### 阶段 0：特征提取 + 图像聚类（`extract_features` + `find_best_image_clustering`）

```python
feature_descriptors = self.extract_features(images)   # [T, C] 每帧全局描述子
distance_matrix = get_distance_matrix(descriptors, whitening=True)  # [T, T]
clusters = find_best_image_clustering(distance_matrix, min_size, max_size)
# → {keyframe_idx: [member_indices]}
```

**聚类算法**（`image_clustering`，运行 50 次取最优）：
1. **初始过聚类**：FPS（最远点采样）确定初始关键帧，每帧分配给最近关键帧
2. **迭代合并**：反复找最小簇，与最高相似度的邻簇合并，直到所有簇 ≥ min_size
3. **精化**：重新分配越簇收益更高的帧，更新 medoid（簇的最中心点为关键帧）

聚类评估指标：簇内紧凑度 + 关键帧间连通性（越低越好）。

**处理每个 cluster**（`process_clusters`）：对每个 cluster 调用基模型推理，取 置信度 × 与其他簇相似度 最高的 cluster 作为初始化种子。

---

### 阶段 1：地图初始化（`initialize_map`）

```python
# 对最佳 cluster 的所有帧排列，找置信度最高的帧作为第一关键帧
for member_idx in cluster_member_indices:
    idx_to_use = [member_idx] + other_members + [cluster_kf_idx]
    res = model.run_amb3r_sfm(views)   # → {world_points, pose, conf}
    # 若第 0 帧置信度更高，则 member_idx 成为新的起始关键帧

self.keyframe_memory.initialize(cluster_pred_all, best_kf_idx)
```

初始化后 `SfMemory`（关键帧内存）持有：初始 cluster 所有帧的 `poses[T,4,4]`、`pts[T,H,W,3]`、`conf[T,H,W]`。

**local_mapping 的通用模式**：每次调用均是将一组帧（KF anchors + 待注册帧）打包成 `views`，送入 `model.run_amb3r_sfm()`，获取所有帧的 pointmap 和 pose，再由 `SfMemory.update_kf()` 对齐到全局坐标系后融合。

---

### 阶段 2：粗配准（`coarse_registration`）

逐 cluster 注册到已有全局地图，支持多轮迭代处理低置信度帧。

**主循环逻辑**：
```
while 还有未注册的 clusters:
    1. rank_clusters_by_distance：按与全局 KF cluster 的视觉距离排序，取 top-k 候选
    2. register_candidate_cluster：将候选 cluster 与最近的 global KF cluster 合并推理
       → views = global_kf_frames + candidate_frames
       → local_mapping → SfMemory.update_kf 对齐 + 更新
       → 若多个 global KF cluster 均与候选帧重叠，保留置信度最高的结果
    3. 将置信度达标帧提交到 SfMemory，低置信度帧延迟处理
    4. 若某帧被多次推理后仍低置信，标记为 unmapped_frames
```

**关键帧晋升机制**：若 cluster 的关键帧本身被融合进全局地图（成为已知 KF），则从剩余成员中选视觉最近者晋升为新 KF。

**最终兜底**（`remap_unmapped_frames`）：对所有未映射帧，挨个尝试与每个全局 KF cluster 配准，取置信度最高的结果。

---

### 阶段 3：全局精化（`global_mapping`）

运行 `max_global_refinement_iters` 轮，每轮依次：

#### 3a. 关键帧精化（`keyframe_mapping`，BFS 遍历）

```
从置信度最高的 KF 出发，BFS 展开：
    当前 KF → 找距离 ≤ max_kf_search_distance 的邻近 KF（最多 max_kf_per_refinement 个）
    → views = [当前KF] + [邻近KF]
    → local_mapping → 对齐后更新当前 KF 的 pts + pose
    → 将邻近 KF 加入队列（优先队列，按置信度降序）
```

#### 3b. 非关键帧精化（`non_keyframe_mapping`）

```
按置信度降序处理所有非 KF 帧：
    → 找同样是非 KF 的邻近帧（non_kf_search_window 内）
    → 再找最近的 KF anchors（max_kf_search_distance 内）
    → views = KF anchors + 非KF窗口帧
    → local_mapping → 以 KF 为基准对齐，更新非KF帧的 pts + pose
```

---

## 模型接口（`run_amb3r_sfm` / `run_amb3r_vo`）

AMB3R-SfM 是一个**通用框架**，任何能输出以下格式的前馈模型都可接入：

```python
def run_amb3r_sfm(self, frames, cfg, keyframe_memory=None):
    images = frames['images']  # (B, T, C, H, W)，[-1, 1]

    # 使用 keyframe_memory 中已有 KF 的 pose/pts 作为条件（anchor）
    # 推理新帧并与 anchor 对齐

    return {
        'world_points': pointmap,         # (B, T, H, W, 3)
        'world_points_conf': confidence,  # (B, T, H, W, 1)
        'pose': pose,                     # (B, T, 4, 4)，c2w
    }
```

`frames['kf_idx']` 指定哪些帧是已知 KF（anchor），模型会以这些帧为条件推理剩余帧。

---

## 输出结构

| 字段 | Shape | 说明 |
|------|-------|------|
| `poses_pred` | `[T, 4, 4]` | 所有帧的 c2w 位姿 |
| `pts_pred` | `[T, H, W, 3]` | 像素级 3D 世界坐标 |
| `conf` | `[T, H, W]` | 置信度（用于过滤噪声点） |
| `kf_idx` | `list` | 关键帧索引 |
| `unmapped_frames` | `set` | 注册失败帧索引（从点云中排除） |

最终保存 `.ply`（过滤 + 降采样后的点云）和 `.npz`（完整原始结果）。
