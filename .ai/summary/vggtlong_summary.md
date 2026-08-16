# VGGT-Long 项目概述

**论文**：[VGGT-Long: Chunk it, Loop it, Align it (arXiv:2507.16443)](https://arxiv.org/abs/2507.16443)（ICRA 2026）  
**关联项目**：Pi-Long / DA3-Streaming / Map-Long

## 定位与核心思想

VGGT-Long 解决的核心问题：**前馈式 3D 模型（VGGT/Pi3/MapAnything）只能处理几十帧，无法直接应用于公里级长序列**（如 KITTI 4500 帧）。

**三步走策略**（标题 "Chunk it, Loop it, Align it"）：

```
1. Chunk it  → 把长序列切成有重叠的短 chunk，分块推理
2. Loop it   → 检测回环帧对，跨 chunk 估计 SIM(3) 约束
3. Align it  → 全局 SIM(3) 位姿图优化，对齐所有 chunk 到统一坐标系
```

**无需**相机标定、深度监督或模型重训练，是纯 inference-time 的工程扩展。

---

## 整体流程（`vggt_long.run()`）

```
输入：图片文件夹（任意长度）
    ↓
Step 0: 回环检测（提前全量处理，节省后续重复推理）
    SALAD (DINOv2) 或 DBoW2 → loop_list = [(frame_i, frame_j), ...]
    ↓
Step 1: 加载基础模型（VGGT / Pi3 / MapAnything）
    ↓
Step 2: process_long_sequence()
    ├─ 2a. 分块推理（Chunk it）
    │       每 chunk：model.infer_chunk(frames) → 结果存磁盘 _tmp_results_unaligned/
    ├─ 2b. 回环对推理（Loop it）
    │       对每个回环帧对（frame_i, frame_j）→ 窗口化为 chunk_a + chunk_b
    │       → 再次推理这两段帧 → 估计 SIM(3)_a 和 SIM(3)_b → 合成 SIM(3)_ab
    ├─ 2c. 相邻 chunk 对齐（顺序 Align）
    │       使用重叠帧的 world_points 估计相邻 chunk 的 SIM(3) 变换
    │       weighted_align_point_maps(overlap_points_1, overlap_points_2) → (s, R, t)
    ├─ 2d. 回环约束全局优化（Loop Optimize）
    │       Sim3LoopOptimizer.optimize(sim3_list, loop_sim3_list) → 全局对齐的 sim3_list
    └─ 2e. 应用对齐，保存点云
            accumulate_sim3_transforms → 每个 chunk 的累积变换
            apply_sim3_direct(world_points) → 变换后保存 .ply
    ↓
Step 3: save_camera_poses()
    对每个 chunk 的 extrinsic，用对应的 SIM(3) 变换到全局坐标
    → camera_poses.txt（每行 4×4 c2w 矩阵展平）
    ↓
Step 4: merge_ply_files() → combined_pcd.ply（所有 chunk 合并）
    ↓
Step 5: close()：删除磁盘临时文件
```

---

## 关键设计细节

### 1. 分块策略（Chunk it）

```python
chunk_size = 60   # 每 chunk 帧数
overlap = 30      # 相邻 chunk 重叠帧数
step = chunk_size - overlap  # = 30，每次滑动 30 帧
chunk_indices = [(0,60), (30,90), (60,120), ...]
```

- **重叠帧的作用**：相邻 chunk 共享 `overlap` 帧，这部分帧会被推理两次，用于估计两个 chunk 坐标系之间的 SIM(3) 变换。
- **磁盘中转**：每个 chunk 推理完立即存磁盘（`.npy`），不在 GPU/CPU 内存中累积，避免长序列 OOM（4500 帧约需 50 GiB 磁盘）。

### 2. SIM(3) 对齐（Align it）

相邻 chunk 的对齐通过**重叠区点云配准**实现：

```python
# 取 chunk_k 末尾 overlap 帧 vs chunk_{k+1} 开头 overlap 帧
s, R, t = weighted_align_point_maps(
    point_map1[-overlap:],   # world_points, (overlap, H, W, 3)
    conf1[-overlap:],         # 置信度加权
    point_map2[:overlap],
    conf2[:overlap],
    conf_threshold = min(median_conf1, median_conf2) * 0.1
)
```

使用 SIM(3)（7-DoF，含尺度）而非 SE(3)（6-DoF），因为不同 chunk 独立推理会有任意尺度差异。  
**MapAnything 有 metric scale 输出时，推荐切换为 SE(3) 对齐**（`using_sim3: False`）。

积累变换：`accumulate_sim3_transforms(sim3_list)` 将 $(s_1, R_1, t_1), (s_2, R_2, t_2), \ldots$ 串联成每个 chunk 相对第 0 chunk 的全局变换。

### 3. 回环检测与优化（Loop it）

**检测**（二选一）：
- **SALAD**（DINOv2，GPU）：相似度 > 0.85 判定为回环，NMS 去重
- **DBoW2**（ORB 词袋，CPU）：适合无 GPU 时

**回环帧对处理**：
```python
loop_list = [(frame_i, frame_j), ...]
→ 对每对回环帧：取各自周围 loop_chunk_size=10 帧窗口
→ 将两段帧拼成一个 batch 推理 → 得到统一坐标系下的 world_points
→ 分别与原 chunk 的重叠区做 SIM(3) 对齐 → 得到 (s_a, R_a, t_a) 和 (s_b, R_b, t_b)
→ 合成 SIM(3)_ab（chunk_a 到 chunk_b 的约束）
```

**全局优化**：`Sim3LoopOptimizer` 以顺序 SIM(3) 链 + 回环 SIM(3) 约束，做非线性最小二乘（LM 算法，支持 C++/Python 两种实现），输出优化后的 `sim3_list`。

---

## 支持的基础模型

通过 Adapter 模式统一接口，只需实现 `model.infer_chunk(image_paths)` 返回：

```python
{
    'world_points':      # [T, H, W, 3]
    'world_points_conf': # [T, H, W]
    'extrinsic':         # [T, 3, 4]（w2c）
    'intrinsic':         # [T, 3, 3]
    'depth':             # [T, H, W, 1]
    'images':            # [T, 3, H, W]
    'mask':              # [T, H, W]（可选，天空/边缘遮罩）
}
```

| 模型 | 特点 | 对齐方式 |
|------|------|---------|
| **VGGT** | 默认，泛用性强 | SIM(3) |
| **Pi3** | 更高精度，Pi-Long 基础 | SIM(3) |
| **MapAnything** | 多模态输入，metric scale | SE(3) 更稳 |

---

## 输出文件结构

```
exps/<scene>/<datetime>/
├── camera_poses.txt        # 每行 16 个数，4×4 c2w 矩阵，全序列所有帧
├── intrinsic.txt           # 每行 fx fy cx cy
├── camera_poses.ply        # 相机位置可视化点云
├── sim3_opt_result.png     # 回环优化前后轨迹对比图（有回环时生成）
└── pcd/
    ├── 0_pcd.ply           # chunk 0 点云（已过滤 + 采样）
    ├── 1_pcd.ply           # chunk 1 点云（已对齐）
    ├── ...
    └── combined_pcd.ply    # 所有 chunk 合并后的完整点云
```

---

## 与同类项目对比

| 项目 | 扩展思路 | 回环 | 尺度对齐 |
|------|---------|------|---------|
| **VGGT-Long** | Chunk + SIM(3) 对齐 + 回环优化 | SALAD/DBoW | SIM(3)/SE(3) |
| VGGT-SLAM | Submap + SL(4) 位姿图 + SALAD 回环 | SALAD+VGGT验证 | SL(4) |
| AMB3R-SfM | 图像聚类 + 多轮全局精化 | 无 | 隐含在模型内 |
| LingBot-Map | KV Cache 因果注意力 + Anchor Context | 无 | 无（端到端） |
