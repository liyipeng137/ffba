1. 🤔 MERG3R 提出了一种无需训练的分而治之框架，旨在使现有神经视觉几何模型能够处理超出其原生内存限制的大规模无序图像集合。
2. 🛠️ 该框架通过将无序图像重排序为伪视频并划分为重叠的子集进行独立重建，随后通过高效的全局对齐和置信度加权Bundle Adjustment来合并局部结果。
3. 🚀 实验结果表明，MERG3R显著提升了7-Scenes、NRGBD、Tanks & Temples和Cambridge Landmarks等大型数据集上的重建精度、内存效率和可扩展性。

---

## 代码主流程（`main.py`）

```
输入图像
  │
  ▼
1. process_images()              # 加载图像，可选下采样 (--subsample)
  │
  ▼
2. create_sequence()             # 图像排序与子集划分
  │  --sequence_type video       →  VideoSequence：基于时序滑动窗口
  │  --sequence_type shortest_path → ShortestPathSequence：DINO相似度矩阵
  │                                  + 最短路径 / MST 排序
  │  --splitting_type            →  interleave(默认) / zigzag / threshold 等
  │  --subset_size=55, --overlap=5
  │
  ▼
3. run_inference_step_by_step()  # 对每个子集独立运行几何基础模型
  │  --model vggt（默认）
  │  输出：每个子集的 extrinsic / intrinsic / depth / depth_conf
  │
  ▼
4. align_extrinsics()            # 子集对齐到统一全局坐标系
  │  --alignment_type weighted_iterative（默认）/ umeyama
  │  对重叠帧用 Huber 损失 + IRLS 求解 Sim(3) 变换
  │
  ▼
5. [可选] --global_ba
  ├─ unproject_depth_map_to_point_map()   # 深度图 → 世界系3D点
  ├─ extract_matches_lightglue()          # video 序列：step=[1,2,3,5,7,10]
  │  graph_extract_matches_lightglue()    # shortest_path + graph 模式：k-NN图
  │  （SuperPoint特征 + LightGlue匹配 + 3D重投影误差过滤 + 并查集轨迹合并）
  └─ global_bundle_adjustment()          # 联合优化 R/t/K/3D点
     │  置信度加权重投影误差，L0.5 范数
     │  --lr 1e-4, --epoch 300, --max_reproj 8.0
  │
  ▼
6. write_recon_to_colmap()       # 输出 COLMAP 格式重建结果
   --stride 100, --point_vis_threshold 50.0, --format txt/bin
```

---

## 子集对齐算法详解（`weighted_iterative_alignment`）

默认对齐方法 `--alignment_type weighted_iterative` 的核心思想：**用模型预测的深度置信度作为初始权重，再用 Huber-IRLS 迭代剔除外点，在 Sim(3) 空间内求解子集间的相似变换。**

### 完整步骤

```
对每条边 (parent_id → child_id)：
  │
  ├─ 1. 取重叠帧的深度图、内外参、置信度
  │      depth_conf 归一化到 [0,1]
  │      conf_threshold = min(70th-percentile_A, 70th-percentile_B)
  │      → 保留置信度最高的 30% 像素
  │
  ├─ 2. 深度图反投影到世界系 3D 点图
  │      A = unproject(depth_A, extri_A, intri_A)   ← 目标帧
  │      B = unproject(depth_B, extri_B, intri_B)   ← 源帧
  │
  ├─ 3. weighted_align_point_maps(A, conf_A, B, conf_B)
  │      ├─ 过滤：保留两侧置信度均 > threshold 的像素
  │      ├─ 点对权重 w_i = sqrt(conf_A_i × conf_B_i)  ← 几何均值
  │      └─ robust_weighted_estimate_sim3(B_pts, A_pts, w)
  │           ├─ 初始化：加权 SVD → Sim(3) (s₀, R₀, t₀)
  │           └─ IRLS 循环（max 5 次）：
  │                residual_i = ||A_i - (s·R·B_i + t)||
  │                huber_w_i  = 1            if r_i ≤ δ
  │                           = δ / r_i      if r_i > δ
  │                combined_w = init_conf_w × huber_w  (归一化)
  │                重新求解加权 Sim(3)
  │                收敛条件：Δparams < tol 且 Δrot < 0.1°
  │
  └─ 4. 存储 (R, t, s)，最后 transform_to_shared_frame()
         将所有局部 Sim(3) 链式传播到统一全局坐标系
```

### 与 Umeyama 方法对比

| | `weighted_iterative`（默认）| `umeyama` |
|---|---|---|
| 点对权重 | sqrt(conf_A × conf_B) | 无（或按深度置信度阈值过滤）|
| 外点处理 | Huber-IRLS 迭代重加权 | RANSAC（可选）|
| 鲁棒性来源 | 模型置信度 + 残差自适应 | 随机采样一致性 |
| 计算开销 | 较低（5次迭代，有Numba加速）| RANSAC需500+次迭代 |

---

## 全局 BA 算法详解（`gradient_bundle_adjustment`）

核心思想：**以 LightGlue 跟踪轨迹为观测，用深度置信度加权的 Smooth-L1（Huber）重投影误差，通过 Adam + 余弦退火梯度下降，联合优化所有相机外参和 3D 点坐标。**

### 完整步骤

```
输入：3D点 (P,3)、外参 (N,3,4)、内参 (N,3,3)
      轨迹 track[N] 各图像的观测像素坐标
      每条轨迹的深度置信度 points_conf (P,)
      最大重投影阈值 max_reproj_error=8.0px
  │
  ├─ 1. 置信度预过滤
  │      保留 points_conf > 30th-percentile 的轨迹
  │
  ├─ 2. 参数化
  │      旋转 R  →  单位四元数 (roma)       ← SO(3) 流形约束
  │      平移 t  →  直接向量
  │      焦距 f  →  log(f)                  ← 防止负焦距
  │      主点 pp →  直接向量
  │      3D点 P  →  直接坐标（requires_grad=True）
  │      默认 shared_camera=True → 全图共享一组 K
  │      默认 optimize_intrinsics=False → K 固定不优化
  │
  ├─ 3. 预过滤轨迹（初始重投影误差过滤）
  │      将初始 3D 点投影到每张图像
  │      投影误差 > max_reproj_error(8px) 的观测标记为无效
  │      生成 track_mask (N, P_max) bool
  │
  ├─ 4. 梯度下降优化（Adam + CosineAnnealing, epoch=300）
  │      每步：
  │        K, w2c = make_intri_extri(log_f, pp, quat, t)
  │        points_pixel = project_3d_points(P[points_id], w2c, K)
  │        valid = track_mask & in_image_mask
  │        │
  │        loss = reprojection_loss(points_pixel, tracks, valid, conf_weights)
  │               ├─ Smooth-L1(δ=2px) 逐观测计算像素误差
  │               ├─ × 深度置信度权重 (P,) per-track
  │               ├─ × valid_mask（过滤出界/标记无效观测）
  │               └─ / 有效权重之和（归一化均值）
  │        │
  │        loss.backward() → Adam.step()
  │        quat ← F.normalize(quat)   ← 保持单位四元数
  │
  └─ 5. 输出优化后的 extrinsic, intrinsic, points3D
```

### 关键设计

| 设计点 | 实现 | 意义 |
|---|---|---|
| 旋转参数化 | 单位四元数 + 每步归一化 | 无约束优化中保持 SO(3) |
| 焦距参数化 | log(f) | 防止优化到负值 |
| 损失函数 | Smooth-L1 (δ=2px) | 像素级 Huber，鲁棒于匹配噪声 |
| 轨迹权重 | 深度置信度 (per-track) | 利用模型预测质量 |
| 预过滤 | 初始重投影 > 8px 屏蔽 | 防止外点主导优化 |
| 优化变量 | R, t, 3D点（K 固定） | 完整 BA，非仅位姿优化 |
| 调度器 | CosineAnnealing (eta_min=lr) | 学习率恒定（eta_min=lr 效果等同常数） |

> **与论文描述的差异**：论文公式使用 L0.5 范数，代码实际为 Smooth-L1（δ=2px），相当于 Huber 损失，对小残差是 L2、大残差线性，比 L0.5 更稳定。

### 关键参数速查

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--subset_size` | 55 | 每个子集的图像数 |
| `--overlap` | 5 | 相邻子集重叠帧数 |
| `--sequence_type` | video | video / shortest_path |
| `--splitting_type` | interleave | 子集划分策略 |
| `--alignment_type` | weighted_iterative | 子集对齐方法 |
| `--global_ba` | False | 是否启用全局BA |
| `--tracking_type` | graph | graph / video（仅 shortest_path 模式） |
| `--model` | vggt | 底层几何基础模型 |

MERG3R论文介绍了一种名为MERG3R的无训练（training-free）分而治之（divide-and-conquer）框架，旨在解决现有神经视觉几何模型在处理大规模无序图像集时面临的内存和可扩展性瓶颈。特别是像VGGT和Pi3这样的基于 Transformer 的模型，由于其完全注意力（full attention）机制，计算和内存成本随输入图像数量呈二次方增长（$O(N^2)$），严重限制了其在实际大规模场景中的应用。MERG3R通过重新组织、分区、局部重建和高效全局对齐，使得几何基础模型（geometric foundation models）能够超越其原生内存限制，实现高质量的大规模三维重建。

该方法的核心思想可以分解为以下几个关键步骤：

1.  **图像集排序与分区（Image Set Ordering and Partitioning）**:
    *   **伪时间序列构建（Pseudo-temporal Ordering）**: 面对大量无序图像 $I = (I_i)_{i=1}^N$，MERG3R首先通过构建一个伪时间序列来最大化视觉连续性。它计算一个密集的视觉相似性矩阵 $M \in R^{N \times N}$，其中 $M_{i,j}$ 表示图像 $I_i$ 和 $I_j$ 之间的DINO-based视觉相似性。这个矩阵被视为一个加权完全图，目标是近似找到一条哈密顿路径 $P^* = (p_1, \dots, p_N)$，使得连续帧之间的相似性之和最大化：
        $$P^* = \arg \max_{P} \sum_{k=1}^{N-1} M_{p_k, p_{k+1}}$$
        对于特别长的序列，还会引入DINO相似性约束来选择下一帧，以避免选择过于不相似的图像。
    *   **交错采样与子集划分（Interleaved Sampling and Partitioning）**: 为了确保每个子集包含足够的视角变化以进行鲁棒的多视角立体（multi-view stereo），并保持相邻子集间的充分重叠以便后续对齐，MERG3R采用交错采样策略。它将有序序列 $P^*$ 重排为 $\tilde{P}$，通过从 $K$ 个目标子序列中循环抽取帧：$\tilde{P}_i = P^*_{\{(i \pmod K) \cdot K + \lfloor i/K \rfloor\}}$. 这种方式确保了每个集群都包含了来自整个序列的丰富视角多样性，而不是只包含时间上相邻的类似视角。然后，通过一个固定长度 $T$ 的滑动窗口，以 $T-O$ 的步长（$O$ 为期望的重叠量）在 $\tilde{P}$ 上滑动，生成一系列局部可管理、相互重叠的子集 $S_k$。每个子集定义为 $S_k = \{ \tilde{P}_i \mid i \in [k(T-O), k(T-O)+T) \}$。

2.  **局部重建（Local Reconstruction）**:
    *   每个分区后的子集 $S_k$ 由一个预训练的几何基础模型 $F_g$ 独立进行重建。模型输出包括相机参数 $G_k$、深度图 $D_k$ 和置信度分数 $C_k$，即 $F_g(S_k) = (G_k, D_k, C_k)$。
    *   通过将图像集划分为大小为 $T$ 的 $K$ 个子集，原始模型的 $O(N^2)$ 注意力复杂度降低到 $O(KT^2)$，相当于 $O(N^2/K)$。这显著减少了峰值GPU内存需求，并允许并行处理不同子集的重建。

3.  **集群对齐（Cluster Alignment）**:
    *   由于子集是独立重建的，它们需要被对齐到一个统一的全局坐标系中。MERG3R采用了VGGT-Long中使用的加权迭代相似变换估计器（weighted iterative similarity-transform estimator）。
    *   对于每对重叠的相邻子集 $S_k$ 和 $S_{k+1}$，首先识别对应的3D点 $\{(p_i^k, p_i^{k+1})\}$ 及其置信度分数 $\{(c_i^k, c_i^{k+1})\}$。滤除置信度低于某个百分位阈值 $\tau_{\text{conf}}$ 的点。
    *   通过求解一个最小化Huber损失（Huber loss）的目标函数，计算将 $S_{k+1}$ 对齐到 $S_k$ 的相似变换 $T \in \text{Sim}(3)$：
        $$T^*_{k,k+1} = \arg \min_{T \in \text{Sim}(3)} \sum_i \rho ||p_i^k - T p_i^{k+1}||^2$$
        其中 $\rho(\cdot)$ 是Huber损失函数。该优化问题通过迭代重加权最小二乘法（IRLS）求解，权重 $w_i^{(t)}$ 在每次迭代中根据残差 $r_i^{(t)}$ 和置信度 $c_i$ 更新：$w_i^{(t)} = c_i \rho'(r_i^{(t)})/r_i^{(t)}$，其中 $r_i^{(t)} = ||p_i^k - T^{(t)} p_i^{k+1}||^2$。

4.  **跟踪（Tracking）**:
    *   为后续的全局光束法平差（Global Bundle Adjustment）提供精确的像素对应关系。为避免二次方复杂度的朴素匹配，对于每个子集 $S_k$，MERG3R使用相似性矩阵 $M$ 构建稀疏 k-NN 图。对于保留的每条边 $(i, j)$，提取 SuperPoint 特征并使用 LightGlue 进行匹配。
    *   为减少错误对应，原始匹配会被提升到3D空间，并通过几何一致性检查进行过滤。具体来说，通过预测的深度图 $D_i$ 将原始对应关系 $\{(x_i^{m,n}, x_j^{u,v})\}$ 反投影到3D，然后将其重投影到配对视图中。双向重投影误差超过阈值 $\tau_{\text{reproj}}$ 的匹配将被丢弃。
    *   剩余的对应关系通过不相交集并操作（disjoint-set union）合并成多视图轨迹 $T_l = (x_{l,i_1}^{m_1,n_1}, x_{l,i_2}^{m_2,n_2}, \dots)$。每条轨迹的3D位置 $x_l$ 和置信度 $C_l$ 通过以下加权平均得到：
        $$x_l = \frac{\sum_{k=1}^{L_l} C_{i_k}[m_k, n_k] x_{l,i_k}^{m_k,n_k}}{\sum_{k=1}^{L_l} C_{i_k}[m_k, n_k]}, \quad C_l = \frac{\sum_{k=1}^{L_l} C_{i_k}[m_k, n_k]}{L_l}$$
        其中 $x_{l,i_k}^{m_k,n_k}$ 是每个像素对应的3D点，$L_l$ 是轨迹的长度。这种跟踪方法的复杂度与图像数量呈线性关系 $O(kN)$。

5.  **全局光束法平差（Global Bundle Adjustment）**:
    *   为进一步提升3D重建质量并保持全局一致性，MERG3R引入了一个高效的全局光束法平差步骤，联合优化相机内参、外参和3D点位置。
    *   优化基于合并后的多视图轨迹 $T$，通过梯度下降迭代 $\nu$ 次，最小化所有3D点在相机上的置信度加权2D重投影误差：
        $$R^*, t^*, K^*, P^* = \arg \min_{R,t,K,P} L_{BA}$$
        $$L_{BA} = \sum_{(T_l,x_l,C_l) \in T} C_l \sum_{y_{l,i} \in T_l} ||y_{l,i} - \pi_i(x_l)||^{0.5}$$
        其中 $\pi_i : \mathbb{R}^3 \to \mathbb{R}^2$ 是3D点在图像 $I_i$ 平面上的投影。与MASt3R-SfM等只优化图像对的方法不同，MERG3R在所有视图上优化，从而提高了全局一致性和准确性。

MERG3R的贡献在于其训练无关、模型无关的特性，可与任何预训练的几何基础模型结合，显著提升其在内存效率、可扩展性和重建精度方面的表现，尤其是在数据集超出GPU内存容量限制时。实验结果表明，MERG3R在7-Scenes、NRGBD、Tanks & Temples和Cambridge Landmarks等大规模数据集上，无论是在相机姿态估计还是点云质量方面，都持续优于或媲美现有State-of-the-Art方法，并大幅降低了运行时长和内存消耗。