# GGPT Dense Point Cloud 精化逻辑

> 核心思路：将稀疏但精确的 DLT 点（产自 BA 优化 pose）与稠密但有偏的 ff 点**混合**送入 Point Transformer，借助 DLT 点的几何约束力，精化 ff 的每像素 3D 坐标。

---

## 数据准备（坐标系对齐）

**文件**：`GGPT/ggpt/dataloader/base_dataset.py` → `load_scene_()` L30–61

```python
# ff_pts 与 geo_pts（DLT）坐标系不同，先用 Umeyama Sim3 对齐
scene['ff_pts'] = umeyama_alignment(
    B=scene['geo_pts'],   # 目标坐标系：DLT（由 BA 优化 pose 三角化）
    A=scene['ff_pts'],    # 待对齐：前馈模型预测的 dense 点
    mask=scene['geo_msks']
)[0]
```

对齐后两类点在同一坐标系下，才能混合输入 Transformer。

---

## 场景切块（Chunk）

**文件**：`GGPT/ggpt/dataloader/base_dataset.py` → `sample_a_chunk()` L71–102

每个 chunk 包含空间邻域内的点：
```python
a_chunk = {
    'ff_pts':      scene['ff_pts'][msk_chunk],       # dense 初值（待精化）
    'ff_pts_conf': scene['ff_conf'][msk_chunk],       # ff 置信度
    'geo_pts':     scene['geo_pts'][msk_chunk],       # DLT 稀疏点（几何约束）
    'geo_msks':    scene['geo_msks'][msk_chunk],      # DLT 有效性 mask
    'msks_in_scene': msk_chunk,                       # 在完整场景 [N,H,W] 中的索引
}
# 归一化到 chunk_center/chunk_radius，使坐标在 [-1,1] 附近
a_chunk[key] = (a_chunk[key] - chunk_center) / chunk_radius
```

---

## 模型输入编码（关键：DLT 如何注入）

**文件**：`GGPT/ggpt/model/base.py` → `embed_input()` L103–161

每个点的特征向量：

```
ff 点的 feature = [
    ff_type_embed,          # 可学习 type token，区分"ff点" vs "dlt点"
    sinusoidal(ff_xyz),     # ff 点自身坐标的位置编码
    ff_conf,                # ff 置信度（1维）
    sinusoidal(dlt_xyz),    # 对应像素位置的 DLT 点坐标编码（几何约束注入）
    (ff_xyz - dlt_xyz),     # ff 到 DLT 的残差向量（显式偏差信号）
]
# shape: [K, type_dim + 3+sin_dim*3 + 1 + 3+sin_dim*3 + 3]

dlt 点的 feature = [
    dlt_type_embed,         # 可学习 type token，区分"dlt点" vs "ff点"
    sinusoidal(dlt_xyz),    # dlt 坐标编码
    (zeros),                # 其余位置填 0
]
```

DLT 约束的注入方式是**双路径**：
1. **隐式**：DLT 点作为独立节点加入点云，与 ff 点一同进入 Point Transformer，通过注意力机制交互
2. **显式**：每个 ff 点的特征中直接编码了 `sinusoidal(dlt_xyz)` 和 `(ff_xyz - dlt_xyz)`，让模型感知到偏差方向和幅度

> 若对应像素无有效 DLT 点（`geo_msks=False`），则 `dlt_xyz_emb` 和 `delta` 置零（zero-pad），不影响其他点。

最终输入 Transformer 的点云 = ff 点 + DLT 点拼接：
```python
feat = torch.cat([ff_pts_feat, dlt_pts_feat], dim=0)  # (K+N, in_channel)
coor = torch.cat([ff_pts_xyz,  dlt_pts_xyz ], dim=0)  # (K+N, 3)
```

---

## 模型前向（残差预测）

**文件**：`GGPT/ggpt/model/base.py` → `forward()` L209–221

```python
pt_out = self.backbone(model_input)          # Point Transformer V3
head_out = self.head(pt_out_feat + model_input['feat'])  # skip connection
delta_xyz, conf = head_out[:,:3], head_out[:,3].exp()+1

# 残差预测：在输入坐标基础上加偏移
xyz_out = model_input['coord'] + delta_xyz   # 对所有点（ff+dlt）都预测残差
```

零初始化最后一层权重：
```python
torch.nn.init.constant_(self.head[-1].weight, 0)
torch.nn.init.constant_(self.head[-1].bias,   0)
# → 训练初始阶段输出纯 0，等价于直接输出对齐后的 ff_pts，稳定训练
```

输出解包时只取 ff 点的部分（DLT 点的输出不用于最终点云）：
```python
ff_pts_out      = xyz_out[start_idx : start_idx+num_ff]    # 精化后的 dense 点
ff_pts_conf_out = conf_out[start_idx : start_idx+num_ff]
```

---

## 多 Chunk 结果聚合

**文件**：`GGPT/utils/points.py` → `aggregate_chunks()` L10–57

```python
# 每个 chunk 通过 msks_in_scene 将点映射回完整场景的 [N*H*W] 索引
pts_sum.index_add_(0, flat_index, chunk_pts)    # 累加（重叠区域多次累加）
counts.index_add_(0, flat_index, ones)

# 取均值；未被覆盖的像素用原始 ff_pts（对齐后）填充
pred_pts[valid] = pts_sum[valid] / counts[valid]
pred_pts = pred_pts.view(N, H, W, 3)
```

---

## 最终输出与保存

**文件**：`GGPT/run_demo.py` L155–164

```python
# 反归一化：chunk 坐标系 → 世界坐标系
pred = unnormalize_pts(chunk, out['ff_pts_out'])
     = out['ff_pts_out'] * chunk_radius + chunk_center

# 置信度过滤 + 随机降采样
pred_mask = filter_points(pred_pts, pred_confs, None,
                          max_pts_num, conf_quantile_thresh)

# 保存三份 ply 做对比
save_xyzrgb_to_ply(sfm_outputs['points'][sfm_masks], ...)   # sfm_dlt_points.ply
save_xyzrgb_to_ply(ff_outputs['points'][pred_mask],  ...)   # ff_points.ply      (对齐前原始ff)
save_xyzrgb_to_ply(pred_pts[pred_mask],              ...)   # ggpt_points.ply    (最终输出)
```

---

## 整体数据流

```
ff_outputs['points']  [N,H,W,3]   前馈初始预测（dense，有系统偏差）
        │
        ▼  Umeyama Sim3 对齐到 SfM 坐标系
        │  base_dataset.py: load_scene_()
        │
        ▼  空间切块（random/octree）
        │  base_dataset.py: split_scenes_*()
        │
        ├── ff_pts（归一化）   ┐
        └── geo_pts（归一化）  ┤ → embed_input()
                               │   双路径注入：隐式混合 + 显式残差编码
                               ▼
                    Point Transformer V3（backbone）
                               ↓
                    head: 预测 Δxyz（残差）
                               ↓
                    ff_pts_out = ff_pts + Δxyz（反归一化）
                               ↓
                    aggregate_chunks(): 多 chunk 均值聚合
                               ↓
pred_pts  [N,H,W,3]   精化后的 dense 点云（ggpt_points.ply）
```
