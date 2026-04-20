# GGPT 项目概述

**论文**：[Geometry-Grounded Point Transformer (CVPR 2026, arXiv:2603.11174)](https://arxiv.org/abs/2603.11174)  
**模型**：[HuggingFace YutongGoose/GGPT](https://huggingface.co/YutongGoose/GGPT)

## 定位

GGPT 是一个**多视图 3D 重建**方法，以前馈模型（VGGT/Pi3 等）的初始预测为基础，通过密集匹配 + 稀疏 BA + DLT 三角化构建稀疏 SfM 约束，再用 Point Transformer 对每像素 3D 点进行几何引导的精化，输出高质量密集点云。

---

## run_demo.py 整体流程

```
输入：图片文件夹
    ↓  1. FeedForward 推理（VGGT/Pi3/DAv3/MapAnything）
    │   → ff_outputs: {images_ff, extrinsics, intrinsics, points[N,H,W,3], points_conf}
    ↓  2. run_sfm()（密集匹配 + 稀疏BA + DLT三角化）
    │   → sfm_outputs: {extrinsics, intrinsics, points[N,H,W,3], point_masks[N,H,W]}
    ↓  3. GGPT Point Transformer 精化
    │   → pred_pts[N,H,W,3]（每像素精化后的3D点）
    ↓  输出：sfm_dlt_points.ply / ff_points.ply / ggpt_points.ply
```

---

## FeedForward 模块（`feedforward/__init__.py`）

`FeedForward_Model` 统一封装多个前馈模型：

| 模型 | 说明 |
|------|------|
| `vggt-point` / `vggt-depth` | VGGT（Facebook），主力方案 |
| `pi3` / `pi3x` | Pi3/Pi3X |
| `dav3` | Depth Anything v3 |
| `ma` | MapAnything |

**统一输出格式** `ff_outputs`：

| 字段 | Shape | 说明 |
|------|-------|------|
| `images_ff` | `[N, H, W, 3]` | 归一化到模型输入分辨率（518×? px）的图像 |
| `extrinsics` | `[N, 4, 4]` | w2c 外参矩阵（OpenCV 惯例） |
| `intrinsics` | `[N, 3, 3]` | 相机内参矩阵（cx = (W-1)/2） |
| `points` | `[N, H, W, 3]` | 每像素 3D 世界坐标（前馈初始预测） |
| `points_conf` | `[N, H, W]` | 置信度 |

---

## run_sfm() 详解

`run_sfm(images, ff_outputs, match_models, cfg)` 是 GGPT 的核心 SfM 模块，负责构建稀疏几何约束。分三步：

### 第 1 步：密集匹配（Dense Matching）

```python
match_results = match_images(match_models, images_hr, lr_h, lr_w, hr_to_lr)
```

**匹配模型**：RoMa / RoMaV2 / UFM，支持多模型集成（按最小循环误差选最优匹配）。

**坐标空间**：匹配在高分辨率图像上进行，结果重新映射到低分辨率（ff 分辨率 `ff_h×ff_w`）。使用 `mres_to_fres` 仿射变换矩阵（含 0.5 像素中心偏移）对应双坐标系。

**`match_results` 结构**（所有 index 均在低分辨率坐标系下）：

| 字段 | Shape | 说明 |
|------|-------|------|
| `pred_matches_lr` | `[Ntgt, Nsrc, H*W, 2]` | 源图每像素在目标图的对应 2D 坐标（xy，低分辨率） |
| `pred_scores` | `[Ntgt, Nsrc, H*W]` | 匹配置信度（0~1） |
| `pred_cycle_error` | `[Ntgt, Nsrc, H*W]` | 循环一致性误差（像素）：src→tgt→src 的往返误差 |
| `sp_scores` | `[Nsrc, H*W]` | SuperPoint 角点响应分，用于选重要 track |

> **关键设计**：匹配结果为稠密格式——源图每个像素都有与所有目标图的对应关系，因此 `H*W` 即为潜在 track 的数量上界（= `N*ff_h*ff_w`）。

---

### 第 2 步：稀疏 BA（Bundle Adjustment）

目标：用一部分高质量 track 优化相机位姿和焦距，以 ff 的预测为初值。

#### 2.1 过滤与构建 BA 用 Track

```
M_ba = (pred_scores > ba_score_thresh) & (pred_cycle_error < ba_cycle_thresh)
M_ba: [Ntgt, Nsrc*H*W]（bool）= [Ntgt, Ntracks_all]
```

每个 `track_id`（0 ~ N×H×W-1）对应源图 `src = track_id // (H*W)` 的像素 `(x,y)`。

**Track 选取策略**（贪心，保证每视图覆盖量）：

```python
for 每个视图 ni:
    to_select_num = mintrack_per_view - tracknum_perview[ni]  # 当前视图还差多少
    candidate_tracks = M_ba[ni]            # 在 ni 中可见
                     & (M_ba.sum(0) >= 2)  # 至少 2 个视图可见（可三角化）
                     & (~selected)         # 未被选中
    # 按 SuperPoint 分数降序选取
    selected[selected_ids] = True
    tracknum_perview += M_ba[:, selected_ids].sum(axis=1)  # 更新所有视图覆盖量
```

**BA 用数据结构**：

```python
tracks_ba:     [N, Ntracks_ba, 2]   # 每个 track 在每个视图的 2D 坐标（低分辨率）
tracks_mask_ba:[N, Ntracks_ba]      # bool，该 track 在该视图是否可见
pts3d_ba:      [Ntracks_ba, 3]      # 3D 点初值 = ff 的 points[...].reshape(-1,3)[selected]
```

> **关键**：3D 点初值直接取 ff 预测的对应像素 3D 坐标，即 `ff_outputs['points'].reshape(-1,3)[selected]`，其中 `selected` 是 track 在平铺后的 `N*H*W` 数组中的索引。

#### 2.2 pycolmap Bundle Adjustment

```python
reconstruction = batch_torch_matrix_to_pycolmap(
    points3d=pts3d_ba,
    tracks=tracks_ba + 0.5,       # 转为像素中心坐标
    masks=tracks_mask_ba,
    extrinsics=ff_outputs['extrinsics'][:,:3,:4],
    intrinsics=ba_intrinsics,     # ff 预测的内参（或 gt）
    camera_type='SIMPLE_PINHOLE', # 或 PINHOLE
    shared_camera=True,           # 所有视图共享内参
)
bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_options, ba_config, reconstruction)
bundle_adjuster.solve()
```

输出：优化后的 `extrinsics[N,4,4]` 和 `intrinsics[N,3,3]` 写入 `sfm_outputs`。

---

### 第 3 步：DLT 三角化（Direct Linear Triangulation）

目标：用 BA 优化后的相机将所有高质量匹配点三角化为稠密 3D 点云，最终将 3D 点坐标**写回像素坐标格**（与 ff_outputs 的 points 同 shape）。

#### 3.1 过滤与构建 DLT Track

```
M_dlt = (pred_scores > dlt_score_thresh) & (pred_cycle_error < dlt_cycle_thresh)
M_dlt: [Ntgt, Nsrc*H*W]（比 BA 用更宽松/更严格的阈值，独立配置）
```

进一步用**极线误差**过滤（双向）：

```python
for 每个视图 ni（作为参考视图）:
    dis_a, dis_b = compute_epipolar_errors(w2c_0, w2c_s, K_0, K_s, matches)
    epipolar_msk = (dis_a < max_epipolar_error) & (dis_b < max_epipolar_error)
    M_dlt[:, ni] &= epipolar_msk
```

#### 3.2 批量 DLT 三角化

```python
tracks_dlt:      [Nview, Ntracks_dlt, 2]   # 2D 坐标（低分辨率）
tracks_mask_dlt: [Nview, Ntracks_dlt]       # 可见性 mask

# 对每个 track：构建 A^T A = sum_views(P_i^T * cross_matrix * P_i)
Ai_chunk = tracks_2d[...,None] * P[:,None,2:3,:] - P[:,None,:2,:]  # (N, Ntracks, 2, 4)
AitAi_sum = sum(AitAi * visibility_weight)                          # (Ntracks, 4, 4)
eigenvectors = eigh(AitAi_sum)                                      # 最小特征值对应的向量
pt3d = eigenvectors[:,0]   # [X,Y,Z,W]（齐次坐标）
xyz = pt3d[:,:3] / pt3d[:,3:4]
```

#### 3.3 三重过滤（顺序执行）

1. **有效解过滤**：`|W| > 1e-10`（避免无穷远点）
2. **重投影误差过滤**：可见视图上的平均重投影误差 < `max_reproj_error`（默认 4px）
3. **三角化角过滤**：任意两可见视图间最大夹角 > `min_tri_angle`（默认 3°，排除低视差）

#### 3.4 结果写回像素坐标格

```python
# 目标：dlt_xyz[N, H, W, 3]，每个有效像素存其三角化 3D 坐标
index1d = view_idx * ff_w * ff_h + y.clamp(0,H-1) * ff_w + x.clamp(0,W-1)
xyz_in_img.scatter_reduce_(dim=0, index=index1d, src=xyz, reduce='sum')
count_in_img.scatter_reduce_(dim=0, index=index1d, src=ones, reduce='sum')
dlt_xyz = (xyz_in_img / count_in_img).view(N, H, W, 3)
dlt_mask = (count_in_img > 0).view(N, H, W)   # 有效像素 mask
```

> **含义**：一个像素可能被多个 track 命中（从不同视图映射过来），取均值。`dlt_mask=True` 的像素才有有效 3D 坐标。

**`sfm_outputs` 结构**：

| 字段 | Shape | 说明 |
|------|-------|------|
| `extrinsics` | `[N, 4, 4]` | BA 优化后的 w2c 外参 |
| `intrinsics` | `[N, 3, 3]` | BA 优化后的内参 |
| `points` | `[N, H, W, 3]` | DLT 三角化的稠密 3D 点（无效处为 0） |
| `point_masks` | `[N, H, W]` | bool，有效 3D 点的 mask |
| `points_success` | `bool` | SfM 是否成功 |

---

## GGPT Point Transformer 精化

```python
demo_dataset = DemoDataset(ff_data=ff_outputs, geo_data=sfm_outputs)
# 将 N 帧分成多个 chunk（Point Transformer 对点云 chunk 处理）
for chunk in scene_chunks:
    out = ggpt_model(chunk)        # out: {ff_pts_out, ff_pts_conf_out}
    # 输出反归一化后的精化 3D 点
# aggregate_chunks: 多 chunk 结果合并，按 msks_in_scene 拼回完整场景
pred_pts: [N*H*W, 3]   # 精化后的密集点云
```

GGPT 以 SfM 提供的稀疏几何约束为条件，对 ff 的每像素 3D 点预测做 Point Transformer 细化，输出精度更高的密集点云。

---

## 可替换的 FeedForward 后端

`run_sfm()` 的 3D 点初值来自 `ff_outputs['points']`，相机来自 `ff_outputs['extrinsics']`，因此 **任何能输出同格式的前馈模型都可接入**（VGGT、Pi3、LingBot-Map 等）。接入要求：

```python
ff_outputs = {
    'images_ff':  Tensor[N, H, W, 3],   # 归一化图像
    'extrinsics': Tensor[N, 4, 4],       # w2c，OpenCV 惯例
    'intrinsics': Tensor[N, 3, 3],       # fx,fy,cx,cy 内参矩阵
    'points':     Tensor[N, H, W, 3],    # 像素级 3D 世界坐标
    'points_conf':Tensor[N, H, W],       # 置信度
}
```
