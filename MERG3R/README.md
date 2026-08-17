# FeedForwardWithBA / MERG3R + GlueMap 主流程说明

本文档以当前主入口 [run_merg3r_gluemap_pipeline.py](run_merg3r_gluemap_pipeline.py) 为准。

这个项目的目标不是保留原始 MERG3R 或 GlueMap 的独立运行形态，而是把前馈式几何模型、MERG3R 的分块合并能力、GlueMap 的稀疏匹配/三角化能力，以及 BAE bundle adjustment 后端缝合成一个可交付 SfM POD 的主流程。

## 核心思想

主流程分为两大阶段：

```text
输入图片
  │
  ▼
两阶段图片处理
  ├─ low images  -> 前馈模型粗估计 pose / intrinsics / depth
  └─ high images -> 稀疏特征、VGGSfM prior tracks、最终 COLMAP 输出
  │
  ▼
Stage A: MERG3R coarse geometry
  ├─ pi3x 前馈预测局部 depth/pose
  ├─ MERG3R subset split + global alignment
  ├─ 生成 BA 前 coarse pose / depth / intrinsics
  └─ 基于 coarse pose 建立共视 pair graph
  │
  ▼
Stage B: GlueMap-style sparse refinement
  ├─ SIFT database
  ├─ VGGSfM prior tracks
  ├─ prior tracks snap 到 SIFT keypoints
  ├─ 合并 SIFT + prior track database
  ├─ 三角化 / track selection / reprojection filtering
  └─ BAE 或 Ceres bundle adjustment
  │
  ▼
refined_gluemap_aba/  # 最终 refined COLMAP model
```

Stage A 用前馈模型快速给出全局可用的 coarse 几何；Stage B 不再完全依赖前馈 depth，而是把 coarse pose 作为先验，交给 GlueMap 风格的稀疏特征、prior track 和 BA 流程去生成最终可交付的 SfM 稀疏重建。

## 当前主入口

推荐直接运行：

```bash
cd MERG3R

python run_merg3r_gluemap_pipeline.py \
  --dataset ../650_data/image_jpg/ \
  --output_dir ./650_data/ \
  --prior_match_topology star \
  --virtual_verify_mode center \
  --ba_backend bae \
  --bae_max_num_iterations 20 \
  --num_refinement_iterations 2 \
  --bae_optimize_intrinsics \
  --bae_robust_loss huber \
  --bae_huber_delta 1.0 \
  --filter_reproj_error_threshold 1.0
```

当前主流程固定使用：

```text
feed-forward model: pi3x
camera model      : SIMPLE_PINHOLE
feature database  : SIFT
prior query source: ALIKED
group strategy    : pose
BA backend        : BAE by default
```

## 输入要求

`--dataset` 指向图片目录：

```text
/path/to/images/
  frame_000000.png
  frame_000001.png
  ...
```

注意：当前 tensor loading 仍要求进入模型的图片尺寸一致。默认开启的 image pyramid 会对图片做 resize/crop，但输入数据最好本身来自同一序列或同一分辨率规范。

## Stage A: 前馈粗几何

代码入口：

```text
run_merg3r_coarse_stage()
```

主要步骤：

1. 可选构建两阶段图片金字塔。
2. 加载 low-resolution 图片。
3. 用 MERG3R 的 sequence split 方式分块。
4. 用 `pi3x` 对每个 subset 做前馈推理。
5. 合并并对齐各 subset 的 pose / depth / intrinsics。
6. 恢复原始图片顺序。
7. 将 low-resolution intrinsics 映射到 high-resolution 工作图。
8. 基于 coarse pose 构建 pair graph。
9. 保存 `pipeline_stage_a_summary.json`。

Stage A 的结果不直接作为最终 SfM 输出。它主要提供：

- 初始 camera pose；
- 初始 intrinsics；
- 原始 dense depth；
- 用于构建 GlueMap pose groups 的 pair graph；
- 后续稀疏精修的几何先验。

## 两阶段图片处理

这是后续新增的重要逻辑，位于 [utils/image_pyramid.py](utils/image_pyramid.py)。

默认开启：

```bash
--image_pyramid
```

作用：

- `low images`：给前馈模型使用，降低显存和推理成本。
- `high images`：给 SIFT、VGGSfM prior tracking、GlueMap refinement 和最终输出使用。
- `manifest`：记录原图、low 图、high 图之间的 resize/crop/scale 关系。
- `scale_intrinsics_low_to_high()`：把 Stage A 得到的 low intrinsics 映射到 high 图坐标系。

相关参数：

```bash
--stage1_downscale_n 4
--stage1_multiple 14
--stage2_scale_factor 0
--image_pyramid_workers 16
```

`--stage2_scale_factor 0` 表示 high 图默认约等于原始尺寸，即使用 `stage1_downscale_n` 作为 low-to-high 放大倍率。

如果要禁用：

```bash
--no-image_pyramid
```

## Pair Graph

Stage A 后会构建用于 GlueMap pose groups 的 pair graph。

默认主要依赖 pose 邻接：

```bash
--pair_k_pose 25
--pair_pose_rotation_threshold 30.0
--pair_pose_fill_unfiltered
```

可额外启用：

```bash
--pair_k_similarity N
--pair_temporal_window N
```

思想是：前馈模型已经提供 coarse pose，因此后续稀疏匹配不需要完全盲目地 all-pairs，而是可以优先在几何邻近、视角相近的帧之间建立关系。

`pair_pose_fill_unfiltered` 会在通过 rotation threshold 的邻居不足时，用 camera center 距离补足候选，避免 pair graph 过稀导致后续 pose groups 覆盖不足。

## Stage B: GlueMap/SPV 精修

代码入口：

```text
utils/gluemap_spv_refine.py::run_gluemap_spv_refinement()
utils/gluemap_refine_core.py
```

这里 GlueMap 被当作工具包使用，而不是作为独立 CLI 运行。

主要步骤：

1. 保存 high-resolution work images 到 `<output_dir>/images/`。
2. 按 pose groups 运行 VGGSfM prior tracking。
   - tracker coarse fmaps 会优先以 BF16（不支持时 FP16）常驻 GPU，避免每个 pose group 重复从 CPU 搬运；若压缩缓存预计占用超过当前空闲显存的 50%，自动回退到原有 FP32 CPU cache。
3. 准备 SIFT database。
4. 统计 SIFT observations 和 prior track observations。
5. 按观测数量过滤低覆盖帧。
6. 导出前馈 depth 到 `<output_dir>/pred_depth/`，供后续分析或 dense 后处理使用。
7. 使用 GlueMap 的 intrinsics averaging 得到 shared `SIMPLE_PINHOLE` intrinsics。
8. 将 VGGSfM prior tracks snap 到 SIFT keypoints。
9. 写出 prior track database。
10. 合并 prior database 和 SIFT database。
11. 写出 coarse COLMAP reconstruction。
12. 进入 augmented refinement loop。

默认配置：

```text
SIFT database mode : sift
VGGSfM query source: aliked
tracker input      : 1024
prior topology     : star
camera model       : SIMPLE_PINHOLE
```

## Augmented Refinement Loop

精修循环位于：

```text
utils/gluemap_refine_core.py::run_merg3r_augmented_refinement_loop()
```

每一轮大致执行：

```text
seed reconstruction
  -> triangulation
  -> select tracks
  -> reprojection / angular error filtering
  -> bundle adjustment
  -> 下一轮继续用优化后的 pose / intrinsics 作为 seed
```

关键参数：

```bash
--num_refinement_iterations 3
--tri_min_angle 1.0
--tri_create_max_angle_error 0.5
--select_track_min_support 512
--filter_reproj_error_type angular
--filter_reproj_error_threshold 0.5
--augmented_ba_max_filter_iterations 3
--augmented_ba_normalized_reproj_threshold 1e-2
```

## BAE 后端

原 GlueMap / pycolmap 路线默认依赖 Ceres solver；当前主流程默认使用 [third_party/bae](third_party/bae) 作为 PyTorch BA 后端：

```bash
--ba_backend bae
```

BAE 默认参数：

```bash
--bae_max_num_iterations 20
--bae_optimize_intrinsics
--bae_fix_gauge two_cams
--bae_robust_loss huber
--bae_huber_delta 1.0
```

当前 BAE 路线的设计取舍：

- 使用真实 sparse tracks 做 refinement。
- 跳过 GlueMap virtual tracks。
- 用 `two_cams` gauge fixing 模拟 COLMAP/Ceres 的 gauge 约束语义。
- 可优化 `SIMPLE_PINHOLE` 的 focal，固定 principal point。
- 对 real-track residual 使用 Huber robust loss。
- 中间轮不强制重复 post-BA filter，最终轮才保留有效过滤结果。

如果显式使用 Ceres：

```bash
--ba_backend ceres
```

Ceres 路线会保留 GlueMap 原本更接近 SPV 的 virtual-track 分支：

- 构建 virtual tracks；
- 初始化 virtual points；
- 同时维护 real reconstruction 和 virtual reconstruction；
- 每轮做 virtual track selection / filtering / BA。

因此 README 里的 `SPV` 指的是 GlueMap 原始 virtual-track 思路；当前默认 BAE 主线实际是 `SP real tracks + BAE BA`。

## 这套项目后来新增/改造的逻辑

相对于原始 MERG3R 或原始 GlueMap，当前主流程主要新增了这些部分：

1. **两阶段图片处理**
   - low 图给前馈模型；
   - high 图给稀疏特征、VGGSfM prior 和最终输出；
   - 显式保存 low/high 对应关系和 intrinsics scale。

2. **前馈粗几何到 GlueMap 的桥接**
   - 把 MERG3R/PI3X 的 coarse pose、intrinsics、depth 整理为 `Merg3rCoarseState`；
   - 用 coarse pose 建 pose-aware pair graph；
   - 让 GlueMap 后端从 coarse reconstruction 启动，跳过原始的每帧构建star图推理阶段。

3. **VGGSfM prior tracks + SIFT snap/merge**
   - 用 ALIKED query point 生成 prior tracks；
   - 将 prior observations snap 到 SIFT keypoints；
   - 合并 `database_vggsfm_prior.db` 与 `database_sift.db`。

4. **BAE 替代 Ceres 作为默认 BA 后端**
   - BAE 位于 `third_party/bae`；
   - 主流程默认 `--ba_backend bae`；
   - 保留 `--ba_backend ceres` 作为对照路径。

5. **低覆盖帧过滤**
   - 统计 SIFT observations 和 prior observations；
   - 对低于 `--min_frame_observations` 的帧做过滤；
   - 避免低观测帧破坏后续三角化和 BA。

6. **Depth 导出与后处理接口**
   - Stage A 的 depth 会导出到 `pred_depth/`；
   - 后续的 depth correction、TSDF、LingBot depth refine 等脚本可以基于这些输出继续处理；
   - 这些 dense/depth 后处理不是 `run_merg3r_gluemap_pipeline.py` 主流程的一部分。

7. **项目结构清理**
   - GlueMap / BAE 作为 `third_party/` 工具包；
   - 通用函数逐步移到 `utils/`；
   - 当前主流程只依赖保留下来的 `pi3x_model` / `vggt_omega` feed-forward 代码，其中主入口固定 `pi3x`。

## 主要输出

一次运行的 `<output_dir>` 典型结构：

```text
output/
  pipeline_config.json
  pipeline_stage_a_summary.json
  refine_stats.json

  image_pyramid/
    low/
    high/
    manifest.json

  images/
    frame_000000.png
    frame_000001.png
    ...

  pred_depth/
    depth_npy/
    depth_u16/
    depth_vis/

  database_sift.db
  database_vggsfm_prior.db
  database_merged.db

  coarse/
    cameras.*
    images.*
    points3D.*

  refined_gluemap_aba/
    cameras.*
    images.*
    points3D.*

  virtual_gluemap_aba/        # 仅 Ceres / virtual-track 路线可能产生
```

最终交付的 SfM sparse reconstruction 通常看：

```text
<output_dir>/refined_gluemap_aba/
```

如果需要 PLY，可用 COLMAP 自带 converter 从 `points3D` 转出。

## 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--dataset` | required | 输入图片目录 |
| `--output_dir` | required | 输出目录 |
| `--device` | `cuda` | 推理和 refinement 设备 |
| `--num_images` | `-1` | 限制图片数量，`-1` 表示全部 |
| `--subsample` | `1` | 按顺序采样图片 |
| `--multi_dirs` | off | 递归读取多级图片目录 |
| `--image_pyramid` | on | 启用 low/high 两阶段图片 |
| `--stage1_downscale_n` | `4` | low 图相对原图的下采样基数 |
| `--stage1_multiple` | `14` | low 图裁剪到该倍数 |
| `--stage2_scale_factor` | `0` | high 图相对 low 图的倍率，`0` 表示使用 `stage1_downscale_n` |
| `--sequence_type` | `shortest_path` | MERG3R subset 构建方式 |
| `--subset_size` | `100` | 每个前馈 subset 的图片数 |
| `--overlap` | `5` | subset 之间的 overlap |
| `--splitting_type` | `interleave` | subset 内图片组织方式 |
| `--alignment_type` | `weighted_iterative` | MERG3R subset pose 对齐方式 |
| `--pair_k_pose` | `25` | 每帧按 coarse pose 选取的邻居数量 |
| `--pair_k_similarity` | `0` | 额外按 DINO similarity 选邻居，默认关闭 |
| `--pair_temporal_window` | `0` | 额外加入时序邻居，默认关闭 |
| `--path_tracker` | required in practice | VGGSfM tracker checkpoint |
| `--neighbors_per_center` | `25` | 每个 pose group 的邻居上限；rotation-valid 优先，同层按 camera-center 距离排序，不足时才取 unfiltered/fill 邻居 |
| `--export_vggsfm_groups_only` | off | Stage A 后导出当前 VGGSfM pose groups、contact sheets、JSON 和人工标签 CSV，然后跳过 tracker/refinement |
| `--vggsfm_query_points` | `1024` | prior tracking query 点数 |
| `--prior_match_topology` | `star` | prior tracks 写入 pair matches 的拓扑 |
| `--min_frame_observations` | `10` | 低覆盖帧过滤阈值 |
| `--ba_backend` | `bae` | `bae` 或 `ceres` |
| `--bae_max_num_iterations` | `20` | BAE 迭代次数 |
| `--bae_fix_gauge` | `two_cams` | BAE gauge fixing 策略 |
| `--bae_robust_loss` | `huber` | BAE robust loss |
| `--num_refinement_iterations` | `3` | augmented refinement 外层轮数 |

## 推荐检查点

运行完成后优先检查：

```text
pipeline_stage_a_summary.json
refine_stats.json
refined_gluemap_aba/
```

重点看：

- Stage A 是否覆盖了所有图片；
- pair graph 是否有 `zero_degree_images`；
- frame filtering 是否丢掉过多帧；
- SIFT / prior track observations 是否足够；
- `vggsfm.neighbor_rank_stats` 中第 13～25 名邻居的通过率和有效 observation；
- `vggsfm.workload.attempted_query_views` 与 `query_track_stats` 的成轨率、track length；
- `augmented_refinement.final.real_by_source` 中最终 `p_only` / `mixed` 点数；
- `augmented_refinement.final.angular_errors_by_track_source` 中最终 P 误差；
- BAE summary 是否收敛；
- `refined_gluemap_aba` 中的 registered images 和 points3D 数量是否合理。

## 当前边界

- `run_merg3r_gluemap_pipeline.py` 是当前主流程。
- `run_merg3r_bae_pipeline.py` 是较早的 MERG3R + BAE pipeline，不是当前 README 描述的主路径。
- `eval/`、旧 README 里的评测命令、旧 `run.sh` 参数不作为当前主流程依据。
- depth correction、TSDF、LingBot depth refine、hybrid point cloud 等属于 BA 后处理链路，可基于主流程输出继续运行，但不属于本 README 的主流程闭环。
