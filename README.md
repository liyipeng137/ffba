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

## 环境准备
```bash
conda env create -f env_base.yaml
pip install trimesh numba "xformers==0.0.32.post1" "git+https://github.com/pypose/pypose.git"
cd third_party/gluemap && pip install .
cd third_party/bae && USE_CUDSS=0 python -m pip install --no-build-isolation -v -e .
# prepare vggsfm_track weights
```

## 当前主入口

运行示例：

```bash
python run_merg3r_gluemap_pipeline.py  \
  --dataset /kiri/tmp/Courtroom540/images \
  --output_dir  /kiri/tmp/Courtroom540/output \
  --prior_match_topology star \
  --ba_backend bae \
  --bae_max_num_iterations 20 \
  --num_refinement_iterations 2  \
  --bae_optimize_intrinsics \
  --bae_robust_loss huber \
  --bae_huber_delta 1.0 \
  --filter_reproj_error_threshold 1.0 \
  --neighbors_per_center 12 \
  --vggsfm_group_strategy projected_overlap \
  --vggsfm_group_batch_size 3

# test on 1000 frames
python run_merg3r_gluemap_pipeline.py    \
  --dataset /kiri/tmp/Courtroom540/images \  
  --output_dir  /kiri/tmp/Courtroom540/output_fk/ \
  --subset_size 200 \
  --overlap 10 \
  --prior_match_topology star   \
  --ba_backend bae   \
  --bae_max_num_iterations 20 \   
  --num_refinement_iterations 2  \   
  --bae_optimize_intrinsics    \
  --bae_robust_loss huber   \
  --bae_huber_delta 1.0   \
  --bae_max_observations  2000000 \
  --filter_reproj_error_threshold 0.5  \
  --neighbors_per_center 12   \
  --select_track_min_support 256  \
  --vggsfm_group_strategy projected_overlap \   
  --vggsfm_group_batch_size 3 
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

- `low images`：在 CPU 内存生成，DINO/Pi3X 按批搬到 GPU；Stage A
  前馈结束后释放。
- `high images`：保留在 CPU 内存，供 SIFT、VGGSfM prior tracking、GlueMap
  refinement 和最终输出按需使用。
- low/high 中间图片不落盘；`images/` 仍由 `_save_work_images()` 保存一次，供
  SIFT/pycolmap 和 group audit 使用。
- `manifest`：记录原图、low 图、high 图之间的 resize/crop/scale 关系。
- `scale_intrinsics_with_pyramid_records()`：把 Stage A 得到的 low intrinsics
  映射到 high 图坐标系。

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
```

思想是：前馈模型已经提供 coarse pose，因此后续稀疏匹配不需要完全盲目地 all-pairs，而是在满足 rotation threshold 的候选中，按 camera center 距离优先选择邻居。`pair_k_pose` 是每帧主动选择的上限；候选不足时不补边，也不要求每个 center 都必须有 pair。

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

如需仅在最后一轮减弱 Huber 降权、增强中等残差的拟合力度，例如
三轮分别使用 `delta=1.0 / 1.0 / 2.0`：

```bash
--num_refinement_iterations 3
--bae_robust_loss huber
--bae_huber_delta 1.0
--final_bae_huber_delta 2.0
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
  prior_pose_import.json       # 仅 prior-pose 模式产生

  image_pyramid/               # 仅 feed-forward 模式产生
    image_pyramid_manifest.json

  images/
    frame_000000.png
    frame_000001.png
    ...

  pred_depth/                  # 仅 feed-forward 模式产生
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

只审计 VGGSfM group、同时比较 pose baseline 与 projected-overlap hybrid：

```bash
python run_merg3r_gluemap_pipeline.py \
  --dataset <images> \
  --output_dir <output> \
  --pair_k_pose 25 \
  --neighbors_per_center 12 \
  --export_vggsfm_groups_only \
  --vggsfm_group_audit_strategy both
```

该模式在相同 Stage A 结果上输出：

```text
vggsfm_group_audit/
  pose_k12/
  projected_overlap_hybrid_k12/
    groups.json
    candidate_scores.json
    labels.csv
    contact_sheets/
    groups/
```

projected-overlap hybrid 的候选池为 rotation-valid pose pairs 与 DINO
top-30 的并集；排序使用 low-resolution depth 的有向 round-trip
reprojection overlap，DINO 只用于候选召回。

正式 refinement 使用 projected-overlap group：

```bash
python run_merg3r_gluemap_pipeline.py \
  --dataset <images> \
  --output_dir <output> \
  --pair_k_pose 25 \
  --neighbors_per_center 12 \
  --vggsfm_group_strategy projected_overlap
```

不传 `--vggsfm_group_strategy` 时仍使用 `pose`，便于和已有结果对照。
正式 projected-overlap 路径复用下方四个 `--projected_overlap_*` 参数；
如果 sequence 阶段未产生 DINO similarity matrix，Stage A 会自动补算一次。

### SIFT-first Center / Group V1

默认 `--vggsfm_schedule_mode legacy` 保留原有 VGGSfM-first、full-center 执行
顺序。实验模式：

```text
sift_first_full
  先构建一次 SIFT DB
  SIFT pairs = 旧 pose candidates + local temporal pairs
  所有帧仍作为 center

sift_first_sparse
  在 sift_first_full 基础上
  用 verified SIFT inliers + 双向 grid coverage 抽稀 center
  group = owned frames + adjacent-center bridges + projected-overlap fill
```

当前质量基线使用 `neighbors_per_center=16`。以下命令使用当前 V1 的默认抽稀阈值，
可作为标准 sparse 实验的命令行参考：

```bash
source /opt/conda/bin/activate
conda activate gluemap-merg3r

python run_merg3r_gluemap_pipeline.py \
  --dataset /path/to/images \
  --output_dir /path/to/output_sparse \
  --prior_match_topology star \
  --ba_backend bae \
  --bae_max_num_iterations 20 \
  --num_refinement_iterations 3 \
  --bae_optimize_intrinsics \
  --bae_robust_loss huber \
  --bae_huber_delta 1.0 \
  --final_bae_huber_delta 2.0 \
  --filter_reproj_error_threshold 1.0 \
  --neighbors_per_center 16 \
  --vggsfm_group_strategy projected_overlap \
  --vggsfm_group_batch_size 3 \
  --vggsfm_schedule_mode sift_first_sparse \
  --sift_temporal_window 2 \
  --sift_schedule_grid_size 8 \
  --sift_schedule_min_inliers_per_cell 2 \
  --sift_schedule_min_pair_inliers 128 \
  --sift_schedule_min_grid_coverage 0.20 \
  --vggsfm_max_center_gap 2
```

启用 sparse 调度必须同时设置 `--vggsfm_schedule_mode sift_first_sparse` 和
`--vggsfm_group_strategy projected_overlap`；后面六项显式写出了当前默认值，便于实验
记录和复现。`sift_first_sparse` 会先构建
SIFT DB，再根据 verified SIFT 强边选择部分帧作为 VGGSfM center；未被选为 center 的帧
仍可作为 group neighbor，并继续参加最终注册和 BA。

对于数百帧以上、BAE observation 可能过多的序列，可在相同命令中增加：

```bash
--bae_max_observations 2000000
```

该参数会在每轮 BAE 前按完整 track 裁剪 observation。启用 cap 后，`final points`
会同时受到 track length 分布影响，因此不能只根据最终点数判断 sparse 是否提升质量。

当前 sparse V1 仍是实验模式，`legacy` 仍为默认和质量回退路径。720_room 的首轮结果中，
默认 `min_pair_inliers=128` 获得约 27% 端到端加速，但 P-only angular median/p90
超过了 3% 退化门槛。下一轮若要测试更保守的 center 抽稀，可仅覆盖：

```bash
--sift_schedule_min_pair_inliers 512
```

这组 `512` 配置目前只是基于 720_room 阈值 sweep 得到的候选（预计保留 508/720 个
center），尚未完成正式质量验收，不应替代上面的可复现实测配置。

SIFT-first 模式额外输出 `vggsfm_schedule.json`，包含 pair 来源、verified
inlier/coverage、阈值 sweep、center/owner 原因和三层 group provenance。

### SIFT + LoMa Prior V1（实验）

通过 `--prior_provider loma` 选择 LoMa-B，默认 provider 仍为 VGGSfM。
LoMa 路径先完成 SIFT 匹配，使用两套 pair 图：

- SIFT：当前 pose neighbors（含 rotation threshold）并上 temporal pairs，默认时序窗口 ±2。
- LoMa：pose pairs ∪ 每图 DINO top-30 ∪ temporal pairs，无向去重后全部匹配。
  SIFT 的 `sufficient / insufficient / untried` 标注仅用于诊断，不删减候选。

每张工作图使用原生路径预处理提取一次特征，保留 LoMa-B 默认 2048 个关键点及
0.1 匹配阈值。固定 feature ID 经过 pycolmap 两视图几何验证后写入 prior DB，
再使用现有数据库合并、三角化、SelectTrack、几何过滤和 BAE 流程。LoMa 坐标不 snap 到 SIFT。
VGGSfM 的 group、center、query 与 `prior_match_topology` 参数不控制 LoMa。

在已具备主流程及 LoMa 依赖的 GPU 环境中运行：

```bash
python run_merg3r_gluemap_pipeline.py \
  --dataset /path/to/images \
  --output_dir /path/to/output_loma \
  --prior_provider loma \
  --ba_backend bae \
  --bae_max_observations 0 \
  --bae_optimize_intrinsics
```

LoMa V1 只支持 BAE，不启用 observation cap；传入正的 `bae_max_observations`
会报错提示，避免沿用旧测试命令时无意裁剪。内参优化默认开启，可用
`--no-bae_optimize_intrinsics` 关闭。其余 BAE 和既有过滤参数沿用当前命令配置。
有深度前馈输入和 `--prior_transforms_json /path/to/transforms.json` 均可使用同一 LoMa pair 规则。

模型从仓库 `third_party/LoMa/src` 加载，依赖见其 `pyproject.toml`；首次加载会使用
上游的权重下载与缓存机制。特征只缓存在本次进程的 CPU 内存中。
输出 `prior_loma_pairs.json`（候选来源、SIFT 标注和匹配结果）、
`prior_loma_stats.json`（分阶段时间、验证图连通性、唯一观测及最终轨长统计）、
`database_loma_prior.db`，并沿用 `refine_stats.json` 和原有 COLMAP 输出目录。

已通过 CPU 上的真实 pycolmap 几何验证、DB 合并/重映射及三图成轨测试；
LoMa 权重推理和 BAE 的 GPU 端到端验证待运行，速度与质量收益尚无实测结论。
详细设计见 [LoMa prior V1](.ai/plan/loma_prior_lite_design.md)。

### Nerfstudio Prior Pose 输入

传入 `--prior_transforms_json` 后，Nerfstudio `transforms.json` 中的 pose 会替代
Pi3X/MERG3R coarse pose。它只作为初值，后续仍由 BAE 优化。该模式：

- 以 JSON `frames[]` 顺序作为图片和时序顺序；
- 直接在原始图片分辨率上运行 SIFT、refinement 和输出 COLMAP model；
- 跳过 Pi3X、subset alignment、depth 生成与 `pred_depth/` 导出；
- 只支持 `--ba_backend bae`；
- VGGSfM 使用 depth-free `sift_pose_dino` group，不支持 `projected_overlap`；LoMa 使用上述独立 pair 图。

789_room sparse 示例：

```bash
source /opt/conda/bin/activate
conda activate gluemap-merg3r

python run_merg3r_gluemap_pipeline.py \
  --dataset /kiri/codex_use_data/789_room/image \
  --prior_transforms_json /kiri/codex_use_data/789_room/transforms.json \
  --output_dir /kiri/tmp/ffba_789_prior_sparse \
  --prior_match_topology star \
  --ba_backend bae \
  --bae_max_num_iterations 20 \
  --bae_max_observations 2000000 \
  --num_refinement_iterations 3 \
  --bae_optimize_intrinsics \
  --bae_robust_loss huber \
  --bae_huber_delta 1.0 \
  --final_bae_huber_delta 2.0 \
  --filter_reproj_error_threshold 1.0 \
  --neighbors_per_center 16 \
  --vggsfm_group_strategy sift_pose_dino \
  --vggsfm_group_batch_size 3 \
  --vggsfm_schedule_mode sift_first_sparse \
  --sift_temporal_window 2 \
  --sift_schedule_min_pair_inliers 128 \
  --sift_schedule_min_grid_coverage 0.20 \
  --vggsfm_max_center_gap 2
```

输入的 Nerfstudio OpenGL camera-to-world pose 会转换为 OpenCV/COLMAP
world-to-camera。内参统一为 `fx=fy=mean((fl_x+fl_y)/2)`、`cx=w/2`、`cy=h/2`，
输出仍使用 shared `SIMPLE_PINHOLE`，并忽略 JSON 中的 distortion 和
`depth_file_path`。DINO retrieval 使用临时长边 512 的缩略图；VGGSfM 仍使用其内部
1024×1024 resize/pad，这两者都不会改变原图 SIFT/COLMAP 坐标。

运行会额外输出 `prior_pose_import.json`，用于审计 frame 映射、坐标转换、共享内参、
轨迹范围和 DINO 临时输入尺寸。

如果需要 PLY，可用 COLMAP 自带 converter 从 `points3D` 转出。

## 常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--dataset` | required | 输入图片目录 |
| `--output_dir` | required | 输出目录 |
| `--prior_transforms_json` | unset | 可选 Nerfstudio transforms.json；启用原图分辨率 prior-pose 模式并跳过 feed-forward/depth |
| `--prior_dino_long_side` | `512` | prior-pose 模式的临时 DINO retrieval 图片长边，不改变几何工作分辨率 |
| `--prior_dino_batch_size` | `16` | prior-pose 模式的 DINO inference batch size |
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
| `--pair_pose_rotation_threshold` | `30.0` | pose pair 允许的最大视角差，单位为度 |
| `--path_tracker` | required in practice | VGGSfM tracker checkpoint |
| `--neighbors_per_center` | `16` | 每个 VGGSfM group 的邻居上限；V1 固定质量基线为 16 |
| `--vggsfm_group_strategy` | `pose` | VGGSfM group 策略；`pose`、depth-based `projected_overlap` 或 depth-free `sift_pose_dino` |
| `--vggsfm_group_batch_size` | `2` | 按 `(group_size, query_points)` 分桶后，每次 VGGSfM forward 的 group 数量；尾桶自动降为较小 batch |
| `--vggsfm_schedule_mode` | `legacy` | `legacy`、`sift_first_full` 或 `sift_first_sparse` |
| `--sift_temporal_window` | `2` | SIFT candidate graph 强制加入的时序窗口 |
| `--sift_schedule_grid_size` | `8` | verified SIFT coverage 网格边数 |
| `--sift_schedule_min_inliers_per_cell` | `2` | cell 被视为 occupied 的最少 verified inliers |
| `--sift_schedule_min_pair_inliers` | `128` | valid schedule edge 的最少 verified inliers |
| `--sift_schedule_min_grid_coverage` | `0.20` | pair 两端最小 grid coverage |
| `--vggsfm_max_center_gap` | `2` | sparse mode 下 selected centers 最大时序间隔 |
| `--export_vggsfm_groups_only` | off | Stage A 后按 audit strategy 导出 VGGSfM groups、contact sheets、JSON 和人工标签 CSV，然后跳过 tracker/refinement |
| `--vggsfm_group_audit_strategy` | `pose` | `pose`、`projected_overlap` 或 `both`；仅影响 group audit 提前退出模式 |
| `--projected_overlap_dino_candidates` | `30` | 每个 center 加入 projected-overlap 候选池的 DINO retrieval 数量 |
| `--projected_overlap_samples` | `2048` | 每个 center 用于有向几何投影的 low-res depth 规则网格采样上限 |
| `--projected_overlap_reproj_threshold` | `4.0` | low-res depth round-trip reprojection 一致性阈值，单位为像素 |
| `--projected_overlap_conf_quantile` | `0.2` | 丢弃每帧最低比例的 depth-confidence 样本 |
| `--vggsfm_query_points` | `1024` | prior tracking query 点数 |
| `--prior_match_topology` | `star` | prior tracks 写入 pair matches 的拓扑 |
| `--min_frame_observations` | `10` | 低覆盖帧过滤阈值 |
| `--ba_backend` | `bae` | `bae` 或 `ceres` |
| `--bae_max_num_iterations` | `20` | BAE 迭代次数 |
| `--bae_max_observations` | `0` | 每轮进入 BAE 的 real observation 硬上限；`0` 表示禁用，超限时按质量排序原地删除完整 track |
| `--bae_fix_gauge` | `two_cams` | BAE gauge fixing 策略 |
| `--bae_robust_loss` | `huber` | BAE robust loss |
| `--final_bae_huber_delta` | unset | 仅在 augmented refinement 最后一轮使用的 BAE Huber delta；未设置时每轮均使用 `--bae_huber_delta` |
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
