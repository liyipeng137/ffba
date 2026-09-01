# Overlap3 panorama rig FFBA

Overlap3 是 ERP 输入的质量消融布局：只保留三个共光心透视视图，减少
up/down 天空和地面方向带来的离散点，同时让相邻水平视图具有真实面积重叠。

## 几何契约

```text
sensor order : center, left, right
yaw          : 0°, -90°, +90°
HFOV / VFOV  : 110° / 100°
translation  : co-located, zero baseline
camera model : SIMPLE_PINHOLE
```

center 与 left/right 各有约20°水平重叠。left 与 right 互为相反方向，
不要求直接重叠。同时间戳的三个虚拟相机仍然共光心，因此不会建立同帧
三角化 pair。

110°×100°与单一 `fx=fy` 焦距共同决定输出宽高比。prepare 默认参考 ERP
宽度四分之一所对应的90°焦距，并选择能被4整除的近似最优尺寸。对于
`4320×2160` ERP，默认输出是 `1548×1292`；其横纵焦距相对差约0.005%。
不要独立拉伸成任意宽高，否则会破坏 `SIMPLE_PINHOLE/shared focal` 假设。

## 准备150帧

```bash
source /opt/conda/bin/activate
python scripts/prepare_pano_rig_from_erp.py \
  --input /kiri/dataset/local_test_erp.mp4 \
  --output-dir /kiri/dataset/local_test_overlap3_150 \
  --layout overlap3 \
  --num-frames 150 \
  --device cuda \
  --batch-size 1
```

prepare 使用 `pytorch360convert.e2p`，并在 `pano_rig_manifest.json` 中记录
实际尺寸、HFOV/VFOV、焦距、sensor 顺序和 `sensor_from_rig` 矩阵。

## 首轮质量消融 FFBA

首轮建议减少连接和 outer refinement，先判断重叠三视图本身是否改善点云
表面厚度和屋瓦等重复纹理：

```bash
source /opt/conda/bin/activate
python run_merg3r_gluemap_pipeline.py \
  --dataset /kiri/dataset/local_test_overlap3_150 \
  --output_dir /kiri/dataset/local_test_overlap3_ffba_150_v1 \
  --path_tracker /root/.cache/torch/hub/checkpoints/vggsfm_v2_tracker.pt \
  --num_images 150 \
  --subset_size 100 \
  --overlap 5 \
  --pair_k_pose 16 \
  --neighbors_per_center 8 \
  --vggsfm_group_strategy projected_overlap \
  --vggsfm_group_batch_size 2 \
  --prior_match_topology star \
  --no-stop_before_bae \
  --ba_backend bae \
  --bae_max_num_iterations 30 \
  --bae_max_observations 2000000 \
  --num_refinement_iterations 1 \
  --augmented_ba_max_filter_iterations 2 \
  --bae_robust_loss huber \
  --bae_huber_delta 1.0 \
  --select_track_min_support 256 \
  --filter_reproj_error_type angular \
  --filter_reproj_error_threshold 1.0
```

Overlap3 下 pair/group 视轴阈值自动使用105°，所以轴线相差约90°但有20°
视锥重叠的 center↔left/right 跨帧候选可以进入匹配。可通过显式传入
`--pano_pair_max_axis_angle` 和 `--pano_group_max_axis_angle` 覆盖。

首轮固定 prepare 给出的精确焦距，以减少实验变量。需要验证全局 shared
focal 优化时再增加 `--bae_optimize_intrinsics`；rig BAE 仍只创建一个全局
焦距参数块。

## 兼容性

- 不传 `--layout` 时 prepare 仍生成旧 Cubemap5。
- FFBA 优先读取 `pano_rig_manifest.json`；旧五面 manifest 和无 manifest 的
  标准 `center/left/right/up/down` 数据仍可读取。
- 三面数据必须带 manifest，避免从目录名猜测 FOV 或虚拟相机旋转。
