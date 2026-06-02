# MERG3R 当前 Pipeline 摘要

本文档按当前 `MERG3R/main.py` 中实际生效的代码整理。已经被手动注释掉的流程和参数不作为当前主流程的一部分描述。

## 入口与默认配置

入口文件：`MERG3R/main.py`

当前流程强制使用 CUDA：

```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required to run this pipeline.")
device = "cuda"
```

主要默认参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `pi3x` | 支持 `vggt / pi3 / pi3x / vggt_omega` |
| `--pi3x_intrinsics_method` | `moge` | pi3x 内参估计方式 |
| `--sequence_type` | `shortest_path` | 图像排序与分块方式 |
| `--subset_size` | `100` | 每个子集图像数 |
| `--overlap` | `5` | 相邻子集重叠帧数 |
| `--splitting_type` | `interleave` | shortest_path 后的分块重排策略 |
| `--global_ba` | `True` | 默认执行 tracking + global BA |
| `--tracking_type` | `graph` | shortest_path 下默认图匹配 |
| `--tracking_matcher` | `loma` | 默认使用 LoMa 做图匹配 |
| `--point_vis_threshold` | `20.0` | dense 点云和 COLMAP 点过滤使用的置信度阈值 |
| `--dense_max_points` | `50_000_000` | `dense_model_points.ply` 最大导出点数 |
| `--save_dense_depth` | `False` | 当前 dense depth 投影导出代码已注释，参数暂不进入主流程 |
| `--lingbot_refine` | `False` | 可选 LingBot-Depth refine |
| `--high-dataset` | `None` | 可选高分辨率输出图像目录 |

运行开始后会把解析后的参数写入：

```text
<output_dir>/config.json
```

## 总体流程

```text
输入图像
  |
  v
process_images()
  |
  v
create_sequence()
  |
  v
load_model() + run_inference_step_by_step()
  |
  v
align_extrinsics() + restore_predictions_order()
  |
  v
[默认] tracking + global_bundle_adjustment()
  |
  v
[可选] run_bae_refinement()
  |
  v
export_prediction_depth_maps()
  |
  v
[可选] 加载 high_dataset 并缩放内参
  |
  v
collect_dense_world_points() + export_dense_world_points_ply()
  |
  v
[可选] run_lingbot_depth_refinement()
  |
  v
computation_stats.txt + write_recon_to_colmap()
```

## 1. 图像加载

`process_images(args.dataset, subsample, device, args.num_images, args.multi_dirs, args.model)` 负责读取输入图像，支持：

- `--dataset`：输入图像目录。
- `--num_images`：限制读入图像数量，`-1` 表示不限制。
- `--subsample`：按固定间隔抽帧。
- `--multi_dirs`：支持多目录输入。

输出：

```python
images, image_names
size_hw = images.shape[-2:]
```

其中 `size_hw` 会作为模型推理的目标尺寸记录。

## 2. 图像排序与子集划分

`create_sequence()` 根据 `--sequence_type` 构造 sequence，并生成 `sequence.image_split`。

当前默认：

```text
sequence_type = shortest_path
subset_size = 100
overlap = 5
splitting_type = interleave
```

默认 `shortest_path` 流程会基于图像相似度构造全局排序，再按重叠子集切分。切分后的每个 batch 会先转到 CPU，避免模型加载前占用过多显存：

```python
batches = sequence.image_split
for i, img in enumerate(batches):
    batches[i] = img.to("cpu")
```

## 3. 子集级模型推理

当前通过 `load_model(args.model)` 加载基础几何模型，然后逐子集推理：

```python
sequence.predictions = run_inference_step_by_step(
    model,
    batches,
    size_hw,
    device,
    need_features=False,
    pi3x_intrinsics_method=args.pi3x_intrinsics_method,
)
```

推理后释放模型，并把每个 prediction 转为 CPU numpy：

```python
prediction[key] = prediction[key].cpu().numpy()
```

典型 prediction 字段包括：

- `extrinsic`
- `intrinsic`
- `depth`
- `depth_conf`
- `local_points`
- `images`

实际字段取决于底层模型返回内容。

## 4. 子集对齐与恢复原始顺序

对每个子集的局部相机结果做全局对齐：

```python
final_predictions, _, _ = align_extrinsics(
    sequence,
    method=args.alignment_type,
    ba=False,
)
restore_predictions_order(final_predictions)
```

默认 `alignment_type=weighted_iterative`。这一步之后，`final_predictions` 已经回到全局 frame 顺序，并拥有对齐后的相机外参、内参、深度、置信度和局部点。

## 5. Tracking 与 Global BA

`--global_ba` 默认开启。当前流程会先根据 `final_predictions['depth']` 反投影出临时世界点图：

```python
final_predictions['world_points'] = unproject_depth_map_to_point_map(
    final_predictions['depth'],
    final_predictions['extrinsic'],
    final_predictions['intrinsic'],
)
```

然后根据 sequence 和 matcher 类型执行 tracking。

### video sequence

`sequence_type == "video"` 时使用 LightGlue 固定步长匹配：

```python
steps = [1, 2, 3, 5, 7, 10]
extract_matches_lightglue(...)
```

### shortest_path sequence

`sequence_type == "shortest_path"` 时当前默认走：

```text
tracking_type = graph
tracking_matcher = loma
```

可选 matcher：

| matcher | 函数 |
|---|---|
| `loma` | `graph_extract_matches_loma()` |
| `hloc` | `graph_extract_matches_hloc()` |
| `lightglue` | `graph_extract_matches_lightglue()` |

LoMa 默认配置：

```text
loma_arch = LoMa-B
loma_filter_threshold = 0.1
k = 5
max_num_keypoints = 4096
```

### 丢弃无 tracking 帧

tracking 后会调用：

```python
drop_untracked_frames(...)
```

如果某些帧没有任何有效 track，会同步从以下内容中移除：

- `final_predictions`
- `images`
- `sequence.images`
- `sequence.image_names`
- `track`
- `points_id`

被移除的 frame id 会写入 `computation_stats.txt`。

### Global BA

随后执行：

```python
global_bundle_adjustment(
    final_predictions,
    track,
    points_id,
    points_3d,
    points_conf,
    max_reproj_error=args.max_reproj,
    lr=args.lr,
    epoch=args.epoch,
)
```

BA 结束后会把所有帧的内参强制设为共享内参：

```python
shared_intrinsic = np.mean(final_predictions['intrinsic'], axis=0, keepdims=True)
final_predictions['intrinsic'] = np.repeat(shared_intrinsic, N, axis=0)
```

这符合当前 pi3x 流程中“BA 后内参全局一致”的假设。

## 6. 可选 BAE refine

如果传入 `--bae_refine`，会在 global BA 后继续执行：

```python
run_bae_refinement(
    final_predictions,
    track,
    points_id,
    valid_track_mask=valid_track_mask,
    iters=args.bae_iters,
    device=device,
    optimize_intrinsics=args.bae_optimize_intrinsics,
)
```

当前 `--bae_refine` 依赖 `--global_ba`，因为它需要 BA 后的 sparse tracks 和 points。

## 7. 单帧预测深度导出

主流程现在总是导出一份来自 `final_predictions['depth']` 的单帧深度：

```python
single_frame_depth_stats = export_prediction_depth_maps(
    final_predictions,
    sequence.image_names,
    os.path.join(args.output_dir, "single_frame_depth"),
    conf_threshold=2.0,
)
```

这份深度来自基础模型的每帧预测，经过 `align_extrinsics` 和后续 frame 顺序恢复；它不是融合后的全局 dense cloud 重投影深度。

导出目录：

```text
<output_dir>/single_frame_depth/
  depth_npy/      # float32 depth
  depth_u16/      # uint16 millimeter PNG 深度图
  depth_vis/      # 可视化深度图
  confidence/     # confidence mask
```

当前 `conf_threshold=2.0` 表示按 `depth_conf` 的第 2 百分位做过滤：

```python
valid_conf = conf_np >= np.percentile(conf_np, 2.0)
```

`confidence/` 目录中的 mask 语义为：

- 白色：高置信度，有效，后续可保留。
- 黑色：低置信度，无效，后续应过滤。

`export_prediction_depth_maps()` 当前采用逐帧写出的方式，避免一次性构造巨大数组导致内存峰值过高。

## 8. 高分辨率输出模式

如果传入 `--high-dataset`，流程会额外加载一套高分辨率图像：

```python
high_output_images, high_output_image_names = load_images_matching_names(
    args.high_dataset,
    sequence.image_names,
    device=sequence.images.device,
)
high_output_intrinsic = scale_intrinsics_between_image_sets(
    low_intrinsic,
    sequence.images,
    high_output_images,
)
```

该模式用于后续 depth refine 或高分辨率相机输出。低分辨率 BA 后内参会按图像尺寸比例缩放到高分辨率。

当前 COLMAP 主输出仍调用低分辨率 `write_recon_to_colmap()`；高分辨率模式下另外写出：

```text
<output_dir>/colmap/high_cameras.txt
```

## 9. Dense model points 导出

如果 `final_predictions` 中包含 `local_points`，当前流程会总是导出融合后的 dense 点云：

```python
dense_world_points, dense_world_colors, dense_world_stats = collect_dense_world_points(
    final_predictions['local_points'],
    final_predictions['extrinsic'],
    sequence.images,
    final_predictions['depth_conf'],
    conf_threshold=args.point_vis_threshold,
    stride=args.dense_depth_stride if args.save_dense_depth else 1,
    max_points=args.dense_max_points if args.dense_max_points > 0 else None,
    include_colors=True,
)
```

输出：

```text
<output_dir>/dense_model_points.ply
```

当前点云导出逻辑要点：

- 输入是每帧的 `local_points`。
- 通过 BA 后 `extrinsic` 变换到全局坐标。
- 使用 `depth_conf` 过滤低置信度点。
- 默认最多导出 `50_000_000` 个点。
- 如果 `--save_dense_depth` 为 false，点采样 stride 固定为 `1`。

注意：当前 `main.py` 中从 merged dense cloud 重投影深度图的 `export_dense_projected_depth_maps()` 调用已经被注释，因此 `--save_dense_depth` 相关参数目前不产生 dense depth 输出。

## 10. LingBot-Depth refine

如果传入 `--lingbot_refine`，当前流程会调用：

```python
run_lingbot_depth_refinement(
    lingbot_images,
    lingbot_image_names,
    os.path.join(args.output_dir, "single_frame_depth", "depth_npy"),
    lingbot_output_dir,
    shared_intrinsic,
    device=device,
    depth_image_names=sequence.image_names,
)
```

当前 LingBot 输入深度为：

```text
<output_dir>/single_frame_depth/depth_npy
```

也就是说，它 refine 的是单帧预测深度，而不是 merged dense cloud 投影深度。

输出目录默认为：

```text
<output_dir>/lingbot_depth/
  depth_npy/
  depth_png/
  depth_vis/
```

如果启用了 `--high-dataset`：

- LingBot 输入图像使用高分辨率图像。
- 内参使用缩放后的高分辨率共享内参。
- 输出深度保持高分辨率图像尺寸。

当前 `dense_lingbot_refined_points.ply` 反投影导出逻辑已注释，因此 LingBot refine 后不会再自动生成 refined dense PLY。

## 11. 统计输出

主流程会写出：

```text
<output_dir>/computation_stats.txt
```

当前统计项包括：

- 总 runtime
- sequence time
- inference time
- tracking time
- dropped untracked frames
- BA time
- BAE time 与 BAE loss/观测统计
- single-frame depth 导出统计
- high dataset 信息
- LingBot refine 统计
- dense world collect time
- dense PLY time
- single-frame depth time
- LingBot depth time
- peak GPU memory

其中 dense depth projection、dense correction、InfiniDepth 等统计写出代码当前已注释或不进入主流程。

## 12. COLMAP 输出

最后调用：

```python
write_recon_to_colmap(
    args.output_dir,
    final_predictions,
    sequence.images,
    sequence.image_names,
    stride=args.stride,
    conf_threshold=args.point_vis_threshold,
    format=args.format,
    shared_camera=args.global_ba,
)
```

输出目录：

```text
<output_dir>/colmap/
```

当前默认：

```text
stride = 10
conf_threshold = 20.0
format = txt
shared_camera = True
```

高分辨率模式下，当前额外写出 `high_cameras.txt`，用于记录缩放后的高分辨率 camera intrinsics。

## 当前主输出清单

一次标准运行后，主要输出为：

```text
<output_dir>/
  config.json
  computation_stats.txt
  dense_model_points.ply
  single_frame_depth/
    depth_npy/
    depth_u16/
    depth_vis/
    confidence/
  colmap/
    cameras.txt / cameras.bin
    images.txt / images.bin
    points3D.txt / points3D.bin
```

启用 `--lingbot_refine` 后额外输出：

```text
<output_dir>/lingbot_depth/
  depth_npy/
  depth_png/
  depth_vis/
```

启用 `--high-dataset` 后额外输出：

```text
<output_dir>/colmap/high_cameras.txt
```

## 当前未纳入主流程的部分

以下能力在代码中仍有 import、helper 或历史实现，但当前 `main.py` 主流程中相关参数或调用已被注释，因此不作为当前版本 pipeline 描述：

- dense debug 输出。
- inverse-depth affine dense correction。
- merged dense cloud 投影导出 `dense_depth`。
- LingBot refined depth 反投影生成 `dense_lingbot_refined_points.ply`。
- InfiniDepth refine 主流程。

备注：当前文件中仍残留一个 `elif args.infinidepth_refine:` 分支，但对应 argparse 参数已被注释。按当前“注释部分忽略”的约定，本文档将 InfiniDepth 视为未启用流程。
