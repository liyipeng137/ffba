# plan_vggtslam_pi3: 将 Pi3/Pi3X 接入 VGGT-SLAM 的分阶段计划

## 0. 当前决策

目前主线改为：

```text
基于 VGGT-SLAM 的 submap + graph 思路
先把基础模型从 VGGT 替换/适配为 Pi3/Pi3X
先跑通普通 submap 建图
再逐步替换 VGGT-specific 的回环逻辑
最后再考虑 LoMa / BAE / dense fusion 增强
```

第一部分不要一次性解决所有问题。先实现最小闭环：

```text
输入图片序列
  -> VGGT-SLAM keyframe/submap 逻辑
  -> Pi3/Pi3X submap inference
  -> 转成 VGGT-SLAM 期望的 predictions
  -> Solver.add_points()
  -> PoseGraph.optimize()
  -> 输出 selected-frame poses + dense point cloud
```

## 1. 我们之前讨论过的 VGGT 强依赖点

VGGT-SLAM 当前依赖 VGGT 的地方主要有三类。

### 1.1 普通 submap 推理依赖 `pose_enc`

当前 `Solver.run_predictions()` 做法：

```python
predictions = model(images)
extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic
```

也就是说 VGGT-SLAM 期待基础模型输出：

```text
pose_enc
depth
depth_conf
```

然后在 `Solver.add_points()` 中用：

```text
depth + extrinsic + intrinsic -> unproject_depth_map_to_point_map()
```

转换成 submap dense point cloud。

Pi3/Pi3X 没有 `pose_enc`，但有：

```text
camera_poses: c2w, OpenCV
local_points: 每帧相机坐标点图
points: world points
conf: confidence logits
```

因此 Phase 1 的核心是写 adapter，把 Pi3 输出转成 VGGT-SLAM 需要的：

```text
extrinsic: w2c, [S, 3, 4]
intrinsic: [S, 3, 3]
depth: [S, H, W, 1]
depth_conf: [S, H, W]
images: [S, 3, H, W]
```

### 1.2 回环验证依赖 VGGT `image_match_ratio`

当前 VGGT-SLAM 对 loop pair 做：

```python
predictions_lc = model(lc_frames, compute_similarity=True)
image_match_ratio = predictions_lc["image_match_ratio"]
```

如果低于阈值就拒绝回环。

Pi3/Pi3X 没有 `compute_similarity=True` 和 `image_match_ratio`。我们之前讨论的替代方案是把回环拆成两层：

```text
LoopCandidateDetector:
  SALAD / DBoW2 / NetVLAD / DINO retrieval

LoopConstraintEstimator:
  LoMa 2D-2D matches
  -> 几何验证
  -> 2D-3D PnP 或 3D-3D Sim(3)
  -> 生成 loop edge
```

但这不是第一阶段要做的。Phase 1 可以先：

```text
max_loops = 0
禁用 loop closure
```

或保留 retrieval 但不插 loop edge。

### 1.3 回环 submap 构造依赖 VGGT pair inference

当前 VGGT-SLAM 不只是检测回环，还会为 loop pair 创建一个额外 loop closure submap：

```text
current loop frame + retrieved frame
  -> VGGT 推理
  -> 得到 pair-local pose/depth
  -> 插入 PoseGraph
```

这部分也依赖 VGGT 能对两帧给出稳定的 pair-local geometry。

Pi3/Pi3X 可以尝试两帧/小窗口推理，但不能直接假设等价。因此后续有两个方向：

```text
A. 保留 loop submap 思路：
   Pi3X 对 loop pair/window 推理
   LoMa 几何验证替代 image_match_ratio

B. 替换为直接 loop edge：
   LoMa matches + Pi3 dense/local points
   -> 估计 Sim(3)/PnP edge
   -> 不创建 VGGT-style loop submap
```

我们之前更倾向于 B，因为它更模型无关。

## 2. Phase 1 目标：Pi3/Pi3X 接入普通 submap 路径

### 2.1 目标

只跑通：

```text
VGGT-SLAM main loop
  -> selected keyframes
  -> Pi3/Pi3X inference per submap
  -> GraphMap/Submap/PoseGraph
  -> dense map export
```

暂时不做：

```text
loop closure
LoMa
BAE
non-keyframe pose recovery
SL(4) 后端替换
```

### 2.2 预期输出

Phase 1 跑完应该能输出：

```text
poses.txt
export_pcd.ply / .pcd
framewise pointcloud logs, optional
```

注意：如果仍然使用 VGGT-SLAM 的 optical-flow keyframe selection，则最终 pose 只包含被选中的 keyframes/submap frames，不包含被跳过的原始视频全帧。

## 3. Phase 1 实现方案

### 3.1 增加 BaseModelAdapter 抽象

建议新增：

```text
VGGT-SLAM/vggt_slam/model_adapters/
  __init__.py
  base.py
  vggt_adapter.py
  pi3_adapter.py
```

统一接口：

```python
class BaseSubmapModel:
    def load(self): ...

    def infer_submap(self, images: torch.Tensor, image_names: list[str]) -> dict:
        """
        images: [S, 3, H, W], [0,1], already preprocessed by VGGT-SLAM loader
        return:
          images: [S, 3, H, W]
          extrinsic: [S, 3, 4]  # w2c OpenCV
          intrinsic: [S, 3, 3]
          depth: [S, H, W, 1]
          depth_conf: [S, H, W]
          optional:
            world_points: [S, H, W, 3]
            local_points: [S, H, W, 3]
            point_conf: [S, H, W]
        """
```

`VGGTAdapter` 先包住当前 VGGT 逻辑，保证原流程不变。

`Pi3Adapter` / `Pi3XAdapter` 将 Pi3 输出转换成统一格式。

### 3.2 修改 `main.py` 模型选择

新增参数：

```text
--base_model vggt|pi3|pi3x
--pi3_ckpt <path optional>
--disable_loop_closure_for_non_vggt 默认 True
```

Phase 1 默认：

```text
--base_model pi3x
--max_loops 0
```

避免先碰 VGGT-specific loop closure。

### 3.3 修改 `Solver.run_predictions()`

当前 `run_predictions()` 直接假设：

```python
predictions = model(images)
pose_encoding_to_extri_intri(...)
```

需要改成：

```python
predictions = model_adapter.infer_submap(images, image_names)
```

并要求 adapter 返回已经转换好的：

```text
extrinsic
intrinsic
depth
depth_conf
images
detected_loops
```

Phase 1 中：

```python
predictions["detected_loops"] = []
```

### 3.4 Pi3/Pi3X 输出转换细节

Pi3/Pi3X 输出：

```text
camera_poses: [B, S, 4, 4] c2w OpenCV
local_points: [B, S, H, W, 3]
points: [B, S, H, W, 3]
conf: [B, S, H, W, 1] logits
```

转换：

```text
c2w = camera_poses[0]
w2c = inverse(c2w)
extrinsic = w2c[:, :3, :4]
depth = local_points[0, ..., 2:3]
depth_conf = sigmoid(conf[0, ..., 0])
```

内参：

```text
优先级 1: 用户提供 intrinsics / transforms / calib
优先级 2: Pi3X conditioning intrinsics
优先级 3: recover_intrinsics_from_output(res, imgs)
优先级 4: 默认 pseudo intrinsics
```

Phase 1 建议先使用已有 `Pi3/pi3/utils/transforms_utils.py::recover_intrinsics_from_output()` 或手动提供 shared K，避免内参不一致导致点云/pose 分解错误。

### 3.5 `Solver.add_points()` 是否使用 Pi3 world points

当前 `add_points()` 总是：

```python
world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
```

Phase 1 有两种选择：

#### 方案 A：最小改动

保持不变。Pi3Adapter 提供：

```text
depth = local_points[..., 2]
extrinsic = inverse(camera_poses)
intrinsic = recovered/provided K
```

让 VGGT-SLAM 自己通过 depth 反投影生成 world_points。

优点：

- 改动最小。
- 兼容 VGGT-SLAM 当前逻辑。

风险：

- 如果 Pi3 local_points 与 recovered K 不完全一致，反投影 world_points 可能不如 Pi3 原始 `points`。

#### 方案 B：允许 adapter 提供 world_points

修改 `add_points()`：

```python
if "world_points" in pred_dict:
    world_points = pred_dict["world_points"]
else:
    world_points = unproject_depth_map_to_point_map(...)
```

优点：

- 直接用 Pi3 模型输出的 dense world points。
- 避免 K/depth 反投影误差。

风险：

- VGGT-SLAM 的 SL(4) / projection matrix 逻辑仍依赖 depth/intrinsic/extrinsic 一致性。

Phase 1 建议先做方案 A，作为最小可跑通版本；随后做 A/B 对比。

## 4. Phase 1 验证清单

### 4.1 单 submap sanity check

使用 `submap_size <= 16`、`max_loops=0`，只跑一个 submap。

检查：

```text
Pi3Adapter 输出 shape 正确
w2c @ c2w 接近 I
depth 全部有限，正值比例合理
depth_conf 分布合理
intrinsics 合理
Solver.add_points 不报错
导出的 point cloud 方向/尺度正常
```

### 4.2 多 submap 无 loop

使用 50-200 帧：

```text
submap_size = 16
overlapping_window_size = 1
max_loops = 0
```

检查：

```text
相邻 submap 能通过 overlapping frame 连接
PoseGraph.optimize 能运行
轨迹连续
点云没有明显翻转/爆炸
```

### 4.3 VGGT baseline 对照

同一短序列：

```text
base_model=vggt
base_model=pi3x
```

对比：

```text
submap 内点云质量
轨迹连续性
跨 submap 对齐质量
处理速度
```

## 5. Phase 2：替换 VGGT-specific 回环逻辑

Phase 2 才处理回环。我们之前讨论过的方案是：

```text
retrieval/SALAD/DBoW 只负责 loop candidate
LoMa 负责 2D-2D 几何验证
Pi3/VGGT dense/local points 负责 lift 到 3D
RANSAC 估计 Sim(3) 或 PnP
将通过验证的相对几何作为 loop edge
```

### 5.1 先替换回环验证

保留 VGGT-SLAM 当前 retrieval candidate：

```text
ImageRetrieval.find_loop_closures()
```

但把：

```text
VGGT image_match_ratio
```

替换成：

```text
LoMa match count
E/F matrix inlier ratio
2D reprojection error
```

这是最小替换，先只决定“接不接受 loop”。

### 5.2 再替换 loop constraint generation

更彻底的方案：

```text
LoMa 2D matches
  -> sample dense 3D from current/historical submap
  -> 3D-3D Sim(3) RANSAC
  -> 生成 relative Sim(3)/homography edge
```

或者：

```text
LoMa 2D matches
  -> old frame 3D + new frame 2D
  -> PnP
  -> 生成 SE(3) loop edge
```

这里需要决定是否继续使用 VGGT-SLAM 的 SL(4) graph，还是新增 Sim(3) graph backend。

Phase 2 初期建议继续保留 SL(4)，把估计出的 Sim(3) 写成 4x4 factor 注入；如果不稳定，再单独实现 Sim(3) backend。

## 6. Phase 3：BAE / sparse BA refinement

在 Pi3 接入和 loop 替换都稳定后，再考虑 BAE。

可能路线：

```text
LoMa tracks
  -> 构造 COLMAP sparse model
  -> 初始 pose 来自 VGGT-SLAM/Pi3 graph
  -> BAE refine selected keyframe poses + sparse points
  -> 用 refined poses 重新放置 Pi3 local_points
```

注意：

```text
BAE 不处理 dense points
dense cloud 来自 Pi3 local_points + optimized c2w
```

## 7. Phase 4：恢复非 keyframe 位姿

当前 VGGT-SLAM 固定 `use_optical_flow_downsample=True` 时，只输出 keyframes/submap frames 的 pose。

如果最终目标需要全帧轨迹，需要补：

```text
non-keyframe tracking/localization
```

候选方案：

```text
1. 关闭 optical-flow keyframe selection，让所有帧进入 submap
2. keyframe 建图 + 非 keyframe 用 PnP/LoMa localized 到附近 keyframes
3. 对跳过帧在相邻 keyframe pose 之间插值，作为低精度 baseline
```

Phase 1 暂不处理。

## 8. 风险与注意事项

### 8.1 Intrinsics 是 Pi3 接入最大风险之一

VGGT-SLAM 的 pose/point/world transformation 依赖：

```text
depth + K + w2c 一致
```

如果 K 恢复错误，点云会明显变形，SL(4) graph 也会受到影响。

建议：

```text
优先用已知 K 或 transforms
其次用 Pi3 local_points recover K
最后才用 pseudo K
```

### 8.2 Pi3 world points 与 VGGT-SLAM depth unprojection 可能不一致

Pi3 原始 `points` 是模型内部用 `camera_poses` 和 `local_points` 得到的；VGGT-SLAM 如果重新用 `depth + recovered K` 反投影，可能不同。

因此 Phase 1 后必须做：

```text
A. add_points 使用 depth unprojection
B. add_points 直接使用 Pi3 points
```

对比点云质量。

### 8.3 SL(4) graph 是否适合 Pi3 需要实验

VGGT-SLAM 使用 SL(4) 是为 VGGT submap 的投影变换误差设计。Pi3/Pi3X 可能更适合 Sim(3)。

Phase 1 先复用 SL(4)，因为改动少。若发现跨 submap 对齐异常，再考虑：

```text
Sim(3) graph backend
```

### 8.4 回环不要直接照搬 VGGT 的 pair inference

Pi3 没有 `image_match_ratio`，即使能两帧推理，也未必能稳定生成 loop submap。

后续回环应改成：

```text
retrieval candidate
-> LoMa geometric verification
-> 3D-3D Sim(3) / PnP constraint
```

## 9. 推荐实施顺序

```text
1. 新增 BaseModelAdapter / VGGTAdapter / Pi3XAdapter
2. main.py 增加 --base_model，默认保留 vggt
3. Solver.run_predictions 改为调用 adapter
4. Pi3XAdapter 输出 VGGT-SLAM predictions schema
5. max_loops=0 跑单 submap
6. max_loops=0 跑多 submap
7. 对比 add_points 使用 depth unprojection vs Pi3 points
8. 再开始 LoMa loop verification
9. 再考虑 BAE refinement
```

## 10. Phase 1 成功标准

满足以下条件即可认为“Pi3 已接入 VGGT-SLAM 普通路径”：

```text
1. base_model=vggt 原流程仍可运行
2. base_model=pi3x max_loops=0 可运行
3. 单 submap 点云方向、尺度、相机姿态无明显错误
4. 多 submap 能连接并优化，不崩溃
5. 能导出 selected-frame poses 和 dense point cloud
6. 对同一序列，Pi3X submap dense quality 至少可与独立 Pi3X 输出一致
```
