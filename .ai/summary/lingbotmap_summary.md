# LingBot-Map 项目概述

**论文**：[Geometric Context Transformer for Streaming 3D Reconstruction (arXiv:2604.14141)](https://arxiv.org/abs/2604.14141)  
**模型**：[HuggingFace robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map)（4.63 GB）

## 定位

LingBot-Map 是一个**前馈式流式 3D 重建基础模型**。给定一段图像序列（图片文件夹或视频），单次前向推理即可输出：
- 每帧相机位姿（外参 + 内参）
- 每帧深度图
- 每帧像素级 3D 世界坐标点云

速度约 **~20 FPS**（518×378 分辨率），支持超过 10,000 帧的超长序列。

## demo.py 完整流程

```
输入：图片文件夹 / 视频文件
    ↓  load_images()
    │  - 视频：用 OpenCV 按目标 fps 抽帧，存为 jpg
    │  - 图片：glob 排序读取
    │  - load_and_preprocess_images()：crop 到 518×378，归一化到 [0,1]
    │  → images: [S, 3, H, W]
    ↓  load_model()
    │  - 构建 GCTStream（或 GCTStreamWindow）
    │  - 加载 checkpoint（strict=False）
    ↓  inference_streaming() 或 inference_windowed()
    │  → predictions: {pose_enc, depth, depth_conf, world_points, world_points_conf}
    ↓  postprocess()
    │  - pose_encoding_to_extri_intri()：pose_enc[B,S,9] → extrinsic(w2c) + intrinsic
    │  - closed_form_inverse_se3()：w2c → c2w（相机到世界坐标系）
    │  → predictions: {extrinsic[B,S,3,4], intrinsic[B,S,3,3], depth, world_points, ...}
    ↓  PointCloudViewer (viser)
    │  - 浏览器端 3D 点云可视化（默认 http://localhost:8080）
    │  - 可选天空点过滤（ONNX skyseg 模型）
```

## 模型架构（GCTStream）

```
GCTBase (nn.Module)
├── aggregator: AggregatorStream          # 特征提取 + 时序聚合
│   ├── backbone: DINOv2 ViT-L/14-reg    # patch_size=14, embed_dim=1024
│   │   └── 提取多尺度特征（层 4,11,17,23）
│   └── 流式 Transformer 块（FlashInfer / SDPA）
│       ├── 时序因果注意力（每帧只看过去帧）
│       └── KV Cache（分页缓存，FlashInfer 后端）
├── camera_head: CameraCausalHead         # 相机位姿预测头
│   └── 4 层 Transformer trunk → 输出 pose_enc[B,S,9]
├── depth_head: DPTHead                   # 深度预测（DPT 解码器）
│   └── 输出 depth[B,S,H,W,1] + depth_conf[B,S,H,W]
└── point_head: DPTHead                   # 世界坐标预测（DPT 解码器）
    └── 输出 world_points[B,S,H,W,3] + world_points_conf[B,S,H,W]
```

## 三大核心设计

### 1. Anchor Context（锚帧 / Scale Frames）
- 序列前 N 帧（默认 8 帧）作为 Scale Frames 整体处理（双向注意力）
- 建立全局尺度基准，始终保留在 KV Cache 中
- 后续每帧推理时可通过注意力回溯锚帧，防止尺度漂移

### 2. Pose-Reference Window（滑动窗口 KV Cache）
- 每个新帧与**最近 N 帧**（默认 `kv_cache_sliding_window=16`）做因果注意力
- 超出窗口的帧被驱逐出 cache（节省显存）
- 保持短程时序一致性

### 3. Trajectory Memory（轨迹记忆 / Cross-Frame Special Tokens）
- 从被驱逐帧中保留"特殊 token"（`kv_cache_cross_frame_special=True`）
- 实现对历史帧的长程漂移纠正，即使该帧已不在滑动窗口内

## 两种推理模式

### Streaming 模式（默认，适合 <3000 帧）
```python
model.inference_streaming(images, num_scale_frames=8, keyframe_interval=1)
```
- **Phase 1**：Scale frames 整组处理（双向注意力，建立尺度参考）
- **Phase 2**：逐帧处理，KV cache 累积
- `keyframe_interval > 1`：每 N 帧才将 KV 写入 cache（非关键帧仍预测但不存储），降低长序列显存

### Windowed 模式（适合 >3000 帧）
```python
model.inference_windowed(images, window_size=64, overlap_size=16)
```
- 将序列分成重叠窗口，每窗口独立推理后拼接

## 输出格式

| 字段 | Shape | 说明 |
|------|-------|------|
| `pose_enc` | [B, S, 9] | 位姿编码：绝对平移(3) + 四元数(4) + FoV(2)（absT_quaR_FoV） |
| `extrinsic` | [B, S, 3, 4] | 相机到世界坐标系的 c2w 矩阵（postprocess 后） |
| `intrinsic` | [B, S, 3, 3] | 相机内参矩阵（postprocess 后） |
| `depth` | [B, S, H, W, 1] | 深度图 |
| `depth_conf` | [B, S, H, W] | 深度置信度 |
| `world_points` | [B, S, H, W, 3] | 像素对应的世界坐标 3D 点 |
| `world_points_conf` | [B, S, H, W] | 3D 点置信度（用于过滤噪声点） |

## 关键依赖

| 依赖 | 作用 |
|------|------|
| PyTorch 2.9+ | 基础框架 |
| DINOv2 ViT-L/14 | 图像特征提取骨干 |
| FlashInfer | 分页 KV Cache 注意力（推荐，可用 SDPA 替代） |
| viser | 浏览器端 3D 点云可视化 |
| onnxruntime | 天空分割（可选） |

## 与 VGGT 的关系

架构上参考并基于 VGGT 改造，核心差异在于将 **非流式全局注意力** 替换为 **因果流式 KV Cache 注意力**，从而支持实时/在线场景。
