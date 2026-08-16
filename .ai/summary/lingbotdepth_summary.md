# LingBot-Depth 项目概述

**论文**：[Masked Depth Modeling for Spatial Perception (arXiv:2601.17895)](https://arxiv.org/abs/2601.17895)  
**模型**：[HuggingFace robbyant/lingbot-depth-pretrain-vitl-14-v0.5](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5)

## 定位

LingBot-Depth 是一个**深度图补全与精化**模型。输入 RGB 图像 + 原始传感器深度图（可含空洞/噪声），输出 metric-scale（真实米制）的精化深度图和对应 3D 点云。主要面向机器人感知场景（4D 点跟踪、灵巧抓取）。

## run.py 流程

```
输入：
  images_dir/    *.jpg          # RGB 图像序列
  depth_dir/     *.npy          # 原始传感器深度图（float32，单位米）
  --fx/fy/cx/cy                 # 相机内参（像素单位）

    ↓ 逐帧处理

1. 读取 RGB：cv2 → [1, 3, H, W] float32 tensor，归一化到 [0,1]
2. 读取深度：.npy → resize 到 (1400, 1904) → [1, H, W] float32 tensor（米）
3. 构造归一化内参矩阵：
     [[fx/W,  0,   cx/W],
      [0,    fy/H, cy/H],
      [0,     0,    1  ]]   形状 [1, 3, 3]
4. model.infer(image, depth_in, intrinsics)
     → output['depth']：精化深度图 [1, H, W]（米）
     → output['points']：3D 点云 [1, H, W, 3]（相机坐标系）

5. 保存三种格式：
   output_dir/depth_npy/{stem}.npy    float32 深度（米）
   output_dir/depth_vis/{stem}.png    Turbo colormap 可视化
   output_dir/depth_png/{stem}.png    uint16 PNG（值 = depth_m × 1000）
```

## 模型架构（`MDMModel`）

- **主干**：DINOv2 ViT-Large（patch_size=14）
- **输入融合**：RGB + 深度图通过 Cross-Modal Attention 在统一 latent space 对齐
- **解码器**：多尺度特征金字塔 + 深度回归头
- **预训练方式**：Masked Depth Modeling（自监督，遮盖深度图后重建）

## 接口

```python
from mdm.model.v2 import MDMModel

model = MDMModel.from_pretrained('robbyant/lingbot-depth-pretrain-vitl-14-v0.5').to(device)

output = model.infer(
    image,              # [B, 3, H, W] float32, [0,1]
    depth_in=depth,     # [B, H, W] float32, 米，0/NaN 表示无效
    intrinsics=K,       # [B, 3, 3]，fx/W 归一化格式
    enable_depth_mask=False,
)
# output['depth']:  [B, H, W]    精化深度（米）
# output['points']: [B, H, W, 3] 相机坐标系 3D 点
```

## 模型变体

| 模型 | HuggingFace | 说明 |
|------|-------------|------|
| v0.5（推荐） | `robbyant/lingbot-depth-pretrain-vitl-14-v0.5` | 通用深度精化，修复 v0.1 的 bug |
| v0.1 | `robbyant/lingbot-depth-pretrain-vitl-14` | 早期版本 |
| DC 版 | `robbyant/lingbot-depth-postrain-dc-vitl14` | 专为稀疏深度补全优化 |

## 训练数据

3M RGB-D 样本：2M 真实室内（居家/办公/商业场所，Intel RealSense / Orbbec / Azure Kinect 采集）+ 1M 仿真渲染。
