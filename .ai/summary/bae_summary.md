# bae 项目概述

**论文**：[Bundle Adjustment in the Eager Mode (arXiv:2409.12190)](https://arxiv.org/abs/2409.12190)

## 定位

`bae` 是一个基于 PyTorch 的稀疏二阶优化库，专门用于计算机视觉和机器人领域的 **Bundle Adjustment (BA)** 和 **Pose Graph Optimization (PGO)**。它的核心创新是在"Eager 模式"下（类似 PyTorch 的即时执行风格）高效构建和求解稀疏 Jacobian 矩阵，而不依赖 COLMAP 等离线 BA 工具。

## 核心架构

### 1. 稀疏 Jacobian 追踪（`bae/autograd/`）

BA 的关键挑战是 Jacobian 矩阵高度稀疏（每个 2D 观测只与对应的一个相机和一个 3D 点相关）。`bae` 用自定义机制解决这个问题：

- **`TrackingTensor`**：对张量进行包装，在计算时记录操作追踪（`optrace`），分两类：
  - `'index'`：索引操作，记录稀疏的对应关系（哪个观测对应哪个相机/点）
  - `'map'`：映射操作（通过 `@map_transform` 装饰器标记），记录可 vmap 的函数
- **`backward()`**：自定义反向追踪，使用 `torch.vmap(jacrev(...))` 计算局部 Jacobian 块，通过链式法则拼接，最终以稀疏 BSR（Block Sparse Row）张量输出
- **`jacobian(output, params)`**：对外接口，执行上述追踪并返回每个参数的稀疏 Jacobian

### 2. 稀疏矩阵运算（`bae/sparse/`）

- 支持 BSR / BSC 格式的稀疏块矩阵
- `CuSparse`：基于 CUDA 的稀疏矩阵乘法（用于构建 Hessian $J^T J$）
- 自定义 CUDA kernel 用于高效稀疏线性代数

### 3. LM 优化器（`bae/optim/optimizer.py`）

继承自 PyPose 的 `LevenbergMarquardt`，关键流程：

```
1. 前向传播得残差 R = model(input)
2. 追踪计算稀疏 Jacobian J
3. 构建 Hessian 近似 A = J^T J（CuSparse）
4. 添加 LM 阻尼：A += λI
5. 用线性求解器求解 A·D = -J^T·R
6. 用 TrustRegion 策略判断接受/拒绝步长
7. 参数更新：pose 用 SE3 流形加法（PyPose），3D 点用欧氏加法
```

**SE3 切空间处理**：位姿用四元数 `[tx,ty,tz,qx,qy,qz,qw]`（7维）存储，但切空间只有 6 维，通过 `trim_SE3_grad=True` 截断 Jacobian 的最后一列。

### 4. 线性求解器（`bae/utils/pysolvers.py`）

- **PCG**（Preconditioned Conjugate Gradient）：纯 PyTorch 实现
- **CUDSS**：可选的 NVIDIA CUDA Sparse Solver 后端

## COLMAP 接入流程（`ba_colmap.py`）

```
COLMAP 模型 (.txt/.bin)
    ↓  read_colmap_data()         # 支持自动检测 txt/bin 格式
数据字典 {camera_params, points_3d, points_2d, 观测索引, intrinsics}
    ↓  Reproj(...)                # nn.Module，相机位姿 + 3D 点为可学习参数
残差 = 投影点 - 观测 2D 点        # PINHOLE 模型：project_colmap()
    ↓  LM.step()                 # Levenberg-Marquardt 迭代
优化后位姿 / 3D 点 / 内参
    ↓  save_colmap_result()
优化后 COLMAP txt 文件
```

**相机模型限制**：目前仅支持单一共享 **PINHOLE** 相机（fx, fy, cx, cy），可选联合优化内参（`--optimize-intrinsics`）。

## 数据集支持

| 数据集 | 格式 | 用途 |
|--------|------|------|
| BAL (Bundle Adjustment in the Large) | 自定义 | BA 基准测试 |
| 1DSfM | 自定义 | 大规模 SfM |
| G2O | 图优化格式 | PGO |
| COLMAP | txt/bin | 实际场景 BA |

## 外部集成

作为 [VGGT](https://github.com/zitongzhan/vggt) 的可选 BA 后端：VGGT 用前馈网络预测相机位姿和 3D 点，再调用 `bae.optim.LM` 细化后导出 COLMAP 格式重建结果。

## 依赖

- PyTorch 2.0+（包含 `torch.vmap`、`torch.func.jacrev`、稀疏张量）
- [PyPose](https://github.com/pypose/pypose)（`bae` 分支）：SE3 位姿表示与 LM 基类
- CUDA 12.x（可选 CUDSS）
