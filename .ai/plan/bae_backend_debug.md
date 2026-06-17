# BAE Backend Debug Notes

更新时间：2026-06-17

## 背景

当前 Merg3r + GlueMap pipeline 已经完成从 Ceres solver 到 BAE backend 的第一版接入。BAE 后端的目标是加速 GlueMap 中耗时较高的 augmented bundle adjustment，但实际测试显示，直接把 GlueMap 的 real + virtual observations 映射到 BAE 后，并不能立即获得与 Ceres 版一致的优化质量。

当前 BAE backend 仍作为独立后端存在，不替换原有 Ceres solver：

```text
--ba_backend ceres
--ba_backend bae
```

相关实现入口：

```text
MERG3R/gluemap/gluemap/estimators/bae_solver.py
MERG3R/gluemap/gluemap/controllers/augmented_bundle_adjustment.py
MERG3R/run_merg3r_gluemap_pipeline.py
```

当前可用的主要 BAE 模式：

```text
--ba_backend bae
--bae_real_only
--bae_optimize_intrinsics
```

其中 `--bae_optimize_intrinsics` 目前仅支持 `SIMPLE_PINHOLE` 的 `f` 优化，并固定 `cx/cy`。

## 已观察到的问题

### 1. 没有 Gauge Fix，内参 f 会累计漂移

当前 BAE backend 第一版没有实现 gauge fixing。开启：

```text
--bae_optimize_intrinsics
```

后，`SIMPLE_PINHOLE` 的 focal length `f` 会随着多轮 BAE 迭代持续下降，和 Ceres solver 的结果相差达到数百像素量级。

已观察到的现象：

```text
f: 1166.07 -> 971.642 -> 952.232 -> 944.408 -> 930.904
```

这类漂移不应被视作正常收敛。更可能的原因是：

- 当前 BAE 没有 fix gauge。
- 当前 BAE 没有 focal prior。
- BA 问题中 pose / point depth / focal length 之间存在尺度耦合。
- 多轮 refinement 会把上一轮漂移后的 intrinsics 继续作为下一轮初值，导致累计偏移。

当前结论：

```text
主实验暂时不建议启用 --bae_optimize_intrinsics。
```

更稳妥的 baseline 是：

```text
--ba_backend bae --bae_real_only
```

后续如果要恢复 BAE 内参优化，至少需要设计其中一种约束：

- 固定 gauge。
- 对 `f` 加 prior。
- 对 `f` 加合理 bounds / clamp。
- 只在特定外层 iteration 优化 `f`，避免每轮累计漂移。
- 和 Ceres 版对齐 camera / pose / point 的固定策略。

### 2. BAE 目前无法等价复刻 Ceres 的两种 loss_type

GlueMap 原版 Ceres augmented BA 中，real 和 virtual residual 进入同一个 Ceres problem，但使用不同的 loss type：

```text
real residual   -> loss_type_normal，例如 huber
virtual residual -> loss_type_virtual，例如 arctan
```

这样做的意义是：

- real tracks 通常来自 SfM / COLMAP reconstruction，可信度更高。
- virtual tracks 来自 GlueMap / Merg3r 增强观测，数量多但噪声和外点风险更高。
- virtual residual 需要更强的 robustification，避免大量 virtual observations 等权硬拟合，把 pose 拉偏。

当前 BAE backend 第一版没有实现这种 per-source robust loss。直接把 real + virtual observations 拼成一个 residual graph 后，本质上更接近：

```text
all residuals -> plain squared loss
```

这和 Ceres 版的优化语义并不一致。

已尝试过的简化方案：

```text
--bae_virtual_weight 0.1
```

测试结果显示，单纯降低 virtual residual 的全局权重并没有明显改善 pose 分层问题，因此该逻辑已经移除。

当前结论：

```text
BAE 的 full real+virtual 模式还不能视作 Ceres augmented BA 的等价替代。
```

如果后续要对齐 Ceres 机制，需要在 BAE 内部实现至少一种 per-residual robustification：

- real / virtual residual 使用不同 robust kernel。
- 根据 residual source 生成不同权重。
- 在 BAE 的 residual/Jacobian 计算后做 IRLS-style reweight。
- 更完整地复刻 Ceres 对 robust loss 的 `rho / rho' / rho''` 处理。

这大概率需要修改 BAE optimizer 或 residual graph 的结构，而不是只在 GlueMap wrapper 层简单拼接输入。

### 3. 当前稳定路径只能使用 real only 模式

测试显示：

```text
--ba_backend bae
```

直接加入 virtual observations 后，最终 pose 质量明显差于 Ceres solver，并出现类似 pose 被分成两层的现象。

而启用：

```text
--ba_backend bae --bae_real_only
```

后，pose 分层问题消失。这说明：

- BAE 对 real reconstruction 的 BA 本身是可用的。
- 当前主要问题不是 pose 参数写回或 BAE 基础投影模型。
- 问题更集中在 virtual residual 如何进入 BAE 优化，以及缺少 Ceres 版 robust loss / gauge 策略。

因此当前建议把 BAE 后端定位为：

```text
real-only BA 加速后端
```

而不是完整的 GlueMap augmented BA 替代。

## Real Only 下 Virtual Reconstruction 的状态

在 `--bae_real_only` 模式下，BAE 优化图只包含 real reconstruction 中的 tracks 和 observations：

```text
real poses
real 3D points
real observations
```

virtual observations 不进入 BAE residual graph，virtual points xyz 也不会被 BAE 优化。

但是每轮 BAE 结束后，当前 GlueMap pipeline 仍会把优化后的 pose / intrinsics 同步到 `virtual_reconstruction`。这会导致 virtual reconstruction 变成：

```text
new poses / new intrinsics
+ old virtual points xyz
+ old virtual observations
```

因此，BA 后的 virtual reconstruction 并不是一个经过联合优化的一致 reconstruction。它更像是用更新后的 real pose 去重新检验旧 virtual xyz。

这会带来一个直接影响：

```text
virtual observations / virtual tracks 在后续 select / filter 阶段可能被大量过滤。
```

这种现象并不一定表示 BAE real-only 的 pose 失败，而是因为 virtual xyz 没有跟随新 pose 一起重新优化或重建。

## 与 Ceres 版的核心差异

Ceres 版 augmented BA 的语义大致是：

```text
real residual
+ virtual residual
-> same Ceres problem
-> shared camera poses / intrinsics
-> real points and virtual points jointly optimized
-> different robust loss for real / virtual
```

当前 BAE real-only 语义是：

```text
real residual only
-> BAE problem
-> optimize real poses / real points
-> sync optimized poses to virtual reconstruction
-> virtual points are not optimized
```

因此二者不是等价替代。

主要差异包括：

- Ceres 会让 virtual residual 直接参与 pose 优化；BAE real-only 不会。
- Ceres 会联合优化 virtual points xyz；BAE real-only 不会。
- Ceres 对 real / virtual 使用不同 loss type；BAE 当前没有。
- Ceres 的过滤发生在一个更一致的 augmented reconstruction 后；BAE real-only 的过滤发生在 `new pose + old virtual xyz` 的组合上。

## 后续可能方向

### 方向 A：维持 BAE Real Only，加速主 BA

这是当前最稳定、最容易继续验证的方向。

优点：

- BAE 速度优势明显。
- pose 分层问题消失。
- 改动边界清晰，不需要深改 BAE optimizer。

缺点：

- 不等价于 Ceres augmented BA。
- virtual observations 不直接参与 pose 优化。
- virtual points 可能在 BA 后因 pose 更新被大量过滤。

### 方向 B：每轮 BA 后重建或重初始化 Virtual Points

如果继续采用 BAE real-only，一个重要改进是：

```text
每轮 BA 更新 pose 后，基于新 pose 重新构建 / 重新三角化 virtual points。
```

这样可以避免长期使用：

```text
new poses + old virtual xyz
```

的组合。

这个方向可能比强行把 virtual residual 拼进当前 BAE plain squared loss 更稳，因为它承认 BAE 当前只负责 real BA，然后让 virtual reconstruction 在每轮外层 refinement 中重新和新 pose 对齐。

需要进一步设计的问题：

- virtual points 是全部重建，还是只重建被过滤掉的部分。
- 重建发生在每轮 BAE 之后，还是下一轮 `triangulate_from_seed_reconstruction()` 前。
- 是否保留上一轮 virtual tracks 的选择结果。
- 如何避免重建后 virtual observations 数量剧烈波动。

### 方向 C：深改 BAE，支持 Ceres-like Robust Loss

如果目标是让 BAE full real+virtual 更接近 Ceres augmented BA，需要在 BAE 侧实现 per-residual robust loss。

可能方案：

- 在 residual metadata 中保留 `is_virtual`。
- BAE optimizer 根据 `is_virtual` 选择不同 robust kernel。
- real residual 使用较温和 robust loss。
- virtual residual 使用更强 robust loss。
- residual 和 Jacobian 同步 reweight。

这个方向工程量更大，但语义上最接近 Ceres。

需要注意的是，简单参考 pycolmap / Ceres 的 Python 接口可能不够，因为 BAE 的求解流程和 Ceres problem block 机制不同。真正关键的是把 robust loss 融入 BAE 自己的 residual/Jacobian/linearization 流程。

### 方向 D：恢复 Intrinsics Optimization，但加约束

当前 `--bae_optimize_intrinsics` 已经支持 `SIMPLE_PINHOLE` 的 `f` 参数化，且固定 `cx/cy`。但由于没有 gauge fix / focal prior，目前结果不稳定。

后续如果要继续优化内参，建议先增加：

- focal prior。
- focal bounds。
- pose gauge fix。
- 只在最后少数 BA iteration 优化 intrinsics。
- 打印每轮 `f` drift 并设异常阈值。

在这些约束完成前，`--bae_optimize_intrinsics` 应只作为 debug 功能，不作为默认实验配置。

## 当前建议实验配置

短期推荐配置：

```text
--ba_backend bae
--bae_real_only
--bae_max_num_iterations 20
```

短期不推荐配置：

```text
--ba_backend bae
--bae_optimize_intrinsics
```

除非当前实验目标就是观测 focal drift。

## 当前结论

从 Ceres solver 切换到 BAE 后，当前主要问题不是 BAE 速度或 basic BA 能力，而是 GlueMap augmented BA 中 virtual observations 的优化语义没有被完整迁移。

第一版 BAE backend 可以作为 real-only BA 加速路径继续验证，但不能直接宣称已经等价替代 Ceres augmented BA。

后续最关键的设计点是：

```text
1. BAE real-only 后，virtual reconstruction 是否每轮重建 / 重初始化。
2. 是否深改 BAE 支持 real / virtual 不同 robust loss。
3. 是否为 BAE intrinsics optimization 增加 gauge / prior / bounds。
```
