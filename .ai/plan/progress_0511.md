# progress_0511：项目阶段进展总结

更新时间：2026-05-11

## 1. 总体目标

本项目一直围绕一个核心目标推进：

```text
在前馈式 3D 重建模型的基础上，构建可处理长序列的高质量 pose 与 dense 点云系统。
```

关注点主要包括：

- 如何把单次或短窗口模型扩展到几百帧级别。
- 如何获得更准确的全局相机位姿。
- 如何导出更稳定、更少重影的 dense 点云。
- 如何在质量、速度、显存和工程复杂度之间取得平衡。

目前已经基本明确：

```text
dense 几何基础模型更适合使用 Pi3 / Pi3X；
后端优化应主要作用在 pose / sparse tracks / submap alignment 上；
dense 点云应尽量由 optimized poses + local_points 重新融合得到。
```

## 2. 早期路线：LingBot-Map + GGPT / BAE / LingBot-Depth

最初设想是：

```text
LingBot-Map 输出 pose / depth / dense points
-> 参考 GGPT 构建 sparse tracks
-> 用 BAE 做 sparse BA
-> 用优化后的 pose 修正 dense
-> 再接 LingBot-Depth refine
```

这一阶段得到的主要结论：

- BAE 不适合直接优化全量 dense points，只适合 sparse BA。
- GGPT 的 BA 本质也是 sparse BA，dense 结果不是由 BA 直接优化出来的。
- dense correction 更合理的方式是 pose-delta 或重新放置 local points。
- LingBot-Map 的基础 dense 质量和可用输出不满足主线需求。

因此，LingBot-Map 后来被降级为可选 prior / candidate proposal，不再作为主 dense geometry 基础模型。

## 3. 主线转向 Pi3 / Pi3X

放弃 LingBot-Map 后，主线切换到 Pi3 / Pi3X。

原因是 Pi3 / Pi3X 能直接提供更适合后端融合的几何表达：

- `camera_poses`
- `local_points`
- `points`
- `conf`

其中最关键的判断是：

```text
local_points 是更基础、更干净的 dense geometry 表达。
如果后端优化了 pose，优先使用：
X_world_i = C_opt_i * X_cam_i
而不是长期依赖原始 world_points 或 depth 反投影。
```

这个结论后续一直保留，并成为当前 MERG3R + Pi3X 方向中导出 dense 点云的核心依据。

## 4. 第一轮系统骨架：VGGT-SLAM + Pi3X

随后尝试把 Pi3X 接入 VGGT-SLAM，目标是复用其：

- submap 构建
- overlap 对齐
- pose graph 优化
- dense 点云导出
- loop closure 框架

已完成的主要工作包括：

- 新增 `Pi3Solver`，隔离 Pi3X 与 VGGT 原始路径。
- 支持 `--base_model vggt|pi3x`。
- 直接消费 Pi3X 的 `local_points / points / conf / camera_poses`。
- 将跨 submap 对齐从 scale-only 升级为 Sim3。
- 支持 multi-overlap 联合估计。
- 增加 depth edge / confidence 过滤。
- 增加 per-submap local/world PLY 导出，用于定位误差来源。
- 接入基于 LingBot pose 的 loop candidate 和 Pi3 小窗口 loop edge。

这一阶段的主要实验结论：

- 不做跨 submap 对齐时，dense 点云完全不可用。
- 单个 submap 内部通常相对正常，主要问题来自跨 submap 全局一致性。
- 只增加 overlap 数量收益有限。
- overlap-only + 当前 pose graph 的质量上限可能已经接近。
- loop edge 已能影响结果，但稳定性不足，有时改善，有时恶化。

因此，当时的判断是：

```text
基础链路已经跑通；
问题从“能不能跑”转向“全局一致性约束质量够不够”。
```

## 5. 离线 KeyFrame / Shared-Anchor 尝试

为了缓解 VGGT-SLAM + Pi3X 的跨 submap dense 重影，之后引入离线 KeyFrame / shared-anchor 方案。

核心思路是：

```text
先全局选择 keyframes；
再构建带 shared anchors 的 local batches；
用 shared keyframes 替代简单 prefix/suffix overlap。
```

已完成内容包括：

- 新增离线 planner，读取 LingBot pose 与 HLoc/COLMAP sparse 结果。
- 基于 pose coverage 和 observation quality 选择 keyframes。
- 生成 `keyframes.json` 与 `local_batches.json`。
- submap 支持 global frame ids、anchor ids、target ids。
- Pi3Solver 支持 shared-anchor Sim3 对齐。
- 增加 anchor-only / keyframe-only 点云导出。

该阶段的重要结论：

- 固定 scale 为 1 没有明显改善，说明主问题不是简单 scale drift。
- 更严格点过滤能降低 residual，但不能根治 full dense 重影。
- shared-anchor transform disagreement 不算特别大，暂时没有必要优先引入 affine/projective 形变。
- keyframe / anchor-only 点云明显好于 full dense。

因此得到更明确的判断：

```text
keyframe / shared-anchor skeleton 相对可靠；
full dense 重影主要来自 non-keyframes / 普通帧点云的大量贡献。
```

当时形成了两条候选路线：

1. 继续基于 VGGT-SLAM/Pi3Solver，强化 batch 构建、shared-anchor reject 和 non-KF dense 过滤。
2. 转向 KeyFrame-only skeleton + BAE，再在 fixed skeleton 下补 non-KF dense 点。

## 5.1 彻底放弃VGGT-SLAM框架

短暂的基于候选路线1尝试了submap间keyframes前后对齐, 在200帧数据上无提升,submap间依然存在重影。
导致完全放弃VGGT-SLAM框架的关键原因是尝试了700帧的中场景数据, 发现误差随着submap的扩展不断累积,导致结果完全崩溃, 即使在开启回环判断的情况下, 遂准备完全放弃VGGT-SLAM框架

## 5.2 LoGeR 短暂尝试

作为自然支持长上下文重建的模型,在700帧数据测试下质量一般, 可以看出大体结构, 变形明显

## 6. AMB3R-SfM 调研与短暂转向

之后开始评估 AMB3R-SfM。

AMB3R 最值得借鉴的是：

```text
先构建全局 keyframe memory / scaffold；
再用 anchors 条件下的局部重推理逐步注册新帧和精化地图。
```

它的思想和前面 keyframe/shared-anchor 结论相符：

- anchor 控制全局结构。
- non-keyframes 不应直接污染全局骨架。
- 局部模型推理结果需要被全局 memory 约束和筛选。

但实测后，AMB3R-SfM 存在两个直接问题：

- 速度较慢。
- 原框架不直接支持 Pi3X，需要较多适配。

因此 AMB3R 没有成为当前主实现框架，而是保留为 anchor memory / fixed skeleton non-KF filler 的思想参考。

## 7. 最新方向调整：转向 MERG3R + Pi3X

最近重新尝试 MERG3R 后，观察到它在结果和效率上优于 AMB3R，因此当前方向再次调整：

```text
不再以 AMB3R-SfM 或 VGGT-SLAM 主框架继续推进；
改为基于 MERG3R 的核心思想和架构，接入 Pi3X，并吸收前面 VGGT-SLAM / AMB3R 中有效的局部思想。
```

MERG3R 的优势在于主流程更简单：

```text
图像排序 / 切分
-> 每个 subset 独立跑基础几何模型
-> overlap 上用 confidence-weighted Sim3 对齐
-> 可选 LightGlue tracks + global BA
-> 输出 COLMAP / 点云
```

相比 AMB3R，MERG3R 更轻量、推理次数更少、工程改造路径更清晰。

相比 VGGT-SLAM，MERG3R 的框架更接近当前需要的分治式 pipeline，不强依赖 VGGT-specific 的 pose encoding / loop submap 逻辑。

## 8. 当前 MERG3R + Pi3X 已做的工作

当前已经开始把 Pi3X 接入 MERG3R。

已完成的主要工程工作：

- 从原版 Pi3X 源码中提取 Pi3X 相关模型代码到独立目录 `MERG3R/pi3x_model`。
- 在 MERG3R 的 `load_model()` 中新增 `pi3x` 分支。
- 在主流程中增加 `--pi3x_ckpt`。
- 在 `run_inference_step_by_step()` 中增加 Pi3X 推理路径。
- 明确 Pi3X 输出的 `camera_poses` 是 OpenCV c2w。
- 明确 MERG3R 内部 `extrinsic` 使用 w2c，因此需要对 Pi3X c2w 求逆。
- 保存 Pi3X 的 `local_points`，为后续 dense fusion 做准备。
- 对 `dense_model_points.ply` 导出改为流式版本，避免 700 帧级别一次性构造完整 dense world point map 导致显存或内存压力过大。

当前已经确认 Pi3X 的几何语义：

```text
points = c2w @ local_points
```

因此理论上：

```text
compute_depth(points, w2c)
```

和：

```text
local_points[..., 2]
```

应当一致，差异主要来自数值误差、尺度处理或后续变换。

## 9. 关于 dense PLY 的当前结论

当前不再满足于导出由 depth 反投影生成的 dense 点云。

更合理的 dense 导出目标是：

```text
使用基础模型原始 local_points；
再用最终优化后的 pose / aligned pose 将每帧 local dense 点变换到全局坐标；
最后按置信度、stride、可选边缘过滤进行流式写出。
```

原因是：

- depth 反投影会丢掉原始模型 pointmap 的一部分表达。
- Pi3X 原生输出 local point map，本身已经是 dense geometry。
- 当 pose 被后端修改后，local_points + optimized poses 是最直接的重融合方式。
- 对 700 帧左右场景，必须避免一次性生成 `[N,H,W,3]` 的全局 dense 数组。

当前已经将 `dense_model_points.ply` 改为流式导出。

仍需注意：

```text
MERG3R 原有 write_recon_to_colmap / points.ply 路径仍可能构造 depth-based world_points；
如果后续 700 帧仍 OOM，需要继续把这一部分也改成流式或可选跳过。
```

## 10. 关于 MERG3R BA 与 BAE 的判断

已对比 MERG3R 的 BA 和 BAE。

二者在问题形式上相似：

```text
sparse 2D observations
-> constrain camera poses and sparse 3D points
```

但实现侧重点不同：

- MERG3R BA 使用 PyTorch / Adam / padding 后的 per-image tensor，工程简单，但大规模帧数下内存压力较大。
- BAE 更接近传统 sparse BA，适合 observation-flat 格式、稀疏 Jacobian、LM / Trust Region / PCG 等大规模优化。

当前判断：

```text
短期保留 MERG3R 原 BA 作为 baseline；
中期可以增加 --ba_backend merg3r|bae；
将 MERG3R tracking 结果转换成 BAE 所需 sparse observation 格式。
```

BAE 仍然定位为 sparse pose / sparse points refinement，不直接优化 dense points。

## 11. 当前已经形成的总体结论

截至目前，几次方向调整可以概括为：

```text
LingBot-Map 主模型
-> 放弃，质量和输出不满足主线

Pi3 / Pi3X dense local geometry
-> 保留，成为基础模型方向

VGGT-SLAM + Pi3X
-> 跑通，但 cross-submap / full dense 一致性达到瓶颈

Offline KeyFrame / Shared-Anchor
-> 证明 keyframe skeleton 比 full dense 更稳定

AMB3R-SfM
-> 思想有价值，但速度慢且 Pi3X 接入成本高

MERG3R + Pi3X
-> 当前主线，兼顾效率、架构简单度和可扩展性
```

当前最重要的技术判断是：

```text
1. Pi3X local_points 是 dense 点云融合的核心输入。
2. dense 点云不应直接参与大规模 BA。
3. sparse BA / Sim3 alignment 负责优化 pose 与全局一致性。
4. optimized/aligned pose + local_points 负责最终 dense fusion。
5. non-keyframes / 普通帧点云是 full dense 重影的重要来源，需要过滤、降权或固定骨架下补点。
```

## 12. 当前正在做的事

当前阶段的主线任务是：

```text
基于 MERG3R 框架接入 Pi3X，
先跑通高效的 subset inference + overlap alignment + dense local_points streaming export。
```

具体正在推进：

- 完善 Pi3X 在 MERG3R 中的模型加载和推理路径。
- 统一 Pi3X `c2w` 与 MERG3R `w2c` 的 pose 约定。
- 确认 `points / local_points / depth` 的一致性。
- 将 dense_model_points 导出改为基于 `local_points + final extrinsic` 的流式版本。
- 评估 MERG3R 原 BA 是否足够，以及后续是否接入 BAE 作为 sparse BA 后端。

## 13. 下一步建议

短期优先级：

1. 先完成 MERG3R + Pi3X baseline 跑通。
2. 用同一组数据对比 VGGT / Pi3X 在 MERG3R 框架下的 pose 和 dense PLY。
3. 检查 `dense_model_points.ply` 与原 depth-based `points.ply` 的差异。
4. 如果 700 帧仍有内存问题，继续流式化或跳过 MERG3R 原 COLMAP dense point export。
5. 保留 MERG3R BA baseline，再设计 BAE backend adapter。
6. 后续再考虑引入 keyframe skeleton / anchor memory / non-KF filler 思路。

当前最合理的方向是：

```text
以 MERG3R 为主框架；
以 Pi3X 为 dense local geometry 模型；
以 Sim3 alignment + sparse BA 解决全局一致性；
以 streaming local_points fusion 输出最终 dense 点云。
```
