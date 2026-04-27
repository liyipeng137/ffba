# VGGT-SLAM 接入 Pi3 当前进度总结

更新时间：2026-04-28

## 1. 本阶段目标

本阶段的主要目标是先将 `VGGT-SLAM` 的前端推理从 `VGGT` 切换到 `Pi3X`，在不引入回环检测的前提下，尽量跑通：

- submap 构建
- submap 间拼接
- 全局图优化
- dense 点云导出

当前策略是：

- 使用 `Pi3X` 替代 `VGGT` 做几何推理
- 使用手工提供的共享先验内参
- 不要求 `Pi3` 返回 `depth`，而是直接消费 `local_points / points / conf / camera_poses`
- 先跑通 overlap 路径，再逐步接入基于 LingBot 先验 pose 的回环约束


## 2. 已完成的代码改造

### 2.1 新增 Pi3 专用 Solver

已新增独立文件：

- [VGGT-SLAM/vggt_slam/pi3_solver.py](/Users/lyp/CodeProject/SelfProject/FeedForwardWithBA/VGGT-SLAM/vggt_slam/pi3_solver.py)

目的：

- 不破坏原始 `solver.py`
- 将 `Pi3` 路径与 `VGGT` 路径隔离
- 单独处理 `Pi3` 的输入输出语义、overlap 对齐和调试日志

当前 `Pi3Solver` 的核心设计：

- 输入图片不做 resize / crop，直接读取原图
- 要求 submap 内所有图像尺寸一致，且高宽均为 14 的倍数
- 使用全局共享先验内参
- 使用 `Pi3X` 输出的：
  - `local_points`
  - `points`
  - `conf`
  - `camera_poses`
- 当前 submap 中保存的是 **camera-local 点图**（`local_points`）
- 图中的节点语义是：
  - `frame local camera coordinates -> global/map coordinates`


### 2.2 main.py 接入 Pi3 路径

已改造：

- [VGGT-SLAM/main.py](/Users/lyp/CodeProject/SelfProject/FeedForwardWithBA/VGGT-SLAM/main.py)

新增能力：

- `--base_model vggt|pi3x`
- `--pi3_ckpt`
- `--fx --fy --cx --cy`
- `--lingbot_transforms_json`
- `--loop_window_radius`
- `--min_loop_frame_gap`
- `--loop_translation_thresh`
- `--loop_rotation_thresh_deg`
- 支持导出单独 submap 的局部/全局点云


### 2.3 Pi3 输入输出适配

已实现：

- 原图加载逻辑，不走 `VGGT` 的预处理路径
- 手工构造 per-frame intrinsics
- 直接调用 `Pi3X` 模型
- 从 `Pi3X` 结果中整理出：
  - `images`
  - `local_points`
  - `world_points_pi3`
  - `point_conf`
  - `camera_poses`
  - `intrinsic`

说明：

- 当前并未强行伪造 `depth/depth_conf`
- `Pi3Solver` 直接消费 `local_points`，避免先压成 depth 再重投影


## 3. submap 间 overlap 处理的改造

### 3.1 初版：scale-only 对齐

最初版本中，`Pi3Solver` 在跨 submap 时只对 overlap 做 `scale` 估计，模仿原版 `solver.py` 的基本结构。

问题：

- 仅靠 scale 无法稳定消除 Pi3 输出在不同 submap 推理上下文下产生的几何差异
- 导出的 dense 点云出现明显重影和分层


### 3.2 引入 overlap 帧 Sim3 估计

后续已将 overlap 对齐升级为：

- 对 overlap 对应点做 `Sim3` 拟合
- 拟合形式为 `target ~= s * R * source + t`

已实现的辅助函数包括：

- `estimate_similarity_transform()`
- `transform_points()`
- `rotation_angle_degrees()`

当前逻辑不是只把该变换作用到 overlap 帧本身，而是：

- 先估计 `current_submap -> prior_submap` 的整体桥接变换
- 再把该变换通过当前 submap 的 graph anchor 和内部 `H_inner` 传播到整个当前 submap


### 3.3 支持 multi-overlap 帧联合估计

已支持 `overlapping_window_size > 1`。

不是简单逐帧独立对齐，而是：

1. 先找出当前 submap 前缀与上一 submap 后缀的真实重叠帧
2. 将每个 overlap 帧的 `local_points` 用各自 `camera_poses` 变到各自 submap 坐标系
3. 聚合全部 overlap 帧的 3D-3D 对应
4. 联合估计一个共享 `Sim3`

这样做的目的是让跨 submap 桥接变换不只受单帧局部误差支配。


### 3.4 overlap 检测逻辑修复

已修复 overlap 识别逻辑：

- 优先按 `img_names` 的 basename 识别重叠帧
- 回退到 `frame_ids` 识别
- 匹配时从最大 overlap 数向下搜索，避免只从 `count=1` 开始导致识别失败

当前已能正确识别例如：

- 当前 submap 前 3 帧
- 上一 submap 后 3 帧

这种 `3-frame overlap`


## 4. Pi3 点置信与边缘过滤

已新增 Pi3 原生的深度边缘过滤：

- 使用 `depth_edge(local_points[..., 2], rtol=0.03)`

过滤策略：

- 对 edge 区域对应的 `conf` 直接置零
- 该过滤同时用于：
  - overlap 对齐点筛选
  - dense 点云导出时的有效点保留

目的：

- 减少深度边缘、遮挡边界、薄结构等区域对 Sim3 和 dense 导出的破坏


## 5. 调试与可分析能力增强

### 5.1 增加大量运行日志

在 `Pi3Solver` 中已加入多种调试信息输出，包括：

- `Pi3 confidence filtering`
- `Pi3 add_points summary`
- `Pi3 overlap detection by image names`
- `Pi3 overlap detection fallback by frame ids`
- `Pi3 overlap frame stats`
- `Pi3 overlap pose diagnostics`
- `Pi3 overlap Sim3`
- `Pi3 overlap anchor`
- `Pi3 extra overlap constraints`

这些日志现在可用于分析：

- overlap 是否识别正确
- 有效点数量是否足够
- Pi3 两次推理对同一 overlap 图的 pose 是否一致
- Sim3 的 scale / rotation / translation / residual 是否合理


### 5.2 支持导出每个 submap 的单独点云

已新增：

- [VGGT-SLAM/vggt_slam/submap.py](/Users/lyp/CodeProject/SelfProject/FeedForwardWithBA/VGGT-SLAM/vggt_slam/submap.py)
- [VGGT-SLAM/vggt_slam/map.py](/Users/lyp/CodeProject/SelfProject/FeedForwardWithBA/VGGT-SLAM/vggt_slam/map.py)
- [VGGT-SLAM/main.py](/Users/lyp/CodeProject/SelfProject/FeedForwardWithBA/VGGT-SLAM/main.py)

支持参数：

- `--export_submap_local_dir`
- `--export_submap_world_dir`

作用：

- 导出每个 submap 在 **local** 坐标系下的点云
- 导出每个 submap 在 **graph/world** 坐标系下的点云

这一步已经帮助定位问题来源。


## 6. 当前实验结论

### 6.1 不做跨 submap 对齐时，结果完全不可用

已验证：

- 如果禁用跨 submap 对齐修正，dense 点云会明显错乱

结论：

- 跨 submap 对齐是必须的


### 6.2 overlap_count=1 与 overlap_count=3 的体感差别不大

目前用户主观观察：

- `overlapping_window_size=1`
- `overlapping_window_size=3`

二者导出的 dense 点云效果差异不明显。

初步结论：

- 简单增加 overlap 帧数，并不能根本解决当前的 cross-submap 重影问题


### 6.3 单个 submap 内部相对正常，重影主要来自跨 submap 对齐

通过导出 `submap local/world point cloud` 后，已基本确认：

- 单个 submap 的 local 点云内部不是主要矛盾
- 重影和分层主要体现在多个 submap 拼接后的 world 结果上

结论：

- 当前瓶颈已经从“单个 submap 推理是否可用”
- 转移到“submap 间连接与全局一致性优化”


### 6.4 overlap-only 对齐的质量上限可能已经接近

目前观察表明：

- 仅靠 overlap 帧 pairwise/multi-frame 对齐
- 即使加入 Sim3、边缘过滤、多 overlap 联合估计

仍然无法彻底消除 dense 重影。

推断：

- 只靠 overlap 的局部桥接约束，上限可能有限
- 后续需要引入更强的全局一致性机制


## 7. 回环检测与回环约束接入进展

### 7.1 已接入基于 LingBot 先验位姿的回环候选检测

当前 `Pi3Solver` 已新增：

- 读取 `LingBot transforms.json`
- 将其中的 `OpenGL c2w` 转回 `OpenCV c2w`
- 基于先验位姿做 loop candidate 粗筛

当前候选筛选使用的约束包括：

- `min_loop_frame_gap`
- `loop_translation_thresh`
- `loop_rotation_thresh_deg`

默认策略：

- 至少隔 `2 * submap_size` 帧
- 相机中心距离小于阈值
- 旋转差小于阈值
- 每个当前 submap 最多只接收 1 个 loop candidate

当前这部分相关改动位于：

- [VGGT-SLAM/vggt_slam/pi3_solver.py](/Users/lyp/CodeProject/SelfProject/FeedForwardWithBA/VGGT-SLAM/vggt_slam/pi3_solver.py)
- [VGGT-SLAM/main.py](/Users/lyp/CodeProject/SelfProject/FeedForwardWithBA/VGGT-SLAM/main.py)


### 7.2 已支持 loop 小窗口复推理

当前回环不是直接把 LingBot 先验 pose 当最终 loop edge，而是采用以下流程：

1. 用 LingBot 先验 pose 找到粗候选回环帧对
2. 以候选中心帧为中心，左右各扩 2 帧
3. 历史端窗口和当前端窗口拼接为一个 5+5 以内的小序列
4. 对这个小窗口单独再跑一次 `Pi3`
5. 从这次统一上下文推理中取中心帧之间的相对位姿
6. 将该相对位姿作为 loop edge 加入图优化

当前窗口半径参数：

- `--loop_window_radius`

默认值：

- `2`

即最多取 5 帧局部窗口。


### 7.3 loop edge 的图优化接入方式

当前 Pi3 路径的 loop 约束实现方式与原版不同：

- 原版：新建一个 2-frame loop closure submap 作为桥接子图
- 当前 Pi3 版：直接对 `historical node <-> current node` 添加一条 `between factor`

当前已新增：

- 单独的 `loop_noise`

位置：

- [VGGT-SLAM/vggt_slam/graph.py](/Users/lyp/CodeProject/SelfProject/FeedForwardWithBA/VGGT-SLAM/vggt_slam/graph.py)

当前做法是：

- 候选回环通过 Pi3 小窗口重推理得到 `relative_h`
- 直接：
  - `graph.add_between_factor(hist_node, curr_node, relative_h, loop_noise)`
- 然后沿用主流程里的 `solver.graph.optimize()`


### 7.4 已补充 loop 调试日志

当前回环相关已新增日志，包括：

- `Loaded LingBot prior poses`
- `Pi3 loop candidate`
- `Pi3 loop verification`
- `Pi3 loop edge added`

用于分析：

- 是否成功读取 LingBot 先验 pose
- 粗筛候选是否合理
- Pi3 小窗口重推理与 LingBot 先验是否一致
- 实际哪两个 graph node 被回环边连接


### 7.5 已修正 LingBot 导出图片命名

之前 `LingBot transforms.json` 默认使用重编号图片名：

- `000000.png`
- `000001.png`

这与 `VGGT-SLAM` 侧原始图像文件名不一致，会导致 basename 匹配失败。

目前已改造：

- [lingbot-map/demo.py](/Users/lyp/CodeProject/SelfProject/FeedForwardWithBA/lingbot-map/demo.py)

新逻辑：

- 导出的预处理图片沿用原始输入图片 basename
- `transforms.json` 里的 `file_path` 也同步保留原图文件名
- 增加重复 basename 检查，避免导出时静默覆盖

这样可以保证：

- `LingBot transforms.json`
- `VGGT-SLAM` 输入图像

在 basename 层面一一对应。


## 8. 最新实验结论

### 8.1 LingBot 先验 pose 已能筛出较准确的回环帧

当前实验表明：

- 通过设置合适的
  - `loop_translation_thresh`
  - `loop_rotation_thresh_deg`
- 可以筛出比较准确的回环候选

用户观察结果：

- 当前筛出来的候选帧已经接近“几乎重叠”的真实回环帧

说明：

- LingBot 先验 pose 作为 loop candidate 粗筛是有效的


### 8.2 Pi3 小窗口复推理已经能产出可用的 loop edge

当前一旦找到回环候选，会：

- 构建历史窗口和当前窗口
- 单独跑一次 Pi3
- 从中心帧之间取相对位姿
- 作为 loop edge 加入图

从日志看：

- `query_frame / detected_frame` 一致性较高
- 中心帧 `conf` 较高
- Pi3 估计的相对位姿和 LingBot 先验量级大体一致，但仍存在一定偏差

当前判断：

- 回环链路已经打通
- 但 loop edge 质量还不够稳定，暂时不能视为强约束


### 8.3 小室内场景下，加入回环后效果有改善也有恶化

当前在偏小的室内数据上，加入回环后的现象是：

- 有些区域的 dense 点云变好
- 有些区域的错位反而变严重

这说明：

- loop constraint 已经开始对全局图施加影响
- 但不同约束之间的折中结果还不稳定

这类现象通常意味着：

- loop candidate 虽然准
- 但 loop edge 的几何质量还不够硬
- 或 loop noise / reject 策略还不够保守


### 8.4 目前尚不能直接下结论“只有 BA 能解决”

当前更合理的判断是：

- 单靠 overlap + 当前这版 pose graph + 当前 loop edge
- 还不能稳定解决所有 dense 错位

但这并不直接等价于：

- “只能靠 BA”

更可能的状态是：

1. overlap-only 的上限已经接近
2. 回环开始起作用，但约束质量还不够稳定
3. 需要更保守的 loop reject / weighting
4. 若这些都做完仍不够，再进一步考虑 BA 或更强的 refinement


## 9. 当前仍未完成的部分

### 9.1 loop verify 还缺少 reject 逻辑

当前 `_verify_loop_candidate()` 的本质是：

- 对粗候选做一次 Pi3 小窗口重推理
- 输出诊断信息
- 直接返回 loop edge

目前还缺：

- 基于 `Pi3 vs LingBot` 一致性的 reject 条件
- 基于中心帧 `conf` 的 reject 条件
- 更严格的窗口稳定性检查


### 9.2 loop edge 权重与过滤仍需继续调

当前虽然已经接入 `loop_noise`，但仍需要继续实验：

- 是否应进一步放松 `loop_noise`
- 是否应拒绝 `Pi3` 和 `LingBot` 偏差过大的 loop
- 是否应限制单个局部区域接收的 loop 数量


### 9.3 尚未做更强的全局 refinement / BA

当前只有：

- submap 内部相对位姿因子
- 相邻 submap 的 overlap 桥接因子
- 少量直接添加的 loop between factors

尚未引入：

- 更系统的 loop-aware refinement
- submap-level global refinement
- 稀疏 anchor BA
- full BA


## 10. 对当前阶段的总体判断

本阶段已经完成了 Pi3 接入的第一轮核心工程工作：

- 跑通了 `VGGT-SLAM -> Pi3X`
- 完成了独立 solver 路径
- 实现了 overlap 的 Sim3 桥接
- 实现了 multi-overlap 联合估计
- 实现了 edge-aware 过滤
- 增强了日志与导出分析能力
- 打通了基于 LingBot 先验 pose 的回环检测链路
- 打通了基于 Pi3 小窗口复推理的 loop edge 注入链路

当前系统已经具备：

- 连续 submap 构建
- 基于图的全局优化
- dense 点云导出
- cross-submap 问题定位能力
- loop candidate 粗筛能力
- loop edge 注入与调试能力

但实验结果也表明：

- 当前主要误差来源已经收敛到 **cross-submap global consistency**
- 单纯继续微调 overlap 对齐，收益可能有限
- 小室内场景下，loop 已经起作用，但结果仍不稳定
- 系统已经进入“全局一致性约束质量”而不是“基础链路是否能跑通”的阶段


## 11. 下一阶段建议

建议下一阶段优先级如下：

1. 给 `_verify_loop_candidate()` 增加 reject 逻辑
2. 继续调 `loop_noise` 和 loop candidate 阈值
3. 对比“无 loop / 保守 loop / 当前 loop”三组结果
4. 若 loop 质量稳定后仍存在明显接缝错位，再评估 submap-level refinement / BA
5. 在更大场景数据上验证回环收益是否更明显

当前较合理的路线是：

- **LingBot 位姿做候选检测**
- **Pi3 或局部几何过程做 loop 验证/估相对约束**
- **把验证后的 loop edge 加入图优化**
