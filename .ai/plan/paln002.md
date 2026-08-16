# paln002: 方案演变与阶段结论汇总

## 1. 文档目的

本文档用于归纳到目前为止讨论过的方案演变、已经排除的路线、已经确认的结论，以及当前建议继续推进的主线。

重点不是给出最终实现，而是明确：

```text
我们为什么从某条路线转向另一条路线
哪些结论已经足够明确，可以当成前提
哪些部分仍然只是候选方案，需要实验验证
```

---

## 2. 最初方案：LingBot-Map + GGPT/BAE + LingBot-Depth

最早设想的链路是：

```text
1. LingBot-Map 长序列前馈推理
   -> pose / depth / dense world points

2. 参考 GGPT run_sfm
   -> 建立 2D-3D 对应 / sparse tracks

3. 用 BAE 替换 GGPT 里的 BA
   -> 优化相机 pose + sparse 3D points

4. 用优化后的 3D / pose 投影 dense depth

5. 把 projected depth 输入 LingBot-Depth refine
```

### 当时得到的关键结论

1. **不能把全量 dense points 直接送进 BAE。**

原因：

- 变量规模过大。
- Jacobian/Hessian 即使稀疏也不现实。
- dense 点里异常点太多。

结论：

```text
BAE 只适合 sparse BA
dense geometry 必须通过“稀疏 BA 驱动的 dense correction/fusion”来处理
```

2. **GGPT 的 BA 本质是 sparse BA，不是 dense BA。**

GGPT 的 dense 点并不是 BA 直接优化出来的，而是：

```text
ff dense points 作为初值
selected sparse tracks 做 BA
DLT 写回 dense-shaped semidense map
再由 Point Transformer 做 dense refinement
```

结论：

```text
如果不使用 GGPT 的 Point Transformer，
仅替换 BA 后端不会自动得到 corrected dense points
需要自己补 dense correction 层
```

3. **Pose-delta dense correction 可以作为 baseline。**

对第 `i` 帧：

```text
X'_i = C'_i * W_i * X_i
```

即：

```text
原始 world point -> 拉回原始相机坐标 -> 放到优化后 pose 下
```

结论：

- 它会改变 dense points 的世界坐标位置。
- 但不会改变这些点相对于来源帧自身的 camera-space depth。
- 它的价值来自跨帧融合和重新投影，不来自“自己投回自己”。

4. **LingBot-Depth refine 应该放在 pose-delta / corrected dense projection 之后。**

推荐顺序：

```text
BAE
-> pose-delta corrected dense cloud
-> z-buffer projected depth
-> LingBot-Depth refine
-> 用 optimized pose 反投影 refined depth
```

而不是先 refine 再做 pose-delta。

---

## 3. LoMa / HLoc / sparse SfM 路线的引入

在继续推进时，引入了 LoMa 和 HLoc 方向：

```text
LingBot-Map pose/K 作为 reference model
-> HLoc + LoMa sparse matching / triangulation
-> COLMAP sparse model
-> BAE sparse BA
-> pose-delta 矫正 LingBot dense points
```

### 当时得到的结论

1. **LoMa 是 2D-2D matcher，不是 dense matcher。**

它不能像 RoMa 那样直接给每像素 dense flow。

但它可以通过 lift 成：

```text
2D-2D -> 3D-3D
2D-2D -> 2D-3D
```

用于：

- Sim(3) RANSAC
- PnP
- sparse BA tracks

2. **传统 sparse SfM 和 LingBot dense geometry 可以完全解耦。**

也就是说：

```text
LoMa/HLoc/BAE 负责 sparse pose correction
LingBot dense points 只在后端做 dense correction / projection
```

这是合理的。

3. **如果 sparse SfM 不用 LingBot pose 作为 reference model，就需要额外 Sim(3) 对齐。**

如果用完整传统 SfM 重建，pose 会处于任意 gauge 下。此时不能直接做：

```text
Delta_i = C'_i * inverse(C_i_lingbot)
```

必须先把 SfM/BAE pose 对齐到 LingBot world frame。

---

## 4. 放弃 LingBot-Map 作为基础模型

后续实测后，决定放弃将 LingBot-Map 作为主基础模型，原因包括：

1. **基础模型质量不够好。**
2. **提供的权重实际不返回有效 `world_points`。**
3. **展示的 dense point 主要来自 pose + depth 投影。**
4. **depth 可能存在跨帧尺度不一致，点云出现明显分层。**

### 因此得到的结论

```text
LingBot-Map 不再作为主 dense geometry 基础模型
```

后续只保留一个开放问题：

```text
是否把 LingBot-Map pose 当成外部粗先验 / pair proposal scaffold
```

但这件事后来也被降级为“可选 ablation”，不是主线。

---

## 5. 主线转向 Pi3 / Pi3X

放弃 LingBot-Map 后，主线切换为 Pi3 / Pi3X，原因：

1. Pi3/Pi3X 已知输出稳定的：

```text
camera_poses
local_points
points
conf
```

2. Pi3X 还支持 multimodal conditioning：

```text
pose
depth
intrinsics
ray
```

3. Pi3/Pi3X 更适合做 dense local geometry 基础模型。

### 与 Pi3 相关的关键结论

1. **长序列问题不应该强行让 Pi3 一次吃完。**

应拆成：

```text
dense local geometry
global pose / loop consistency
dense fusion
```

2. **如果后端 pose 被优化，最好用 Pi3 `local_points` 重新放置 dense points。**

比起总是用原始 `points` 做 pose-delta，更干净的做法是：

```text
X_world_i = C_opt_i * X_cam_i
```

其中 `X_cam_i = local_points_i`。

这意味着：

```text
Pi3 local_points 是更基础的 dense geometry 表达
```

---

## 6. 引入 VGGT-SLAM / VGGT-Long / AMB3R 的比较

### 6.1 VGGT-SLAM

VGGT-SLAM 提供的是：

```text
submap
keyframe selection
overlap
loop retrieval
pose graph optimization
dense map accumulation
```

优点：

- 更像完整系统。
- 实测 dense point cloud 质量优于 VGGT-Long。

问题：

- 当前实现对 VGGT 有明显依赖。
- 尤其体现在 loop verification 和 loop closure submap 构造。

### 6.2 VGGT-Long

VGGT-Long 提供的是：

```text
Chunk it
Loop it
Align it
```

优点：

- 更 model-agnostic。
- 已经支持 adapter 风格接入 Pi3。
- chunk + overlap + loop Sim(3) 结构清晰。

问题：

- 更偏 chunk-level Sim(3) 对齐。
- 缺少 VGGT-SLAM 那种 submap graph 的细粒度一致性。
- 实测点云更容易没对齐。

### 6.3 AMB3R-SfM

AMB3R 更像：

```text
image clustering
keyframe memory
以 anchor 为条件进行多轮全局精化
```

它最值得借鉴的是：

```text
先有一个全局 scaffold
再让前馈模型在 anchor 条件下重推局部 chunk
```

但它不适合作为第一版直接复现。

### 比较后的结论

1. **VGGT-Long 的思想适合作为 chunk / loop / Sim(3) 的参考。**
2. **VGGT-SLAM 更适合作为当前主系统骨架。**
3. **AMB3R 的 memory/anchor 思想更适合作为后期增强。**

---

## 7. 关于回环：已经明确的认识

### 7.1 为什么需要回环

长序列仅依赖相邻帧/相邻 submap 约束时，会有累计漂移：

```text
0 -- 1 -- 2 -- ... -- 200
```

回环提供的是：

```text
frame / submap 200 与 frame / submap 10 的长程几何约束
```

帮助图优化把漂走的轨迹拉回。

### 7.2 回环不是“pose 相等”

回环约束不应该写成：

```text
pose_i == pose_j
```

而应该写成：

```text
relative_transform(i, j) == measured_loop_transform
```

也就是：

```text
SE(3) / Sim(3) / SL(4) relative edge
```

### 7.3 回环检测和回环约束估计是两回事

已经明确：

```text
回环检测:
  只是提出 candidate

回环约束估计:
  才决定是否加入 graph edge，
  以及 edge 的测量值是多少
```

### 7.4 LoMa 在回环中的角色

LoMa 不能直接做 3D-3D matching，但可以：

```text
2D-2D matches
  -> lift with dense/local points
  -> 3D-3D Sim(3)
  -> 或 2D-3D PnP
```

因此 LoMa 很适合替换 VGGT-SLAM 中 VGGT-specific 的回环验证部分。

---

## 8. 当前主线：基于 VGGT-SLAM，先接入 Pi3/Pi3X

在比较后，当前决定的主线是：

```text
基于 VGGT-SLAM 的 submap + graph 思路
先把基础模型从 VGGT 适配为 Pi3/Pi3X
先跑通普通 submap 建图
回环先禁用或最小化
然后再逐步替换回环逻辑
最后再考虑 BAE
```

### 已确认的 VGGT 强依赖点

当前 VGGT-SLAM 对 VGGT 的主要依赖包括：

1. **普通 submap 推理依赖 `pose_enc`。**
2. **回环验证依赖 `image_match_ratio`。**
3. **回环 submap 构造依赖 VGGT pair inference。**

### 我们之前讨论过的替代方案

```text
LoopCandidateDetector:
  SALAD / DBoW2 / NetVLAD / DINO retrieval

LoopConstraintEstimator:
  LoMa 2D-2D matches
  -> 几何验证
  -> 2D-3D PnP 或 3D-3D Sim(3)
  -> loop edge
```

当时的共识是：

```text
对于 Pi3/Pi3X，更倾向于“不再继续 VGGT-style loop closure submap”
而是直接估计 loop edge
```

---

## 9. 关于 LingBot-Map 作为前置 pose prior 的最终态度

后续我们又单独讨论了：

```text
是否让 LingBot-Map 先跑全序列，输出粗 pose，
再把它作为 Pi3/VGGT-SLAM 的先验
```

当时得出的结论是：

### 它理论上能做什么

1. loop candidate proposal  
2. chunk global initialization  
3. weak pose prior  
4. Pi3X pose/intrinsics conditioning  

### 但它不是必要组件

如果主系统已经有：

```text
retrieval + LoMa + overlap + graph optimization
```

那么 LingBot-Map 的主要收益只剩：

```text
pair/loop 候选更快
chunk 初始化更好
可能的弱先验
```

而代价是：

```text
额外一次完整长序列推理
更多系统复杂度
先验错误可能污染后端
```

因此最终判断为：

```text
LingBot-Map 作为前置 pose prior 不是主线
最多作为后续 ablation / optional module
```

---

## 10. 当前已经明确的主结论

以下结论可以视为当前阶段的“已定前提”：

### A. 主 dense 基础模型

```text
不再使用 LingBot-Map
主 dense geometry 基础模型切换到 Pi3/Pi3X
```

### B. 长序列系统骨架

```text
优先沿用 VGGT-SLAM 的 submap + graph 思路
不是先复刻 VGGT-Long，也不是先复刻 AMB3R
```

### C. 第一阶段工作重点

```text
先把 Pi3/Pi3X 接入 VGGT-SLAM 的普通 submap 流程
暂时不处理回环
暂时不处理 BAE
```

### D. 回环替换思路

```text
VGGT-specific 的回环检测/验证不是必须
后续可替换为 retrieval + LoMa + Sim(3)/PnP
```

### E. BAE 的定位

```text
BAE 是后续增强项
用于 sparse pose / sparse points refinement
不用于 dense points optimization
```

### F. Dense 点的最终表达

```text
如果 pose 后端被优化，
优先使用 optimized poses + Pi3 local_points 重建 dense world points
```

### G. Loop closure 的理解

```text
回环约束是 relative transform edge
不是“两个 pose 应该相等”
```

### H. LingBot 前置 pose prior 的定位

```text
不是主线
如有必要，只作为 optional prior / proposal source
```

---

## 11. 当前仍未定论、需要实验的问题

以下问题还没有结论，只能实验判断：

1. **Pi3/Pi3X 接入 VGGT-SLAM 后，SL(4) graph 是否仍稳定。**
2. **`Solver.add_points()` 是更适合使用 depth 反投影还是直接使用 Pi3 `points`。**
3. **Pi3X 的 pose/depth/intrinsics conditioning 在长序列 submap 场景下是否有效。**
4. **LoMa + dense/local points 的 3D-3D Sim(3) 回环边是否比 VGGT-style loop submap 更稳。**
5. **后续是否需要把 VGGT-SLAM 的 SL(4) backend 换成 Sim(3) backend。**
6. **LingBot-Map pose prior 是否在 pair selection / chunk initialization 上有实测收益。**

---

## 12. 当前推荐推进顺序

### Step 1

实现：

```text
Pi3/Pi3X -> VGGT-SLAM adapter
```

目标：

```text
跑通无 loop 的 submap 建图
```

### Step 2

验证：

```text
单 submap
多 submap
导出 selected-frame poses 和 dense point cloud
```

### Step 3

开始替换回环：

```text
retrieval candidate
-> LoMa verification
-> Sim(3)/PnP loop edge
```

### Step 4

如果普通 graph 稳定，再引入：

```text
BAE sparse refinement
```

### Step 5

最后才考虑：

```text
Pi3X conditioning
AMB3R-like anchor memory
LingBot prior ablation
```

---

## 13. 一句话总结

到目前为止，路线已经从：

```text
LingBot-Map + GGPT/BAE + LingBot-Depth
```

演变为：

```text
Pi3/Pi3X 作为 dense local geometry 基础模型
+ VGGT-SLAM 作为长序列 submap + graph 骨架
+ LoMa 作为后续回环/几何验证模块
+ BAE 作为后续 sparse refinement 模块
```

当前阶段最明确的结论是：

```text
先把 Pi3/Pi3X 塞进 VGGT-SLAM 普通流程跑通，
其他增强项全部后置。
```
