# MERG3R + GlueMap BAE 后端优化记录

更新时间：2026-06-21
关联文档：[`merg3r_gluemap_bae_backend.md`](merg3r_gluemap_bae_backend.md)、[`merg3r_gluemap.md`](merg3r_gluemap.md)

本文件记录本次会话对 `--ba_backend bae` 路径做的一系列优化与修复。所有改动默认行为对 `ceres` 后端**零影响**（除非显式标注），目的是让 BAE-SP 路径在质量、稳定性、速度上更可用。

---

## 改动总览

| # | 优化 | 类型 | 影响后端 | 文件 |
|---|------|------|---------|------|
| 1 | BAE Huber robust loss（可选） | 质量 | bae | `bae_solver.py`、`augmented_bundle_adjustment.py`、`gluemap_refine_core.py`、`gluemap_spv_refine.py`、`run_merg3r_gluemap_pipeline.py` |
| 2 | 中间轮跳过 post-BA filter | 速度 | bae | `augmented_bundle_adjustment.py`、`gluemap_refine_core.py` |
| 3 | 关闭 filter 后的 re-BA | 速度 | bae | `augmented_bundle_adjustment.py`、`gluemap_refine_core.py` |
| 4 | 跨轮 GPU 显存释放（OOM 修复） | 稳定性 | bae | `bae_solver.py` |
| 5 | `refine_stats.json` 序列化健壮性修复 | bugfix | 全部 | `algos/utils.py`、`gluemap_spv_refine.py` |
| — | depth-pose 一致性诊断脚本（独立调查） | 工具 | — | `scripts/diagnose_depth_pose_consistency.py` |

---

## 1. BAE Huber robust loss（可选）

### 动机

BAE 第一版用 plain squared (L2) loss，无 robust kernel。对照 Ceres：real track 用 Huber、virtual track 用 Arctan。BAE-SP 只喂 real track，缺少鲁棒化，导致：
- 中等 outlier 把 L2 解带偏 → pose 相对一致性变差；
- 下一轮三角化 / angular filter 产出更少点（real points 比 ceres 少）。

把鲁棒性放进 loss（而不是靠 pre-BA 硬过滤），对应 Ceres `loss_type_normal="huber"`。BAE 无 virtual，故只需 Huber 一种。

### 实现要点

在 BAE 的 LM optimizer 里，R 与 J 都来自同一个 forward 残差（[`optim/optimizer.py`](../../MERG3R/bae/bae/optim/optimizer.py) 的 `step`：`R = model(input)`、`J = jacobian(R, params)`）。因此**只要让 forward 返回加权残差，J 会被同样加权**，无需改 BAE 库。

IRLS：forward 返回 `base * sqrt(w)`，其中
```
s = ||residual||            # 每观测残差范数（像素）
w = 1                       if s <= delta
w = delta / s               if s >  delta
```
`sqrt(w)` 是 detached plain 张量（无 optrace），会被 BAE 的 map-edge backward（[`autograd/graph.py`](../../MERG3R/bae/bae/autograd/graph.py)）排除在 argnums 外、当成常数缩放。由此得到 `A = JᵀWJ`、`rhs = -JᵀWr`，即标准 Huber IRLS。`__mul__` 在 `autograd/function.py` 的 `WHITELISTED_MAPS` 内，确保乘法被 trace。

`delta` 单位是**像素**，默认 1.0，语义对齐 Ceres `Huber params=[1.0]`（两边残差都在像素空间）。

### 改动位置

- `gluemap/estimators/bae_solver.py`
  - `_make_bae_model(...)` 新增 `robust_loss`、`huber_delta`；`GluemapBaeResidual` 拆出 `_parse_input` / `_project` / `_huber_weight_sqrt` / `forward` / `robust_debug_stats`；`forward` 在 `robust_loss=="huber"` 时返回 `base * _huber_weight_sqrt(base)`。
  - `bundle_adjustment_bae(...)` 新增 `robust_loss`、`huber_delta` 参数 + 校验；逐轮/首末打印 down-weighting 与 raw/weighted MSE；`summary["robust"]` 记录 initial/final 统计。
- `gluemap/controllers/augmented_bundle_adjustment.py`：`IterativeBAOptions` 新增 `bae_robust_loss`、`bae_huber_delta`，bae 分支透传。
- `utils/gluemap_refine_core.py`：从 `args` 设进 `IterativeBAOptions`。
- `utils/gluemap_spv_refine.py`：`GluemapSpvRefineConfig` 字段 + `_make_refine_args` 透传 + 写入 `stats`。
- `run_merg3r_gluemap_pipeline.py`：CLI `--bae_robust_loss {none,huber}`、`--bae_huber_delta`。

### 已知近似（写入实验记录）

- accept/reject 用的 `self.model.loss` 在 huber 下是 **IRLS 代理代价** `Σ w·r²`（权重每步重算），非严格 Huber 代价。标准 IRLS 行为，实践可用。
- 一阶 IRLS，无 Triggs 二阶修正。

---

## 2. 中间轮跳过 post-BA filter（bae）

### 动机

外层 `num_refinement_iterations` 每轮开头 `triangulate_from_seed_reconstruction(..., clear_points=True)` 会**清空 points3D 重新三角化**，只用 pose + 数据库。因此：
- **非最终轮**内层 filter 删的 points 在下一轮三角化时被丢弃 → 纯浪费（每趟 filter 因 pycolmap 原生 filter 不可用退回纯 Python，约 ~20s/趟）。
- 只有**最终轮**的删除能进 `refined_gluemap_aba` 输出。
- 当前 `augmented_ba_max_filter_iterations=3` 下，内层 filter 从不触发 re-BA（撞 cap），故 filter 不改 pose → 跳过中间轮 filter 对下一轮**零影响**（pose、数据库都不变，三角化逐字节一致）。

> 前提：上述"零影响"仅当内层 filter 不触发 re-BA。若调大 `augmented_ba_max_filter_iterations` 使 re-BA 能触发，中间轮跳过会通过"re-BA 改 pose"变成真实（但很小）的行为改变。

### 改动位置

- `IterativeBAOptions.run_post_ba_filter: bool = True`（默认保持原行为）。
- `iterative_bundle_adjustment`：BA 后若 `not run_post_ba_filter` 则 log 并 `break`（BA 跑一次，不 filter）。
- `gluemap_refine_core.py` 每外层轮设：`ba_options.run_post_ba_filter = use_virtual_tracks or is_final_round`
  - ceres（`use_virtual_tracks=True`）恒 True，**不变**；
  - bae 仅最终轮 True。
- `iter_stats["bundle_adjustment"]["post_ba_filter_ran"]` 记录该轮是否跑了 filter。

---

## 3. 关闭 filter 后的 re-BA（bae）

### 动机

内层"删够 `convergence_threshold`(1%) → 重跑 BA"对 ceres 是 virtual 驱动的 trimming 机制；对 **bae+huber 近乎无用**：outlier 已被 huber 降权到 ≈0，删掉再 BA 几乎不动 pose。实测（3 轮、δ=1.5、gate=1.0 的最终轮）：被触发的 BA#2 `loss 2.762483 → 2.742522`（−0.7%，噪声级），`raw_mse 9.16→9.50`（反升），白花 ~12s BA + 一趟额外 filter。

正确做法不是减少 filter 档（会丢掉 1x 的最终清理），而是：**BA 一次 → 3x/2x/1x 三档 filter 全做完 → 永不 re-BA**。

### 改动位置

- `IterativeBAOptions.allow_re_ba_after_filter: bool = True`（默认保持 ceres 行为）。
- `iterative_bundle_adjustment` 内层："删够 1%"分支前置 `options.allow_re_ba_after_filter`；为 False 时永远走 else（收紧阈值继续 filter，不 break 去 re-BA），跑满 `max_filter_iterations` 后正常退出。
- `gluemap_refine_core.py` 构造 `IterativeBAOptions` 时 `allow_re_ba_after_filter=(ba_backend == "ceres")`。

### 效果

bae 最终轮变为：`BA 一次 → filter 3x → filter 2x → filter 1x → 结束`。保留完整 outlier 清理（含 1x 那趟），砍掉无用的 re-BA，质量零损失。

---

## 4. 跨轮 GPU 显存释放（OOM 修复）

### 现象

`--num_refinement_iterations 3` 时第 3 轮 BA 首步 OOM，炸在 cuSPARSE SpGEMM（构建 `J^T J`）的 external buffer 申请：
```
cusparseSpGEMMreuse_nnz(): all externalBuffer must be != NULL
CUDA error: out of memory → internal error → illegal memory access
```

### 根因

**不是问题变大**：第 1 轮（360k tracks、参数量 ~1.08M）最大却跑通，第 3 轮（218k tracks、~0.66M）更小却炸。`bundle_adjustment_bae` 每轮新建 model（cuda 参数）、input_dict（cuda 张量）、LM optimizer（`self.mm = CuSparse()` 持 workspace），跑完**直接 return，既不 del 也不 empty_cache**。torch 缓存分配器保留并碎片化，optimizer↔model 还有引用环 → 跨轮累积，到第 3 轮没有连续空间给 SpGEMM buffer。放宽 gate（obs↑）进一步压缩 headroom。

### 修复

`bae_solver.py`：`import gc`；`bundle_adjustment_bae` 返回前
```python
del optimizer, model, input_dict, camera_params, points_3d, intrinsics
gc.collect()                 # 打破 optimizer<->model 引用环
torch.cuda.empty_cache()     # 把块还给 CUDA
```
放在所有写回与统计之后，避免 NameError。每个 BA 调用自我收尾，跨轮不累积。

### 备选省显存杠杆（未实现）

- gate 别放太松（obs↓ → `J^T J`↓）。
- LM 的 `matrix_free_normal=True`：用 `NormalMatVec` 无矩阵 PCG，**绕过显式 `J^T J` 的 SpGEMM**（即 OOM 那一步）。如需可加 `--bae_matrix_free` 透传。

---

## 5. `refine_stats.json` 序列化健壮性修复

### 现象

末尾写 `refine_stats.json` 抛 `TypeError: Object of type PosixPath is not JSON serializable`。与 huber 无关：`export_prediction_depth_maps` 返回的 `output_dir` 是 `PosixPath`，进了 `stats["depth_export"]`。

### 修复

- `algos/utils.py`：`export_prediction_depth_maps` 返回 `"output_dir": str(output_dir)`。
- `utils/gluemap_spv_refine.py`：`_write_json` 加 `default=str`，作为安全网，避免任何非可序列化值（Path/np 标量）在昂贵计算后炸掉整份 stats。

> BA 输出 `refined_gluemap_aba` 在此崩溃前已写完，崩的只是 stats JSON。

---

## 附：depth-pose 一致性诊断脚本（独立调查，未接入 pipeline）

`scripts/diagnose_depth_pose_consistency.py`：针对"前馈 depth 与 BA 后 pose 不再尺度一致 → TSDF 局部错层"问题的**前置诊断**（不依赖 pycolmap，直读 COLMAP .bin/.txt）。

输入：`refined_gluemap_aba`（BA 后 pose + points3D）、`pred_depth`（低分辨率 FF depth）、可选 `coarse`（BA 前 pose）。
输出：逐帧尺度比 `r = z_ba / d_ff` 的统计 + 平面拟合 R²、错层世界离散度、Sim3(coarse→refined) 残差，给出 `global_scalar / per_frame_scalar / per_frame_smooth_field / model_based` 的建议。

> 该方向（用 sparse 锚点估计平滑尺度场来矫正 dense depth；lingbot-depth / GGPT 仅作参考）当前优先级低于 pose 优化，脚本就绪待用。

---

## 新增 / 相关 CLI 参数

```bash
# 本次新增（仅影响 bae）
--bae_robust_loss {none,huber}   # 默认 none；huber = IRLS Huber 加权
--bae_huber_delta 1.0            # Huber δ，单位像素

# 既有、本次用于调参的旋钮
--filter_reproj_error_threshold 0.5   # pre-BA angular 闸门（注意：ceres/bae 共享）
--num_refinement_iterations 2         # 外层重三角化轮数
--bae_max_num_iterations 20           # 单次 BAE LM 迭代上限
--bae_optimize_intrinsics             # 仅 SIMPLE_PINHOLE 优化 f，固定 cx/cy
--bae_fix_gauge two_cams              # gauge 固定策略
```

内部自动按 backend 设置、无 CLI：
- `run_post_ba_filter`：ceres 恒 True；bae 仅最终轮。
- `allow_re_ba_after_filter`：ceres True；bae False。

当前推荐组合（gate=1.0 已验证正向，δ=1.0 与 1.5 基本无差别，用默认 1.0）：
```bash
--ba_backend bae --bae_max_num_iterations 20 --num_refinement_iterations 3 \
--bae_optimize_intrinsics --bae_robust_loss huber --bae_huber_delta 1.0 \
--filter_reproj_error_threshold 1.0
```

---

## 关键实验结论

### Huber vs 默认 L2（同数据 390 帧，δ=1/gate=0.5，2 轮）

- inlier 重投影**明显改善**：iter2（round-1 BA 后）P-only mean `0.2235°→0.1918°`（−14%）、`<0.5°` `90.7%→92.8%`；S-only 同向改善；iter2 final real `0.217°→0.186°`。
- raw_mse 反而更高（4.61 vs 3.76）：**符合预期**——L2 直接最小化平方和（连 outlier 一起拟合），huber 不追 outlier，均值被尾巴抬高；median/inlier 才是 huber 该赢的地方。
- 最终点更少（225025 vs 232542，−3.2%）：huber 把 L2 当 inlier 硬塞进解的 outlier 识别并由后续 filter 清掉 → "少而净"。最终优劣**需下游 mesh/Gaussian 判**。
- down-weighting 随轮次下降（iter1 92%→39%，iter2 74%→55%），符合 IRLS。
- δ=1px 在粗位姿起点偏激进（init 92% 被降权、iter1 raw_mse 瞬时尖刺），可与 gate 一起调；建议 A/B `gate ∈ {0.5°,1.5°} × δ ∈ {1,2,3}`。

### 内层 filter 的真实角色

- ceres 的多轮 filter+re-BA 由 **virtual track 驱动**；bae 无 virtual。
- bae+none：内层 filter 近乎空转；bae+huber：它变成有用的**最终轮 outlier 清理**（删的是 huber 降权后仍大残差的 obs），但**不驱动 pose 再优化**。
- 故采用：中间轮跳过（优化 2）+ 最终轮跑满三档但不 re-BA（优化 3）。

### gate vs δ 单变量拆解（gate 是正向的，δ≈无差别）

补做了第三个 run 后，凑齐 2×2 两条边，能把 gate 与 δ 干净分开。三个 run 均 `--bae_optimize_intrinsics`，在唯一可比的 **iter=2（round-1 BA 后；round 1 不受总轮数影响）** 对照（数据均为 390 帧）：

| run | δ | gate | real mean | real median | round-1 BA 喂入 obs |
|---|---|---|---|---|---|
| A `390_bae_huber_loss` | 1.0 | **0.5** | 0.1863° | 0.1040° | 1.57M |
| B `390_bae_huber_custom_1` | 1.0 | **1.0** | **0.1744°** | **0.0989°** | **1.84M（+17%）** |
| C `390_bae_huber_custom_new` | **1.5** | 1.0 | 0.1765° | 0.0999° | 1.84M |

- **gate 效应（A→B，δ 固定 1.0）**：real mean **−6.4%**、median −4.9%，S-only/P-only 同向；且 round-1 BA 多保留 **+17%** 观测。→ **`--filter_reproj_error_threshold 1.0` 是正向的**，且它正是早先 ~5% 提升的真正来源。
- **δ 效应（B→C，gate 固定 1.0）**：real mean **+1.2%**（δ=1.5 反而略差），到 iter=3 两者重合（B 0.1566° vs C 0.1563°），最终点数几乎相同（217243 vs 217429）。→ **δ∈{1.0,1.5} 基本无差别**。
- 机理佐证：B（gate=1.0）起步残差更高（init mean 6.06px vs A 4.66px，93% 被降权），却收敛到更低的最终重投影——**放宽硬闸门多留覆盖、鲁棒性交给 huber** 的设计假设成立。

**结论**：提升来自 **gate（pre-BA 闸门 0.5°→1.0°）**，不是 δ。推荐默认 **δ=1.0 + gate=1.0**（即 `custom_1` 这组），δ 无需调到 1.5。

保留意见：仅 iter=2 干净可比（A 是 2 轮、B/C 是 3 轮，后续点不可比）；`<Xdeg%` 因阈值随 gate 变不可比，只看 mean/median；绝对差 sub-0.01°，最终仍由 mesh/Gaussian 判。

---

## 验证状态

- 所有改动 `python -m py_compile` 通过。
- huber、3 轮、OOM 修复、序列化修复均已在目标 conda 环境完整跑通（用户侧）。
- `ruff` / 完整 SfM 链路以云端/目标环境为准。

## 待办 / 下一步

- bae vs ceres **同数据**对照（real angular / 点数 / 时间），口径对齐（P-only bucket vs ceres real bucket）。
- 用 **mesh/Gaussian** 判定 huber"少而净"、3 轮 vs 2 轮、以及 gate=1.0 的 ~6% reprojection 增益是否真有下游收益。
- gate/δ 拆解已做（gate 正向、δ 无差别）；若想继续，可再试 gate=1.5° 看是否还有增益、何时过松。
- 视需要：`--bae_matrix_free`（省显存）、把 re-BA 前提补进注释。
