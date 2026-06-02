# Merg3r → Gluemap Refinement 验证计划

## Summary

在 `.ai/plan/merg3r_gluemap.md` 写入本计划。目标是先验证 **Merg3r coarse pose + COLMAP/pycolmap refinement** 是否能提升 pose 质量，再决定是否加入 Gluemap 的 `V` virtual tracks 和 augmented BA。

第一阶段拆成两个脚本，不构建完整 pipeline：

- **A 脚本，位于 MERG3R 下**：运行 Merg3r 到 `align_extrinsics()` 完成；基于 aligned pose 构建 pairs；提取 SuperPoint/LightGlue tracks；导出 B 脚本所需 artifacts。
- **B 脚本，位于 gluemap 下**：读取 A artifacts；运行 pose-guided VGGSfM prior tracks；写 COLMAP database；基于 Merg3r coarse pose 做 pycolmap triangulation + BA。

v1 只做 `S + P`：`S = SuperPoint/LightGlue`，`P = VGGSfM prior tracks`。暂不接入 `V` 和 Gluemap augmented BA。

## Key Changes

- 新增 A 脚本：`MERG3R/export_merg3r_refine_inputs.py`
  - 复用 `main.py` 中图像加载、sequence 构建、model inference、`align_extrinsics()`、`restore_predictions_order()`。
  - 不执行 Merg3r 当前 gradient BA。
  - 使用 aligned `extrinsic/intrinsic` 和图像相似度/pose-guided 规则选 pairs。
  - 提取 SuperPoint features + LightGlue matches，导出为中间格式，不直接依赖 Merg3r dense 3D 点作为 BA 点。
  - 输出目录建议：`<output_dir>/gluemap_refine_inputs/`。

- A artifacts 固定为：
  - `metadata.json`：image names、image size、camera model、shared/per-image intrinsics 策略、pair 选择参数。
  - `coarse_poses.npz`：`extrinsic` `(N,3,4)` world-to-camera、`intrinsic` `(N,3,3)`、可选 `image_ids`。
  - `pairs.npy`：`(M,2)` int image-index pairs。
  - `features_lightglue/`：每张图 keypoints/descriptors/scores。
  - `matches_lightglue.npz`：每个 pair 的 match indices。
  - `images/` 或 `image_paths.json`：B 脚本可直接读取的图像路径；默认优先保存/引用与 A 推理一致的低分辨率图像，避免坐标尺度不一致。

- 新增 B 脚本：`gluemap/run_merg3r_pycolmap_refine.py`
  - 读取 A artifacts，构建 `global_rotations/global_centers/global_intrinsics/intrinsics_mapping`。
  - 基于 A 的 pairs 构建 pose-guided VGGSfM groups：每个 center 取 top-k 邻居，输入 `[center, neighbors...]`，query points 来自 center。
  - 将 LightGlue tracks 和 VGGSfM prior tracks 分别写入 COLMAP database，再 merge 成 `database_merged.db`。
  - 用 Merg3r coarse pose 写 `coarse/` COLMAP reconstruction。
  - 调用 pycolmap triangulation，然后运行标准 pycolmap bundle adjustment。
  - 输出 `refined_pycolmap/` reconstruction，并导出 timing 与 track coverage 统计。

- 不直接复用 `prepare_sift_database()` 作为 S 源。
  - 原因：它硬编码 SIFT extraction/matching。
  - v1 需要新增一个 LightGlue/SuperPoint COLMAP DB writer，保证 keypoints、matches、two-view geometry、image ids、camera ids 与 coarse reconstruction 一致。

## Feasibility Review

- 合理之处：
  - Merg3r 提供快速 coarse pose，避免 Gluemap 最慢的 Two-view inference 和 full star feedforward。
  - SuperPoint/LightGlue 作为 `S` 可替代 SIFT 的“真实图像特征约束”角色，只要写成 COLMAP DB。
  - Pose-guided VGGSfM 作为 `P` 比全帧全 query 更可控，能补充长 track 和 chunk 边界覆盖。
  - pycolmap triangulation 让 real track 的 3D 点由 2D tracks + coarse pose 产生，不再强依赖 Merg3r dense 3D 初始化。

- 风险与规避：
  - 如果 Merg3r coarse pose 局部错误，pose-guided pair 可能漏掉正确连接；pair selection 必须混合 image similarity/top-k temporal pairs，不只依赖几何投影。
  - LightGlue 不是 SIFT database 的 drop-in replacement；必须严谨处理 keypoint 坐标、image id、pair id、match index offset。
  - VGGSfM groups 不能无限大；默认 `neighbors_per_center=8`，`query_points=1024`，后续根据显存调。
  - v1 不加入 `V`，因此不会验证 Gluemap augmented BA 的全部收益；这是刻意降低变量数量。

## Test Plan

- 小序列 smoke test：20-50 帧，确认 A artifacts 完整、B 能生成 COLMAP DB、triangulation、BA 输出。
- 中序列验证：100-300 帧，对比：
  - Merg3r aligned coarse pose
  - Merg3r 原 gradient BA
  - 新 `S+P + pycolmap BA`
- 记录指标：
  - pair 数、每 pair matches 数、triangulated points 数、BA 前后 reprojection error。
  - 每帧 track coverage，尤其是 chunk 边界帧。
  - BA 后是否出现明显 pose collapse、scale drift、错误相机翻转。
- Ablation：
  - `S only`
  - `P only`
  - `S + P`
  - `S + P` with temporal-only pairs vs mixed pose/similarity pairs。

## Assumptions

- v1 不接入 `V` virtual tracks，也不调用 Gluemap augmented BA。
- v1 的 BA 使用 pycolmap 标准 triangulation + bundle adjustment。
- 默认使用 shared camera intrinsics，取 Merg3r aligned 后 `intrinsic` 的均值；后续如多相机数据再扩展 `intrinsics_mapping`。
- B 脚本运行在 gluemap 环境下，但不会依赖 Gluemap star inference 的 `predictions_dict`。
- `.ai/plan/merg3r_gluemap.md` 当前为空文件；执行阶段可直接覆盖写入本计划。
# Merg3r -> Gluemap Refinement Verification Plan

## Summary

目标是先验证 **Merg3r coarse pose + COLMAP/pycolmap refinement** 是否能提升 pose 质量，再决定是否加入 Gluemap 的 `V` virtual tracks 和 augmented BA。

第一阶段拆成两个脚本，不构建完整 pipeline：

- **A 脚本，位于 MERG3R 下**：运行 Merg3r 到 `align_extrinsics()` 完成；基于 aligned pose 构建 pairs；提取 SuperPoint/LightGlue tracks；导出 B 脚本所需 artifacts。
- **B 脚本，位于 gluemap 下**：读取 A artifacts；运行 pose-guided VGGSfM prior tracks；写 COLMAP database；基于 Merg3r coarse pose 做 pycolmap triangulation + BA。

v1 只做 `S + P`：`S = SuperPoint/LightGlue`，`P = VGGSfM prior tracks`。暂不接入 `V` 和 Gluemap augmented BA。

## Key Changes

- 新增 A 脚本：`MERG3R/export_merg3r_refine_inputs.py`
  - 复用 `main.py` 中图像加载、sequence 构建、model inference、`align_extrinsics()`、`restore_predictions_order()`。
  - 不执行 Merg3r 当前 gradient BA。
  - 使用 aligned `extrinsic/intrinsic` 和图像相似度/pose-guided 规则选 pairs。
  - 提取 SuperPoint features + LightGlue matches，导出为中间格式，不直接依赖 Merg3r dense 3D 点作为 BA 点。
  - 输出目录：`<output_dir>/gluemap_refine_inputs/`。

- A artifacts：
  - `metadata.json`：image names、image size、camera model、shared/per-image intrinsics 策略、pair 选择参数。
  - `coarse_poses.npz`：`extrinsic` `(N,3,4)` world-to-camera、`intrinsic` `(N,3,3)`、`image_ids`。
  - `pairs.npy`：`(M,2)` int image-index pairs。
  - `features_lightglue/`：每张图 keypoints/descriptors/scores。
  - `matches_lightglue.npz`：每个 pair 的 match indices。
  - `images/`：与 A 推理一致的低分辨率图像，避免坐标尺度不一致。

- 新增 B 脚本：`gluemap/run_merg3r_pycolmap_refine.py`
  - 读取 A artifacts，构建 `global_rotations/global_centers/global_intrinsics/intrinsics_mapping`。
  - 基于 A 的 pairs 构建 pose-guided VGGSfM groups：每个 center 取 top-k 邻居，输入 `[center, neighbors...]`，query points 来自 center。
  - 将 LightGlue tracks 和 VGGSfM prior tracks 分别写入 COLMAP database，再 merge 成 `database_merged.db`。
  - 用 Merg3r coarse pose 写 `coarse/` COLMAP reconstruction。
  - 调用 pycolmap triangulation，然后运行标准 pycolmap bundle adjustment；若运行环境的 pycolmap 没有便捷 BA API，则 fallback 到 Gluemap real-only BA wrapper。
  - 输出 `refined_pycolmap/` reconstruction，并导出 timing 与 track coverage 统计。

## Feasibility Review

- 合理之处：
  - Merg3r 提供快速 coarse pose，避免 Gluemap 最慢的 Two-view inference 和 full star feedforward。
  - SuperPoint/LightGlue 作为 `S` 可替代 SIFT 的“真实图像特征约束”角色，只要写成 COLMAP DB。
  - Pose-guided VGGSfM 作为 `P` 比全帧全 query 更可控，能补充长 track 和 chunk 边界覆盖。
  - pycolmap triangulation 让 real track 的 3D 点由 2D tracks + coarse pose 产生，不再强依赖 Merg3r dense 3D 初始化。

- 风险与规避：
  - 如果 Merg3r coarse pose 局部错误，pose-guided pair 可能漏掉正确连接；pair selection 混合 image similarity、temporal pairs 和 camera-center proximity，不只依赖几何投影。
  - LightGlue 不是 SIFT database 的 drop-in replacement；必须严谨处理 keypoint 坐标、image id、pair id、match index offset。
  - VGGSfM groups 不能无限大；默认 `neighbors_per_center=8`，`query_points=1024`，后续根据显存调。
  - v1 不加入 `V`，因此不会验证 Gluemap augmented BA 的全部收益；这是刻意降低变量数量。

## Test Plan

- 小序列 smoke test：20-50 帧，确认 A artifacts 完整、B 能生成 COLMAP DB、triangulation、BA 输出。
- 中序列验证：100-300 帧，对比：
  - Merg3r aligned coarse pose
  - Merg3r 原 gradient BA
  - 新 `S+P + pycolmap BA`
- 记录指标：
  - pair 数、每 pair matches 数、triangulated points 数、BA 前后 reprojection error。
  - 每帧 track coverage，尤其是 chunk 边界帧。
  - BA 后是否出现明显 pose collapse、scale drift、错误相机翻转。
- Ablation：
  - `S only`
  - `P only`
  - `S + P`
  - `S + P` with temporal-only pairs vs mixed pose/similarity pairs。

## Assumptions

- v1 不接入 `V` virtual tracks，也不调用 Gluemap augmented BA。
- v1 的 BA 使用 pycolmap 标准 triangulation + bundle adjustment。
- 默认使用 shared camera intrinsics，取 Merg3r aligned 后 `intrinsic` 的均值；后续如多相机数据再扩展 `intrinsics_mapping`。
- B 脚本运行在 gluemap 环境下，但不会依赖 Gluemap star inference 的 `predictions_dict`。
