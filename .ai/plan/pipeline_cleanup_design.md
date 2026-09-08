# 正式管道整理

分支：`codex/pipeline-cleanup`，基于 `codex/loma-prior-lite` 的 `eb3cb19`。

## 约定

- CLI 只暴露配置文件、图片、输出、可选 transforms、ordered/unordered、standard/lite。
- standard 固定 SIFT-first sparse-center VGGSfM；lite 固定 LoMa-B + sift-guided 3/5/5 + batch 2/4、workers 4/4、CUDA cache、内部 torch 线程 8。
- 两种输入顺序都保留原有 ShortestPath 子集划分、Pi3X 和 weighted iterative 对齐算法。
- Stage A 无论前馈或导入先验都提供 DINO 相似度。前馈复用现有 low images 的相似度；先验模式仍用临时 512 长边、batch 16 的 retrieval images。
- ordered 使用输入时序；unordered 不添加 SIFT/LoMa temporal pairs，不使用 LoMa 序号间隔去冗余。
- 无序 VGGSfM 内部调度使用确定性的 DINO 贪心路径：从相似度总和最大的图片开始，每步选择未访问的最高相似度邻居，平分按稳定图片索引。调度数组不改变 image IDs。
- center 抽稀仍依据上一个 center 的有效 SIFT 边与最大调度间隔；owner 保存真实图片索引，三层 group 沿 selected centers 的调度顺序寻找有效 bridge。
- group 由配置选择 `projected_overlap` 或 `sift_pose_dino`。请求前者而实际无深度时回退后者；其他错误不吞掉。保存 requested/effective/fallback reason。
- 只有真实-track BAE；忽略两视图成轨、全来源 SelectTrack、原有过滤阈值、gauge 和 Huber 行为不变。
- 标准配置保留 BAE observation budget，默认 0；lite 禁止正预算。

## 配置与接口

`config.yaml` 是唯一数值默认来源。`ffba/config.py` 负责严格合并、类型/范围校验，拒绝未知字段、重复键以及旧 CLI。
部分实验 YAML 叠加到仓库 YAML；CLI 仅覆盖运行选择。`GluemapSpvRefineConfig` 的历史类名适配仍从相同默认配置取值，不维护另一套数值默认。

`resolved_config.yaml` 包含 requested 配置、生效 temporal window、实际 group 策略、源路径、Git commit 和 dirty 状态。它包含审计字段，不能直接当作输入覆盖文件。
`pipeline_stage_a_summary.json` 保存源图片顺序及其来源；`vggsfm_schedule.json` 保存独立的内部调度数组。

取消正式的 legacy/full-center、全候选 LoMa、Ceres/virtual 运行分支。保留旧模块路径的薄转发便于历史脚本导入，但不保留旧 CLI 兼容层。
旧 README 完整归档在 `docs/legacy_pipeline.md`，当前 README 只说明正式入口。

## 模块职责

- `ffba/pipeline.py`：顶层编排；`ffba/refinement.py` 连接 Stage B 的匹配与重建。
- `ffba/initialization/`：工作图、先验导入、DINO、Pi3X、分块和对齐。原始 `algos/` 留存，正式依赖图不再导入它。
- `ffba/matching/`：SIFT、center/owner、group、prior 推理及 pair/track 观测转换。
- `ffba/reconstruction/`：数据库、三角化、SelectTrack、观测预算、共同 refinement、BAE 控制器。
- `ffba/reporting/`：来源/数值统计、深度导出、历史 group 可视化帮助函数。
- `ffba/api.py`：编排与历史分析工具的显式函数汇总，无独立算法实现。
- `ffba/refinement_config.py`：后端参数对象及 GlueMap 参数适配。

BAE 控制器直接调用现有 `bundle_adjustment_bae`，不经过支持 Ceres 的旧 augmented BA controller。
保留单次 BAE + 最终轮 3x/2x/1x 过滤语义，三角化、相机初始化和 solver 数值实现不变。
第三方包中的旧实现保留供其自身代码使用，正式管道不导入 Ceres BA 控制器。

## 验证

- 原有 SIFT/LoMa 调度、成轨、匹配列反序、来源统计和真实数据库测试迁移后继续执行。
- 新增 8 种流程组合配置测试、配置错误/覆盖测试、DINO 调度与 owner 映射、无序 temporal 禁用、深度 fallback 测试。
- BAE 新旧控制器对照：最终/非最终轮、1/3/5 次过滤，验证 solver 调用、阈值序列与摘要一致，且始终只优化一次。
- `--help` 在禁止 torch 导入时仍可运行。
- GPU 验证在独立目录 `/kiri/tmp/ffba_pipeline_cleanup_validation` 执行；不改动远程原 checkout 或历史产物。

### 完成结果

- 本地与远程均通过语法/模块检查；`ruff check ffba` 通过。
- 最终远程测试：**106 passed**（真实 torch 2.8.0、pycolmap、pygluemap）；只有既有 SWIG 类型弃用提示。
- 真实 RTX 4090 全流程 smoke 共 5 次，均成功，8 张输入全部注册。前四组使用默认数值配置，仅限制输入数量；最后一组额外设 subset_size=4 / overlap=2，实际形成 `[4,4,4]` 三个子集以覆盖跨子集对齐。

| 流程 | 实际 group | 点数 | 观测数 | pipeline 耗时 |
| --- | --- | ---: | ---: | ---: |
| lite + ordered + prior pose | LoMa | 2,357 | 10,854 | 31.20 s |
| standard + unordered + prior pose | fallback sift_pose_dino | 2,191 | 9,232 | 19.19 s |
| standard + ordered + feedforward | projected_overlap | 2,145 | 8,834 | 69.64 s |
| lite + unordered + feedforward | LoMa | 2,398 | 10,959 | 78.14 s |
| lite + ordered + 多子集 feedforward | LoMa | 2,447 | 11,107 | 79.99 s |

产物审计确认：每组都有 DINO；三轮都忽略两视图成轨、执行全来源 SelectTrack；post-BA filter 只在最终轮执行；乱序组的 temporal pair 数为 0。

远程目录：`/kiri/tmp/ffba_pipeline_cleanup_validation/`。汇总审计同步到本地 `logs/pipeline_cleanup_validation/validation_audit.json`。
前四组运行的源文件快照保留为远程 `tested_four_smokes_source.tar.gz`，后续模块路径和日志整理后再次完成全量单元测试及模块导入检查。

这些是 8 图链路验证，不是速度排名或 789 图质量等价证明；不同初始化和输入语义的点数不能直接用于判断质量优劣。大场景质量回归仍需后续运行。
