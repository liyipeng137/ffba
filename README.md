# FeedForwardWithBA

使用 Pi3X 前馈重建或已有相机初值，结合 SIFT、VGGSfM / LoMa 和 BAE，生成 COLMAP 稀疏重建。
正式入口是 [run.py](run.py)，算法与执行参数统一在 [config.yaml](config.yaml)。

## 运行方式

| 选择 | 配置 / CLI | 行为 |
| --- | --- | --- |
| 标准模式 | `--mode standard` | SIFT-first，固定抽稀 center，再运行 VGGSfM |
| 轻量模式 | `--mode lite` | LoMa-B，SIFT-guided 3/5/5，matching batch=2、extract batch=4、workers=4/4、CUDA cache |
| 有序输入 | `--input_order ordered` | SIFT 与 prior 允许真实时序召回，默认窗口 ±2 |
| 乱序输入 | `--input_order unordered` | 禁用 temporal pairs 和 LoMa 序号间隔偏好；VGGSfM 使用 DINO 内部调度顺序 |
| 有相机初值 | `--prior_transforms_json /path/to/transforms.json` | 导入 Nerfstudio pose / intrinsics，不运行 Pi3X、不生成初始深度 |
| 纯图片 | 省略 `--prior_transforms_json` | 当前 ShortestPath 子集划分 + Pi3X + weighted iterative Sim3 对齐 |

默认 `standard + ordered`。所有组合均在 Stage A 提供 DINO 相似度；有序与乱序保持相同的前馈子集划分算法。
DINO 内部调度顺序只用于 center / owner 选择，不改变图片身份、数据库 ID，也不会给乱序输入添加 temporal pairs。

有序模式使用图片路径排序，或 transforms 的 `frames` 列表顺序；该顺序需要符合实际采集时序。
输入图片需满足当前公共尺寸要求。前馈模式保留 low/high 图片金字塔；先验 pose 模式保留原图工作分辨率。
先验 pose 是优化初值，BAE 后相机仍会更新；`bae.optimize_intrinsics` 控制内参优化。

### 轻量模式 CLI（云端 789_room）

在已配置的 Ubuntu / RTX 4090 环境中，从当前整理分支运行：

```bash
source /opt/conda/bin/activate && conda activate gluemap-merg3r
cd /kiri/FeedForwardWithBA
set -o pipefail
mkdir -p /kiri/tmp/ffba_789_lite_cleanup

python -u run.py \
  --config config.yaml \
  --dataset /kiri/codex_use_data/789_room/image \
  --prior_transforms_json /kiri/codex_use_data/789_room/transforms.json \
  --output_dir /kiri/tmp/ffba_789_lite_cleanup \
  --input_order ordered \
  --mode lite \
  2>&1 | tee /kiri/tmp/ffba_789_lite_cleanup/run.log
```

其他数据请修改输入、先验和输出路径；每次实验使用独立输出目录。
纯图片标准模式：

```bash
python -u run.py \
  --dataset /path/to/images \
  --output_dir /path/to/output \
  --input_order ordered \
  --mode standard
```

## 配置

CLI 仅保留 `--config`、`--dataset`、`--output_dir`、`--prior_transforms_json`、`--input_order`、`--mode`。
旧的 `--ba_backend`、`--vggsfm_schedule_mode`、`--loma_match_batch_size` 等参数已移除；请在 YAML 中配置。

可以直接编辑 `config.yaml`，也可以通过 `--config experiment.yaml` 提供部分覆盖。
优先级为仓库默认配置 → 实验 YAML → CLI 的 mode / input_order。
未知字段、重复键、错误类型和非法范围会在模型加载前报错。

```yaml
# experiment.yaml：只覆盖以下值，其余继承 config.yaml。
prior:
  vggsfm:
    group_strategy: sift_pose_dino
    group_batch_size: 3
  loma:
    match_batch_size: 2
bae:
  optimize_intrinsics: true
```

| 分组 | 内容 |
| --- | --- |
| `pipeline` | standard/lite、ordered/unordered |
| `input` | 图片采样、low/high 分辨率 |
| `retrieval` | DINO 相似度混合；long_side/batch_size 保留先验 pose 路径的预处理设置，前馈路径仍复用 low images |
| `initialization.feedforward` | 内参估计、子集大小、重叠和现有划分设置 |
| `sift` | pose 邻居、旋转阈值、时序窗口、强边的内点和覆盖判据 |
| `prior.vggsfm` | group 策略、center gap、query、阈值、batch、权重路径 |
| `prior.loma` | DINO top-k、3/5/5、batch、workers、cache |
| `refinement` | 三角化、全来源 SelectTrack、过滤和外层迭代 |
| `bae` | 优化轮数、内参、gauge、Huber 和观测预算 |
| `runtime` | device、图片 workers、日志 |

### VGGSfM group

center 始终采用 SIFT 支持引导的抽稀。默认 `prior.vggsfm.max_center_gap: 3`，相邻 center 的调度位置差最多为 3，中间最多跳过 2 帧；SIFT 支持不足时提前选 center。有序输入按帧序、无序输入按 DINO 内部顺序计数。SIFT 支持充分时约每三帧选一个 center，此参数不影响 LoMa。

group 保留两种选择：

| `prior.vggsfm.group_strategy` | 有深度 | 无深度 |
| --- | --- | --- |
| `projected_overlap`（默认） | 深度投影重叠 | fallback 到 `sift_pose_dino` |
| `sift_pose_dino` | SIFT / pose / DINO | SIFT / pose / DINO |

两种策略都保留 owned frames、有效的相邻 center bridge、候选补充三层 group。
输出记录 requested/effective strategy 与 fallback reason；深度数据错配或计算异常不会被当成“缺少深度”吞掉。

### 共同后端与轻量约定

唯一 BA 后端是 BAE，只处理真实 tracks；正式流程已去掉 Ceres BA 和 virtual-track 构建。
仍使用 PyCOLMAP/GlueMap 的匹配、几何验证、三角化、SelectTrack 和模型读写。

- 三角化固定 `ignore_two_view_tracks=True`，不保证后续过滤后的所有 track 均有至少 3 个观测。
- SIFT 和 prior 全部参与 SelectTrack，支持阈值默认 512；“共同筛选”不是保留全部 tracks。
- 默认三轮 refinement，每轮 BAE 优化 20 次；仅最终轮执行原有逐级 post-BA filter。
- 默认优化内参，Huber delta 为 1，最终轮为 2；前置 angular filter 为 1°。
- 轻量模式不设置 observation cap，`bae.max_observations` 必须 ≤0；标准模式默认也关闭预算，可按需要配置。
- LoMa 内部 PyTorch CPU 线程固定 8，退出 prior 阶段时恢复原值。CPU/CUDA cache 均为进程内缓存，在进入 BAE 前释放。
- LoMa pair 池为 pose ∪ DINO top-30 ∪ temporal（仅有序）；保留 temporal，每图对其余 sufficient/insufficient/untried 候选最多选 3/5/5，任一方向选中即执行。

## 输出

- `resolved_config.yaml`：本次最终配置、顺序语义、实际 group fallback、代码版本与 dirty 状态；用于审计，包含运行元数据，不作为输入覆盖文件。
- `pipeline_config.json` / `pipeline_stage_a_summary.json` / `pipeline_run_summary.json`：阶段配置、图片名称顺序、计时与输出位置。
- `refine_stats.json`：SIFT/prior 来源、筛选前后数量、覆盖、优化和计时。
- `vggsfm_schedule.json` 或 `prior_loma_pairs.json` / `prior_loma_stats.json`：各自调度、pair 与执行明细。
- `database_sift.db`、prior DB、`database_merged.db`：匹配与合并产物。
- `refined_gluemap_aba/`：最终 COLMAP 模型；有深度时继续导出 `pred_depth/`。

## 代码结构

```text
ffba/
  config.py                 # YAML / CLI / 校验 / 生效配置
  pipeline.py               # 主流程调度
  refinement.py             # Stage B：连接匹配与重建
  types.py                  # Stage A 状态
  refinement_config.py      # 后端配置适配（默认值来自同一 YAML）
  initialization/           # 图片、DINO、Pi3X、pose 导入、划分、Sim3 对齐
  matching/                 # SIFT、center 调度、group、VGGSfM、LoMa
  reconstruction/           # 数据库、三角化、SelectTrack、过滤、BAE
  reporting/                # 统计、深度导出、group 可视化
```

旧 `algos/` 暂时保留，正式管道不再导入它。旧 `utils` 路径仅保留少量转发，避免历史分析脚本复制实现。
[旧流程与实验 CLI](docs/legacy_pipeline.md) 仅供历史对照，不能直接用于新入口。
设计和验证记录见 [整理方案](.ai/plan/pipeline_cleanup_design.md)。

## 环境与验证

复用已有 `gluemap-merg3r` GPU 环境。新配置加载需要 `PyYAML`；LoMa 的依赖见
[third_party/LoMa/pyproject.toml](third_party/LoMa/pyproject.toml)，首次加载按上游机制下载权重。
VGGSfM 权重位置在 `prior.vggsfm.weights` 配置；GlueMap、BAE、Pi3X 的安装需求与整理前相同。

```bash
python run.py --help
python -m pytest -q tests
```

`--help` 不加载 torch 或模型。完整测试需要 torch、pycolmap 与 pygluemap；CPU 主机可运行配置及匹配数据逻辑测试，不能替代 GPU 端到端验收。

当前整理版已通过远程 **106 项测试**及 **5 次 8 图 GPU 全流程验证**，覆盖先验/前馈、两种 prior、有序/乱序、深度 fallback 与多子集对齐。详细产物与结论边界见整理方案；尚未完成 789 图质量 A/B。
