# LoMa 批处理与执行流水加速设计

状态：已实现并完成 Ubuntu/4090 的两轮 789 图运行；固定输入的 GPU 数值一致性 A/B 仍待验收。

更新时间：2026-09-08

2026-09-08 实测后的执行修订：主流程在整个 LoMa prior 阶段固定 PyTorch intra-op
线程数为 8，模型构建前设置，在成功或异常退出时恢复原线程数，不新增 CLI 参数。
依据为远程 64 逻辑 CPU / 13.6 核配额下的 CPU batch 上传准备小测试。
本次组合实验 matching batch 改为 2，其余沿用提取 batch=4、预处理/几何 workers=4、
CUDA cache；上传 buffer 实现保持当前版本，以便限定实验因素。

本轮完整运行成功（退出码 0），远程目录 `/kiri/tmp/ffba_789_loma_355_batch2_t8/`。
相对上一轮 `/kiri/tmp/ffba_789_loma_355_batch/` 的记录：

| 指标 | 上一轮 batch=4 | 本轮 batch=2、LoMa CPU threads=8 |
| --- | ---: | ---: |
| LoMa 模型加载 | 32.05 s | 11.32 s |
| 特征提取 | 106.92 s | 57.23 s |
| 提取准备 + H2D 计时区间 | 50.95 s | 5.25 s |
| Matching | 153.61 s | 161.15 s |
| Matching + geometry 墙钟 | 154.47 s | 164.01 s |
| Prior 总计 | 293.66 s | 232.79 s |
| Stage A | 152.05 s | 98.10 s |
| SIFT 阶段 | 133.76 s | 169.06 s |
| 后端 refinement | 505.04 s | 539.63 s |
| 端到端 | 1151.43 s | 1106.53 s |

日志确认线程 `64 -> 8 -> 64`。Matching 实际执行 4702 个双 pair batch 与 1 个尾 pair，
共 9405 pairs（上一轮 9348）；按 pair 归一的 matching 为 17.13 ms，对照 16.43 ms。
8 线程下提取准备区间显著下降，本轮 batch=2 未显示 matching 速度优势；两项同时变化，
不据此单独量化 batch 因果效果。Stage A 不受本次线程作用域影响，其耗时变化也不能计作该改动收益。

本次重新运行 SIFT，matches 为 1,752,322，对照 1,542,334（+13.6%）；最终 495,773 点、
2,255,088 观测、无掉帧，其中 S-only 351,439 / P-only 144,333 / mixed 1。
P-only 角度均值为 0.119569°，对照 0.118762°。这些是完整运行的观测结果，不是固定 SIFT、
固定 selected pairs 的严格质量/速度 A/B。详细统计保存在本地 `logs/ffba_789_loma_355_batch2_t8/`。

关联：[LoMa prior 主设计](loma_prior_lite_design.md)。本方案在现有 3/5/5 pair
选择后改变执行方式，保留 LoMa-B、2048 关键点、原生检测/描述分辨率、0.1 匹配
阈值、两视图几何验证、固定 feature ID、成轨与 BAE 设置。

## 1. 基线与目标

用户云端 Ubuntu / RTX 4090 的 789 图 3/5/5 结果：

| 指标 | 已测结果 |
| --- | ---: |
| 执行 pairs | 9,331 |
| 特征提取 | 234.96 s |
| Matching | 204.35 s |
| 几何验证 | 178.76 s |
| Prior 总时间 | 644.41 s |
| 端到端 | 1,442.05 s |

来源：[prior_loma_stats.json](../../logs/ffba_789_loma_355/prior_loma_stats.json)、
[run.log](../../logs/ffba_789_loma_355/run.log)。这是串行执行基线，不是本方案实测。

目标是减少小批量 GPU 推理、重复特征搬运及 GPU 等待 CPU 的成本。GPU 利用率约
50%、显存占用较低是用户观察，不能据此推算线性加速倍数或最佳 batch 大小。
本轮不切换 B128、不减少 pairs/关键点、不改变精度配置，也不同时加入 compile。

## 2. 参数与生效语义

参数贯通主入口、`GluemapSpvRefineConfig` 和 `LoMaBackend`/执行器，仅作用于 LoMa。
初始默认保持原执行方式；加速配置显式启用，验证后再决定是否调整默认。

| 已新增参数 | 默认/对照值 | 首轮组合试验值 | 语义 |
| --- | ---: | ---: | --- |
| `--loma_match_batch_size` | 1 | 8 | 一次 matcher forward 中的 pair 数，整数且 >=1 |
| `--loma_extract_batch_size` | 1 | 2 | 一次检测/描述的图像数，整数且 >=1 |
| `--loma_preprocess_workers` | 0 | 4 | CPU 解码/缩放线程数；0 为主线程同步预处理，禁止负值 |
| `--loma_geometry_workers` | 1 | 4 | 独立 pair 几何验证的并发数，整数且 >=1；1 保持串行验证路径 |
| `--loma_feature_cache` | `cpu` | `cuda` | 归一化关键点与描述子的驻留位置，仅允许 cpu/cuda |

`feature_cache=cuda` 要求当前计算设备为 CUDA，使用同一 device index。
匹配和提取 batch 独立设置，不能因 matcher batch=8 就按 8 图运行重型 descriptor。
每个几何任务内部 RANSAC `num_threads=1`，避免外部 worker 与内部线程相乘。
CPU worker 数以云端分配的 CPU 核数为依据，不由显卡型号推导。

全默认值应保留原生逐图路径和逐对匹配路径，作为数值与性能回退。
CPU 单图预处理已与 vendored 原生加载函数逐元素对照（横/竖图、小图放大、RGB/灰度/RGBA），
输入一致；GPU 提取数值仍需独立检查，不能仅凭 batch=1 宣称整条推理等价。

## 3. 提取：一次解码、双路输入、按尺寸分桶

1. 每张工作图解码一次 RGB，分别从该原图生成 detector 与 descriptor 输入。
2. Detector 沿用 DaD 的长边 1024、宽高向下对齐到 8、RGB 转换、PIL resize
   及归一化顺序；小图仍按原生规则放大，不引入 hloc 的 resize_max 语义。
3. Descriptor 从原工作图独立生成 784×784 输入，保留原生 resize 参数和
   NumPy 除法/float 转换顺序；不能复用 detector 缩小后的图片再缩一次。
4. CPU 线程仅做 I/O 和预处理，不加载模型或调用 CUDA。主进程汇总张量，
   在适当的批次 staging buffer 上使用 pinned memory/异步 H2D。
5. 按 detector 的实际 H×W 分桶，桶内批量调用现有 DaD `detect` 和 DeDoDe
   `describe_keypoints`。两者都显式处于 eval/inference mode；不为拼批额外 pad。
6. 本次 789 数据尺寸一致，可形成同一尺寸桶；尾批和单图桶正常运行。
7. 结果按原 image index 回填，每图只提取一次。关键点仍映射回原工作图，
   保留本次提取的行号，不做坐标聚类、重排、snap 或跨图融合。

预处理采用有限在途队列，初版最多预取 2 个提取 batch（运行中的任务也计入上限）；
不能把全部 789 张双路输入预处理后堆在 RAM/pinned memory。
分桶可基于工作图尺寸预先生成稳定任务顺序，完成顺序不改变 image ID。
Detector 和 descriptor 本轮共用 extract_batch_size，不再增加两个子模型参数。

## 4. 特征缓存生命周期

- `cpu`：沿用 CPU 完整特征缓存，按 batch 拼装并上传需要的归一化坐标与描述子。
  不要求将整套特征常驻 pinned memory；仅为有限批次分配传输缓冲。
- `cuda`：提取结果保持原生 dtype，归一化坐标和描述子常驻指定 GPU；matching
  根据 pair 索引组装 batch。约 `789×2048×256×4` 字节，即 1.54 GiB 的描述子
  本体仅是 FP32 估算，不包含模型、中间张量和 allocator reserved memory。
- CPU 始终保留工作图像素坐标，供几何验证与 DB writer 使用。CUDA 模式不再保留
  无用的整份 CPU descriptor 副本，也不因队列等待保留每批 score matrix。
- 所有匹配结果回收到 CPU 后释放模型引用、GPU descriptor/cache/staging buffers，
  在进入现有 BAE 前沿用既有 CUDA cleanup。不得在每对/每批中调用 empty_cache。
- 本轮不新增磁盘缓存复用。验收用的固定特征快照需包含图片身份、分辨率、模型与
  dtype 信息；属于测试夹具，不自动复用为正式输入。
- 不做隐式缓存迁移或自动减小 batch。OOM 应报告阶段、实际 batch、缓存规模和
  已完成进度，并传播错误；用户可显式调小 batch 或切回 CPU cache。

## 5. Matching：独立 pair batch

1. 只处理 3/5/5 选中的固定 pair 列表，以稳定 pair ID 建立任务。
2. 按 `(N0, N1, descriptor_dim, dtype)` 分桶，拼装 `[B,N,2]` 坐标和
   `[B,N,D]` 描述子，调用现有 LoMa forward/filter_matches。
3. 不将不同 pair 合并成多图 joint tracking，不修改模型 attention 或匹配阈值。
   数量不一致时分桶处理，不直接补零影响 softmax/mutual match。
4. 逐 batch 元素提取独立匹配索引与分数；修正当前只读取 `indices[0]` 的适配层。
   批量回传紧凑索引/分数到 CPU，不回传完整 `[B,N0,N1]` score matrix。
5. 原始每图 feature ID 不变；共享图像参加多对匹配不会重新提取或生成新身份。
6. 空特征 pair 生成空匹配并完成任务；不足一个 batch 的尾批不丢弃、不重复填充。
7. CUDA 单一模型由主进程控制；本轮先不使用多个并发模型副本或多 CUDA 推理流。

## 6. 几何验证：有限队列与 CPU/GPU 重叠

```text
GPU：batch 1 matching → batch 2 matching → batch 3 matching → …
CPU：                 verify batch 1 / verify batch 2 / …
汇总：按固定 pair ID 收集结果 → 原有 indexed DB → triangulation / BAE
```

`geometry_workers=1` 保持匹配一批后串行验证该批，便于隔离 batch 的收益。
大于 1 时启用并发验证和与下一批 GPU matching 的流水重叠。

- 实现采用线程池：已核实目标 [pycolmap 4.1.0 绑定](https://github.com/colmap/colmap/blob/4.1.0/src/pycolmap/estimators/two_view_geometry.cc)
  的 `estimate_two_view_geometry` 显式使用 `py::gil_scoped_release`。各任务独立创建
  相机/options；本机 pycolmap 4.1.1 的合成投影数据串行/并行内点对照通过。
  这证明当前输入的结果契约，实际并发吞吐仍需云端 CPU 配额下测试。未引入进程池或自动回退。
- 每个任务独立构造 geometry options/相机输入，避免并发修改共享的 pycolmap 对象；
  keypoints 是只读数组。实际阈值、模型选择和内点准入沿用当前标准验证。
- 待完成任务（含正在运行的任务）上限为
  `max(2 × match_batch_size, geometry_workers)` 个 pairs；达到上限时先回收结果。
  队列限制只控制在途内存，不删 pair，也不改变匹配数量。
- 完成顺序可以不同，最终结果、pair JSON 和 DB 写入顺序按原 pair ID 恢复。
  DB 由主线程统一写，不让 workers 并发写 SQLite。
- worker 异常必须携带 pair ID 传播，取消待执行任务并回收资源；不能把任务错误
  记成零匹配，也不能在任务未完成时将 run 标为成功。
- RANSAC 随机性和调度可能影响结果。正式默认不擅自更改随机种子；一致性测试
  可为两条路径指定相同的逐 pair 种子，区分算法随机波动与并发错误。

## 7. 可观测性与验证

运行配置同时记录请求值和实际 batch/worker/cache；每个分桶记录尺寸、数量、
forward 次数、有效 batch 分布与尾批。统计缓存实际字节、模型加载与提取/匹配
阶段的 CUDA peak allocated/reserved、CPU 在途任务峰值和错误上下文。

新执行路径计时分开：解码/预处理、H2D、detector、descriptor、matcher、D2H、几何验证、
队列等待、整个 prior wall time。GPU 使用 CUDA events，在适当批次边界回收事件；
不为了逐 kernel 计时增加全局同步破坏流水。CPU task 累计时长和 CUDA 工作时长
不等于墙钟时间，并发后不得直接相加；在 timing 中显式区分口径并标注 schema 版本。
初始化/冷启动与 steady-state 分开记录，完整端到端仍包含本次实际发生的启动成本。
生产日志的首次 forward 成本包含在各阶段内；独立 GPU 检查将 matching 首次调用和后续
重复调用分开。默认原生提取保留其路径 API，`native_extract_wall` 不细分内部 kernel；
不能把未记录字段当作零耗时。`memory.stages` 在阶段边界重置 peak，统计进程 allocator，
包含其他模块仍存活的分配，不声称是 LoMa 模型的独占显存。

验证顺序：

| 实验 | 固定项 | 改动项 | 主要检查 |
| --- | --- | --- | --- |
| M | 同一份特征、selected pairs、几何设置 | match batch=1/4/8，先 CPU cache + 单几何 worker | 每对索引/分数、阈值附近变化、吞吐与显存 |
| C | M 的 batch 和输入 | cpu/cuda feature cache | dtype/特征身份、匹配一致性、传输时间、BAE 前释放 |
| E | 同一批工作图片及模型 | 原生路径/新预处理；extract batch=1/2/4，workers=0/4 | 输入张量、关键点集合/位移、描述子、下游匹配 |
| G | 固定 pair 匹配结果和几何配置 | geometry workers=1/4，随后验证流水 | 单独的几何结果一致性、线程实际吞吐、队列与异常传播 |
| 综合 | 固定 SIFT DB、图片与相机、selected pairs、3/5/5 和后端 | 推荐组合 | 789 端到端、成轨、覆盖、残差、pose/focal、内存 |

不能重新运行 SIFT 后假定输入一致；已有两次 789 的 SIFT matches 相差约 10%。
GPU batch 可能改变浮点求值/TopK 并列结果，尤其 detector 的 feature 顺序；测试既看
集合与位置，也看重映射后的匹配，不能假定跨运行行号天然对应。本次运行内部 ID 必须稳定。
验收不要求事先承诺 bitwise identical，但显著差异必须解释并完成几何质量检查。

CPU 测试覆盖分桶/尾批/空输入、顺序映射、队列上限、异常收敛和 JSON 口径；
真实模型与并发收益必须在用户 Ubuntu/4090 环境验证。重点跟踪第 261 帧及其他
弱帧的最终观测与空间覆盖，不能只看全局平均残差。

## 8. 实施位置与交付边界

- `run_merg3r_gluemap_pipeline.py`：CLI、校验、配置与 summary。
- `utils/gluemap_spv_refine.py`：配置透传、生效设置与 prior/后端生命周期衔接。
- `utils/loma_prior.py`：提取批处理、cache、matching batch、验证调度及统计。
- `utils/loma_execution.py`：无 Torch 依赖的原生预处理、分桶、有限预取和几何队列。
- `tests/test_loma_prior.py`、`tests/test_loma_execution.py`：CPU 调度契约、真实 pycolmap 与 DB 衔接。
- `scripts/check_loma_execution.py`：真实 GPU 数值 smoke；同一份内存特征对照 M/C，
  E 对照按互为最近邻映射回原生 feature ID，输出每图/每 pair 差异。快照元数据记录图片
  SHA256、分辨率、模型及原生 dtype（cache 转换用逐元素检查），不生成磁盘特征缓存。
- 上游模型权重和算法、SIFT、pair selection、COLMAP 成轨、BAE 保持原实现。

M/C/E/G 均已接入，每项可回到对照设置。运行示例见 README 的 LoMa 小节。
本机无 Torch/CUDA，63 项定向 CPU 测试通过；远程真实 PyTorch 线程恢复检查通过，
789 图完整运行已完成，结果见文首。固定输入的 GPU 数值一致性与 BAE 质量 A/B 仍待验证。
