# TAPNet 项目梳理（逐帧 + 高精度）

> 基于 [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet)  
> 目标：在视频中**逐帧**跟踪任意点，并优先选择**精度最高**的现成 pipeline。

---

## 1. 项目是做什么的

**TAP（Tracking Any Point）**：给定视频 + 若干**查询点**（query point，在某帧上的像素位置），模型输出这些点在**每一帧**的 2D 轨迹，以及是否可见（遮挡）。

任务特点：

- 点级精度（不是框/分割跟踪）
- 类别无关（任意可跟踪表面上的点）
- 支持长时序、遮挡、形变表面

本仓库包含：模型权重、Colab Demo、本地 Live Demo、TAP-Vid / RoboTAP / TAPVid-3D 评测代码与数据接口。

---

## 2. 术语：Online vs Offline

仓库里有两套含义，容易混淆：

| 术语 | 含义 | 典型入口 |
|------|------|----------|
| **Online / 逐帧** | 因果推理：每来一帧只处理当前帧，依赖 `state` / `causal_context` 传递历史 | `colabs/torch_tapnextpp_demo.ipynb`、`colabs/causal_tapir_demo.ipynb`、`tapnet/live_demo.py` |
| **Offline / 整段** | 非因果：一次性看完整段视频，双向时序 refine | `colabs/tapir_demo.ipynb`、`colabs/torch_tapir_demo.ipynb` |

你的需求是**逐帧 + 高精度** → 选 **Online** 路线，并在其中选精度最高的 checkpoint。

---

## 3. 模型选型（精度优先 + 逐帧）

TAP-Vid DAVIS First（Average Jaccard，越高越好）：

| 模型 | 推理方式 | DAVIS First | 本地脚本 | 推荐度（你的场景） |
|------|----------|-------------|----------|-------------------|
| **TAPNext++** | 逐帧 Online | **65.6%** | Colab only（PyTorch） | ⭐ 首选 |
| BootsTAPNext | 逐帧 Online | 65.25% | Colab（Jax / PyTorch） | 次选 |
| Online BootsTAPIR | 逐帧 Online | 59.7% | `live_demo.py` / Colab | 有本地实时 demo，精度较低 |
| BootsTAPIR（Offline） | 整段 | 62.4% | Colab | 精度不错，但**非逐帧** |
| TAPIR（Offline） | 整段 | 58.5% | Colab | 不推荐 |

**结论：逐帧高精度 → 用 TAPNext++，入口为 `colabs/torch_tapnextpp_demo.ipynb`。**

TAPNext++ 额外能力：更长稳定跟踪、遮挡穿越、重检测（在 1024 帧合成序列上微调）。

---

## 4. 推荐 Pipeline：TAPNext++ 逐帧推理

### 4.1 流程概览

```mermaid
flowchart LR
    A[加载 checkpoint] --> B[准备 video + query_points]
    B --> C["第 0 帧: model(video, query_points)"]
    C --> D[得到 tracks / logits / state]
    D --> E["第 t 帧: model(video_t, state=state)"]
    E --> F[拼接全序列 tracks + visible]
    F --> G[可选: TAP-Vid 指标 / 可视化]
```

### 4.2 依赖安装

```bash
git clone https://github.com/google-deepmind/tapnet.git
cd tapnet
pip install .
pip install "numpy<2.1.0"
pip install git+https://github.com/google-deepmind/recurrentgemma.git@main
pip install torch torchvision  # TAPNext++ 需要 CUDA
```

### 4.3 Checkpoint

```bash
wget https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt
```

### 4.4 核心代码路径

| 模块 | 路径 |
|------|------|
| TAPNext++ 模型 | `tapnet/tapnext/tapnext_torch.py` → `TAPNext` |
| 逐帧推理示例 | `colabs/torch_tapnextpp_demo.ipynb` → `run_eval_per_frame()` |
| 评测指标 | `tapnet/tapvid/evaluation_datasets.py` |
| 确定性/置信度 | `tapnet/tapnext/tapnext_torch_utils.py` → `tracker_certainty()` |

---

## 5. 输入 / 输出规范

### 5.1 坐标约定（重要）

- **2D 点**：一般用 `(x, y)` 表示横纵像素
- **3D 查询点**：仓库统一用 **`(t, y, x)`** 顺序  
  - `t`：帧索引（0 = 第一帧，可为浮点）
  - `y, x`：栅格坐标，`(0,0)` 为左上角像素角，`(h, w)` 为右下角
- **TAPNext 输出的 tracks**：`(x, y)` 顺序（与 query 的 y,x 相反）
- 评测代码里常做 `[..., ::-1]` 把 `(x,y)` 转成 `(y,x)` 再算指标

### 5.2 TAPNext++ 输入

#### 首次调用（初始化，通常第 0 帧）

```python
model(video, query_points=query_points)
```

| 参数 | Shape | 类型 / 范围 | 说明 |
|------|-------|-------------|------|
| `video` | `[B, T, H, W, 3]` | float，`[-1, 1]` | 首帧时 `T=1`；ImageNet 式归一化：`x/255*2-1` |
| `query_points` | `[B, Q, 3]` | float | `(t, y, x)`，像素坐标；首帧查询通常 `t=0` |

#### 后续逐帧调用

```python
model(video=frame_t, state=tracking_state)
```

| 参数 | Shape | 说明 |
|------|-------|------|
| `video` | `[B, 1, H, W, 3]` | 当前单帧 |
| `state` | `TAPNextTrackingState` | 上一步返回的状态，**不再传 query_points** |

`TAPNextTrackingState` 字段：

- `step: int` — 已处理帧数
- `query_points: [B, Q, 3]` — 原始查询点（内部会按 step 回拨时间）
- `hidden_state` — 12 层 RecurrentBlock 的 SSM cache

#### 分辨率

- 默认训练/推理：**256×256**
- `TAPNext(image_size=(256, 256))` 需与输入 resize 一致

### 5.3 TAPNext++ 输出

每次 `forward` 返回 4 元组：

```python
tracks, track_logits, visible_logits, tracking_state = model(...)
```

| 输出 | Shape | 含义 |
|------|-------|------|
| `tracks` | `[B, T, Q, 2]` | 当前 batch 内每帧预测位置，**(x, y)** 像素坐标 |
| `track_logits` | `[B, T, Q, 512]` | 坐标 softmax logits（x/y 各 256） |
| `visible_logits` | `[B, T, Q, 1]` | 可见性 logit，`> 0` 即可见 |
| `tracking_state` | `TAPNextTrackingState` | 传给下一帧 |

#### 拼成完整视频轨迹

逐帧循环后 concat，再 transpose：

```python
# 逐帧收集 pred_tracks, pred_visible
tracks   = torch.cat(pred_tracks, dim=1).transpose(1, 2)   # [B, Q, T_total, 2]
visible  = torch.cat(pred_visible, dim=1).transpose(1, 2)  # [B, Q, T_total, 1]
occluded = ~visible                                        # 或结合 certainty 阈值
```

#### 可选后处理

- **可见性**：`visible = visible_logits > 0`
- **确定性过滤**（demo 中可选）：`tracker_certainty(tracks, track_logits, radius=8)` 与 visible 相乘后阈值化，减少漂移点

### 5.4 评测数据集 batch 格式（TAP-Vid）

`create_davis_dataset()` 产出的 `batch`（在 `batch['davis']` 或展平后）：

| 字段 | Shape | 说明 |
|------|-------|------|
| `video` | `[1, T, H, W, 3]` | float，`[-1, 1]` |
| `query_points` | `[1, Q, 3]` | `(t, y, x)` 像素坐标 |
| `target_points` | `[1, Q, T, 2]` | GT 轨迹，**(x, y)** |
| `occluded` | `[1, Q, T]` | GT 遮挡，`True`=被挡 |

送入模型前 demo 会 `.cuda().float()`；预测 tracks 需 `[..., ::-1]` 转成 `(y,x)` 再调用 `compute_tapvid_metrics()`。

---

## 6. 备选 Pipeline：Online BootsTAPIR（本地可跑）

若需要**本地摄像头实时 demo**（精度略低于 TAPNext++）：

| 项目 | 内容 |
|------|------|
| 入口 | `tapnet/live_demo.py`（Jax）、`tapnet/pytorch_live_demo.py`（PyTorch） |
| Checkpoint | `causal_tapir_checkpoint.npy` 或 `causal_bootstapir_checkpoint.npy` |
| 推理模式 | 三阶段：`get_feature_grids` → `get_query_features`（首帧）→ `estimate_trajectories`（逐帧 + `causal_context`） |

### 输入

| 阶段 | 输入 | Shape |
|------|------|-------|
| 预处理帧 | `frames` | `[T, H, W, 3]` uint8 `[0,255]` → `preprocess_frames` → `[-1,1]` |
| 查询点 | `query_points` | `[Q, 3]` 或 `[1,Q,3]`，**(t, y, x)** 像素 int/float |
| 逐帧预测 | `causal_context` | 由上一步 `estimate_trajectories` 返回 |

### 输出（单帧 `online_model_predict`）

| 字段 | Shape | 说明 |
|------|-------|------|
| `tracks` | `[1, Q, 1, 2]` | 当前帧位置 **(x, y)** |
| `occlusion` | `[1, Q, 1]` | 遮挡 logit |
| `expected_dist` | `[1, Q, 1]` | 不确定性 |
| `visibles` | `[Q]` | `postprocess_occlusions(occlusion, expected_dist)` |
| `causal_context` | — | 下一帧状态 |

---

## 7. Offline Pipeline（对比用，非逐帧）

`colabs/tapir_demo.ipynb`（BootsTAPIR 默认）一次性推理：

```python
outputs = tapir(
    video=frames,           # [1, T, H, W, 3], [-1, 1]
    query_points=points,    # [1, Q, 3], (t, y, x)
    is_training=False,
    query_chunk_size=32,
)
tracks         = outputs['tracks']          # [1, Q, T, 2] (x, y)
occlusions     = outputs['occlusion']         # [1, Q, T]
expected_dist  = outputs['expected_dist']   # [1, Q, T]
visibles       = postprocess_occlusions(occlusions, expected_dist)
```

优点：同系列里 Offline BootsTAPIR 精度高于 Online 版；缺点：**不能逐帧因果推理**，无法用于严格 online 场景。

---

## 8. 仓库结构速览

```
tapnet/
├── colabs/                    # 各模型 Demo（推荐从这里入手）
│   ├── torch_tapnextpp_demo.ipynb   # ★ 逐帧高精度首选
│   ├── tapnext_demo.ipynb           # BootsTAPNext (Jax)
│   ├── causal_tapir_demo.ipynb      # Online TAPIR/BootsTAPIR
│   └── tapir_demo.ipynb             # Offline TAPIR/BootsTAPIR
├── tapnet/
│   ├── tapnext/               # TAPNext / TAPNext++ 实现
│   ├── models/                # TAPIR (Jax)
│   ├── torch/                 # TAPIR (PyTorch)
│   ├── tapvid/                # TAP-Vid 数据集 & 评测
│   ├── tapvid3d/              # 3D 点跟踪 benchmark
│   ├── robotap/               # 机器人模仿相关
│   ├── live_demo.py           # Online TAPIR 摄像头 demo
│   └── pytorch_live_demo.py   # Online BootsTAPIR 摄像头 demo
├── configs/                   # 训练配置
└── README.md
```

---

## 9. 针对你需求的最小落地路径

1. **跑通官方逐帧高精度 demo**  
   打开 `colabs/torch_tapnextpp_demo.ipynb`，或本地按其中 `run_eval_per_frame()` 逻辑复现。

2. **换成自己的视频**  
   - resize 到 256×256  
   - 构造 `query_points`: `[1, Q, 3]`，`(t=0, y, x)`  
   - 视频 tensor: `[1, T, 256, 256, 3]`，范围 `[-1, 1]`  
   - 循环：`state=None` 首帧 → 之后只传 `state`

3. **读结果**  
   - `tracks[b, q, t]` → `(x, y)` 像素  
   - `visible_logits > 0` → 是否可见  
   - 需要更稳的可见性时加 `tracker_certainty`

4. **若要本地实时但接受更低精度**  
   用 `live_demo.py` + `causal_bootstapir_checkpoint`（BootsTAPIR Online 版）。

---

## 10. 参考链接

- 仓库：https://github.com/google-deepmind/tapnet  
- Checkpoint 汇总：[HuggingFace google/tapnet](https://huggingface.co/google/tapnet)  
- TAPNext++ Colab：[torch_tapnextpp_demo.ipynb](https://colab.research.google.com/github/deepmind/tapnet/blob/main/colabs/torch_tapnextpp_demo.ipynb)
