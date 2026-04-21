# LoMa 项目概述

**论文**：[LoMa: Local Feature Matching Revisited (arXiv:2604.04931)](https://arxiv.org/abs/2604.04931)  
**机构**：Chalmers / Linköping / Amsterdam / Lund 大学联合

## 定位

LoMa 是一个**局部特征匹配**模型（Local Feature Matcher），定位与 LightGlue 相同（Detect → Describe → Match 三阶段），但在多个基准上优于 LightGlue 和 RoMa/RoMaV2，尤其在困难匹配场景（WxBS benchmark）上表现突出。可直接作为 SfM / Visual Localization 流水线中 LightGlue 的替代品。

---

## 接口

### 最简用法

```python
from loma import LoMa, LoMaB
import cv2

model = LoMa(LoMaB())
kptsA, kptsB = model.match("img_A.jpg", "img_B.jpg")
# kptsA, kptsB: np.ndarray, 像素坐标，形状 [N, 2]

F, mask = cv2.findFundamentalMat(kptsA, kptsB, method=cv2.USAC_MAGSAC, ...)
```

### 分步用法（demo.py 流程）

```python
model = LoMa(LoMaB())

# Step 1: 检测 + 描述（冻结的 detector + descriptor，各自独立）
kpts_A, desc_A, h1, w1 = model.detect_and_describe("img_A.jpg")
kpts_B, desc_B, h2, w2 = model.detect_and_describe("img_B.jpg")

# Step 2: Transformer 匹配打分
with torch.inference_mode():
    scores = model(kpts_A, kpts_B, desc_A, desc_B)["scores"]  # [1, N, M]

# Step 3: 过滤 mutual best matches
m0, *_ = filter_matches(scores, threshold=0.1)
valid = m0[0] > -1
matched_A = to_pixel_coords(kpts_A[0][valid], h1, w1)       # [K, 2]
matched_B = to_pixel_coords(kpts_B[0][m0[0][valid]], h2, w2) # [K, 2]
```

---

## 模型架构

```
输入图像 A / B
    ↓  DaD（冻结，不参与训练）
关键点坐标 kpts [1, N, 2]（归一化，范围 [-1, 1]）
    ↓  DeDoDeDescriptor（冻结，不参与训练）
局部描述子 desc [1, N, 256]
    ↓  input_proj（线性层，可选维度对齐）
    ↓  LearnableFourierPositionalEncoding（2D 位置编码）→ encoding [2, 1, N, head_dim]
    ↓  N 层 TransformerLayer（Self-Attn + Cross-Attn 交替，这部分是可训练的）
        SelfBlock:  desc_A 内部自注意力（RoPE 位置编码）
        CrossBlock: desc_A ↔ desc_B 跨图像交叉注意力
    ↓  MatchAssignment（最后一层）
        scores = softmax(sim, dim=2) * softmax(sim, dim=1)  # dual-softmax
        → [1, N, M]
    ↓  filter_matches（互相最优 + 阈值过滤）
输出匹配点对 (kptsA, kptsB)
```

**关键点**：Detector（DaD）和 Descriptor（DeDoDeB/G）均为**冻结的预训练模型**，只有 Transformer matcher 部分（`transformers` + `log_assignment`）参与训练。

---

## 模型变体

| 变体 | embed_dim | heads | 描述子 | 说明 |
|------|-----------|-------|--------|------|
| **LoMa-B**（推荐） | 256 | 4 | DeDoDeG-256 | 与 LightGlue 同等体量，通用首选 |
| LoMa-B128 | 256 | 4 | DeDoDeB-128 | 更轻量（128 维描述子） |
| LoMa-L | 512 | 8 | DeDoDeG-256 | 更大容量 |
| LoMa-G | 1024 | 16 | DeDoDeG-256 | 最高精度，超越 RoMa 家族 |
| **LoMa-R** | 256 | 4 | DeDoDeG-256 | 旋转不变，适合航拍/卫星图像 |

权重从 GitHub Releases 自动下载（`torch.hub.load_state_dict_from_url`）。

---

## 与 RoMa 的对比

| 特性 | LoMa | RoMa / RoMaV2 |
|------|------|----------------|
| 匹配范式 | 局部特征（Detect→Describe→Match） | 稠密匹配（Dense Flow） |
| 速度 | 快（类 LightGlue） | 较慢 |
| WxBS 精度 | 超过 RoMa 家族 | 次之 |
| SfM 接入 | 直接替换 LightGlue | 需要稠密匹配接口 |
| 旋转不变 | LoMa-R 支持 | 不支持 |

---

## 外部集成

- **HLoc**：[davnords/Hierarchical-Localization fork](https://github.com/davnords/Hierarchical-Localization)
- **vismatch**：[PR #63](https://github.com/gmberton/vismatch/pull/63)
- 可直接替换 GGPT 等流水线中的 RoMa 匹配器
