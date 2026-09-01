# INSV 双鱼眼 9-camera Rig：进展记录与实施计划

> 当前可执行代码、两份 prepare manifest、ERP Cubemap5 基线和 Fish9
> 未实现清单的权威交接入口是
> [DUAL_FISHEYE_9CAM_HANDOFF.md](DUAL_FISHEYE_9CAM_HANDOFF.md)。本文保留为
> INSV 字段调查和历史设计记录；若状态冲突，以新交接文档为准。

更新日期：2026-08-31

本文记录从原始 INSV 双鱼眼视频生成无拼接、去畸变的 pinhole 输入，并接入完整 FFBA/BAE rig 流程的当前结论。为避免把猜测写成事实，文中使用以下状态：

- **已验证**：已经直接从当前 INSV 样本、代码或运行结果中确认。
- **推导**：由现有标定和几何关系计算得到，仍需通过图像实验验证坐标约定。
- **计划**：尚未实现或尚未完成测试。

当前样本：`/kiri/dataset/insv_data/VID_20260824_182629_00_004.insv`

## 1. 当前目标与设计决定

目标是跑通以下完整流程：

1. 从原始 INSV 的两个同步鱼眼视频流直接生成 pinhole 图像，不经过 ERP 拼接。
2. 每个时间戳生成 9 个虚拟相机输入：
   - 前鱼眼四个对角视角；
   - 后鱼眼四个对角视角；
   - 前鱼眼额外一个 center 视角。
3. 只用 `front_center` 进行 Pi3X/MERG3R 前馈重建。
4. 补入其余 8 个视角，执行 SIFT、VGGSfM、三角化和筛选。
5. BAE 每帧只优化一个 rig pose，并用固定的 9-camera rig 外参约束全部观测。

这套设计的核心不是“9 个独立运动的相机”，而是：

- 每帧有 **1 个 rig pose**；
- 有 **2 个物理光心**，分别对应前、后鱼眼镜头；
- 有 **9 个固定虚拟视角**；
- 同一鱼眼生成的多个虚拟视角共享同一个物理光心，只具有不同固定旋转；
- 前、后两组虚拟视角之间保留双鱼眼的真实基线。

## 2. 当前 INSV 中已经获得的数据

### 2.1 视频和音频流

**已验证：**

- 文件大小：2,783,891,804 bytes。
- 容器：MP4/QuickTime，包含 Insta360 私有尾部数据。
- 设备型号：Insta360 X5。
- 固件：`v1.11.10_build1`。
- 视频流 0：HEVC/H.265，3840×3840，约 29.97 fps，3513 帧。
- 视频流 1：HEVC/H.265，3840×3840，约 29.97 fps，3513 帧。
- 两个视频流的 PTS、DTS 和 duration 序列完全一致，可进行逐帧一一对应。
- 视频时长约 117.217 秒。
- 音频流：AAC LC，48 kHz，4 声道。
- 画面检查确认两个视频流是相反朝向的独立圆形鱼眼画面，并在圆周附近具有重叠区域。

**尚未完全确认：**

- 容器没有提供明确的 `front`/`back` 语义标签。当前把 stream 0 作为前镜头、stream 1 作为后镜头是合理候选，但在正式固化命名之前，必须用一个已知方向的地标或官方导出的 ERP 做一次方向核验。

### 2.2 INSV 私有数据区

**已验证：**该文件含有 INSV v3 私有 trailer，可解析到 31 个记录入口。对当前工作直接有用的记录包括：

| 记录 | 当前识别结果 | 数量或大小 |
| --- | --- | ---: |
| metadata | protobuf/相机元数据 | 4004 bytes |
| thumbnail | H.264 缩略视频 | 1,228,840 bytes |
| gyro + accel | 原始 IMU | 2,346,240 bytes |
| exposure | 曝光记录 | 56,336 bytes |
| AAA | 自动曝光等统计 | 168,672 bytes |
| anchors | 时间或数据锚点 | 35 bytes |
| AAA simulation | 私有模拟记录 | 938,238 bytes |

X5 还包含若干尚未解释的较新记录（例如 id 22、28、29）。第一版投影和 rig 实现不依赖它们，但准备脚本不得破坏或错误改写源文件。

当前样本没有可用 GPS 记录，metadata 中的 GPS 字段也为零。

### 2.3 图像、时间和传感器元数据

**已验证：**

- 编码图像：3840×3840。
- 标定/处理链中的尺寸：原始 5376×5376，目标裁剪 5312×5312，最终编码 3840×3840。
- 帧率字段：30。
- gamma：`standard`。
- 支持 raw gyro，IMU 中包含陀螺仪和加速度计。
- IMU 约 117,312 条，时间跨度约 117.462 秒，中位采样间隔 1000 μs，约 1 kHz。
- X5 IMU 轴排列在现有开源解析器中标记为 `yzX`，仍需在运动补偿前核验其坐标系定义。
- rolling-shutter 字段值为约 21.244；它很可能以毫秒为单位，但在用于逐行姿态补偿前需要用官方定义或实验确认。
- exposure 约 3521 条，中位间隔约 33.368 ms。
- AAA 约 3514 条，当前样本 ISO 约 100–101。

这些信息使后续可以把每个导出帧和原始视频 PTS、曝光及 IMU 时间关联起来，而不是只按解码顺序猜测时间。

### 2.4 双鱼眼内参、畸变和相对外参

**已验证：**metadata 中包含 `offset`、`offset_v2` 和 `offset_v3` 标定串；原始值与当前值一致。`offset_v3` 的每个镜头记录可解释为：

```text
xi, fx, fy, cx, cy,
yaw, pitch, roll,
tx, ty, tz,
k1, k2, k3, p1, p2,
width, height, lens_type
```

当前样本的主要标定值为：

| 字段 | 镜头 0 | 镜头 1 |
| --- | ---: | ---: |
| xi | 2.000000 | 2.000000 |
| fx | 4266.770 | 4262.700 |
| fy | 4268.000 | 4263.370 |
| cx（组合标定画布） | 2691.560 | 8074.190 |
| cy | 2704.980 | 2700.330 |
| yaw | -0.041° | -0.070° |
| pitch | 0.079° | 0.091° |
| roll | 90.821° | 88.608° |
| tx | 0 | 0.000688 |
| ty | 0 | 0.000183 |
| tz | 0 | -0.032500 |
| k1 | 0.17725553 | 0.19437833 |
| k2 | 2.05762434 | 2.01216817 |
| k3 | -3.17907715 | -3.04443812 |
| p1 | 0.00043737 | 0.00073542 |
| p2 | -0.00108740 | 0.00092976 |

**推导：**

- 标定使用 10752×5376 的双镜头组合画布；镜头 1 的局部 `cx` 需要减去 5376。
- 将 5376→5312→3840 的裁剪和缩放纳入后，编码视频上的近似统一模型参数为：

| 字段 | stream 0 | stream 1 |
| --- | ---: | ---: |
| fx | 3084.4120 | 3081.4699 |
| fy | 3085.3012 | 3081.9542 |
| cx | 1922.5429 | 1927.2786 |
| cy | 1932.1286 | 1928.8071 |

- 镜头 1 相对镜头 0 的平移向量范数约为 0.03251。若标定单位为米，则前后镜头物理基线约为 32.5 mm。单位和左右乘坐标约定仍需通过 SDK 或重投影实验确认。

注意：上表中的 `K` 属于 Insta360 的统一鱼眼模型参数，不能直接当作普通 pinhole 内参使用。

## 3. 从鱼眼直接生成真正的 pinhole 图像

### 3.1 目标定义

每个虚拟相机都明确保存：

- 固定输出宽高；
- pinhole 内参 `K_out`；
- 水平和垂直 FOV；
- 相对其物理鱼眼镜头的固定旋转；
- 相对 rig 的固定旋转和平移；
- 输出畸变参数为零；
- 有效像素 mask。

只要一张输出图的所有像素都来自同一个鱼眼光心，它就可以被建模为一个标准中央投影 pinhole 相机。它不是“视觉上看起来较直”的图片，而是每个像素都有明确 pinhole 射线的图像。

### 3.2 推荐的逆向重映射

第一版应从目标 pinhole 像素反查原鱼眼像素，而不是先把整个鱼眼展开后再裁图：

1. 对输出像素 `(u, v)` 用 `K_out^-1` 得到虚拟相机单位射线：

   ```text
   d_view = normalize(K_out^-1 [u, v, 1]^T)
   ```

2. 用固定旋转将射线变换到对应的物理鱼眼镜头坐标系：

   ```text
   d_lens = R_lens_from_view · d_view
   ```

3. 使用 `xi + k1/k2/k3 + p1/p2` 的 Insta360 统一鱼眼模型，把 `d_lens` 投影到标定画布坐标。
4. 应用组合画布拆分、5376→5312 裁剪以及 5312→3840 缩放，得到编码视频坐标。
5. 对原始鱼眼帧做 bilinear 或 bicubic 采样。
6. 同时生成有效性 mask：要求映射点位于对应镜头的标定有效区，并和鱼眼圆周保留足够安全边距。

这一方式每个输出只进行一次必要的图像重采样，也不会在两个镜头之间混合像素。

### 3.3 “去畸变”能保证什么

正确完成上述映射后：

- 输出几何模型是普通 pinhole，标定畸变为零；
- 直线在理想静态场景中应映射为直线；
- SIFT、VGGSfM 和 BA 可以直接使用固定 `K_out`；
- 所有像素都可以追溯到一个物理镜头和一个明确射线。

但它不能消除：

- 原始标定误差；
- 插值、降采样和混叠；
- 色差、炫光和鱼眼边缘成像质量下降；
- rolling shutter、运动模糊和曝光差异；
- 两个物理镜头之间的视差。

特别地，把前后鱼眼像素混合到同一张图中，即使画面没有弯曲，也不能被严格表示成单光心 pinhole。当前 9-camera 设计正是通过“每个视图只取一个鱼眼”来避免这个问题。

## 4. 双鱼眼 9-camera 设计

### 4.1 虚拟相机列表

建议的稳定命名为：

| 虚拟相机 | 来源 | 角色 | 物理光心 |
| --- | --- | --- | --- |
| `front_center` | 前鱼眼 | 唯一前馈输入 | front |
| `front_ul` | 前鱼眼 | 补充/BA | front |
| `front_ur` | 前鱼眼 | 补充/BA | front |
| `front_dl` | 前鱼眼 | 补充/BA | front |
| `front_dr` | 前鱼眼 | 补充/BA | front |
| `back_ul` | 后鱼眼 | 补充/BA | back |
| `back_ur` | 后鱼眼 | 补充/BA | back |
| `back_dl` | 后鱼眼 | 补充/BA | back |
| `back_dr` | 后鱼眼 | 补充/BA | back |

`ul/ur/dl/dr` 应以最终 rig 坐标系中的屏幕方向定义，而不能简单照搬原始鱼眼画面的像素象限；原始视频可能带有镜头旋转或镜像。

### 4.2 与“正八面体”的关系

用户提出的几何直觉是成立的，但需要准确描述：

- 如果 8 个对角视轴取为 `(±1, ±1, ±1) / sqrt(3)`，它们在方向上是立方体的 8 个顶点；
- 同一组方向也正好是正八面体 8 个三角面的外法线；
- 因此可以称为“**正八面体面法向式 8-view 外壳**”；
- 再额外加入前镜头 `front_center = (0, 0, +1)`，完整的 9-view 集合本身不再是一个正多面体，而是“8-view 外壳 + 前馈中心参考视角”。

精确的正八面体面法向方案具有：

- 每个对角视轴相对前/后主轴偏转约 `54.7356°`；
- 相邻对角视轴夹角约 `70.5288°`。

需要注意，90° 方形 pinhole 图像从中心到角点的最大射线角也正好约为 `54.7356°`。如果视轴本身已经偏离鱼眼主轴 `54.7356°`，再使用 90° 输出，最外角射线通常会超出单个鱼眼的可靠覆盖范围。因此“视轴正八面体对称”和“每面 90°”不能同时直接假设成立。

### 4.3 两个待比较的视角方案

在实现时保留两个候选，不提前用主观观感决定：

#### A. 精确正八面体面法向

- 8 个对角轴严格取 `(±1, ±1, ±1) / sqrt(3)`。
- 初始输出 FOV 预计只能取约 65°–70°，具体由鱼眼有效 mask 决定。
- 优点：方向分布均匀，球面几何清晰。
- 风险：如果 FOV 过窄，视图间重叠和球面覆盖可能不足。

#### B. 内缩对角轴

- 每个鱼眼的四个对角轴在其切平面上取约 `(±tan(30°), ±tan(30°), 1)` 后归一化。
- 视轴相对鱼眼主轴约偏转 39.23°，可尝试更大的输出 FOV，例如 90°。
- 优点：更容易远离鱼眼圆周，保留较好的单镜头图像质量和视图重叠。
- 风险：不再是精确的正八面体对称，球面采样更集中在前后主轴附近。

最终选择依据应包括：有效像素比例、距鱼眼圆周的最小边距、球面覆盖率、视图重叠图、SIFT 密度和几何内点，而不是只看单张图是否“平”。

### 4.4 Rig 位姿模型

统一使用与现有代码一致的 `sensor_from_world` 方向。每个虚拟相机的位姿应写成：

```text
sensor_from_world(frame, sensor)
    = sensor_from_rig(sensor) · rig_from_world(frame)
```

其中：

```text
sensor_from_rig
    = view_from_lens · lens_from_rig
```

- `rig_from_world(frame)`：每一帧唯一的待优化 rig pose。
- `lens_from_rig`：前/后物理镜头相对 rig 的固定外参。
- `view_from_lens`：虚拟 pinhole 相对来源鱼眼的固定纯旋转。
- 前鱼眼的 5 个虚拟相机具有相同平移。
- 后鱼眼的 4 个虚拟相机具有相同平移。
- 前后两组保留标定给出的固定相对平移和旋转。

BAE 不应该为 9 张图分别优化 9 个独立 pose；它只优化一个 frame pose，再通过固定 `sensor_from_rig` 产生 9 个相机位姿。

## 5. 输入数据和 manifest 建议

建议的准备结果结构：

```text
dataset/
  front_center/
  front_ul/
  front_ur/
  front_dl/
  front_dr/
  back_ul/
  back_ur/
  back_dl/
  back_dr/
  masks/
    front_center.png
    front_ul.png
    ...
  dual_fisheye_rig_manifest.json
```

如果视频分辨率和映射参数不变，每个 sensor 的有效 mask 是静态的，只需保存一次。

manifest 至少应记录：

- 源 INSV 路径、设备型号和标定串摘要；
- stream index 到 `front/back` 的映射及其验证状态；
- 原始 frame index、PTS 和时间戳；
- 9 个 sensor 的固定顺序、名称和角色；
- 每个 sensor 的来源物理镜头；
- 输出宽高、`K_out`、FOV、零畸变声明；
- `view_from_lens`、`lens_from_rig` 和最终 `sensor_from_rig`；
- 坐标系、四元数/旋转矩阵和左右乘约定；
- 投影模型和准备脚本版本；
- mask 路径、有效像素比例、最小鱼眼边缘余量；
- 帧过滤规则以及 9 张图的完整分组关系。

## 6. 对当前 FFBA 流程的影响

### 6.1 前馈和帧分组

- Stage A 只读取 manifest 中 `role=feed_forward_reference` 的 `front_center`。
- center 过滤时必须按时间戳保留或丢弃整个 9-image group。
- 前馈深度只属于 `front_center`，不能直接复制给其他旋转视角或后镜头视角。

### 6.2 SIFT / VGGSfM pair 与 group

当前 Cubemap5 的 5 个面被建模为同一光心，因此同帧面间 pair 不能提供三角化基线。新的 9-camera rig 需要按物理镜头区分：

- 同一时间、同一物理鱼眼来源的虚拟相机之间：光心相同，不应作为三角化 pair；
- 同一时间、不同物理鱼眼来源的虚拟相机之间：存在约 32.5 mm 候选基线，若视锥有足够重叠，可以作为近距离 stereo pair；
- 跨时间 pair：继续结合中心轨迹邻近、时间邻近、视轴夹角和视锥重叠选择；
- VGGSfM group 可以覆盖全部 9 个 sensor，但 group 构造需要避免只按目录名硬编码 center/left/right/up/down。

前后鱼眼的重叠主要位于两个镜头的交界带。四对角视轴及其 roll 应兼顾跨镜头 overlap，使 front/back 两组特征图真正连通。

### 6.3 BAE rig 约束和审计

现有 BAE “每帧一个 pose block + 固定 sensor extrinsics” 的核心模型可以扩展到任意数量的 sensor。需要将当前输入层和审计中针对 3/5 个固定面名称的逻辑改为 manifest 驱动。

9-camera 审计至少检查：

- 同帧前组 5 个虚拟相机的 camera center 完全一致；
- 同帧后组 4 个虚拟相机的 camera center 完全一致；
- 每帧前后 camera center 的距离和方向符合固定 rig baseline；
- 9 个 sensor 相对 rig 的旋转不随帧漂移；
- 每帧只存在一个被优化的 rig pose；
- COLMAP/BAE 导入导出后，9 个相机仍保持上述固定关系。

## 7. 已有 ERP Cubemap5 基线

在切换到原始双鱼眼之前，已有一组可用于 A/B 对比的 150 帧 ERP Cubemap5 FFBA 结果：

- 输入：`/kiri/dataset/local_test_cubemap5_150`
- 输出：`/kiri/dataset/local_test_cubemap5_ffba_150`
- 日志：`/kiri/dataset/ffba_logs/local_test_cubemap5_ffba_150.log`
- 150 个时间帧、750 张图，全部注册。
- 最终约 400,373 个点、1,804,506 个观测。
- S-only 点约 227,253，P-only 点约 173,120。
- 最终角度误差：
  - S：median 约 0.0729°，p90 约 0.3724°；
  - P：median 约 0.0907°，p90 约 0.3030°；
  - 全部低于 2°。
- rig center 最大 spread 约 `1.61e-14`。
- rig orientation 最大误差约 `2.41e-6°`。
- `up` 只有 128/150 帧包含最终观测，其余面为 150/150。

这组结果是当前功能正确性基线。双鱼眼 9-camera 方案除了比较最终点数，还要重点比较各 sensor 的有效区域、匹配连通性和交界方向的误差。

## 8. 接下来的实施计划

### Phase 1：固化 INSV 只读探测与双流解码

- 实现只读 INSV metadata/trailer probe，输出 JSON，不重写源文件。
- 记录视频流参数、逐帧 PTS、标定串、IMU/曝光可用性和未知记录目录。
- 同步解码两个 HEVC 流，并验证选取的每一帧 PTS 一致。
- 用已知地标或官方 ERP 核验 stream 0/1 的前后语义和图像旋转。

交付：一个样本 probe JSON、同步双鱼眼帧和方向核验记录。

### Phase 2：单帧鱼眼到 pinhole 映射

- 实现 Insta360 unified/fisheye 投影及其坐标缩放链。
- 采用逆向映射，一次采样生成 pinhole 图和静态 mask。
- 首先只生成 `front_center`，验证中心、四角射线和直线保持。
- 用 raw→pinhole→ray 的数值 round-trip 检查投影误差。
- 与官方 ERP 或现有 ERP cubemap 中对应方向做几何对照，但不把其拼接像素当作真值。

交付：可复用的 remap、mask、固定 `K_out` 和投影单元测试。

### Phase 3：9-camera 几何搜索

- 同时生成“精确正八面体面法向”和“内缩对角轴”两组候选。
- 网格搜索轴偏角、输出 FOV 和安全边距。
- 计算：
  - 每面有效像素比例；
  - 最小鱼眼边缘距离；
  - 球面覆盖率；
  - 任意两视图的视锥重叠；
  - front/back 跨物理镜头连接图；
  - 单位球面角分辨率。
- 再用少量真实帧比较 SIFT 数量、匹配内点和图像边缘质量。

交付：确定最终 8 个对角轴、每面 FOV、roll 和输出分辨率。

### Phase 4：数据准备脚本

计划新增类似：

```text
scripts/prepare_dual_fisheye_rig_from_insv.py
```

第一轮生成 20 或 50 帧，确认目录、manifest、mask 和前后镜头语义；通过后再生成与 ERP 基线相同时间戳的 150 帧数据。

交付：可重现的 INSV→dual-fisheye-9cam 数据集。

### Phase 5：FFBA manifest 化和 9-camera rig

- 将 `utils/pano_rig.py` 和 pipeline runner 中固定 Cubemap5 面名改为 manifest 驱动。
- Stage A 只处理 `front_center`。
- Stage B 加入其余 8 个 sensor。
- pair/group 选择改为根据物理光心、实际视轴、FOV 和时间关系判断。
- 建立 9 个固定 `sensor_from_rig`，但每帧只有一个 pose block。
- 将过滤、注册、导入导出和审计全部按 9-image group 处理。

交付：9-camera 的完整 SIFT + VGGSfM + BAE rig 流程。

### Phase 6：测试和 A/B 评估

单元/集成测试包括：

- INSV trailer 和标定字段解析 fixture；
- 两视频流 PTS 对齐和抽帧；
- 投影中心/边角、有效 mask 和射线 round-trip；
- 9 个 `sensor_from_rig`、两个物理光心和 baseline；
- 同帧同光心/异光心 pair 规则；
- 9-camera COLMAP round-trip；
- BAE shared-pose 和固定外参审计。

在相同时间戳上比较 ERP Cubemap5 与 raw dual-fisheye 9cam：

- 每个 sensor 的有效像素比例和 SIFT 密度；
- pair 匹配数、几何内点率和跨镜头连接性；
- track 长度、三角化角和最终观测数；
- 每个 sensor 的注册帧数及观测覆盖；
- BA loss、重投影/角度误差；
- rig 固定外参误差；
- 鱼眼交界方向和 ERP seam 附近的局部质量；
- 必要时继续比较下游 Gaussian Splatting 质量。

## 9. 初版验收标准

- 每张输出图只从一个物理鱼眼采样，不发生前后镜头像素混合。
- 所有输出像素有效，或由 mask 明确标记为无效；不得静默填黑后当作正常特征区。
- 每个 sensor 的 pinhole `K_out` 固定且 distortion 为零。
- 每个观测明确属于 front 或 back 物理光心。
- 每个导出帧记录原始 stream、frame index 和 PTS。
- 150 帧全部注册，或对未注册帧给出可解释的过滤原因。
- 不存在长期零观测的 sensor；若某面低于阈值，几何/FOV 必须重新调整。
- BAE 后相对旋转、同组光心一致性和前后 baseline 保持在数值精度范围。
- 相比 ERP Cubemap5，raw 9cam 至少不能显著退化 center 重建和整体 BA；预期优势应体现在无 seam、交界方向几何一致性及 front/back 真实基线上。

## 10. 当前风险和待确认项

1. stream 0/1 的 front/back 命名和画面朝向尚需一次实景核验。
2. `offset_v3` 的坐标轴、Euler 顺序、平移单位和左右乘约定需要通过数值重投影验证，不能只凭字段名决定。
3. X5 的较新私有记录尚未全部解释；第一版不依赖它们，但若官方结果和当前投影存在系统误差，需要继续解析。
4. 精确正八面体方向可能要求较窄 FOV；最终几何需由有效 mask 和匹配连通性共同决定。
5. 32.5 mm 左右的前后基线对远景三角化能力有限，但仍是必须正确保留的物理约束。
6. rolling shutter 和 IMU 补偿先不进入首版，否则会把投影模型验证与时域补偿耦合；在静态几何正确后再单独评估。

## 11. 参考实现和资料

- 当前五面 rig 设计与运行记录：[`PANO_RIG_5FACE.md`](PANO_RIG_5FACE.md)
- 历史三面 handoff：[`PANO_RIG_HANDOFF.md`](PANO_RIG_HANDOFF.md)
- [Insta360 Desktop SDK / MediaSDK 文档](https://github.com/Insta360Develop/Insta360-Developer_Docs/blob/main/docs/en/x/desktop/guide/index.md)
- [telemetry-parser 项目及 X5 支持说明](https://github.com/AdrianEddy/telemetry-parser/blob/master/README.md)
- [telemetry-parser 的 Insta360 解析和 offset_v3 字段实现](https://github.com/AdrianEddy/telemetry-parser/blob/master/src/insta360/mod.rs)
- [INSV record 类型定义](https://github.com/AdrianEddy/telemetry-parser/blob/master/src/insta360/record.rs)

