# COLMAP Bundle Adjustment Gauge Fix

本文档说明 `BundleAdjustmentConfig::FixGauge()` 与 `BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD` 的设计动机、调用链、实现细节与边界情况，供开发者或其他 AI 快速理解 COLMAP / pycolmap 中的 gauge fixing 逻辑。

---

## 1. 背景：什么是 Gauge Freedom

Structure-from-Motion 的 Bundle Adjustment（BA）在重投影误差意义下存在 **7 个 gauge 自由度**：

| 自由度 | 含义 |
|--------|------|
| 3 | 整体旋转 |
| 3 | 整体平移 |
| 1 | 整体尺度 |

若不施加额外约束，BA 问题在数学上是欠定的：整体刚体变换 + 统一缩放不会改变任何重投影误差，优化器可能发散或数值不稳定。

COLMAP 通过 `BundleAdjustmentGauge` 枚举提供两种显式 gauge fix 策略：

```cpp
// src/colmap/estimators/bundle_adjustment.h
MAKE_ENUM_CLASS_OVERLOAD_STREAM(
    BundleAdjustmentGauge, -1, UNSPECIFIED, TWO_CAMS_FROM_WORLD, THREE_POINTS);
```

| 枚举值 | 整数值 | 策略 |
|--------|--------|------|
| `UNSPECIFIED` | -1 | 不主动 fix gauge |
| `TWO_CAMS_FROM_WORLD` | 0 | 固定两个 frame 的 `rig_from_world` 位姿（部分固定第二个） |
| `THREE_POINTS` | 1 | 固定 3 个不共线的 3D 点坐标 |

---

## 2. API 层：`fix_gauge` 本身做什么

### 2.1 C++

```cpp
// src/colmap/estimators/bundle_adjustment.h
class BundleAdjustmentConfig {
 public:
  void FixGauge(BundleAdjustmentGauge gauge);
  BundleAdjustmentGauge FixedGauge() const;
 private:
  BundleAdjustmentGauge fixed_gauge_ = BundleAdjustmentGauge::UNSPECIFIED;
};
```

```cpp
// src/colmap/estimators/bundle_adjustment.cc
void BundleAdjustmentConfig::FixGauge(BundleAdjustmentGauge gauge) {
  fixed_gauge_ = gauge;  // 仅写入配置，不修改 Reconstruction
}

BundleAdjustmentGauge BundleAdjustmentConfig::FixedGauge() const {
  return fixed_gauge_;
}
```

**关键点**：`FixGauge()` 只是设置一个配置标志，**不会立即修改** `Reconstruction` 中的相机位姿或 3D 点。真正的约束在 `BundleAdjuster::Solve()` 构建 Ceres/Caspar 优化问题时施加。

### 2.2 pycolmap 绑定

```cpp
// src/pycolmap/estimators/bundle_adjustment.cc
auto PyBundleAdjustmentGauge =
    py::enum_<BundleAdjustmentGauge>(m, "BundleAdjustmentGauge")
        .value("UNSPECIFIED", BundleAdjustmentGauge::UNSPECIFIED)
        .value("TWO_CAMS_FROM_WORLD", BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD)
        .value("THREE_POINTS", BundleAdjustmentGauge::THREE_POINTS);

PyBundleAdjustmentConfig.def(py::init<>())
    .def("fix_gauge", &BACfg::FixGauge)
    .def_property_readonly("fixed_gauge", &BACfg::FixedGauge);
```

Python 用法：

```python
import pycolmap

ba_config = pycolmap.BundleAdjustmentConfig()
ba_config.add_image(image_id)
ba_config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)

options = pycolmap.BundleAdjustmentOptions()
options.refine_rig_from_world = True  # 必须为 True，见 §5.1

summary = pycolmap.bundle_adjustment(reconstruction, ba_config, options)
```

---

## 3. 调用链（Ceres 后端，默认路径）

```
用户代码
  ba_config.FixGauge(TWO_CAMS_FROM_WORLD)
    ↓
CreateDefaultBundleAdjuster(options, ba_config, reconstruction)
    ↓
CeresBundleAdjuster::SetUpProblem()   // bundle_adjustment_ceres.cc
    ├─ AddImageToProblem()            // 添加重投影残差
    ├─ AddPointToProblem()
    ├─ ParameterizeCameras()
    ├─ ParameterizeRigsAndFrames()    // 设置 rig_from_world 参数化/manifold
    ├─ ParameterizePoints()
    └─ switch (config_.FixedGauge())
         case TWO_CAMS_FROM_WORLD:
           FixGaugeWithTwoCamsFromWorld(...)   // ← 核心实现
         case THREE_POINTS:
           FixGaugeWithThreePoints(...)
    ↓
ceres::Solve()
```

对应源码位置：

| 步骤 | 文件 | 行号（约） |
|------|------|-----------|
| gauge switch | `src/colmap/estimators/bundle_adjustment_ceres.cc` | 626–643 |
| TWO_CAMS 实现 | 同上 | 299–408 |
| THREE_POINTS 实现 | 同上 | 261–292 |
| rig 参数化 | 同上 | 453–526 |

---

## 4. COLMAP 内部使用场景

| 模块 | Gauge 策略 | 原因 |
|------|-----------|------|
| `global_mapper` 全局 BA | `TWO_CAMS_FROM_WORLD` | 多 frame 全局优化，固定两个 frame 更稳定 |
| `incremental_mapper` 全局 BA | `TWO_CAMS_FROM_WORLD` | 注释：比 THREE_POINTS 收敛更快更稳 |
| `incremental_mapper` 局部 BA | `THREE_POINTS` | 局部窗口内 3D 点丰富，固定点更简单 |
| `bundle_adjuster` CLI | `TWO_CAMS_FROM_WORLD` | 独立 BA 命令默认策略 |
| `python/examples/custom_bundle_adjustment.py` | `TWO_CAMS_FROM_WORLD` | 示例代码 |

全局 BA 示例（`global_mapper.cc`）：

```cpp
BundleAdjustmentConfig ba_config;
for (const auto& [image_id, image] : reconstruction.Images()) {
  if (image.HasPose()) {
    ba_config.AddImage(image_id);
  }
}
ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
auto ba = CreateDefaultBundleAdjuster(options, ba_config, reconstruction);
ba->Solve();
```

局部 BA 示例（`incremental_mapper.cc`）：

```cpp
ba_config.FixGauge(BundleAdjustmentGauge::THREE_POINTS);
// ... 添加 local bundle 中的 images
```

---

## 5. `TWO_CAMS_FROM_WORLD` 详细算法

函数签名：

```cpp
void FixGaugeWithTwoCamsFromWorld(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    const std::set<image_t>& image_ids,
    const std::unordered_map<point3D_t, size_t>& point3D_num_observations,
    Reconstruction& reconstruction,
    ceres::Problem& problem);
```

### 5.1 前置条件：必须优化 `rig_from_world`

```cpp
if (!options.refine_rig_from_world) {
  return;  // 所有 frame pose 已在 ParameterizeRigsAndFrames 中设为 constant
}
```

若 `refine_rig_from_world = false`，所有 `rig_from_world` 已在参数化阶段固定，gauge 天然确定，无需额外 fix。

### 5.2 坐标系与参数块

COLMAP 使用 **Rig/Frame** 模型：

- 每个 `Frame` 有一个 `rig_from_world`（7-DOF：`Rigid3d`，四元数 4 + 平移 3）
- 每个非 reference sensor 有 `sensor_from_rig`
- Reference sensor 的 `sensor_from_rig` 为单位变换，不参与优化

BA 中优化的位姿参数块是 `rig_from_world.params.data()`（7 维 double 数组）。

### 5.3 选择两个 camera（实际是两个不同 Frame）

目标：找到 `image1` 和 `image2`，分别来自**不同 Frame**，且 sensor 侧已"固定"。

**Sensor 已固定的判定**（`IsParameterizedConstSensor`）：

1. 该 image 是 frame 的 **reference sensor**；或
2. 其 `sensor_from_rig` 参数块在 problem 中已是 constant；或
3. `config.HasConstantSensorFromRigPose(sensor_id)`；或
4. `!options.refine_sensor_from_rig`

**选择流程**：

```
Step A: 遍历已 constant 的 rig_from_world
  → 若找到两个不同 Frame 的 image，直接 return（gauge 已足够固定）

Step B: 遍历 variable 的 rig_from_world
  → 第一个满足 IsParameterizedConstSensor 的 image → image1
  → 再找一个不同 Frame、满足 IsParameterizedConstSensor 的 image → image2
  → 计算两 Frame 基线向量 baseline = T1 * inv(T2) 的 translation
  → 取 baseline 绝对值最大分量对应的轴 → frame2_from_world_fixed_dim
  → 该轴用于后续只固定 frame2 的一个平移分量（约束尺度）
```

### 5.4 施加约束

**Frame 1（image1 所属 Frame）**：完全固定

```cpp
if (!config.HasConstantRigFromWorldPose(image1->FrameId())) {
  const Rigid3d& frame1_from_world = image1->FramePtr()->RigFromWorld();
  problem.SetParameterBlockConstant(frame1_from_world.params.data());
}
```

固定 7 个 DOF 中的 6 个（3 旋转 + 3 平移中的 2 个平移由 frame1 全固定 + frame2 部分固定共同完成；frame1 全固定提供 6 DOF，frame2 再固定 1 个平移轴提供尺度）。

**Frame 2（image2 所属 Frame）**：部分固定

```cpp
Rigid3d& frame2_from_world = image2->FramePtr()->RigFromWorld();
if (options.constant_rig_from_world_rotation) {
  // 旋转 + 一个平移轴全部固定（subset manifold）
  SetManifold(&problem, frame2_from_world.params.data(),
      CreateSubsetManifold(7, {0, 1, 2, 3, 4 + frame2_from_world_fixed_dim}));
} else {
  // 旋转用 quaternion manifold（自由），平移只固定 baseline 最大轴
  SetManifold(&problem, frame2_from_world.params.data(),
      CreateProductManifold(
          CreateEigenQuaternionManifold(),
          CreateSubsetManifold(3, {frame2_from_world_fixed_dim})));
}
```

**Gauge 自由度消除计数**：

| 约束 | 消除 DOF |
|------|---------|
| Frame1 `rig_from_world` 全 constant | 6（3 rot + 3 trans，四元数用 constant block 表示） |
| Frame2 固定 1 个平移轴 | 1（尺度） |
| **合计** | **7** |

### 5.5 失败回退

```cpp
if (image1 == nullptr || image2 == nullptr) {
  LOG(WARNING) << "Failed to fix Gauge with two cameras. "
                  "Falling back to fixing Gauge with three points.";
  FixGaugeWithThreePoints(point3D_num_observations, reconstruction, problem);
  return;
}
```

常见失败原因：

- BA 问题中只有 1 个 Frame
- 找不到 baseline 足够大的第二 Frame（`|baseline| < 1e-9`）
- 非 reference sensor 的 `sensor_from_rig` 未被固定，导致无法选出合法 image pair

源码注释也指出：当前实现**未完美处理**所有退化情况（如两相机缺乏共视约束、multi-camera rig 选 pair 策略较粗糙）。

---

## 6. 回退策略：`THREE_POINTS` 简述

当 `TWO_CAMS_FROM_WORLD` 失败时，或用户显式选择 `THREE_POINTS`：

```cpp
struct FixedGaugeWithThreePoints {
  Eigen::Index num_fixed_points = 0;
  Eigen::Matrix3d fixed_points = Eigen::Matrix3d::Zero();
  bool MaybeAddFixedPoint(const Eigen::Vector3d& point) {
    // 用 QR 秩检测确保 3 个点不共线
    fixed_points.col(num_fixed_points) = point;
    if (fixed_points.colPivHouseholderQr().rank() > num_fixed_points) {
      ++num_fixed_points;
      return true;
    }
    return false;
  }
};
```

逻辑：

1. 先检查 problem 中**已经 constant** 的 3D 点是否足够（≥3 且不共线）→ 满足则直接 return
2. 否则遍历 variable 3D 点，逐个 `SetParameterBlockConstant`，直到凑够 3 个线性无关点
3. 仍不足则 `LOG(WARNING)`

固定 3 个不共线 3D 点 = 固定 9 个坐标 = 消除 7 个 gauge DOF（冗余 2 个，但保证数值稳定）。

---

## 7. Caspar 后端的差异

若 `BundleAdjustmentBackend::CASPAR`，gauge fix 逻辑在 `bundle_adjustment_caspar.cc`：

```cpp
case BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD:
  FixGaugeWithOneFrameFromWorld();  // 注意：只固定一个 frame，不是两个
  break;
```

Caspar 的 `TWO_CAMS_FROM_WORLD` **仅固定一个 ref-sensor frame 的 `rig_from_world`**，尺度 DOF 故意保留（注释说明 Caspar 表达第二个 camera 的 1-DOF translation manifold 收益不大）。这与 Ceres 后端的完整两-frame fix **行为不同**。

默认 pycolmap / CLI 使用 Ceres 后端。

---

## 8. 与 `ParameterizeRigsAndFrames` 的关系

Gauge fix **发生在参数化之后**。`ParameterizeRigsAndFrames` 负责：

```cpp
// 对每个 frame 的 rig_from_world：
if (!options.refine_rig_from_world ||
    config.HasConstantRigFromWorldPose(image.FrameId())) {
  problem.SetParameterBlockConstant(rig_from_world.params.data());
} else if (options.constant_rig_from_world_rotation) {
  // 只优化平移，旋转固定
  SetManifold(..., CreateSubsetManifold(7, {0, 1, 2, 3}));
} else {
  // 旋转 quaternion manifold + 平移 euclidean manifold
  SetManifold(..., CreateProductManifold(
      CreateEigenQuaternionManifold(), CreateEuclideanManifold<3>()));
}
```

`FixGaugeWithTwoCamsFromWorld` 在此基础上进一步：

- 将某个 variable frame 改为 **full constant**（frame1）
- 或将某个 variable frame 的 manifold 改为 **更严格的 subset**（frame2）

也可通过 `ba_config.set_constant_rig_from_world_pose(frame_id)` 手动固定 frame，此时 gauge fix 会检测到"已有两个 frame constant"而提前 return。

---

## 9. 参数与配置速查

### BundleAdjustmentConfig

| 方法 | 作用 |
|------|------|
| `FixGauge(gauge)` | 设置 gauge 策略 |
| `FixedGauge()` | 读取当前策略 |
| `SetConstantRigFromWorldPose(frame_id)` | 手动固定 frame 位姿 |
| `SetConstantSensorFromRigPose(sensor_id)` | 手动固定 sensor 外参 |
| `AddImage(image_id)` | 必须先于 FixGauge 生效的 image 集合 |

### BundleAdjustmentOptions（影响 gauge fix）

| 选项 | 默认值 | 对 gauge fix 的影响 |
|------|--------|-------------------|
| `refine_rig_from_world` | true | false 时 TWO_CAMS 直接 skip |
| `refine_sensor_from_rig` | true | 影响 IsParameterizedConstSensor 判定 |
| `constant_rig_from_world_rotation` | false | 影响 frame2 是固定旋转还是只固定平移轴 |

---

## 10. 完整核心源码（Ceres 路径）

### 10.1 Gauge dispatch

```cpp
// src/colmap/estimators/bundle_adjustment_ceres.cc (SetUpProblem 末尾)
switch (config_.FixedGauge()) {
  case BundleAdjustmentGauge::UNSPECIFIED:
    break;
  case BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD:
    FixGaugeWithTwoCamsFromWorld(options_,
                                 config_,
                                 parameterized_image_ids_,
                                 point3D_num_observations_,
                                 reconstruction,
                                 *problem_);
    break;
  case BundleAdjustmentGauge::THREE_POINTS:
    FixGaugeWithThreePoints(
        point3D_num_observations_, reconstruction, *problem_);
    break;
  default:
    LOG(FATAL_THROW) << "Unknown BundleAdjustmentGauge";
}
```

### 10.2 FixGaugeWithTwoCamsFromWorld（精简版）

```cpp
void FixGaugeWithTwoCamsFromWorld(...) {
  if (!options.refine_rig_from_world) return;

  Image* image1 = nullptr;
  Image* image2 = nullptr;

  auto IsParameterizedConstSensor = [&](const Image& image) {
    const sensor_t sensor_id = image.CameraPtr()->SensorId();
    if (image.FramePtr()->RigPtr()->IsRefSensor(sensor_id)) return true;
    const Rigid3d& sensor_from_rig =
        image.FramePtr()->RigPtr()->SensorFromRig(sensor_id);
    if (problem.HasParameterBlock(sensor_from_rig.params.data()) &&
        problem.IsParameterBlockConstant(sensor_from_rig.params.data()))
      return true;
    if (config.HasConstantSensorFromRigPose(sensor_id) ||
        !options.refine_sensor_from_rig)
      return true;
    return false;
  };

  // 已有两个 constant frame → 无需 fix
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    if (config.HasConstantRigFromWorldPose(image.FrameId()) &&
        IsParameterizedConstSensor(image)) {
      if (image1 == nullptr) {
        image1 = &image;
      } else if (image1->FrameId() != image.FrameId()) {
        return;
      }
    }
  }

  // 选择 image1, image2
  int frame2_from_world_fixed_dim = 0;
  for (const image_t image_id : image_ids) {
    Image& image = reconstruction.Image(image_id);
    const Rigid3d& rig_from_world = image.FramePtr()->RigFromWorld();
    if (image1 == nullptr && IsParameterizedConstSensor(image)) {
      image1 = &image;
    } else if (image1 != nullptr && image1->FrameId() != image.FrameId() &&
               IsParameterizedConstSensor(image) &&
               problem.HasParameterBlock(rig_from_world.params.data())) {
      const Eigen::Vector3d baseline =
          (image1->FramePtr()->RigFromWorld() *
           Inverse(image.FramePtr()->RigFromWorld()))
              .translation();
      Eigen::Index max_coeff_idx = 0;
      if (baseline.cwiseAbs().maxCoeff(&max_coeff_idx) > 1e-9) {
        image2 = &image;
        frame2_from_world_fixed_dim = max_coeff_idx;
        break;
      }
    }
  }

  if (image1 == nullptr || image2 == nullptr) {
    FixGaugeWithThreePoints(point3D_num_observations, reconstruction, problem);
    return;
  }

  // 固定 frame1 全部参数
  if (!config.HasConstantRigFromWorldPose(image1->FrameId())) {
    problem.SetParameterBlockConstant(
        image1->FramePtr()->RigFromWorld().params.data());
  }

  // 部分固定 frame2
  if (!config.HasConstantRigFromWorldPose(image2->FrameId())) {
    Rigid3d& frame2_from_world = image2->FramePtr()->RigFromWorld();
    if (options.constant_rig_from_world_rotation) {
      SetManifold(&problem, frame2_from_world.params.data(),
          CreateSubsetManifold(7, {0, 1, 2, 3, 4 + frame2_from_world_fixed_dim}));
    } else {
      SetManifold(&problem, frame2_from_world.params.data(),
          CreateProductManifold(
              CreateEigenQuaternionManifold(),
              CreateSubsetManifold(3, {frame2_from_world_fixed_dim})));
    }
  }
}
```

---

## 11. 给 AI / 自动化脚本的要点摘要

1. **`fix_gauge` 是延迟生效的配置项**，只在 BA `Solve()` 建 problem 时读取。
2. **`TWO_CAMS_FROM_WORLD` 固定的是 Frame 级 `rig_from_world`**，不是单个 Camera 内参。
3. **需要 `refine_rig_from_world=True`**，否则该策略为空操作。
4. **Ceres 后端**：frame1 全固定 + frame2 固定一个平移轴 → 消除 7 DOF。
5. **失败自动 fallback 到 `THREE_POINTS`**（固定 3 个 3D 点）。
6. **单 Frame / 退化 baseline** 是常见失败场景。
7. **Caspar 后端的 `TWO_CAMS_FROM_WORLD` 语义不同**（只固定一个 frame）。
8. COLMAP 全局 BA 默认用 `TWO_CAMS_FROM_WORLD`，局部 BA 默认用 `THREE_POINTS`。

---

## 12. 相关源文件索引

| 文件 | 内容 |
|------|------|
| `src/colmap/estimators/bundle_adjustment.h` | 枚举、`BundleAdjustmentConfig` 声明 |
| `src/colmap/estimators/bundle_adjustment.cc` | `FixGauge()` 实现 |
| `src/colmap/estimators/bundle_adjustment_ceres.cc` | Ceres gauge fix 核心实现 |
| `src/colmap/estimators/bundle_adjustment_caspar.cc` | Caspar gauge fix（简化版） |
| `src/pycolmap/estimators/bundle_adjustment.cc` | Python 绑定 |
| `src/colmap/sfm/global_mapper.cc` | 全局 SfM BA 使用 TWO_CAMS |
| `src/colmap/sfm/incremental_mapper.cc` | 局部/全局 BA gauge 选择 |
| `src/colmap/controllers/bundle_adjustment.cc` | `colmap bundle_adjuster` CLI |
| `python/examples/custom_bundle_adjustment.py` | pycolmap 使用示例 |
| `src/colmap/estimators/bundle_adjustment_ceres_test.cc` | 单元测试（含 `FixGaugeWithTwoCamsFromWorld`） |

---

## 13. 测试参考

`bundle_adjustment_ceres_test.cc` 中：

- `FixGaugeWithTwoCamsFromWorld`：验证 `refine_rig_from_world` 开关与 effective parameters 变化
- `FixGaugeWithTwoCamsFromWorldFixSensorFromRig`：multi-camera rig 场景
- `FixGaugeWithTwoCamsFromWorldFallback`：两相机 fix 失败时 fallback 到 THREE_POINTS

运行：

```bash
cd build
ctest -R "bundle_adjustment_ceres_test" --output-on-failure
```
