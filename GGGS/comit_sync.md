# 提交 `f727413` 移植说明

## 提交信息

- Commit: `f727413db213e6fc1851e9ea61686a42dd589c28`
- 时间: `2026-03-16 16:08:08 +0800`
- Message: `fix: correct typos, fix a bug, and optimize backward pass`

## 这次提交实际修复了什么

1. 修复了 `train.py` 里的一个真实训练逻辑 bug：
   `if gaussians.max_sh_degree == gaussians.max_sh_degree` 这个条件恒为真，导致 SG degree 每 1000 iter 都会被解锁，而不是在 SH 全部解锁后再解锁。
2. 优化了 PatchMatch 的执行路径：
   现在可以分别控制 `geo` 和 `ncc`，只开其中一个时不会白白构建另一部分计算图，也可以直接跳过 NCC 计算。
3. 优化了 CUDA backward：
   把原来的逐通道 / 逐法线分量累计，改成了点积式累计，warp reduce 也更直接，目标是降低 backward 的开销。
4. 清理了 typo 和死代码：
   `get_apperance_embedding` 改成 `get_appearance_embedding`，删掉不再使用的 patch helper，并修正文档拼写。

## 涉及文件

- `train.py`
- `utils/loss_utils.py`
- `scene/gaussian_model.py`
- `utils/graphics_utils.py`
- `submodules/diff-gaussian-rasterization/cuda_rasterizer/render_backward.cu`
- `README.md`

## 建议移植顺序

1. 先一起移植 `scene/gaussian_model.py` 和 `utils/loss_utils.py`
原因：方法名改了，调用方也跟着变了。
2. 再移植 `train.py`
原因：它依赖新的 `PatchMatch(..., optimize_geo=..., optimize_ncc=...)` 接口。
3. 然后移植 `utils/graphics_utils.py`
原因：PatchMatch 重构后，里面那两个 helper 已经变成死代码。
4. 最后移植 `render_backward.cu`
原因：这是子模块 / 扩展代码，改完通常还要重新编译验证。
5. `README.md` 属于文档清理，可选移植。

---

## 1. `scene/gaussian_model.py`

### 改动目的

- 修正 appearance embedding 接口名的拼写错误。
- 这个改动本身很小，但所有调用方都要同步修改。

### 精确改动

```diff
-    def get_apperance_embedding(self, idx):
+    def get_appearance_embedding(self, idx):
         return self._appearance_embeddings[idx]
```

### 移植注意

- 在目标仓库里全局搜索 `get_apperance_embedding`，统一改名。
- 本次提交只改了 `utils/loss_utils.py` 里的一个调用点，但目标仓库可能不止一个。

---

## 2. `utils/loss_utils.py`

### 改动目的

- 同步使用新的 `get_appearance_embedding` 接口名。
- 重构 `PatchMatch`，支持 `geo` / `ncc` 分开控制。
- 当 `geo` 没开时，不再为几何分支保留梯度。
- 当 `ncc` 没开时，直接跳过后续 NCC 计算。
- 删除已经不再使用的 import。

### 2.1 删除无用 import

```diff
 from fused_ssim import fused_ssim
 from scene import GaussianModel, Camera
-from utils.graphics_utils import (
-    patch_offsets,
-    patch_warp,
-)
 from gaussian_renderer import sample_depth, render
 import warp_patch_ncc
```

### 2.2 更新 appearance embedding 调用

```diff
 def L1_loss_appearance(image, gt_image, gaussians, view_idx):
     app_model = gaussians.app_model
     if app_model is GaussianModel.App_model.NO:
         return l1_loss(image, gt_image)
-    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
+    appearance_embedding = gaussians.get_appearance_embedding(view_idx)
```

### 2.3 重构 `PatchMatch.__init__`

```diff
 class PatchMatch:
-    def __init__(self, patch_size, pixel_noise_th, kernel_size, pipe, debug=True, model_path=None):
+    def __init__(
+        self,
+        patch_size,
+        pixel_noise_th,
+        kernel_size,
+        pipe,
+        debug=True,
+        model_path=None,
+        optimize_geo=True,
+        optimize_ncc=True,
+    ):
         self.patch_size = patch_size
         self.total_patch_size = (patch_size * 2 + 1) ** 2
         self.pixel_noise_th = pixel_noise_th
-        self.offsets = patch_offsets(patch_size, device="cuda") * 0.5
-        self.offsets.requires_grad_(False)
         self.kernel_size = kernel_size
         self.pipe = pipe
         self.debug = debug
         self.model_path = model_path
+        self.optimize_geo = optimize_geo
+        self.optimize_ncc = optimize_ncc
         if debug:
             os.makedirs(os.path.join(model_path, "debug"), exist_ok=True)
```

### 2.4 重构 `PatchMatch.__call__`

这是本文件最关键的一段移植内容。

```diff
-    def __call__(self, gaussians: GaussianModel, render_pkg: dict, viewpoint_cam: Camera, nearest_cam: Camera, iteration=0, depth_normal=None):
+    def __call__(
+        self,
+        gaussians: GaussianModel,
+        render_pkg: dict,
+        viewpoint_cam: Camera,
+        nearest_cam: Camera,
+        iteration=0,
+        depth_normal=None,
+    ):
         if nearest_cam is None:
             return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")
         H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
         ## compute geometry consistency mask
         with torch.no_grad():
             ix = (torch.arange(W, device="cuda", dtype=torch.float32) - viewpoint_cam.Cx) / viewpoint_cam.Fx
             iy = (torch.arange(H, device="cuda", dtype=torch.float32) - viewpoint_cam.Cy) / viewpoint_cam.Fy
             view_to_nearest_T = (
                 -viewpoint_cam.world_view_transform[:3, :3].T @ nearest_cam.R @ nearest_cam.T + viewpoint_cam.world_view_transform[3, :3]
             )
             nearest_to_view_R = nearest_cam.R.transpose(1, 0) @ viewpoint_cam.world_view_transform[:3, :3]

-        # pts = (rays_d * render_pkg["median_depth"].squeeze().unsqueeze(-1)).reshape(-1, 3)
-        depth_reshape = render_pkg["median_depth"].squeeze().unsqueeze(-1)
-        pts = torch.cat([depth_reshape * ix[None, :, None], depth_reshape * iy[:, None, None], depth_reshape], dim=-1)
-
-        R = viewpoint_cam.R
-        T = viewpoint_cam.T
-        pts = (pts - T) @ R.T
-        sampled_pkg = sample_depth(
-            pts,
-            nearest_cam,
-            gaussians,
-            self.pipe,
-            self.kernel_size,
-        )
-
-        pts_in_nearest_cam = sampled_pkg["sampled_depth"]
-        R = nearest_cam.R
-        T = nearest_cam.T
-
-        pts_in_view_cam = view_to_nearest_T + pts_in_nearest_cam @ nearest_to_view_R
-        pts_projections = pts_in_view_cam[..., :2] / torch.clamp_min(pts_in_view_cam[..., 2:], 1e-7)
-        pts_projections = torch.addcmul(
-            pts_projections.new_tensor([viewpoint_cam.Cx, viewpoint_cam.Cy]),
-            pts_projections.new_tensor([viewpoint_cam.Fx, viewpoint_cam.Fy]),
-            pts_projections,
-        )
-
-        ix, iy = torch.meshgrid(
-            torch.arange(W, device="cuda", dtype=torch.int32),
-            torch.arange(H, device="cuda", dtype=torch.int32),
-            indexing="xy",
-        )
-        pixels = torch.stack([ix, iy], dim=-1)
-        pixel_f = pixels.type(torch.float32).requires_grad_(False)
-        pixel_noise = torch.pairwise_distance(pts_projections, pixel_f)
+        with torch.set_grad_enabled(self.optimize_geo):
+            depth_reshape = render_pkg["median_depth"].squeeze().unsqueeze(-1)
+            pts = torch.cat(
+                [
+                    depth_reshape * ix[None, :, None],
+                    depth_reshape * iy[:, None, None],
+                    depth_reshape,
+                ],
+                dim=-1,
+            )
+
+            R = viewpoint_cam.R
+            T = viewpoint_cam.T
+            pts = (pts - T) @ R.T
+            sampled_pkg = sample_depth(
+                pts,
+                nearest_cam,
+                gaussians,
+                self.pipe,
+                self.kernel_size,
+            )
+
+            pts_in_nearest_cam = sampled_pkg["sampled_depth"]
+            R = nearest_cam.R
+            T = nearest_cam.T
+
+            pts_in_view_cam = view_to_nearest_T + pts_in_nearest_cam @ nearest_to_view_R
+            pts_projections = pts_in_view_cam[..., :2] / torch.clamp_min(pts_in_view_cam[..., 2:], 1e-7)
+            pts_projections = torch.addcmul(
+                pts_projections.new_tensor([viewpoint_cam.Cx, viewpoint_cam.Cy]),
+                pts_projections.new_tensor([viewpoint_cam.Fx, viewpoint_cam.Fy]),
+                pts_projections,
+            )
+
+            ix, iy = torch.meshgrid(
+                torch.arange(W, device="cuda", dtype=torch.int32),
+                torch.arange(H, device="cuda", dtype=torch.int32),
+                indexing="xy",
+            )
+            pixels = torch.stack([ix, iy], dim=-1)
+            pixel_f = pixels.type(torch.float32).requires_grad_(False)
+            pixel_noise = torch.pairwise_distance(pts_projections, pixel_f)
```

### 2.5 debug 图像写盘只是格式调整

这段不是逻辑变更，只是改成了多行写法：

```diff
-                cv2.imwrite(os.path.join(self.model_path, "debug", "%05d" % iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)
+                cv2.imwrite(
+                    os.path.join(
+                        self.model_path,
+                        "debug",
+                        "%05d" % iteration + "_" + viewpoint_cam.image_name + ".jpg",
+                    ),
+                    image_to_show,
+                )
```

### 2.6 增加 `geo` / `ncc` 条件执行

```diff
         ################## Compute NCC for warped patches ##################
         if not d_mask.any():
             return torch.tensor([0], dtype=torch.float32, device="cuda"), torch.tensor([0], dtype=torch.float32, device="cuda")

-        geo_loss = ((weights * pixel_noise)[d_mask]).mean()
+        if self.optimize_geo:
+            geo_loss = (weights * pixel_noise)[d_mask].mean()
+        else:
+            geo_loss = torch.tensor([0], dtype=torch.float32, device="cuda")
+
+        if not self.optimize_ncc:
+            return torch.tensor([0], dtype=torch.float32, device="cuda"), geo_loss
+
         with torch.no_grad():
             d_mask = torch.flatten(d_mask)
             valid_indices = torch.argwhere(d_mask).squeeze(1)
             weights = torch.flatten(weights)[valid_indices]
```

### 移植注意

- `patch_offsets` / `patch_warp` 在这次重构之后已经不再使用。
- 真正重要的不是格式，而是这两个行为变化：
  - `with torch.set_grad_enabled(self.optimize_geo):`
  - `if not self.optimize_ncc: return zero_ncc, geo_loss`
- 如果目标仓库的 PatchMatch 实现不同，优先把这两个机制移植过去。

---

## 3. `train.py`

### 改动目的

- 始终初始化 `PatchMatch`，但内部通过 flag 控制实际是否优化。
- 修复 SH/SG 解锁逻辑 bug。
- 把多视图条件写得更显式。
- 略微简化 normal loss 的 mask 写法。

### 3.1 始终构造 `PatchMatch`，并传入优化开关

```diff
-    if opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo > 0:
-        patchmatch = PatchMatch(
-            opt.multi_view_patch_size,
-            opt.multi_view_pixel_noise_th,
-            kernel_size=kernel_size,
-            pipe=pipe,
-            debug=True,
-            model_path=dataset.model_path,
-        )
+    patchmatch = PatchMatch(
+        opt.multi_view_patch_size,
+        opt.multi_view_pixel_noise_th,
+        kernel_size=kernel_size,
+        pipe=pipe,
+        debug=True,
+        model_path=dataset.model_path,
+        optimize_geo=opt.lambda_multi_view_geo > 0,
+        optimize_ncc=opt.lambda_multi_view_ncc > 0,
+    )
```

### 3.2 修复 SH/SG 解锁 bug

这是本次提交里最重要的功能修复。

```diff
         # Every 1000 its we increase the levels of SH up to a maximum degree
         if iteration % 1000 == 0:
-            if gaussians.max_sh_degree == gaussians.max_sh_degree:
+            if gaussians.active_sh_degree == gaussians.max_sh_degree:
                 gaussians.unlockSGdegree(100)
             gaussians.oneupSHdegree()
```

### 为什么这段重要

- 旧逻辑里，条件恒为真。
- 新逻辑里，只有当 `active_sh_degree` 已经达到 `max_sh_degree` 时，才开始解锁 SG。
- 如果你时间有限，只想先移植最关键的功能修复，这一段优先级最高。

### 3.3 normal loss 的 mask 写法调整

```diff
             depth_normal, valid_points = depth_to_normal(viewpoint_cam, depth_map)
             normal_error_map = 1 - torch.linalg.vecdot(rendered_normal, depth_normal, dim=0)
-            depth_normal_loss = torch.where(valid_points.squeeze(), normal_error_map, torch.zeros_like(normal_error_map)).mean()
+            depth_normal_loss = normal_error_map.masked_fill_(~valid_points.squeeze(), 0.0).mean()
```

### 3.4 多视图条件写法更明确

```diff
         # patch match loss
-        if reg_kick_on and (opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo):
+        if reg_kick_on and (opt.lambda_multi_view_ncc > 0 or opt.lambda_multi_view_geo > 0):
             nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[sample(viewpoint_cam.nearest_id, 1)[0]]
             ncc_loss, geo_loss = patchmatch(gaussians, render_pkg, viewpoint_cam, nearest_cam, iteration, depth_normal)
```

### 移植注意

- 这份 `train.py` 依赖新的 `PatchMatch(..., optimize_geo=..., optimize_ncc=...)` 签名。
- 如果目标仓库仍然是“有需要时才构造 PatchMatch”，也可以保留，但要保证调用时一定已经初始化。

---

## 4. `utils/graphics_utils.py`

### 改动目的

- 删除 PatchMatch 重构后不再使用的辅助函数。

### 精确删除内容

```diff
-def patch_offsets(h_patch_size, device):
-    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device, dtype=torch.float32)
-    return torch.stack(torch.meshgrid(offsets, offsets, indexing="xy")[::-1], dim=-1).view(1, -1, 2)
-
-
-def patch_warp(H, uv):
-    B, P = uv.shape[:2]
-    H = H.view(B, 3, 3)
-    ones = torch.ones((B, P, 1), device=uv.device)
-    homo_uv = torch.cat((uv, ones), dim=-1)
-
-    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
-    grid_tmp = grid_tmp.reshape(B, P, 3)
-    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
-    return grid
```

### 移植注意

- 在目标仓库删除前，先全局搜索确认是否还有其他地方在用。
- 本次提交里它们对应的 import 也已经从 `utils/loss_utils.py` 删除了。

---

## 5. `submodules/diff-gaussian-rasterization/cuda_rasterizer/render_backward.cu`

### 改动目的

- 优化 backward pass 的累计逻辑。
- 让 `warpSum` 的数组重载更适合固定大小数组。
- 用点积累计替代逐通道 / 逐法线分量累计，降低寄存器和中间状态开销。

### 5.1 修改 `warpSum` 签名

```diff
 template <uint32_t DIM, class WarpT>
-__forceinline__ __device__ void warpSum(float* val, WarpT& warp) {
+__forceinline__ __device__ void warpSum(float (&val)[DIM], WarpT& warp) {
 #pragma unroll
     for (uint32_t i = 0; i < DIM; i++) {
         val[i] = cg::reduce(warp, val[i], cg::plus<float>());
     }
 }
```

### 5.2 替换颜色 / 法线累计状态变量

```diff
-    float accum_rec[C] = {0};
+    float accum_color_dot = 0;
     float dL_dpixel[C];
     float dL_dfinalT;
-    [[maybe_unused]] float accum_t_rec = 0;
     [[maybe_unused]] float dL_dpixel_t;
     [[maybe_unused]] float dL_dpixel_mt;
-    [[maybe_unused]] float accum_normal_rec[3] = {0};
+    [[maybe_unused]] float accum_normal_dot = 0;
     [[maybe_unused]] float dL_dpixel_normal[3];
```

### 5.3 将像素法线临时变量从 `glm::vec3` 换成普通数组

```diff
-            glm::vec3 dL_dpixel_normaln = glm::vec3(dL_dpixel_normals[pix_id],
-                                                    dL_dpixel_normals[H * W + pix_id],
-                                                    dL_dpixel_normals[2 * H * W + pix_id]);
-            glm::vec3 normaln           = glm::vec3(normalmap[pix_id],
-                                                    normalmap[H * W + pix_id],
-                                                    normalmap[2 * H * W + pix_id]);
+            float dL_dpixel_normaln[3] = {dL_dpixel_normals[pix_id],
+                                          dL_dpixel_normals[H * W + pix_id],
+                                          dL_dpixel_normals[2 * H * W + pix_id]};
+            float normaln[3]           = {normalmap[pix_id],
+                                          normalmap[H * W + pix_id],
+                                          normalmap[2 * H * W + pix_id]};
```

### 5.4 用点积状态替换 `last_color[]` / `last_normal[]`

```diff
-    float last_alpha                      = 0;
-    float last_color[C]                   = {0};
-    [[maybe_unused]] float last_t         = 0;
-    [[maybe_unused]] float last_normal[3] = {0};
+    float last_alpha                       = 0;
+    float last_color_dot                   = 0;
+    [[maybe_unused]] float last_t          = 0;
+    [[maybe_unused]] float last_normal_dot = 0;
```

### 5.5 用点积式累计替换原来的逐通道颜色累计

这是 CUDA 优化里的核心改动之一。

```diff
-                float dL_dopa = 0.0f;
-                for (int ch = 0; ch < C; ch++) {
-                    const float c = collected_colors[ch * BLOCK_SIZE + j];
-                    // Update last color (to be used in the next iteration)
-                    accum_rec[ch]  = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
-                    last_color[ch] = c;
-
-                    const float dL_dchannel = dL_dpixel[ch];
-                    dL_dopa += (c - accum_rec[ch]) * dL_dchannel;
-                    dL_dcolors_local[ch] = blending_weight * dL_dchannel;
-                }
+                float dL_dopa   = 0.0f;
+                accum_color_dot = last_alpha * last_color_dot + (1.f - last_alpha) * accum_color_dot;
+
+                float c_dot = 0.f;
+#pragma unroll
+                for (int ch = 0; ch < C; ++ch) {
+                    const float c        = collected_colors[ch * BLOCK_SIZE + j];
+                    const float g        = dL_dpixel[ch];
+                    c_dot                = fmaf(c, g, c_dot);
+                    dL_dcolors_local[ch] = blending_weight * g;
+                }
+
+                dL_dopa += (c_dot - accum_color_dot);
+                last_color_dot = c_dot;
```

### 5.6 用点积替换原来的逐分量法线累计

```diff
-                    const float3 normal          = collected_normals[j];
-                    const float* normal_ptr      = reinterpret_cast<const float*>(&normal);
-                    float* dL_dnormals_local_ptr = reinterpret_cast<float*>(&dL_dnormals_local);
-#pragma unroll
-                    for (int ch = 0; ch < 3; ch++) {
-                        const float n           = normal_ptr[ch];
-                        accum_normal_rec[ch]    = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
-                        last_normal[ch]         = n;
-                        const float dL_dchannel = dL_dpixel_normal[ch];
-                        dL_dopa += (n - accum_normal_rec[ch]) * dL_dchannel;
-                        dL_dnormals_local_ptr[ch] = blending_weight * dL_dchannel;
-                    }
+                    const float3 normal = collected_normals[j];
+
+                    accum_normal_dot    = last_alpha * last_normal_dot + (1.f - last_alpha) * accum_normal_dot;
+                    const float n_dot_g = fmaf(normal.x, dL_dpixel_normal[0],
+                                               fmaf(normal.y, dL_dpixel_normal[1],
+                                                    normal.z * dL_dpixel_normal[2]));
+                    dL_dopa += (n_dot_g - accum_normal_dot);
+                    last_normal_dot     = n_dot_g;
+                    dL_dnormals_local.x = blending_weight * dL_dpixel_normal[0];
+                    dL_dnormals_local.y = blending_weight * dL_dpixel_normal[1];
+                    dL_dnormals_local.z = blending_weight * dL_dpixel_normal[2];
```

### 5.7 更新 `warpSum` 调用方式

```diff
-            warpSum<C>(dL_dcolors_local, warp);
+            warpSum(dL_dcolors_local, warp);
```

### 移植注意

- 这是扩展 / 子模块代码，移植后大概率需要重新编译。
- 虽然这个改动主要是性能优化，但它确实改了累计结构，移植后建议做一次训练稳定性验证。
- 如果目标仓库的 kernel 已经和这里分叉很多，建议按下面顺序分块移植：
  1. `warpSum` 签名
  2. 状态变量替换
  3. 颜色累计块
  4. 法线累计块
  5. `warpSum(dL_dcolors_local, warp)` 调用点

---

## 6. `README.md`

### 改动目的

- 纯文档清理，没有运行时影响。

### 精确改动

```diff
 This project builds on **Gaussian Splatting** and **RaDe-GS**:  
-https://github.com/graphdeco-inria/gaussian-splatting
-https://github.com/HKUST-SAIL/RaDe-GS
+- https://github.com/graphdeco-inria/gaussian-splatting
+- https://github.com/HKUST-SAIL/RaDe-GS
```

```diff
- Spherical Gaussian apperance model from **RayGauss** and **RayGaussX**
-  https://github.com/hugobl1/ray_gauss
-  https://github.com/hugobl1/raygaussx
+- Spherical Gaussian appearance model from **RayGauss** and **RayGaussX**
+  - https://github.com/hugobl1/ray_gauss
+  - https://github.com/hugobl1/raygaussx
```

---

## 最小移植清单

- [ ] 将 `get_apperance_embedding` 重命名为 `get_appearance_embedding`
- [ ] 同步修改 `utils/loss_utils.py` 中的调用方
- [ ] 给 `PatchMatch` 增加 `optimize_geo` / `optimize_ncc`
- [ ] 将几何分支包进 `with torch.set_grad_enabled(self.optimize_geo):`
- [ ] 增加 `if not self.optimize_ncc: return zero_ncc, geo_loss`
- [ ] 修复 `train.py` 中 `active_sh_degree == max_sh_degree` 的判断
- [ ] 删除已经不用的 patch helper
- [ ] 移植 CUDA backward 优化并重新编译扩展

## 如果时间有限，优先移植这些

1. `train.py` 里的 SH/SG 解锁 bug 修复
2. `utils/loss_utils.py` 里的 PatchMatch 条件执行逻辑
3. `scene/gaussian_model.py` 的接口改名
4. `render_backward.cu` 的 backward 优化

