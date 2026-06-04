# 改动记录

## 第 1 次改动：搭建 Taichi 静态 3DGS 基础流水线

### 主要内容

- 新增 `requirements.txt`，声明 `taichi` 和 `numpy` 依赖。
- 新增 `main.py`，作为当前工程入口。
- 新增 `gs_particle_pipeline/` 包，拆分为配置、粒子数据、物理接口、合成场景、渲染器几个模块。
- 新增 `GaussianParticleSet`，用 Taichi field 保存 Gaussian/粒子状态。
- 新增 `PhysicsBridge`，当前是静态 no-op，但已经规定未来物理仿真的 step 入口。
- 新增 `GaussianRenderer`，实现简化版屏幕空间 Gaussian splatting。
- 新增 `README.md`，说明运行方式和当前范围。

### 本次保留的物理仿真接口

- `position[p]`: 粒子位置。
- `velocity[p]`: 粒子速度。
- `deformation_gradient[p]`: 变形梯度 `F_p`。
- `mass[p]`: 质量。
- `volume[p]`: 初始体积。
- `base_covariance[p]`: 初始 Gaussian 形状 `A_p`。
- `current_covariance[p]`: 当前 Gaussian 形状 `a_p(t)`。

未来接入物理时，预期在 `PhysicsBridge.step(dt)` 中更新：

```text
position
velocity
deformation_gradient
current_covariance = F * base_covariance * F^T
```

### 当前实现的限制

- 当前不是完整 3DGS 训练管线，只是静态 Gaussian 数据到图像的基础渲染管线。
- 当前输入是合成 Gaussian 场景，不是 COLMAP/PLY/真实训练结果。
- 渲染器没有深度排序，也没有严格的 3D covariance 到 2D covariance 投影。
- 当前物理模块是 no-op，还没有 MPM。
- 当前输出是 PPM 图片，目的是避免 Pillow 等额外依赖。

### 后续方向

- 添加 PLY/NPZ Gaussian 参数读取器，接入真实 3DGS 训练结果。
- 给 `PhysicsBridge` 增加简单手工运动，例如平移、旋转、缩放，用于验证动态 covariance 传递。
- 实现 `current_covariance = F A F^T` 的 Taichi kernel，并用手工 `F` 验证椭球形变。
- 添加深度排序或 tile/bin 近似排序，提高 splatting 正确性。
- 接入最小 MPM：particle-to-grid、grid update、grid-to-particle。
- 实现 internal filling，让表面 Gaussian 可以扩展成体积粒子。

## 第 2 次改动：加入 3DGS PLY 导入，并把渲染器改为规范 GS 数学流程

### 主要内容

- 新增 `gs_particle_pipeline/ply_loader.py`。
- `main.py` 新增 `--ply` 参数，可以从 3DGS 导出的 PLY 文件读取 Gaussian 数据。
- `renderer.py` 从第 1 轮的简化屏幕空间 splat，改为教学版标准 3DGS 渲染流程。
- `README.md` 增加 PLY 导入运行方式和支持字段说明。
- `pipeline.md` 增加第 2 轮 pipeline，记录 PLY 字段转换、投影协方差、conic、深度排序、alpha 合成。

### PLY 导入的数据转换

当前支持常见 3DGS ASCII PLY 字段：

```text
x, y, z
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
f_dc_0, f_dc_1, f_dc_2
f_rest_*
```

转换关系：

```text
position = [x, y, z]
opacity = sigmoid(opacity)
scale = exp(scale_*)
R = quaternion_to_rotation(rot_*)
base_covariance = R * diag(scale^2) * R^T
sh_coefficients[0] = clamp(0.5 + SH_C0 * f_dc)
```

物理接口保持不变，PLY 导入后的 Gaussian 仍然进入：

```text
GaussianHostData
  -> GaussianParticleSet
  -> PhysicsBridge.step(dt)
  -> GaussianRenderer
```

### 渲染器改动

第 1 轮渲染器：

```text
用 cov_xx/cov_yy 估计屏幕半径
没有深度排序
直接做简化 alpha 累积
```

第 2 轮渲染器：

```text
3D covariance
  -> 透视投影雅可比 J
  -> Sigma_2d = J Sigma_3d J^T
  -> conic = inverse(Sigma_2d)
  -> 按 depth 近到远排序
  -> color += T * alpha * rgb
  -> T *= 1 - alpha
```

新增中间变量：

- `projected_xy`: 屏幕中心。
- `depth`: 相机空间深度。
- `conic`: 2D Gaussian 逆协方差。
- `screen_radius`: 屏幕影响半径。
- `visible`: 可见性标记。
- `transmittance`: 像素透射率。

### 当前限制

- 仍不是高性能 CUDA/tile-based 3DGS rasterizer。
- 当前深度排序在 CPU/Numpy 中完成，适合学习和小规模调试，不适合大量 Gaussian 实时渲染。
- 当前只支持 ASCII PLY；二进制 PLY 需要后续扩展。
- 当前相机是固定简化相机，没有读取 COLMAP 相机参数。
- 当前 SH 是简化一阶表达，没有完整 3DGS 的高阶 SH。

### 后续方向

- 扩展 binary_little_endian PLY 读取，适配更多真实 3DGS 输出。
- 添加 COLMAP camera 读取，支持真实训练视角复现。
- 将 CPU depth sort 改为 tile binning + tile 内排序。
- 扩展完整 SH 维度，例如 16 项三阶球谐。
- 添加手工 deformation_gradient 测试，验证 `current_covariance = F A F^T` 对渲染形状的影响。
- 在 `PhysicsBridge.step(dt)` 中接入最小 MPM。
