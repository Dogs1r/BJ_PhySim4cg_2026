# 当前 Pipeline：Taichi 静态 3DGS 基础流水线（第 1 轮）

## 目标

本轮先搭建一个可以运行静态 Gaussian Splatting 思路的最小工程。重点不是完整复现 3DGS 训练，而是把数据流设计成之后可以接入 PhysGaussian/MPM 物理仿真的形式。

当前数据主线：

```text
合成 Gaussian 输入
  -> GaussianHostData
  -> GaussianParticleSet(Taichi fields)
  -> PhysicsBridge.step(dt)
  -> GaussianRenderer.splat()
  -> PPM 图像输出
```

未来物理主线预留为：

```text
GaussianParticleSet
  -> MPM particle-to-grid
  -> 网格动量/应力更新
  -> grid-to-particle
  -> 更新 position / velocity / deformation_gradient
  -> current_covariance = F * base_covariance * F^T
  -> GaussianRenderer 渲染
```

## 模块说明

### 1. `main.py`

工作内容：

- 初始化 Taichi。
- 创建相机参数和渲染参数。
- 生成合成 Gaussian 粒子。
- 创建粒子容器 `GaussianParticleSet`。
- 调用 `PhysicsBridge.step(0.0)`，让数据先经过物理接口。
- 调用 `GaussianRenderer.render_frame()` 输出静态图像。

关键变量：

- `particles`: 全局粒子/Gaussian 状态容器。
- `physics`: 物理接口对象，目前是静态 no-op。
- `renderer`: splatting 渲染器。

### 2. `gs_particle_pipeline/config.py`

工作内容：

- 定义相机和渲染配置。

变量定义：

- `CameraConfig.width / height`: 输出图像尺寸。
- `CameraConfig.focal_length`: 简化针孔相机焦距。
- `CameraConfig.camera_z`: 相机与场景的 z 方向距离。
- `RenderConfig.max_particles`: 最大 Gaussian/粒子数。
- `RenderConfig.tile_radius_scale`: 屏幕空间 splat 半径系数。
- `RenderConfig.background_*`: 背景色。

### 3. `gs_particle_pipeline/particles.py`

工作内容：

- 定义 CPU 侧输入结构 `GaussianHostData`。
- 定义 Taichi 侧粒子容器 `GaussianParticleSet`。
- 统一管理渲染变量和未来物理变量。

核心变量：

- `position[p]`: 第 `p` 个 Gaussian/粒子的世界位置，是渲染和物理共同使用的核心变量。
- `base_covariance[p]`: 初始 3D Gaussian 协方差矩阵，对应论文中的 `A_p`。
- `current_covariance[p]`: 当前协方差矩阵，对应动态状态下的 `a_p(t)`。
- `opacity[p]`: Gaussian 不透明度。
- `sh_coefficients[p, k]`: 简化 SH 系数，当前保留常数项和一阶方向项。

物理预留变量：

- `velocity[p]`: 粒子速度，未来 MPM 时间积分使用。
- `deformation_gradient[p]`: 变形梯度 `F_p`，未来用于表达局部拉伸、旋转、剪切。
- `mass[p]`: 粒子质量。
- `volume[p]`: 粒子初始体积 `V_p^0`。

当前变量传递：

```text
GaussianHostData.position
  -> GaussianParticleSet.position
  -> GaussianRenderer.splat

GaussianHostData.base_covariance
  -> GaussianParticleSet.base_covariance
  -> PhysicsBridge
  -> GaussianParticleSet.current_covariance
  -> GaussianRenderer.splat

GaussianHostData.sh_coefficients / opacity
  -> GaussianParticleSet
  -> GaussianRenderer.splat
```

### 4. `gs_particle_pipeline/physics_interface.py`

工作内容：

- 定义 `PhysicsBridge`，作为未来物理仿真的统一入口。
- 当前版本不改变粒子状态，只把 `current_covariance` 保持为 `base_covariance`。

当前实现：

```text
F = I
current_covariance = base_covariance
position 不变
velocity 不变
```

未来替换点：

```text
position[p] += dt * velocity[p]
deformation_gradient[p] = update_F(...)
current_covariance[p] = F_p * base_covariance[p] * transpose(F_p)
```

该模块的意义是让后续 MPM 接入时只替换物理 step，不改变渲染器读取数据的方式。

### 5. `gs_particle_pipeline/sample_scene.py`

工作内容：

- 生成一个合成 Gaussian 场景。
- 为每个 Gaussian 同时生成渲染字段和物理字段。

当前生成字段：

- `position`: 环状分布的点。
- `base_covariance`: 各向异性椭球形状。
- `opacity`: 固定透明度。
- `sh_coefficients`: 简化视角相关颜色。
- `mass`: 默认质量。
- `volume`: 默认体积。

### 6. `gs_particle_pipeline/renderer.py`

工作内容：

- 实现教学版静态 splatting。
- 将 3D Gaussian 中心投影到屏幕。
- 使用 covariance 的 x/y 分量估计屏幕空间半径。
- 在局部像素窗口中进行 alpha 合成。
- 输出 PPM 图像。

当前渲染公式的简化理解：

```text
u = focal_length * x / z + width / 2
v = focal_length * y / z + height / 2

sigma_x = sqrt(cov_xx) * focal_length / z
sigma_y = sqrt(cov_yy) * focal_length / z

weight = exp(-0.5 * (dx^2 / sigma_x^2 + dy^2 / sigma_y^2))
alpha = opacity * weight
color += (1 - accumulated_alpha) * alpha * SH(view_dir)
```

当前局限：

- 没有真实 3DGS 的深度排序。
- 没有完整 3D covariance 到 2D covariance 的投影推导。
- 没有 tile-based GPU 优化。
- 合成数据不是训练得到的真实 Gaussian。

## 数据接口设计原则

本轮最重要的工程设计是：渲染器只读取 `GaussianParticleSet` 中的当前状态，不关心这些状态来自静态输入还是物理仿真。

因此后续接入物理仿真时，目标接口保持为：

```text
PhysicsBridge.step(dt)
  输入：position, velocity, deformation_gradient, mass, volume, base_covariance
  输出：position, velocity, deformation_gradient, current_covariance

GaussianRenderer.render_frame()
  输入：position, current_covariance, opacity, sh_coefficients
  输出：image
```

这样可以让工程逐步演进：

```text
第 1 轮：静态 F = I
第 2 轮：手工速度/刚体运动
第 3 轮：简单弹性或重力积分
第 4 轮：MPM particle-grid-particle
第 5 轮：塑性模型和 internal filling
```

# 当前 Pipeline：加入 3DGS PLY 导入与规范 GS 渲染（第 2 轮）

## 本轮回答的两个问题

### 1. 当前输入是否是预设场景？

是。第 1 轮的输入来自 `gs_particle_pipeline/sample_scene.py`，它生成一个合成 Gaussian 场景，用来验证粒子字段、物理接口和渲染链路。

第 2 轮新增了 `gs_particle_pipeline/ply_loader.py`，现在入口支持两种输入：

```text
无 --ply 参数：
  make_demo_scene()
  -> GaussianHostData
  -> GaussianParticleSet

有 --ply 参数：
  load_3dgs_ply(path)
  -> GaussianHostData
  -> GaussianParticleSet
```

运行示例：

```powershell
python main.py --cpu --ply path\to\point_cloud.ply --particles 20000 --output outputs/ply_demo.ppm
```

当前 PLY 读取器支持 ASCII PLY，预期字段来自常见 3DGS 导出：

```text
x, y, z
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
f_dc_0, f_dc_1, f_dc_2
f_rest_*
```

字段到内部变量的转换：

```text
x/y/z
  -> position[p]

opacity
  -> sigmoid(opacity)
  -> opacity[p]

scale_0/1/2
  -> exp(scale)
  -> 三轴标准差 s

rot_0/1/2/3
  -> quaternion_to_rotation()
  -> 旋转矩阵 R

R 和 s
  -> base_covariance[p] = R * diag(s^2) * R^T

f_dc_0/1/2
  -> rgb = clamp(0.5 + SH_C0 * f_dc)
  -> sh_coefficients[p, 0]

f_rest_*
  -> sh_coefficients[p, 1..3]
```

这里最重要的是 `base_covariance` 的构造。3DGS 通常不直接保存 3x3 协方差，而是保存：

```text
scale: 三个轴向尺度
rotation: Gaussian 局部坐标到世界坐标的旋转
```

因此内部协方差由下面公式恢复：

```text
A = R * diag(sx^2, sy^2, sz^2) * R^T
```

这个 `A` 就是之后物理仿真中用于：

```text
current_covariance = F * A * F^T
```

的初始 Gaussian 形状。

### 2. 当前渲染是否已经改成规范 GS 渲染？

已改成教学版规范 GS 渲染。它仍然不是原论文 CUDA tile rasterizer，但数学流程已经从“简单屏幕圆斑”改为 3DGS 的核心流程：

```text
GaussianParticleSet
  -> project()
      3D 中心投影到 2D
      3D covariance 投影为 2D covariance
      2D covariance 求逆得到 conic
      计算屏幕影响半径
  -> CPU depth sort
      按 camera-space depth 从近到远排序
  -> splat_one()
      计算 2D Gaussian 权重
      front-to-back alpha compositing
  -> tonemap()
      输出 PPM
```

## 第 2 轮模块更新

### 1. `gs_particle_pipeline/ply_loader.py`

新增模块，负责把 3DGS PLY 参数转成 `GaussianHostData`。

输入：

```text
path: PLY 文件路径
max_particles: 可选最大读取数量
```

输出：

```text
GaussianHostData(
  position,
  base_covariance,
  opacity,
  sh_coefficients,
  mass,
  volume
)
```

关键中间变量：

- `properties`: PLY header 中的字段名列表。
- `table`: PLY vertex 数据表。
- `scales`: 由 `exp(scale_*)` 得到的三轴标准差。
- `R`: 由四元数 `rot_*` 得到的旋转矩阵。
- `base_covariance`: 由 `R diag(s^2) R^T` 得到的 3D Gaussian 协方差。

### 2. `main.py`

新增参数：

```text
--ply
```

数据分支：

```text
if args.ply:
    host_data = load_3dgs_ply(args.ply)
else:
    host_data = make_demo_scene()
```

这让当前工程既能继续使用合成场景做快速调试，也能逐步接入真实 3DGS 数据。

### 3. `gs_particle_pipeline/renderer.py`

渲染器被重写为规范 GS 数学流程。

新增渲染中间变量：

- `projected_xy[p]`: 第 `p` 个 Gaussian 的屏幕中心。
- `depth[p]`: 相机空间深度，用于排序。
- `conic[p]`: 2D covariance 逆矩阵的紧凑形式 `(A, B, C)`。
- `screen_radius[p]`: 屏幕影响半径。
- `visible[p]`: 可见性标记。
- `transmittance[i, j]`: 像素剩余透射率 `T`。

#### 3D 到 2D covariance 投影

当前相机默认没有旋转，世界空间 covariance 直接视为相机空间 covariance：

```text
Sigma_camera = Sigma_world
```

透视投影为：

```text
u = f * x / z + width / 2
v = f * y / z + height / 2
```

投影雅可比：

```text
J = [ f/z,   0, -f*x/z^2
        0, f/z, -f*y/z^2 ]
```

2D covariance：

```text
Sigma_2d = J * Sigma_3d * J^T
```

直观理解：

- `Sigma_3d` 描述 Gaussian 在世界中的椭球形状。
- `J` 描述当前位置附近的透视投影如何把 3D 小位移变成屏幕小位移。
- `Sigma_2d` 就是这个椭球投影到屏幕后的椭圆。

#### conic 权重

设：

```text
Sigma_2d = [ a b
             b c ]
```

逆矩阵：

```text
Sigma_2d^-1 = 1 / (a*c - b*b) * [  c -b
                                  -b  a ]
```

代码中保存为：

```text
conic = (c/det, -b/det, a/det)
```

像素偏移：

```text
d = pixel_center - projected_center
```

Gaussian 权重：

```text
weight = exp(-0.5 * d^T * Sigma_2d^-1 * d)
```

展开后：

```text
weight = exp(-0.5 * (A*dx^2 + 2*B*dx*dy + C*dy^2))
```

#### 深度排序与 alpha 合成

当前排序在 CPU/Numpy 中完成：

```text
sorted_indices = argsort(depth)
```

然后按近到远对每个 Gaussian 调用 `splat_one(p)`。

像素合成公式：

```text
color += T * alpha * rgb
T *= (1 - alpha)
```

变量含义：

- `T`: 当前像素剩余透射率，初始为 1。
- `alpha`: 当前 Gaussian 对这个像素的不透明贡献。
- `rgb`: 当前 Gaussian 从视角方向看到的颜色。

这对应 3DGS/体渲染中常见的 front-to-back compositing。

## 当前仍未实现的标准 3DGS 部分

本轮已经把核心公式改规范，但仍保留一些教学化简：

- 未实现 CUDA tile rasterizer。
- 未实现 tile 内并行排序。
- 未读取 COLMAP 相机外参/内参。
- 当前相机固定看向 `+z` 方向，没有相机旋转矩阵。
- 当前 SH 只保留常数项和一阶近似，没有完整 3 阶 16 项 SH。
- PLY 读取器当前只支持 ASCII PLY。

这些限制不会破坏数据接口，后续都可以在现有模块上替换或扩展。
