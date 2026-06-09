# 当前核心 Pipeline

本文件描述精简后的当前版本。历史变更记录见 `changes.md`。

## 1. 保留目标

当前程序只保留三件核心能力：

```text
1. 动态渲染
2. 规范 Gaussian 渲染
3. 只基于 PLY 文件导入场景
```

当前不再保留：

```text
main.py 单帧入口
sample_scene.py 合成场景
demo/static 物理模式
points 快速点渲染模式
```

## 2. 文件分层

### 核心入口

```text
simulate.py
```

作用：

- 解析命令行参数。
- 读取 PLY 场景。
- 初始化粒子、物理模块和 Gaussian 渲染器。
- 执行连续时间步。
- 显示 GUI、保存逐帧图片或导出视频。

### 场景生成工具

```text
make_elastic_bar.py
assets/elastic_bar.ply
make_elastic_sheet.py
assets/elastic_sheet.ply
```

这些脚本只用于生成测试 PLY。运行时核心链路仍然只读取 PLY。

### 核心库

```text
gs_particle_pipeline/config.py
gs_particle_pipeline/particles.py
gs_particle_pipeline/ply_loader.py
gs_particle_pipeline/physics_interface.py
gs_particle_pipeline/elastic_mpm.py
gs_particle_pipeline/mass_spring.py
gs_particle_pipeline/dense_cloth.py
gs_particle_pipeline/renderer.py
```

### 非核心内容

```text
outputs/
tandt_db/
__pycache__/
2311.12198v3.pdf
prompt_for_codex
```

- `outputs/`: 输出目录。
- `tandt_db/`: 当前核心程序不读取。
- `__pycache__/`: Python 缓存。
- PDF 和 prompt 是资料，不参与运行。

## 3. 总数据流

```text
simulate.py
  -> load_3dgs_ply(args.ply)
  -> GaussianHostData
  -> sim_particles = GaussianParticleSet.load_from_host()
  -> PhysicsBridge(mode="elastic" or "mass-spring")
  -> optional render_particles = DenseClothRenderLayer(sim_particles)
  -> GaussianRenderer(render_particles or sim_particles)
  -> GUI / save_ppm / ffmpeg
```

逐帧循环：

```text
for frame in frames:
    for substep in substeps:
        physics.step(dt)
    if dense_render:
        dense_layer.update()
    image = renderer.render_frame()
    display/save/export
```

## 4. 命令行参数传递

### 输入

```text
--ply -> load_3dgs_ply(path)
```

`--ply` 是必需参数。

### 相机参数

```text
--width        -> CameraConfig.width
--height       -> CameraConfig.height
--focal-length -> CameraConfig.focal_length
--camera-z     -> CameraConfig.camera_z
--pitch        -> CameraConfig.pitch_degrees
```

### 渲染参数

```text
--particles    -> RenderConfig.max_particles
--radius-scale -> RenderConfig.tile_radius_scale
--low-pass     -> RenderConfig.low_pass_variance
--dense-render -> 启用 Mass-Spring 粗物理网格到密集渲染 Gaussian 的插值层
--render-upsample    -> DenseClothConfig.upsample
--render-splat-scale -> DenseClothConfig.splat_scale
--render-opacity     -> DenseClothConfig.opacity
```

### 物理参数

```text
--dt       -> PhysicsBridge.step(dt)
--substeps -> 每帧 MPM 子步数
--mpm-grid -> ElasticMPMConfig.grid_size
--gravity  -> ElasticMPMConfig.gravity
--spring-nx / --spring-ny -> MassSpringConfig cloth grid
--spring-stiffness        -> MassSpringConfig.stiffness
--spring-damping          -> MassSpringConfig.damping
--spring-gravity          -> MassSpringConfig.gravity
```

### 输出参数

```text
--no-window
--save-frames
--video
--fps
--keep-frames
--progress
--profile
```

## 5. PLY 到内部数据

文件：

```text
gs_particle_pipeline/ply_loader.py
```

接口：

```text
load_3dgs_ply(path, max_particles) -> GaussianHostData
```

支持 Gaussian 字段：

```text
x, y, z
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
f_dc_0, f_dc_1, f_dc_2
f_rest_*
```

支持物理字段：

```text
mass
volume
density
youngs_modulus
poisson_ratio
material_id
pinned
```

关键转换：

```text
position = [x, y, z]
opacity = sigmoid(opacity)
scale = exp(scale_*)
R = quaternion_to_rotation(rot_*)
base_covariance = R * diag(scale^2) * R^T
sh_coefficients[0] = clamp(0.5 + SH_C0 * f_dc)
```

## 6. 共享数据接口

文件：

```text
gs_particle_pipeline/particles.py
```

`GaussianHostData` 是 CPU 侧加载结果。

`GaussianParticleSet` 是 Taichi 侧共享状态，物理和渲染都通过它通信。

核心字段：

```text
rest_position
position
velocity
affine_C
deformation_gradient
mass
volume
material_id
pinned
density
youngs_modulus
poisson_ratio
base_covariance
current_covariance
opacity
sh_coefficients
```

物理模块写：

```text
position
velocity
affine_C
deformation_gradient
current_covariance
```

渲染模块读：

```text
position
current_covariance
opacity
sh_coefficients
```

## 7. 密集布料渲染层

文件：

```text
gs_particle_pipeline/dense_cloth.py
```

用途：

```text
粗物理粒子：Mass-Spring 求解器实际更新，例如 22 x 22 = 484 个节点
密集渲染 Gaussian：从粗网格双线性插值得到，例如 upsample=2 时为 43 x 43 = 1849 个 Gaussian
```

接口：

```text
make_dense_cloth_host_data(source, DenseClothConfig) -> GaussianHostData
DenseClothRenderLayer.update()
```

数据传递：

```text
sim_particles.position
  -> bilinear interpolation
  -> render_particles.position
  -> GaussianRenderer.render_frame()
```

这一层只改善当前小规模布料的视觉连续性，不改变 Mass-Spring 的物理自由度。它也不是正式 CUDA 3DGS rasterizer；后续接入真实高密度 3DGS 时，应替换或扩展渲染后端。

## 8. 物理链路

文件：

```text
gs_particle_pipeline/physics_interface.py
gs_particle_pipeline/elastic_mpm.py
```

接口：

```text
PhysicsBridge(particles, mode="elastic", elastic_config)
PhysicsBridge.step(dt)
```

当前支持：

```text
mode="elastic"
mode="mass-spring"
```

每个 MPM 子步：

```text
clear grid
P2G
grid velocity update
G2P
current_covariance = F * base_covariance * F^T
```

Mass-Spring 模式参考 lab0：

```text
结构弹簧：相邻粒子
弯曲弹簧：跨两格粒子
剪切弹簧：对角粒子
固定点：由 PLY 的 pinned 字段定义
```

弹性本构：

```text
fixed-corotated elasticity
```

材质参数：

```text
E  = youngs_modulus
nu = poisson_ratio
mu = E / (2 * (1 + nu))
lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
```

## 9. Gaussian 渲染链路

文件：

```text
gs_particle_pipeline/renderer.py
```

接口：

```text
GaussianRenderer(particles, camera, render)
GaussianRenderer.render_frame()
save_ppm(path, image)
```

渲染流程：

```text
clear
project 3D Gaussian to 2D Gaussian
depth sort
splat_one for each visible Gaussian
tonemap
```

核心公式：

```text
u = f * x / z + width / 2
v = f * y / z + height / 2
Sigma_2d = J * Sigma_3d * J^T
conic = inverse(Sigma_2d)
alpha = opacity * exp(-0.5 * d^T conic d)
color += T * alpha * rgb
T *= 1 - alpha
```

## 10. 当前运行命令

生成测试 PLY：

```powershell
python make_elastic_bar.py --output assets/elastic_bar.ply
python make_elastic_sheet.py --output assets/elastic_sheet.ply
```

运行动态 Gaussian 渲染：

```powershell
python simulate.py --cpu --ply assets/elastic_bar.ply --particles 300 --frames 30 --dt 0.0005 --substeps 10 --camera-z 3.0 --focal-length 1100 --profile
```

运行 432 粒子弹性布片：

```powershell
python simulate.py --cpu --ply assets/elastic_sheet.ply --particles 432 --frames 30 --dt 0.0005 --substeps 8 --camera-z 3.0 --focal-length 950 --profile
```

运行接近 lab0 MassSpring 的布片：

```powershell
python make_elastic_sheet.py --output assets/mass_spring_sheet.ply --nx 22 --ny 22 --pin-mode top-corners --splat-scale 0.08
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

运行密集渲染版 MassSpring 布片：

```powershell
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 2 --render-splat-scale 0.10 --render-opacity 0.45 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

增大 Mass-Spring 重力：

```powershell
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 2 --spring-gravity 4.0 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

导出视频：

```powershell
python simulate.py --cpu --no-window --ply assets/elastic_bar.ply --particles 300 --frames 120 --dt 0.0005 --substeps 10 --video outputs/elastic_bar.mp4
```
