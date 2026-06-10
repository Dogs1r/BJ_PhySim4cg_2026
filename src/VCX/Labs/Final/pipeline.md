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
gs_particle_pipeline/camera.py
gs_particle_pipeline/particles.py
gs_particle_pipeline/ply_loader.py
gs_particle_pipeline/physics_interface.py
gs_particle_pipeline/elastic_mpm.py
gs_particle_pipeline/mass_spring.py
gs_particle_pipeline/dense_cloth.py
gs_particle_pipeline/cuda_rasterizer_adapter.py
gs_particle_pipeline/cuda_renderer.py
gs_particle_pipeline/render_backend.py
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
  -> create_renderer(args.renderer, render_particles or sim_particles)
  -> GUI / save_ppm / ffmpeg
```

默认官方 CUDA 3DGS rasterizer 数据流：

```text
render_particles + GaussianCamera
  -> build_cuda_rasterizer_payload()
  -> torch CUDA tensors
  -> diff_gaussian_rasterization.GaussianRasterizer
```

Taichi reference 数据流：

```text
render_particles + CameraConfig
  -> GaussianRenderer
  -> Taichi teaching splatting
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
--renderer     -> create_renderer("gs" or "reference")
--gs-device    -> CudaGaussianRenderer torch device
--gs-use-covariance -> 使用 current_covariance 作为 cov3D_precomp
--dense-render -> 启用 Mass-Spring 粗物理网格到密集渲染 Gaussian 的插值层
--render-upsample    -> DenseClothConfig.upsample
--render-splat-scale -> DenseClothConfig.splat_scale，占 dense 网格间距的比例
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

如果命令行没有显式传入 `--dt` 和 `--substeps`，`simulate.py` 会按物理模式选择稳定默认值：

```text
mode="mass-spring": dt=0.002,  substeps=4
mode="elastic":     dt=0.0005, substeps=10
```

原因是当前 Mass-Spring 是显式求解器，通用的 `dt=1/60` 对当前弹簧刚度会在数帧内发散。

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

当前 PLY 顶点表支持：

```text
format ascii
format binary_little_endian
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
scale = exp(scale_*)
rotation = [rot_0, rot_1, rot_2, rot_3]
base_covariance = R * diag(scale^2) * R^T
sh_coefficients[0] = f_dc
sh_coefficients[1..15] = f_rest_0..44
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
scale
rotation
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

Taichi reference 渲染模块读：

```text
position
current_covariance
opacity
sh_coefficients
```

CUDA rasterizer adapter 读：

```text
position
opacity
scale
rotation
sh_coefficients[0..15]
current_covariance
```

其中 `current_covariance` 用于后续选择 `cov3D_precomp` 路径；`scale/rotation` 用于官方 renderer 默认路径。

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

## 9. 可选渲染后端

文件：

```text
gs_particle_pipeline/render_backend.py
```

接口：

```text
create_renderer(backend, particles, reference_camera, gaussian_camera, render)
```

当前后端：

```text
backend="gs"        默认路径，复用官方 CUDA 3DGS rasterizer
backend="reference" Taichi 教学 renderer，用于 CPU/调试/对照
```

`simulate.py` 的主循环不关心具体后端，只要求 renderer 提供：

```text
render_frame() -> uint8 image [height, width, 3]
```

这保证仿真后的粒子状态可以直接传给不同渲染管线。

## 10. Taichi Reference Gaussian 渲染链路

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

## 11. 官方 CUDA 3DGS 接入边界

参考文件：

```text
D:\git_repo\gaussian-splatting\gaussian_renderer\__init__.py
D:\git_repo\gaussian-splatting\gaussian_renderer\network_gui.py
```

当前已预留的接口：

```text
gs_particle_pipeline/camera.py
  -> GaussianCamera.from_config()
  -> image_width / image_height
  -> fov_x / fov_y
  -> world_view_transform
  -> projection_matrix
  -> full_proj_transform
  -> camera_center

gs_particle_pipeline/cuda_rasterizer_adapter.py
  -> build_cuda_rasterizer_payload()
  -> means3D / opacities / shs / scales / rotations
  -> optional cov3D_precomp

gs_particle_pipeline/cuda_renderer.py
  -> CudaGaussianRenderer
  -> GaussianRasterizationSettings
  -> GaussianRasterizer
```

后续真正接入 CUDA 时，目标是把 payload 转成：

```text
means3D         -> pc.get_xyz
opacities       -> pc.get_opacity
scales          -> pc.get_scaling
rotations       -> pc.get_rotation
shs             -> pc.get_features
camera matrices -> viewpoint_camera
```

然后调用官方 `GaussianRasterizer`。本轮没有引入 torch，也没有编译 `diff_gaussian_rasterization`。

当前设计已经在代码层尝试运行时导入 `torch` 和 `diff_gaussian_rasterization`。如果本机没有 CUDA 环境，使用：

```powershell
python simulate.py --renderer reference ...
```

后续可以在单独虚拟环境或云服务器安装官方依赖后运行：

```powershell
python simulate.py --renderer gs ...
```

## 12. 当前运行命令

生成测试 PLY：

```powershell
python make_elastic_bar.py --output assets/elastic_bar.ply
python make_elastic_sheet.py --output assets/elastic_sheet.ply
```

运行动态 Gaussian 渲染：

```powershell
python simulate.py --renderer reference --cpu --ply assets/elastic_bar.ply --particles 300 --frames 30 --dt 0.0005 --substeps 10 --camera-z 3.0 --focal-length 1100 --profile
```

运行 432 粒子弹性布片：

```powershell
python simulate.py --renderer reference --cpu --ply assets/elastic_sheet.ply --particles 432 --frames 30 --dt 0.0005 --substeps 8 --camera-z 3.0 --focal-length 950 --profile
```

运行接近 lab0 MassSpring 的布片：

```powershell
python make_elastic_sheet.py --output assets/mass_spring_sheet.ply --nx 22 --ny 22 --pin-mode top-corners --splat-scale 0.08
python simulate.py --renderer reference --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

运行密集渲染版 MassSpring 布片：

```powershell
python simulate.py --renderer reference --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 2 --render-splat-scale 0.80 --render-opacity 0.85 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

增大 Mass-Spring 重力：

```powershell
python simulate.py --renderer reference --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 2 --spring-gravity 4.0 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

导出视频：

```powershell
python simulate.py --cpu --no-window --ply assets/elastic_bar.ply --particles 300 --frames 120 --dt 0.0005 --substeps 10 --video outputs/elastic_bar.mp4
```
## 13. CUDA GS 布料诊断命令

当 `--renderer gs` 的结果和 `--renderer reference` 明显不一致时，先用下面命令检查 CUDA rasterizer 实际收到的粒子数、投影范围和 Gaussian 尺寸：

```powershell
python simulate.py --renderer gs --no-window --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 10 --render-splat-scale 0.80 --render-opacity 0.85 --frames 1 --spring-nx 22 --spring-ny 22 --width 1280 --height 720 --camera-z 3.0 --focal-length 1100 --pitch -28 --profile --diagnose-state --diagnose-render
```

需要重点确认：

```text
render_particles=44521, dense=True
[diagnose-render] n=44521 ... visible_ndc=...
```

## 14. 布料 GS 仿真阶段性总结

当前 Mass-Spring 布料链路已经跑通：

```text
22 x 22 Mass-Spring physics nodes
  -> dense render layer, for example 211 x 211 = 44521 Gaussians
  -> official CUDA GS rasterizer
  -> continuous cloth-like rendered surface
  -> frame sequence / video export
```

本阶段修复过的关键问题：

```text
1. Mass-Spring 默认 dt/substeps 过大导致显式求解发散。
2. 原始 484 粒子直接渲染只能看到彩色点阵，因此加入 dense render layer。
3. CUDA GS projection matrix 需要保持官方 row-vector 约定，即 projection/world-view 都使用 transpose 后的矩阵。
4. 布料测试 PLY 使用自生成 SH/scale，CUDA GS 路径建议用 --gs-color-mode rgb 解耦 SH 颜色解释。
5. 为了让 dense Gaussian 连续成面，建议使用 --gs-scale-modifier 2.0 左右。
6. CUDA GS 图像上下方向与 reference 不一致时，使用 --gs-flip-y，而不是移除 projection transpose。
```

当前推荐的 CUDA GS 布料视频命令：

```bash
python simulate.py --renderer gs --no-window --mode mass-spring \
  --ply assets/mass_spring_sheet.ply \
  --particles 484 \
  --dense-render \
  --render-upsample 10 \
  --render-splat-scale 0.80 \
  --render-opacity 0.85 \
  --gs-scale-modifier 2.0 \
  --gs-color-mode rgb \
  --gs-flip-y \
  --frames 120 \
  --video outputs/cloth_gs.mp4 \
  --spring-nx 22 \
  --spring-ny 22 \
  --width 1280 \
  --height 720 \
  --camera-z 3.0 \
  --focal-length 1100 \
  --pitch -28 \
  --profile
```

参数含义：

```text
--dense-render         coarse physics nodes -> dense visual Gaussians
--render-upsample      dense visual grid subdivision
--render-splat-scale   Gaussian scale relative to dense spacing
--gs-scale-modifier    official rasterizer global scale multiplier
--gs-color-mode rgb    use precomputed RGB for synthetic cloth tests
--gs-flip-y            fix vertical image convention mismatch
```

## 15. 扫描物体到 GS PLY 再做弹性体仿真的规划

目标链路：

```text
real object scan / video capture
  -> COLMAP camera reconstruction
  -> train 3D Gaussian Splatting
  -> exported point_cloud.ply
  -> build physics representation
  -> elastic simulation
  -> render deformed Gaussian state with CUDA GS
```

需要注意：训练得到的 3DGS PLY 是视觉表示，不等价于物理离散体。它通常只覆盖可见表面，点的密度由视角和纹理决定，不包含体积网格、质量、材料、约束和碰撞信息。因此不能把任意 trained GS PLY 直接当作弹性体 MPM 粒子集使用。

后续需要补的模块：

```text
1. GS PLY visual loader
   读取真实 trained 3DGS 的 xyz / opacity / scale / rotation / SH。

2. Physics proxy builder
   从 GS 表面或额外 mesh/depth 数据构建物理代理：
   - surface shell, for cloth/thin sheet
   - tetrahedral/voxel particles, for volumetric elastic object
   - optional cages or embedded deformation graph

3. Visual-to-physics binding
   建立 visual Gaussian 到 physics proxy 的绑定关系：
   - nearest particle
   - barycentric coordinate in tet/triangle
   - skinning weights
   - deformation graph weights

4. Elastic solver
   对体积物体使用 MPM/FEM/PBD 等弹性体求解器。

5. Deformed GS update
   每帧由 physics proxy 更新 Gaussian position，并按局部 deformation gradient 更新 covariance/scale/rotation。
```

当前已有的 Mass-Spring 不是通用物体弹性体求解器。它专门适合 cloth/sheet：

```text
2D grid topology
springs between neighbor/bending/shear nodes
pinned corners or rows
explicit integration
```

对扫描得到的一般三维物体，应优先使用：

```text
volumetric MPM    适合大变形、拓扑简单、粒子法教学扩展
tet FEM           适合弹性体形变更准确，但需要网格化和求解线性系统
PBD/XPBD          适合稳定实时交互，物理精度较低但工程上直接
embedded graph    适合把高密 GS 绑定到低维控制节点
```

推荐下一阶段先做最小可行版本：

```text
trained/static GS PLY
  + manually generated low-res elastic proxy, for example voxel/tet particles
  + nearest-neighbor or kNN binding from each Gaussian to proxy particles
  + proxy simulation updates Gaussian positions
  + CUDA GS renders updated Gaussians
```

这样可以避免一开始就同时解决“真实扫描、GS 训练、几何重建、体积网格化、弹性仿真、高质量绑定”五个困难问题。

如果 `visible_ndc` 很小，优先检查相机参数；如果 `scale_minmax` 很小，优先增大 `--render-splat-scale`。CUDA GS 相机矩阵采用官方 gaussian-splatting 的 row-vector 约定，`projection_matrix` 和 `world_view_transform` 都需要 transpose 后再计算 `full_proj_transform`。
