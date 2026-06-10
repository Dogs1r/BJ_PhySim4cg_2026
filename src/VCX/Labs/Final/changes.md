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

## 第 3 次改动：新增动态渲染链路、视频导出和清晰度参数

### 主要内容

- 新增 `simulate.py`，支持连续时间步渲染。
- `PhysicsBridge` 新增 `mode` 参数，支持 `static` 和 `demo` 两种模式。
- `GaussianParticleSet` 新增 `rest_position`，用于从参考构型生成动态状态。
- `demo` 模式会更新 `position / velocity / deformation_gradient / current_covariance`。
- `main.py` 暴露 `--focal-length`、`--camera-z`、`--low-pass`、`--radius-scale`。
- `RenderConfig` 新增 `low_pass_variance` 和 `alpha_threshold`。
- `renderer.py` 新增 `render_frame_float()`，用于 GUI 显示。
- `README.md` 增加动态窗口渲染、逐帧导出、视频导出和清晰度参数说明。
- `pipeline.md` 增加第 3 轮 pipeline。

### 动态链路

新增连续时间步流程：

```text
for frame in frames:
    for substep in substeps:
        PhysicsBridge.step(dt)
    image = GaussianRenderer.render_frame()
    GUI 显示 / 保存 frame_xxxx.ppm
```

视频导出流程：

```text
frame_0000.ppm
frame_0001.ppm
...
  -> ffmpeg
  -> demo.mp4
```

运行示例：

```powershell
python simulate.py --cpu --mode demo --frames 180
python simulate.py --cpu --no-window --mode demo --frames 180 --video outputs/demo.mp4
```

### demo 物理模式

当前 `demo` 模式不是 MPM，也不是物理真实仿真。它的作用是验证接口：

```text
rest_position
  -> position(t)
  -> velocity(t)
  -> deformation_gradient(t)
  -> current_covariance(t) = F(t) * base_covariance * F(t)^T
  -> render_frame()
```

这条链路跑通后，后续可以把 `demo` 内部替换为 MPM，而不用改渲染器。

### 清晰度改动

第 2 轮渲染器中低通项固定为 `0.3`，会让画面更稳定但偏糊。本轮改成配置：

```text
RenderConfig.low_pass_variance
```

默认：

```text
0.05
```

并通过命令行暴露：

```text
--low-pass
```

同时暴露：

```text
--width
--height
--focal-length
--radius-scale
```

建议提高清晰度时优先尝试：

```powershell
python main.py --cpu --width 1280 --height 720 --focal-length 920 --low-pass 0.03
```

### 关于 Taichi 的使用

当前 Taichi 已用于 field、投影、splat、tonemap 和动态 step kernel。但 Taichi 本身不是 3DGS 渲染器，也不会自动提供 Gaussian projection、conic rasterization 或 alpha compositing。

因此当前工程仍然需要手写 3DGS 渲染管线。这样做的意义是：

- 学习和控制 3DGS 的核心数据流。
- 让渲染变量和未来物理变量共用同一个 `GaussianParticleSet`。
- 后续 MPM 能直接在 Taichi field 上更新粒子状态。

### 当前限制

- `demo` 模式只是动态接口测试，不代表真实物理。
- GUI 使用 `ti.GUI` 显示 numpy 图像，不是高性能交互式 3DGS viewer。
- 视频导出依赖系统 `ffmpeg`。
- 当前渲染仍未实现 tile-based 高性能排序。

### 后续方向

- 接入最小 MPM，使 `PhysicsBridge.step(dt)` 产生真实物理状态。
- 增加相机控制，让 GUI 中可以旋转/缩放观察场景。
- 增加二进制 PLY 和 COLMAP 相机读取。
- 优化渲染性能：tile binning、tile 内排序、减少 CPU/GPU 同步。

## 第 4 次改动：修复动态 GUI 图像维度不匹配

### 问题

运行 `simulate.py` 时，`ti.GUI.set_image()` 报错：

```text
AssertionError: Image resolution does not match GUI resolution
```

原因是图像维度约定不同：

```text
GaussianRenderer.render_frame()
  -> 为了保存 PPM，返回 [height, width, channel]

ti.GUI.set_image()
  -> 期望 [width, height, channel]
```

### 修改内容

在 `simulate.py` 中，对 GUI 显示路径单独转置：

```text
image_float = transpose(image_u8, (1, 0, 2)) / 255
```

保存图片和视频帧仍然使用 `render_frame()` 返回的 `[height, width, channel]`，不受影响。

### 后续方向

- 可以进一步在 `GaussianRenderer` 中提供两个明确接口：
  - `render_frame_for_file()`
  - `render_frame_for_gui()`
- 或者统一内部图像维度约定，减少转置位置。

## 第 5 次改动：加入教学版弹性 MPM 和长方条 PLY 测试场景

### 主要内容

- 新增 `gs_particle_pipeline/elastic_mpm.py`，实现教学版弹性 MPM。
- `PhysicsBridge` 新增 `mode="elastic"`，调用 `ElasticMPMSolver.substep(dt)`。
- `GaussianParticleSet` 新增物理属性字段：
  - `affine_C`
  - `material_id`
  - `pinned`
  - `density`
  - `youngs_modulus`
  - `poisson_ratio`
- `GaussianHostData` 增加对应 CPU 侧可选物理字段。
- `ply_loader.py` 支持读取 PLY 中的物理字段：
  - `mass`
  - `volume`
  - `density`
  - `youngs_modulus`
  - `poisson_ratio`
  - `material_id`
  - `pinned`
- 新增 `make_elastic_bar.py`，生成带物理字段的长方条 ASCII PLY。
- `simulate.py` 支持：
  - `--mode elastic`
  - `--mpm-grid`
  - `--gravity`
- `README.md` 增加弹性长方条测试命令。

### 当前弹性 MPM 数据流

```text
GaussianParticleSet
  position / velocity / deformation_gradient / affine_C
  mass / volume / youngs_modulus / poisson_ratio / pinned
    -> P2G
    -> grid_v, grid_m
    -> grid gravity / boundary
    -> G2P
    -> update position / velocity / F / C
    -> current_covariance = F * base_covariance * F^T
    -> GaussianRenderer.render_frame()
```

### 材质参数体现方式

当前弹性体使用 fixed-corotated 弹性模型。每个粒子的：

```text
youngs_modulus
poisson_ratio
```

会转换为 Lame 参数：

```text
mu = E / (2 * (1 + nu))
lambda = E * nu / ((1 + nu) * (1 - 2 * nu))
```

并进入应力计算：

```text
stress = 2 * mu * (F - R) * F^T + I * lambda * J * (J - 1)
```

其中：

- `F`: 当前粒子变形梯度。
- `R`: `F` 的极分解旋转部分。
- `J`: `det(F)`，体积变化。
- `E`: 杨氏模量，越大越硬。
- `nu`: 泊松比，越接近 0.5 越接近不可压。

### 长方条测试

生成场景：

```powershell
python make_elastic_bar.py --output assets/elastic_bar.ply
```

运行连续多帧：

```powershell
python simulate.py --cpu --mode elastic --ply assets/elastic_bar.ply --particles 600 --frames 240 --dt 0.0005 --substeps 20 --camera-z 3.0 --focal-length 1100
```

长方条左端 `pinned=1`，其余粒子在重力和弹性应力下运动，因此可以观察下垂和振动。

### 当前限制

- 这是教学版弹性 MPM，不是完整生产级 MPM。
- 当前只实现 fixed-corotated 弹性体。
- 没有塑性、流体、沙土、多相耦合。
- 没有碰撞物体，只做网格边界处理。
- 当前 `material_id` 已存在，但还没有用于多模型分派。

### 后续方向

- 根据 `material_id` 分派不同本构模型。
- 加入 Neo-Hookean、von Mises、Drucker-Prager、Herschel-Bulkley。
- 增加刚体/地面碰撞 SDF。
- 增加 internal filling，让真实 3DGS 表面点变成体积粒子。
- 将 MPM 参数独立为 `materials.json` 或场景配置文件。

## 第 6 次改动：重整当前文件结构和核心流水线说明

### 背景

随着工程加入静态渲染、PLY 导入、动态渲染、弹性 MPM、长方条场景生成，`Final` 目录中的文件已经分化为：

- 核心入口。
- 核心库代码。
- 测试/场景生成脚本。
- 文档和记录。
- 输出/数据/缓存目录。

继续只看早期分轮 pipeline 容易误解当前主链路。

### 修改内容

- 在 `pipeline.md` 顶部新增“当前权威流水线总览”。
- 明确当前核心入口：
  - `main.py`
  - `simulate.py`
- 明确当前核心库：
  - `config.py`
  - `particles.py`
  - `ply_loader.py`
  - `sample_scene.py`
  - `physics_interface.py`
  - `elastic_mpm.py`
  - `renderer.py`
- 明确测试/场景生成文件：
  - `make_elastic_bar.py`
  - `assets/elastic_bar.ply`
- 明确当前核心流水线不依赖：
  - `outputs/`
  - `tandt_db/`
  - `__pycache__/`
- 系统整理了接口和参数传递：
  - 命令行参数到 `CameraConfig / RenderConfig / ElasticMPMConfig`
  - `GaussianHostData` 到 `GaussianParticleSet`
  - `PhysicsBridge` 到 `ElasticMPMSolver`
  - `GaussianParticleSet` 到 `GaussianRenderer`

### 当前结论

以后判断“当前怎么运行、核心文件是什么、参数在哪里传递”，优先看 `pipeline.md` 顶部的“当前权威流水线总览”。后面的第 1-5 轮记录保留作为历史演进。

### 后续方向

- 可以进一步把历史 pipeline 移入单独的 `pipeline_history.md`，让 `pipeline.md` 只保留当前版本。
- 可以新增 `materials.json` 或 `scene_config.json`，把物理参数从 PLY 和命令行中拆出来。

## 第 7 次改动：为长方条弹性测试增加进度输出和快速点渲染模式

### 问题

长方条测试命令在 CPU 上容易表现为“程序未响应”：

```powershell
python simulate.py --cpu --mode elastic --ply assets/elastic_bar.ply --particles 300 --frames 30 --dt 0.0005 --substeps 20 --camera-z 3.0 --focal-length 1100
```

主要原因：

- Taichi 第一次运行 kernel 会编译，启动阶段可能没有输出。
- 弹性 MPM 每帧要执行 `substeps` 次。
- 原有 Gaussian 渲染每个可见 Gaussian 都会调用一次 `splat_one` kernel，CPU 上开销很大。
- GUI 更新期间窗口可能表现为未响应。

### 修改内容

- `simulate.py` 新增启动参数摘要输出。
- `simulate.py` 新增：
  - `--progress`
  - `--profile`
  - `--render-mode gaussian|points`
  - `--point-radius`
- 默认动态渲染模式改为 `points`，用于快速观察物理是否在动。
- `renderer.py` 新增：
  - `splat_points_fast()`
  - `render_fast_points()`

### 推荐调试命令

先用快速点渲染判断 MPM 是否在跑：

```powershell
python simulate.py --cpu --mode elastic --ply assets/elastic_bar.ply --particles 300 --frames 30 --dt 0.0005 --substeps 10 --camera-z 3.0 --focal-length 1100 --profile
```

如果确认物理正常，再切回 Gaussian 渲染：

```powershell
python simulate.py --cpu --mode elastic --ply assets/elastic_bar.ply --particles 300 --frames 30 --dt 0.0005 --substeps 10 --camera-z 3.0 --focal-length 1100 --render-mode gaussian --profile
```

### 当前取舍

`points` 模式不是 3DGS 渲染，只是调试显示。它的作用是把“物理是否运行”和“Gaussian 渲染是否太慢”拆开检查。

正式观察 3DGS 效果仍应使用：

```text
--render-mode gaussian
```

## 第 8 次改动：精简为 PLY 动态 Gaussian 核心框架

### 目标

保留：

- 动态渲染。
- 规范 Gaussian 渲染。
- 只基于 PLY 文件导入场景。

移除冗余：

- `main.py` 单帧入口。
- `sample_scene.py` 合成场景。
- `demo/static` 物理模式。
- `points` 快速点渲染模式。

### 修改内容

- 删除 `main.py`。
- 删除 `gs_particle_pipeline/sample_scene.py`。
- `simulate.py` 改为必须提供 `--ply`。
- `simulate.py` 只保留 `--mode elastic`。
- `simulate.py` 移除 `--render-mode` 和 `--point-radius`。
- `simulate.py` 每帧固定调用 `GaussianRenderer.render_frame()`。
- `physics_interface.py` 删除 `static/demo` 分支，只保留弹性 MPM。
- `renderer.py` 删除 `render_fast_points()` 和 `splat_points_fast()`。
- 重写 `README.md`，只保留当前有效命令。
- 重写 `pipeline.md`，只保留当前核心流水线。

### 当前核心命令

```powershell
python make_elastic_bar.py --output assets/elastic_bar.ply
python simulate.py --cpu --ply assets/elastic_bar.ply --particles 300 --frames 30 --dt 0.0005 --substeps 10 --camera-z 3.0 --focal-length 1100 --profile
```

### 验证要求

本轮精简后需要验证：

- PLY 加载可用。
- 弹性 MPM 子步可运行。
- Gaussian 渲染可运行。
- GUI/无窗口输出链路仍可用。

## 第 9 次改动：新增 400-600 粒子的弹性布片 PLY 场景

### 主要内容

- 新增 `make_elastic_sheet.py`。
- 默认生成 `24 x 18 = 432` 个 Gaussian 粒子。
- 输出 `assets/elastic_sheet.ply`。
- 顶部一行 `pinned=1`，其余粒子受重力和弹性 MPM 影响。
- 每个粒子包含当前核心 PLY 导入器支持的 Gaussian 字段和物理字段。
- 更新 `README.md` 和 `pipeline.md`，加入布片生成和测试命令。

### 场景用途

该场景用于检验 400-600 粒子级别的连续动态 Gaussian 渲染。相比长方条，布片更容易观察：

- 固定边界。
- 重力下垂。
- 弹性振动。
- Gaussian covariance 随 MPM 变形传递。

### 使用命令

```powershell
python make_elastic_sheet.py --output assets/elastic_sheet.ply
python simulate.py --cpu --ply assets/elastic_sheet.ply --particles 432 --frames 30 --dt 0.0005 --substeps 8 --camera-z 3.0 --focal-length 950 --profile
```

### 后续方向

- 可以用 `--nx 25 --ny 24` 生成 600 粒子布片。
- 可以增加固定两角或固定上边两端的模式，让布料形变更明显。

## 第 10 次改动：加入 lab0 风格 Mass-Spring 布料并改善 Gaussian 清晰度

### 主要内容

- 阅读 `Labs/0-GettingStarted/CaseMassSpring.cpp` 和 `MassSpringSystem.h`。
- 新增 `gs_particle_pipeline/mass_spring.py`。
- `PhysicsBridge` 支持 `mode="mass-spring"`。
- `simulate.py` 新增：
  - `--spring-nx`
  - `--spring-ny`
  - `--spring-stiffness`
  - `--spring-damping`
  - `--spring-gravity`
- `make_elastic_sheet.py` 调整为更接近 lab0：
  - 初始布片为水平网格。
  - 默认 `22 x 22 = 484` 粒子。
  - 默认固定顶部两角。
  - 新增 `--pin-mode top-corners|top-row`。
  - 新增 `--splat-scale` 控制 Gaussian 尺寸。
- `renderer.py` 修正背景合成：
  - 以前 clear 时直接把背景写入 color，后续 alpha 会叠加在背景上，画面偏灰偏糊。
  - 现在 color 从 0 开始，tonemap 时使用 `color + transmittance * background`。
- 默认 `low_pass_variance` 从 `0.05` 降到 `0.005`。

### lab0 MassSpring 关键点

lab0 布料结构：

```text
(n+1) x (n+1) 网格
相邻结构弹簧
跨两格弯曲弹簧
对角剪切弹簧
固定两个角点
```

当前实现复用了这个拓扑，但求解器不是 lab0 的 Eigen 隐式求解，而是 Taichi 显式小步长求解。

### 高清 Gaussian 改善

本轮没有接入外部 CUDA 3DGS rasterizer，因为那需要重写渲染后端和数据格式。先做低成本但有效的清晰度修复：

- 减小 PLY 中 Gaussian `scale_*`。
- 减小 renderer 的 `low_pass`。
- 修正背景透射合成。
- 推荐使用更高分辨率和更大 focal length。

### 推荐命令

```powershell
python make_elastic_sheet.py --output assets/mass_spring_sheet.ply --nx 22 --ny 22 --pin-mode top-corners --splat-scale 0.18
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --frames 120 --dt 0.003 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --low-pass 0.003 --profile
```

### 后续方向

- 如果要真正接近官方 3DGS 高清效果，需要接入 tile-based CUDA/Taichi rasterizer，避免当前逐 Gaussian kernel 调用。
- 可以增加线框/弹簧可视化辅助，但核心渲染仍保持 Gaussian。

## 第 11 次改动：修复布料“看起来不动”的诊断与视角问题

### 问题

用户观察到 Mass-Spring 布料保持水平静止。诊断后发现物理实际在运动，但之前有两个问题：

- 相机正对 z 方向，布料初始为水平 x-z 平面，y 方向下垂在屏幕上几乎不可见。
- 默认重力较小，短时间内位移不明显。

### 修改内容

- `CameraConfig` 新增 `pitch_degrees`。
- `simulate.py` 新增 `--pitch`，默认 `-28`，从上方斜看布料。
- `renderer.py` 增加简单 x 轴 pitch 变换，并同步旋转 covariance。
- `simulate.py` 新增 `--diagnose-state`。
- `GaussianParticleSet` 新增 `state_summary()`，打印：
  - 粒子包围盒最小值/最大值。
  - 平均速度。
  - 最大速度。
- Mass-Spring 默认参数调整：
  - `stiffness = 80`
  - `damping = 1.5`
  - `gravity = 2.5`
  - `velocity_damping = 0.985`
- `spring-nx/spring-ny` 默认改为 `22/22`，匹配当前默认布片。

### 诊断结果

使用命令：

```powershell
python simulate.py --cpu --no-window --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --frames 20 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --width 120 --height 90 --camera-z 3.0 --focal-length 160 --pitch -28 --low-pass 0.003 --profile --diagnose-state
```

结果显示：

```text
frame 1:  bbox_min.y = 0.480
frame 20: bbox_min.y = 0.458
mean_v 从 0.019 增长到 0.229
```

说明物理确实在下垂；原先主要是视角和参数导致视觉上不明显。

### 推荐命令

```powershell
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

如果需要确认运动：

```powershell
python simulate.py --cpu --no-window --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --frames 20 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --diagnose-state --profile
```

## 第 12 次改动：加入 Mass-Spring 布料的密集 Gaussian 渲染层

### 背景

400 多个 Mass-Spring 节点可以验证布料动力学，但直接把这些粗节点渲染成 Gaussian 时，画面会表现为离散彩球，视觉上不连续，也容易出现“模糊点阵”的不适感。

本轮不进入真正高密度 3DGS / GPU rasterizer 的大改造，只做第一步：保持粗物理求解不变，在渲染端派生一个更密的 Gaussian 层。

### 修改内容

- 新增 `gs_particle_pipeline/dense_cloth.py`。
- 新增 `DenseClothConfig`，配置 `sim_nx / sim_ny / upsample / splat_scale / opacity`。
- 新增 `make_dense_cloth_host_data()`，从 PLY 导入的粗布料节点双线性生成密集渲染 Gaussian。
- 新增 `DenseClothRenderLayer.update()`，每帧把粗物理网格的当前位置插值到密集渲染层。
- `simulate.py` 新增：
  - `--dense-render`
  - `--render-upsample`
  - `--render-splat-scale`
  - `--render-opacity`
- `simulate.py` 的 pipeline 拆分为：
  - `sim_particles`：物理求解粒子。
  - `render_particles`：实际送入 GaussianRenderer 的粒子；默认等于 `sim_particles`，启用 dense render 后改为密集渲染粒子。
- `README.md` 和 `pipeline.md` 追加 dense render 使用方式和参数传递说明。

### 数据流

```text
22 x 22 sim_particles
  -> MassSpringSolver.step()
  -> DenseClothRenderLayer.update()
  -> 43 x 43 render_particles
  -> GaussianRenderer.render_frame()
```

其中 `upsample=2` 时，`22 x 22 = 484` 个物理粒子会生成 `43 x 43 = 1849` 个渲染 Gaussian。

### Mass-Spring 参数位置

Mass-Spring 的参数在 `simulate.py` 命令行中调整：

```text
--spring-stiffness  弹簧刚度
--spring-damping    弹簧阻尼
--spring-gravity    Mass-Spring 重力
```

如果希望布料下垂更明显，可以增大：

```powershell
--spring-gravity 4.0
```

### 推荐命令

生成更小 splat 的 Mass-Spring 布片：

```powershell
python make_elastic_sheet.py --output assets/mass_spring_sheet.ply --nx 22 --ny 22 --pin-mode top-corners --splat-scale 0.08
```

运行密集渲染布片：

```powershell
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 2 --render-splat-scale 0.10 --render-opacity 0.45 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

更强重力：

```powershell
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 2 --spring-gravity 4.0 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

### 当前结论

本轮改善的是“粗物理节点到更密渲染 Gaussian”的中间表达，适合作为真正高密度 3DGS 接入前的过渡层。它没有改变当前 GaussianRenderer 的教学实现性质，也没有接入官方 CUDA 3DGS rasterizer。

## 第 13 次改动：为官方 CUDA 3DGS rasterizer 接入预备数据和相机接口

### 背景

后续最终目标仍然是接入现有 CUDA 3DGS rasterizer，而不是长期维护当前教学版逐 Gaussian renderer。真正接入前，可以先完成不依赖 CUDA 编译的准备工作：

- 保留完整 3DGS PLY 数据。
- 提供官方风格相机矩阵。
- 明确当前 Taichi 粒子数据到官方 rasterizer 输入的转换边界。

参考文件：

```text
D:\git_repo\gaussian-splatting\gaussian_renderer\__init__.py
D:\git_repo\gaussian-splatting\gaussian_renderer\network_gui.py
```

### 修改内容

- `GaussianHostData` 新增：
  - `scale`
  - `rotation`
- `GaussianParticleSet` 新增：
  - `scale`
  - `rotation`
  - `16` 项 RGB SH 系数容量。
- `ply_loader.py` 支持：
  - `format ascii`
  - `format binary_little_endian`
  - `f_dc_0..2`
  - `f_rest_0..44`
- `ply_loader.py` 不再只把 DC 项提前转成 RGB，而是保留原始 SH 系数：

```text
sh_coefficients[0]    = f_dc
sh_coefficients[1..15] = f_rest_0..44
```

- `renderer.py` 调整教学版 SH 评估：
  - 使用 `0.5 + SH_C0 * f_dc` 的标准 DC 还原。
  - 当前只评估到一阶 SH，但数据层已经保留三阶 SH。
- 新增 `gs_particle_pipeline/camera.py`：
  - `GaussianCamera`
  - `focal_to_fov()`
  - `get_projection_matrix()`
  - `world_view_from_simple_camera()`
- 新增 `gs_particle_pipeline/cuda_rasterizer_adapter.py`：
  - `build_cuda_rasterizer_payload()`
  - 导出 `means3D / opacities / shs / scales / rotations`
  - 可选导出 `cov3D_precomp`
- `simulate.py` 构建 `GaussianCamera` 并在启动信息中打印 FoV。
- `README.md` 和 `pipeline.md` 记录 CUDA rasterizer 接入边界。

### 与官方 renderer 的对应关系

官方 `gaussian_renderer.render()` 中的关键输入：

```text
viewpoint_camera.FoVx / FoVy
viewpoint_camera.world_view_transform
viewpoint_camera.full_proj_transform
viewpoint_camera.camera_center
pc.get_xyz
pc.get_opacity
pc.get_scaling
pc.get_rotation
pc.get_features
optional pc.get_covariance()
```

当前工程预备接口：

```text
GaussianCamera.fov_x / fov_y
GaussianCamera.world_view_transform
GaussianCamera.full_proj_transform
GaussianCamera.camera_center
payload["means3D"]
payload["opacities"]
payload["scales"]
payload["rotations"]
payload["shs"]
payload["cov3D_precomp"]
```

### 当前结论

本轮仍未接入 `torch` 和 `diff_gaussian_rasterization`，也没有编译 CUDA extension。它完成的是正式接入前的数据和相机边界整理，使后续可以把当前 `GaussianParticleSet` 转成官方 rasterizer 的输入。

## 第 14 次改动：修复 Mass-Spring 省略步长参数时的数值发散

### 问题

用户观察到当前版本运行几帧后粒子飞散，表现为上方固定角附近先出现异常。自查后发现：

- 最近 GS 数据和相机接口改动没有直接修改 Mass-Spring 求解公式。
- 在显式传入 `--dt 0.002 --substeps 4` 时，120 帧 Mass-Spring dense render 保持稳定。
- 如果省略 `--dt` 和 `--substeps`，旧默认值是：

```text
dt = 1 / 60
substeps = 2
```

这个步长对当前显式 Mass-Spring 求解器过大，会在第 4-5 帧开始速度暴涨，随后出现 `inf/nan`。

### 复现现象

省略步长参数时，诊断输出会迅速变成：

```text
frame 4: mean_v 明显暴涨
frame 5: bbox 扩展到百级尺度
frame 11: mean_v=inf
frame 12: bbox_min=(nan,nan,nan)
```

### 修改内容

`simulate.py` 将 `--dt` 和 `--substeps` 改为 mode-aware 默认值：

```text
mode="mass-spring": dt=0.002,  substeps=4
mode="elastic":     dt=0.0005, substeps=10
```

如果用户显式传入 `--dt` 或 `--substeps`，仍然以用户命令行为准。

### 结论

这次问题不是 MPM 或 Mass-Spring 算法公式被 GS 接口改动破坏，而是显式求解器使用了过大的通用默认步长。后续做真正 GS 接入前，应继续保留 `--diagnose-state` 来区分“物理状态发散”和“渲染显示异常”。

## 第 15 次改动：建立可选择渲染后端并接入官方 CUDA GS 接口边界

### 目标

在真正编译和运行官方 CUDA rasterizer 之前，先完成渲染层接口设计：

- 默认渲染管线是 `gs`，对应真正的官方 CUDA 3DGS rasterizer。
- 保留当前 Taichi 教学 renderer，命名为 `reference`，作为调试和对照路径。
- `simulate.py` 主循环不直接依赖具体渲染器，只调用统一的 `render_frame()`。
- 仿真后的 `render_particles` 状态直接送入后端；后续 CUDA 后端可读取同一份动态粒子状态。

### 修改内容

- 新增 `gs_particle_pipeline/render_backend.py`：

```text
create_renderer("gs" or "reference", ...)
```

- 新增 `gs_particle_pipeline/cuda_renderer.py`：
  - `CudaGaussianRenderer`
  - 运行时导入 `torch` 和 `diff_gaussian_rasterization`
  - 使用官方 `GaussianRasterizationSettings`
  - 使用官方 `GaussianRasterizer`
  - 输出统一为 `uint8 [height, width, 3]`
- `simulate.py` 新增命令行参数：

```text
--renderer gs|reference
--gs-device
--gs-use-covariance
```

- `--renderer` 默认值改为：

```text
gs
```

- `renderer.py` 继续保留为 Taichi reference renderer。
- `cuda_rasterizer_adapter.py` 将 `current_covariance` 压缩为官方常用的 6 项对称 covariance 格式，用于 `cov3D_precomp`。
- `README.md` 和 `pipeline.md` 更新渲染后端说明。

### 当前运行方式

如果本机还没有 CUDA rasterizer 依赖，使用 reference 路径：

```powershell
python simulate.py --renderer reference --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 2 --frames 20
```

如果后续在虚拟环境或云服务器上安装好官方依赖，则使用默认 GS 路径：

```powershell
python simulate.py --renderer gs --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484
```

### 设计结论

这一步没有复制整个 `D:\git_repo\gaussian-splatting` 仓库，也没有引入训练 pipeline。当前只复用官方渲染后端的接口形态，并在 `cuda_renderer.py` 中预留直接调用官方 rasterizer 的最小路径。后续真正要跑 `--renderer gs` 时，需要先在本机虚拟环境或云服务器中安装 `torch` CUDA 和 `diff_gaussian_rasterization`。

## 第 16 次改动：修正密集布料 GS 渲染的默认覆盖比例

### 问题

云端使用官方 GS rasterizer 渲染布料时，画面几乎全黑，只能看到少量白点。自查后发现核心原因不是物理节点数，而是密集渲染层的 Gaussian 尺寸默认值过小：

```text
旧默认 render_splat_scale = 0.10
此前测试命令甚至使用过 0.06
```

`render_splat_scale` 的含义是“占 dense 渲染网格间距的比例”。例如 `22 x 22` 物理网格、`upsample=10` 时，dense 间距约为原网格间距的 1/10；如果再乘 `0.06`，每个 Gaussian 会变成极小的亚像素点，无法互相覆盖成布料面。

### 修改内容

- `DenseClothConfig.splat_scale` 默认值从 `0.10` 改为 `0.80`。
- `DenseClothConfig.opacity` 默认值从 `0.45` 改为 `0.85`。
- `simulate.py` 命令行默认值同步修改：

```text
--render-splat-scale 0.80
--render-opacity 0.85
```

- `README.md` 和 `pipeline.md` 同步更新推荐命令。

### 推荐范围

用于形成连续布料面时，建议：

```text
--render-upsample 5 到 10
--render-splat-scale 0.60 到 1.20
--render-opacity 0.70 到 0.95
```

如果仍然看到离散点，优先增大 `--render-splat-scale`，而不是盲目增加物理 `nx/ny`。
## 第 17 次改动：自查 CUDA GS 布料视角和稠密渲染诊断

### 问题

云端使用 `--renderer gs` 渲染 Mass-Spring 布料时，画面表现为少量白点、点延伸到屏幕外，和 Taichi reference 中能看到完整布料面的效果差异很大。

### 自查结论

- Mass-Spring 粗物理层仍然只更新 `22 x 22 = 484` 个仿真节点，这是设计预期。
- 启用 `--dense-render --render-upsample 10` 后，渲染层会生成 `211 x 211 = 44521` 个 Gaussian。已用 reference 后端本地验证启动日志显示 `render_particles=44521, dense=True`。
- 因此“看起来只有 484 个点”不是稠密层没有生成，而更可能是 CUDA GS 路径的相机投影、屏幕范围、scale 或 opacity 设置导致视觉上没有连成面。
- CUDA GS 相机发现一处确定问题：官方 `world_view_transform` 和 `projection_matrix` 都采用 transpose 后的 row-vector 约定；当前工程此前只对 world-view 做了 transpose，projection 没有 transpose，可能导致 GS 与 reference 视角不一致。

### 修改内容

- 修正 `gs_particle_pipeline/camera.py` 中 `GaussianCamera.from_config()` 的 projection 矩阵约定：

```text
projection_matrix = get_projection_matrix(...).transpose()
full_proj_transform = world_view_transform @ projection_matrix
```

- 新增 `--diagnose-render` 参数，仅用于 CUDA GS 后端首帧诊断，打印：

```text
n
world_bbox_min / world_bbox_max
camera_z_minmax
screen_bbox_x / screen_bbox_y
visible_ndc
opacity_minmax
scale_minmax
```

### 推荐诊断命令

```bash
python simulate.py --renderer gs --no-window --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 10 --render-splat-scale 0.80 --render-opacity 0.85 --frames 1 --spring-nx 22 --spring-ny 22 --width 1280 --height 720 --camera-z 3.0 --focal-length 1100 --pitch -28 --profile --diagnose-state --diagnose-render
```

启动日志必须包含：

```text
render_particles=44521, dense=True
```

诊断日志中如果 `visible_ndc` 很小，优先调整 `--camera-z`、`--focal-length`、`--pitch`；如果 `scale_minmax` 很小，优先增大 `--render-splat-scale`。
