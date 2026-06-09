# Dynamic Gaussian Splatting + Elastic MPM

当前工程保留一条核心链路：

```text
PLY scene
  -> GaussianParticleSet
  -> Elastic MPM / Mass-Spring step
  -> optional dense cloth Gaussian render layer
  -> standard Gaussian renderer
  -> GUI / frames / video
```

## Install

```powershell
pip install -r requirements.txt
```

## Generate Test Scene

```powershell
python make_elastic_bar.py --output assets/elastic_bar.ply
python make_elastic_sheet.py --output assets/elastic_sheet.ply
```

这些脚本会生成带物理字段的 ASCII PLY，包括：

```text
x, y, z
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
f_dc_0, f_dc_1, f_dc_2
mass, volume, density
youngs_modulus, poisson_ratio
material_id, pinned
```

## Run Dynamic Rendering

```powershell
python simulate.py --cpu --ply assets/elastic_bar.ply --particles 300 --frames 30 --dt 0.0005 --substeps 10 --camera-z 3.0 --focal-length 1100 --profile
```

400-600 粒子布片测试：

```powershell
python simulate.py --cpu --ply assets/elastic_sheet.ply --particles 432 --frames 30 --dt 0.0005 --substeps 8 --camera-z 3.0 --focal-length 950 --profile
```

接近 lab0 MassSpring 的布片效果：

```powershell
python make_elastic_sheet.py --output assets/mass_spring_sheet.ply --nx 22 --ny 22 --pin-mode top-corners --splat-scale 0.08
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

为了避免 400 多个粗粒子直接渲染成明显的“彩球”，可以启用密集渲染层。物理仍然只在 22 x 22 个 Mass-Spring 节点上运行；渲染端每帧从粗网格双线性插值得到更密的 Gaussian：

```powershell
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 2 --render-splat-scale 0.10 --render-opacity 0.45 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

Mass-Spring 参数在 `simulate.py` 的命令行参数里调整：

```text
--spring-stiffness  弹簧刚度
--spring-damping    弹簧阻尼
--spring-gravity    Mass-Spring 重力
```

如果希望布料下垂更明显，可以增大 `--spring-gravity`，例如：

```powershell
python simulate.py --cpu --mode mass-spring --ply assets/mass_spring_sheet.ply --particles 484 --dense-render --render-upsample 2 --spring-gravity 4.0 --frames 120 --dt 0.002 --substeps 4 --spring-nx 22 --spring-ny 22 --camera-z 3.0 --focal-length 1100 --pitch -28 --low-pass 0.003 --profile
```

如果怀疑物理没有运动，加入：

```powershell
--diagnose-state
```

它会打印粒子包围盒和速度。布料下垂时，`bbox_min` 的 y 值会逐帧下降。

Notes:

- `simulate.py` requires `--ply`.
- Rendering always uses the Gaussian renderer. Lower `--low-pass` and smaller generated `--splat-scale` make the image sharper.
- The first frame can be slow because Taichi compiles kernels lazily.
- `--profile` prints per-frame physics/render/gui/save timings.

## Export Video

```powershell
python simulate.py --cpu --no-window --ply assets/elastic_bar.ply --particles 300 --frames 120 --dt 0.0005 --substeps 10 --video outputs/elastic_bar.mp4
```

Video export requires `ffmpeg` on `PATH`.

## Core Files

```text
simulate.py                              dynamic entry point
make_elastic_bar.py                      PLY test-scene generator
make_elastic_sheet.py                    400-600 particle sheet generator
gs_particle_pipeline/config.py           camera/render config
gs_particle_pipeline/particles.py        shared particle/Gaussian fields
gs_particle_pipeline/ply_loader.py       ASCII PLY loader
gs_particle_pipeline/physics_interface.py physics bridge
gs_particle_pipeline/elastic_mpm.py      elastic MPM solver
gs_particle_pipeline/mass_spring.py     lab0-style Mass-Spring cloth solver
gs_particle_pipeline/dense_cloth.py     dense render Gaussian layer for cloth
gs_particle_pipeline/renderer.py         standard Gaussian renderer
```
