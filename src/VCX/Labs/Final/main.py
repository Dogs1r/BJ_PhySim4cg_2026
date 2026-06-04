import argparse

import taichi as ti

from gs_particle_pipeline.config import CameraConfig, RenderConfig
from gs_particle_pipeline.particles import GaussianParticleSet
from gs_particle_pipeline.physics_interface import PhysicsBridge
from gs_particle_pipeline.ply_loader import load_3dgs_ply
from gs_particle_pipeline.renderer import GaussianRenderer, save_ppm
from gs_particle_pipeline.sample_scene import make_demo_scene


def parse_args():
    parser = argparse.ArgumentParser(description="Taichi 静态 3DGS 基础流水线")
    parser.add_argument("--output", default="outputs/static_demo.ppm", help="输出 PPM 图像路径")
    parser.add_argument("--particles", type=int, default=512, help="合成 Gaussian 粒子数量上限")
    parser.add_argument("--width", type=int, default=800, help="图像宽度")
    parser.add_argument("--height", type=int, default=600, help="图像高度")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 后端")
    parser.add_argument("--ply", default="", help="可选：导入 3DGS 导出的 ASCII PLY 文件")
    return parser.parse_args()


def main():
    args = parse_args()
    ti.init(arch=ti.cpu if args.cpu else ti.gpu)

    camera = CameraConfig(width=args.width, height=args.height)
    render = RenderConfig(max_particles=args.particles)

    particles = GaussianParticleSet(max_particles=render.max_particles)
    if args.ply:
        host_data = load_3dgs_ply(args.ply, max_particles=render.max_particles)
    else:
        host_data = make_demo_scene(render.max_particles)
    particles.load_from_host(host_data)

    # 目前是静态 no-op，但数据已经经过物理接口，后续可替换为 MPM step。
    physics = PhysicsBridge(particles)
    physics.step(0.0)

    renderer = GaussianRenderer(particles, camera, render)
    image = renderer.render_frame()

    import os

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_ppm(args.output, image)
    print(f"已输出静态 3DGS 演示图像: {args.output}")


if __name__ == "__main__":
    main()
