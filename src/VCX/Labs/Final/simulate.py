import argparse
import os
import subprocess
import time

import numpy as np
import taichi as ti

from gs_particle_pipeline.config import CameraConfig, RenderConfig
from gs_particle_pipeline.dense_cloth import DenseClothConfig, DenseClothRenderLayer, make_dense_cloth_host_data
from gs_particle_pipeline.elastic_mpm import ElasticMPMConfig
from gs_particle_pipeline.mass_spring import MassSpringConfig
from gs_particle_pipeline.particles import GaussianParticleSet
from gs_particle_pipeline.physics_interface import PhysicsBridge
from gs_particle_pipeline.ply_loader import load_3dgs_ply
from gs_particle_pipeline.renderer import GaussianRenderer, save_ppm


def parse_args():
    parser = argparse.ArgumentParser(description="Taichi 动态 3DGS/MPM 渲染链路")
    parser.add_argument("--ply", required=True, help="导入 3DGS/测试场景 ASCII PLY 文件")
    parser.add_argument("--particles", type=int, default=512, help="Gaussian 粒子数量上限")
    parser.add_argument("--width", type=int, default=1280, help="窗口/输出图像宽度")
    parser.add_argument("--height", type=int, default=720, help="窗口/输出图像高度")
    parser.add_argument("--focal-length", type=float, default=920.0, help="针孔相机焦距，越大画面越放大")
    parser.add_argument("--camera-z", type=float, default=3.2, help="相机到场景的 z 方向距离")
    parser.add_argument("--pitch", type=float, default=-28.0, help="相机俯仰角，负值从上方观察布料")
    parser.add_argument("--low-pass", type=float, default=0.005, help="2D covariance 低通项，越小越锐利")
    parser.add_argument("--radius-scale", type=float, default=3.0, help="Gaussian 屏幕影响半径倍数")
    parser.add_argument("--fps", type=int, default=30, help="导出视频帧率")
    parser.add_argument("--frames", type=int, default=150, help="最多渲染帧数")
    parser.add_argument("--dt", type=float, default=1.0 / 60.0, help="物理子步长")
    parser.add_argument("--substeps", type=int, default=2, help="每个渲染帧包含的物理子步数")
    parser.add_argument("--mode", choices=["elastic", "mass-spring"], default="elastic", help="物理接口模式")
    parser.add_argument("--mpm-grid", type=int, default=32, help="弹性 MPM 网格分辨率")
    parser.add_argument("--gravity", type=float, default=-9.8, help="弹性 MPM 重力加速度")
    parser.add_argument("--spring-nx", type=int, default=22, help="Mass-Spring 布料水平方向粒子数")
    parser.add_argument("--spring-ny", type=int, default=22, help="Mass-Spring 布料竖直方向粒子数")
    parser.add_argument("--spring-stiffness", type=float, default=80.0, help="Mass-Spring 弹簧刚度")
    parser.add_argument("--spring-damping", type=float, default=1.5, help="Mass-Spring 弹簧阻尼")
    parser.add_argument("--spring-gravity", type=float, default=2.5, help="Mass-Spring 重力，参考 lab0 的量级")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 后端")
    parser.add_argument("--no-window", action="store_true", help="不打开 GUI，仅导出帧/视频")
    parser.add_argument("--save-frames", default="", help="可选：逐帧 PPM 输出目录")
    parser.add_argument("--video", default="", help="可选：用 ffmpeg 将逐帧 PPM 合成为 mp4")
    parser.add_argument("--keep-frames", action="store_true", help="生成视频后保留逐帧图片")
    parser.add_argument("--progress", type=int, default=1, help="每隔多少帧打印一次进度，0 表示不打印")
    parser.add_argument("--profile", action="store_true", help="打印每帧 physics/render/gui/save 耗时")
    parser.add_argument("--diagnose-state", action="store_true", help="打印粒子包围盒和速度，检查物理是否在动")
    parser.add_argument("--dense-render", action="store_true", help="为 Mass-Spring 布料生成密集渲染 Gaussian")
    parser.add_argument("--render-upsample", type=int, default=2, help="密集渲染层相对物理网格的细分倍数")
    parser.add_argument("--render-splat-scale", type=float, default=0.10, help="密集渲染 Gaussian 尺寸")
    parser.add_argument("--render-opacity", type=float, default=0.45, help="密集渲染 Gaussian 不透明度")
    return parser.parse_args()


def build_pipeline(args):
    camera = CameraConfig(
        width=args.width,
        height=args.height,
        focal_length=args.focal_length,
        camera_z=args.camera_z,
        pitch_degrees=args.pitch,
    )
    render_capacity = args.particles
    if args.dense_render:
        render_capacity = ((args.spring_nx - 1) * args.render_upsample + 1) * ((args.spring_ny - 1) * args.render_upsample + 1)

    render = RenderConfig(
        max_particles=render_capacity,
        tile_radius_scale=args.radius_scale,
        low_pass_variance=args.low_pass,
    )

    sim_particles = GaussianParticleSet(max_particles=args.particles)
    sim_host_data = load_3dgs_ply(args.ply, max_particles=args.particles)
    sim_particles.load_from_host(sim_host_data)

    render_particles = sim_particles
    dense_layer = None
    if args.dense_render:
        dense_config = DenseClothConfig(
            sim_nx=args.spring_nx,
            sim_ny=args.spring_ny,
            upsample=args.render_upsample,
            splat_scale=args.render_splat_scale,
            opacity=args.render_opacity,
        )
        render_host_data = make_dense_cloth_host_data(sim_host_data, dense_config)
        render_particles = GaussianParticleSet(max_particles=render_capacity)
        render_particles.load_from_host(render_host_data)
        dense_layer = DenseClothRenderLayer(sim_particles, render_particles, dense_config)

    elastic_config = ElasticMPMConfig(grid_size=args.mpm_grid, gravity=args.gravity)
    mass_spring_config = MassSpringConfig(
        nx=args.spring_nx,
        ny=args.spring_ny,
        stiffness=args.spring_stiffness,
        damping=args.spring_damping,
        gravity=args.spring_gravity,
    )
    physics = PhysicsBridge(
        sim_particles,
        mode=args.mode,
        elastic_config=elastic_config,
        mass_spring_config=mass_spring_config,
    )
    renderer = GaussianRenderer(render_particles, camera, render)
    return sim_particles, render_particles, physics, renderer, dense_layer


def encode_video(frame_dir: str, video_path: str, fps: int) -> None:
    """调用 ffmpeg 把 PPM 图片序列合成为 mp4。"""

    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)
    pattern = os.path.join(frame_dir, "frame_%04d.ppm")
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        pattern,
        "-pix_fmt",
        "yuv420p",
        video_path,
    ]
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    ti.init(arch=ti.cpu if args.cpu else ti.gpu)

    particles, render_particles, physics, renderer, dense_layer = build_pipeline(args)
    print(
        "启动动态渲染: "
        f"mode={args.mode}, render=gaussian, sim_particles={args.particles}, "
        f"render_particles={render_particles.count[None]}, dense={dense_layer is not None}, "
        f"frames={args.frames}, substeps={args.substeps}, dt={args.dt}, "
        f"size={args.width}x{args.height}, cpu={args.cpu}",
        flush=True,
    )
    frame_dir = args.save_frames
    if args.video and not frame_dir:
        frame_dir = os.path.join("outputs", "frames_for_video")
    if frame_dir:
        os.makedirs(frame_dir, exist_ok=True)

    gui = None
    if not args.no_window:
        gui = ti.GUI("Dynamic 3D Gaussian Splatting", res=(args.width, args.height))

    frame = 0
    while frame < args.frames:
        frame_t0 = time.perf_counter()
        if gui is not None and not gui.running:
            break

        physics_t0 = time.perf_counter()
        for _ in range(args.substeps):
            physics.step(args.dt)
        if dense_layer is not None:
            dense_layer.update()
        physics_time = time.perf_counter() - physics_t0

        render_t0 = time.perf_counter()
        image_u8 = renderer.render_frame()
        render_time = time.perf_counter() - render_t0

        gui_time = 0.0
        if gui is not None:
            gui_t0 = time.perf_counter()
            # renderer.render_frame() 为文件保存返回 [height, width, channel]；
            # ti.GUI.set_image() 需要 [width, height, channel]，这里单独转置给 GUI。
            image_float = np.transpose(image_u8, (1, 0, 2)).astype(np.float32) / 255.0
            gui.set_image(image_float)
            gui.show()
            gui_time = time.perf_counter() - gui_t0

        save_time = 0.0
        if frame_dir:
            save_t0 = time.perf_counter()
            save_ppm(os.path.join(frame_dir, f"frame_{frame:04d}.ppm"), image_u8)
            save_time = time.perf_counter() - save_t0

        frame_time = time.perf_counter() - frame_t0
        if args.progress and (frame % args.progress == 0 or frame == args.frames - 1):
            if args.profile:
                state_text = ""
                if args.diagnose_state:
                    summary = particles.state_summary()
                    pmin = summary["min"]
                    pmax = summary["max"]
                    state_text = (
                        f" bbox_min=({pmin[0]:.3f},{pmin[1]:.3f},{pmin[2]:.3f})"
                        f" bbox_max=({pmax[0]:.3f},{pmax[1]:.3f},{pmax[2]:.3f})"
                        f" mean_v={summary['mean_speed']:.5f}"
                        f" max_v={summary['max_speed']:.5f}"
                    )
                print(
                    f"frame {frame + 1}/{args.frames}: "
                    f"physics={physics_time:.3f}s render={render_time:.3f}s "
                    f"gui={gui_time:.3f}s save={save_time:.3f}s total={frame_time:.3f}s"
                    f"{state_text}",
                    flush=True,
                )
            else:
                print(f"frame {frame + 1}/{args.frames} done ({frame_time:.3f}s)", flush=True)
        frame += 1

    if args.video:
        encode_video(frame_dir, args.video, args.fps)
        if not args.keep_frames:
            # 只清理本脚本创建的逐帧图片目录内容。
            for name in os.listdir(frame_dir):
                if name.startswith("frame_") and name.endswith(".ppm"):
                    os.remove(os.path.join(frame_dir, name))

    print(f"动态渲染完成：{frame} 帧")


if __name__ == "__main__":
    main()
