import numpy as np
import taichi as ti
import math

from .config import CameraConfig, RenderConfig
from .particles import GaussianParticleSet


@ti.data_oriented
class GaussianRenderer:
    """教学版标准 3DGS 渲染器。

    这里实现的是便于学习的 3DGS 核心数学流程，而不是原论文的高性能 CUDA
    tile rasterizer。关键步骤与标准 Gaussian Splatting 一致：

    1. 把 3D Gaussian 中心投影到屏幕；
    2. 用投影雅可比 J 将 3D covariance 投影为 2D covariance；
    3. 对 2D covariance 求逆，得到 conic 参数；
    4. 按深度从近到远 alpha 合成。

    当前为了代码清楚，排序在 CPU/Numpy 中完成，逐 Gaussian splat 在 Taichi kernel
    中完成。后续若要提速，可以把它替换为 tile binning + tile 内排序。
    """

    def __init__(self, particles: GaussianParticleSet, camera: CameraConfig, render: RenderConfig):
        self.particles = particles
        self.camera = camera
        self.render = render
        self.width = camera.width
        self.height = camera.height
        self.max_particles = render.max_particles
        pitch = math.radians(camera.pitch_degrees)
        self.pitch_cos = float(math.cos(pitch))
        self.pitch_sin = float(math.sin(pitch))

        self.color = ti.Vector.field(3, dtype=ti.f32, shape=(self.width, self.height))
        self.transmittance = ti.field(dtype=ti.f32, shape=(self.width, self.height))
        self.image_u8 = ti.Vector.field(3, dtype=ti.u8, shape=(self.width, self.height))

        # 投影阶段的中间变量。它们不是物理状态，只服务本帧渲染。
        self.projected_xy = ti.Vector.field(2, dtype=ti.f32, shape=self.max_particles)
        self.depth = ti.field(dtype=ti.f32, shape=self.max_particles)
        self.conic = ti.Vector.field(3, dtype=ti.f32, shape=self.max_particles)
        self.screen_radius = ti.field(dtype=ti.i32, shape=self.max_particles)
        self.visible = ti.field(dtype=ti.i32, shape=self.max_particles)

    @ti.kernel
    def clear(self):
        for i, j in self.color:
            self.color[i, j] = ti.Vector([0.0, 0.0, 0.0])
            # transmittance 表示当前像素还剩多少背景光能穿过，初始为 1。
            self.transmittance[i, j] = 1.0
            self.image_u8[i, j] = ti.cast(ti.Vector([0, 0, 0]), ti.u8)

    @ti.func
    def _world_to_camera(self, pos: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
        # 绕 x 轴做简单俯仰，让水平布料的下垂在画面中可见。
        y = self.pitch_cos * pos.y - self.pitch_sin * pos.z
        z = self.pitch_sin * pos.y + self.pitch_cos * pos.z + self.camera.camera_z
        return ti.Vector([pos.x, y, z])

    @ti.func
    def _covariance_to_camera(self, cov: ti.types.matrix(3, 3, ti.f32)) -> ti.types.matrix(3, 3, ti.f32):
        r = ti.Matrix(
            [
                [1.0, 0.0, 0.0],
                [0.0, self.pitch_cos, -self.pitch_sin],
                [0.0, self.pitch_sin, self.pitch_cos],
            ]
        )
        return r @ cov @ r.transpose()

    @ti.func
    def _evaluate_sh(self, p: ti.i32, view_dir: ti.types.vector(3, ti.f32)) -> ti.types.vector(3, ti.f32):
        # 当前保留一阶 SH 教学近似：c0 + cx*x + cy*y + cz*z。
        # 真正 3DGS 可扩展为 16 项或更高阶 SH。
        c = self.particles.sh_coefficients[p, 0]
        c += self.particles.sh_coefficients[p, 1] * view_dir.x
        c += self.particles.sh_coefficients[p, 2] * view_dir.y
        c += self.particles.sh_coefficients[p, 3] * view_dir.z
        return ti.max(ti.Vector([0.0, 0.0, 0.0]), ti.min(c, ti.Vector([1.0, 1.0, 1.0])))

    @ti.kernel
    def project(self):
        """把 3D Gaussian 投影成屏幕空间 2D Gaussian。

        公式：
            p_screen = project(p_camera)
            Sigma_2d = J * Sigma_3d * J^T

        其中 J 是透视投影对相机空间位置的雅可比。当前相机没有旋转，
        因此 world covariance 等于 camera covariance；之后接真实相机时需要
        使用 Sigma_camera = R_camera * Sigma_world * R_camera^T。
        """

        for p in range(self.particles.count[None]):
            pos = self.particles.position[p]
            cam = self._world_to_camera(pos)
            z = cam.z
            self.visible[p] = 0
            self.depth[p] = z

            if z > self.camera.near and z < self.camera.far:
                f = self.camera.focal_length
                u = f * cam.x / z + self.width * 0.5
                v = f * cam.y / z + self.height * 0.5

                cov3 = self._covariance_to_camera(self.particles.current_covariance[p])

                j00 = f / z
                j02 = -f * cam.x / (z * z)
                j11 = f / z
                j12 = -f * cam.y / (z * z)

                # 展开 Sigma_2d = J Sigma_3d J^T，只保留 2x2。
                a = (
                    j00 * j00 * cov3[0, 0]
                    + 2.0 * j00 * j02 * cov3[0, 2]
                    + j02 * j02 * cov3[2, 2]
                )
                b = (
                    j00 * j11 * cov3[0, 1]
                    + j00 * j12 * cov3[0, 2]
                    + j02 * j11 * cov3[2, 1]
                    + j02 * j12 * cov3[2, 2]
                )
                c = (
                    j11 * j11 * cov3[1, 1]
                    + 2.0 * j11 * j12 * cov3[1, 2]
                    + j12 * j12 * cov3[2, 2]
                )

                # 低通项用于避免过小 Gaussian 造成数值问题。
                # 数值越大画面越平滑但越容易糊；数值越小越锐利但更容易闪烁/破碎。
                a += self.render.low_pass_variance
                c += self.render.low_pass_variance
                det = a * c - b * b

                if det > 1e-8:
                    inv_det = 1.0 / det
                    self.conic[p] = ti.Vector([c * inv_det, -b * inv_det, a * inv_det])
                    self.projected_xy[p] = ti.Vector([u, v])

                    trace = 0.5 * (a + c)
                    delta = ti.sqrt(ti.max(0.01, trace * trace - det))
                    lambda_max = trace + delta
                    radius = ti.cast(ti.ceil(self.render.tile_radius_scale * ti.sqrt(lambda_max)), ti.i32)
                    self.screen_radius[p] = radius

                    if radius > 0 and u + radius >= 0 and u - radius < self.width and v + radius >= 0 and v - radius < self.height:
                        self.visible[p] = 1

    @ti.kernel
    def splat_one(self, p: ti.i32):
        if self.visible[p] == 1:
            center = self.projected_xy[p]
            radius = self.screen_radius[p]
            conic = self.conic[p]
            pos = self.particles.position[p]
            z = self.depth[p]
            cam = self._world_to_camera(pos)
            view_dir = ti.Vector([-cam.x, -cam.y, -cam.z]).normalized()
            rgb = self._evaluate_sh(p, view_dir)

            min_x = ti.max(0, ti.cast(ti.floor(center.x), ti.i32) - radius)
            max_x = ti.min(self.width - 1, ti.cast(ti.floor(center.x), ti.i32) + radius)
            min_y = ti.max(0, ti.cast(ti.floor(center.y), ti.i32) - radius)
            max_y = ti.min(self.height - 1, ti.cast(ti.floor(center.y), ti.i32) + radius)

            for i, j in ti.ndrange((min_x, max_x + 1), (min_y, max_y + 1)):
                d = ti.Vector([ti.cast(i, ti.f32) + 0.5 - center.x, ti.cast(j, ti.f32) + 0.5 - center.y])
                power = -0.5 * (conic.x * d.x * d.x + 2.0 * conic.y * d.x * d.y + conic.z * d.y * d.y)
                if power > -16.0:
                    alpha = ti.min(0.99, self.particles.opacity[p] * ti.exp(power))
                    if alpha > self.render.alpha_threshold:
                        # 近到远合成：C += T * alpha * color, T *= (1-alpha)。
                        t = self.transmittance[i, j]
                        self.color[i, j] += t * alpha * rgb
                        self.transmittance[i, j] = t * (1.0 - alpha)

    @ti.kernel
    def tonemap(self):
        for i, j in self.color:
            bg = ti.Vector([self.render.background_r, self.render.background_g, self.render.background_b])
            rgb = self.color[i, j] + self.transmittance[i, j] * bg
            rgb = ti.max(ti.Vector([0.0, 0.0, 0.0]), ti.min(rgb, ti.Vector([1.0, 1.0, 1.0])))
            self.image_u8[i, j] = ti.cast(rgb * 255.0, ti.u8)

    def render_frame(self) -> np.ndarray:
        self.clear()
        self.project()

        visible = self.visible.to_numpy()[: self.particles.count[None]]
        depth = self.depth.to_numpy()[: self.particles.count[None]]
        # z 越小越靠近相机，front-to-back alpha 合成需要从近到远。
        sorted_indices = np.argsort(depth)
        for p in sorted_indices:
            if visible[p] != 0:
                self.splat_one(int(p))

        self.tonemap()
        # Taichi field 维度是 [x, y]，保存图像时转为 [y, x, channel]。
        return np.transpose(self.image_u8.to_numpy(), (1, 0, 2))

    def render_frame_float(self) -> np.ndarray:
        """返回 [0, 1] float 图像，适合 ti.GUI 实时显示。"""

        return self.render_frame().astype(np.float32) / 255.0


def save_ppm(path: str, image: np.ndarray) -> None:
    """不用 Pillow，直接保存二进制 PPM，降低环境依赖。"""

    height, width, _ = image.shape
    with open(path, "wb") as f:
        f.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        f.write(image.astype(np.uint8).tobytes())
