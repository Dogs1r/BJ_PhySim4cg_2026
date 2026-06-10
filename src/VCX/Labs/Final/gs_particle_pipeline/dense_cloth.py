from dataclasses import dataclass

import numpy as np
import taichi as ti

from .particles import GaussianHostData, GaussianParticleSet


@dataclass
class DenseClothConfig:
    """由粗 Mass-Spring 网格生成密集渲染 Gaussian 的配置。"""

    sim_nx: int
    sim_ny: int
    upsample: int = 2
    splat_scale: float = 0.80
    opacity: float = 0.85


def make_dense_cloth_host_data(source: GaussianHostData, config: DenseClothConfig) -> GaussianHostData:
    """从 PLY 导入的粗布料节点生成更密的渲染 Gaussian 初始数据。

    物理仍然只在粗网格上进行；这里生成的 Gaussian 只负责视觉连续覆盖。
    """

    if config.upsample <= 1:
        return source

    render_nx = (config.sim_nx - 1) * config.upsample + 1
    render_ny = (config.sim_ny - 1) * config.upsample + 1
    render_count = render_nx * render_ny
    source_count = config.sim_nx * config.sim_ny
    if source.position.shape[0] < source_count:
        raise ValueError("源 PLY 粒子数少于 sim_nx * sim_ny，无法生成密集布料渲染层")

    position = np.zeros((render_count, 3), dtype=np.float32)
    sh = np.zeros((render_count, 16, 3), dtype=np.float32)
    opacity = np.full((render_count,), config.opacity, dtype=np.float32)

    def sim_idx(i: int, j: int) -> int:
        return j * config.sim_nx + i

    idx = 0
    for rj in range(render_ny):
        fy = rj / config.upsample
        j0 = min(int(np.floor(fy)), config.sim_ny - 2)
        ty = fy - j0
        if rj == render_ny - 1:
            j0 = config.sim_ny - 2
            ty = 1.0
        for ri in range(render_nx):
            fx = ri / config.upsample
            i0 = min(int(np.floor(fx)), config.sim_nx - 2)
            tx = fx - i0
            if ri == render_nx - 1:
                i0 = config.sim_nx - 2
                tx = 1.0

            ids = [
                sim_idx(i0, j0),
                sim_idx(i0 + 1, j0),
                sim_idx(i0, j0 + 1),
                sim_idx(i0 + 1, j0 + 1),
            ]
            weights = np.array(
                [(1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty],
                dtype=np.float32,
            )
            position[idx] = sum(weights[k] * source.position[ids[k]] for k in range(4))
            sh[idx] = sum(weights[k] * source.sh_coefficients[ids[k]] for k in range(4))
            idx += 1

    # 以渲染网格间距设置 Gaussian。splat_scale 是 dense spacing 的覆盖比例；
    # 过小会在官方 GS rasterizer 中变成亚像素亮点，无法连成布料面。
    spacing_x = np.linalg.norm(source.position[sim_idx(1, 0)] - source.position[sim_idx(0, 0)]) / config.upsample
    spacing_y = np.linalg.norm(source.position[sim_idx(0, 1)] - source.position[sim_idx(0, 0)]) / config.upsample
    sigma = max(min(spacing_x, spacing_y) * config.splat_scale, 1e-4)
    covariance = np.tile(np.diag([sigma * sigma, sigma * sigma, sigma * sigma]).astype(np.float32), (render_count, 1, 1))

    mass = np.ones((render_count,), dtype=np.float32)
    volume = np.full((render_count,), sigma * sigma * sigma, dtype=np.float32)
    scale = np.full((render_count, 3), sigma, dtype=np.float32)
    rotation = np.zeros((render_count, 4), dtype=np.float32)
    rotation[:, 0] = 1.0
    return GaussianHostData(
        position=position,
        base_covariance=covariance,
        opacity=opacity,
        sh_coefficients=sh,
        scale=scale,
        rotation=rotation,
        mass=mass,
        volume=volume,
    )


@ti.data_oriented
class DenseClothRenderLayer:
    """每帧把密集渲染 Gaussian 绑定到粗物理布料网格上。"""

    def __init__(
        self,
        sim_particles: GaussianParticleSet,
        render_particles: GaussianParticleSet,
        config: DenseClothConfig,
    ):
        self.sim_particles = sim_particles
        self.render_particles = render_particles
        self.config = config
        self.render_nx = (config.sim_nx - 1) * config.upsample + 1
        self.render_ny = (config.sim_ny - 1) * config.upsample + 1

    @ti.kernel
    def update(self):
        for p in range(self.render_particles.count[None]):
            ri = p % self.render_nx
            rj = p // self.render_nx

            fx = ti.cast(ri, ti.f32) / ti.cast(self.config.upsample, ti.f32)
            fy = ti.cast(rj, ti.f32) / ti.cast(self.config.upsample, ti.f32)
            i0 = ti.min(ti.cast(ti.floor(fx), ti.i32), self.config.sim_nx - 2)
            j0 = ti.min(ti.cast(ti.floor(fy), ti.i32), self.config.sim_ny - 2)
            tx = fx - ti.cast(i0, ti.f32)
            ty = fy - ti.cast(j0, ti.f32)
            if ri == self.render_nx - 1:
                i0 = self.config.sim_nx - 2
                tx = 1.0
            if rj == self.render_ny - 1:
                j0 = self.config.sim_ny - 2
                ty = 1.0

            id00 = j0 * self.config.sim_nx + i0
            id10 = id00 + 1
            id01 = id00 + self.config.sim_nx
            id11 = id01 + 1
            w00 = (1.0 - tx) * (1.0 - ty)
            w10 = tx * (1.0 - ty)
            w01 = (1.0 - tx) * ty
            w11 = tx * ty

            self.render_particles.position[p] = (
                w00 * self.sim_particles.position[id00]
                + w10 * self.sim_particles.position[id10]
                + w01 * self.sim_particles.position[id01]
                + w11 * self.sim_particles.position[id11]
            )
            self.render_particles.current_covariance[p] = self.render_particles.base_covariance[p]
