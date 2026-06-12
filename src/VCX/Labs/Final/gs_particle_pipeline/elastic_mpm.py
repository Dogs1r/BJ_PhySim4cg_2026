from dataclasses import dataclass

import taichi as ti

from .particles import GaussianParticleSet


@dataclass
class ElasticMPMConfig:
    """教学版弹性 MPM 参数。"""

    grid_size: int = 32
    domain_min: float = -1.0
    domain_max: float = 1.0
    gravity: float = -9.8
    boundary_width: int = 3
    grid_v_damping_scale: float = 0.999
    rpic_damping: float = 0.0


@ti.data_oriented
class ElasticMPMSolver:
    """最小弹性体 MPM，用于验证 Gaussian 粒子和物理状态的传递。

    当前实现采用 MLS-MPM/APIC 风格的 P2G/Grid/G2P 循环和 fixed-corotated
    弹性模型。它是教学实现，不追求生产级稳定性和速度。
    """

    def __init__(self, particles: GaussianParticleSet, config: ElasticMPMConfig):
        self.particles = particles
        self.config = config
        self.n_grid = config.grid_size
        self.dx = (config.domain_max - config.domain_min) / config.grid_size
        self.inv_dx = 1.0 / self.dx
        self.stress_scale = 4.0 * self.inv_dx * self.inv_dx

        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_grid, self.n_grid, self.n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.n_grid, self.n_grid, self.n_grid))

    @ti.kernel
    def substep(self, dt: ti.f32):
        for i, j, k in self.grid_m:
            self.grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_m[i, j, k] = 0.0

        self._p2g(dt)
        self._grid_op(dt)
        self._g2p(dt)

    @ti.func
    def _weights(self, fx: ti.types.vector(3, ti.f32)):
        return [
            0.5 * (1.5 - fx) ** 2,
            0.75 - (fx - 1.0) ** 2,
            0.5 * (fx - 0.5) ** 2,
        ]

    @ti.func
    def _fixed_corotated_stress(self, p: ti.i32):
        f = self.particles.deformation_gradient[p]
        u, sig, v = ti.svd(f)
        r = u @ v.transpose()
        j = ti.max(0.2, f.determinant())

        e = self.particles.youngs_modulus[p]
        nu = self.particles.poisson_ratio[p]
        mu = e / (2.0 * (1.0 + nu))
        la = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        return 2.0 * mu * (f - r) @ f.transpose() + ti.Matrix.identity(ti.f32, 3) * la * j * (j - 1.0)

    @ti.func
    def _p2g(self, dt: ti.f32):
        for p in range(self.particles.count[None]):
            x = (self.particles.position[p] - ti.Vector([self.config.domain_min] * 3)) * self.inv_dx
            base = ti.cast(x - 0.5, ti.i32)
            fx = x - ti.cast(base, ti.f32)
            w = self._weights(fx)

            stress = self._fixed_corotated_stress(p)
            stress = (-dt * self.particles.volume[p] * self.stress_scale) * stress
            affine = stress + self.particles.mass[p] * self.particles.affine_C[p]

            if self.particles.pinned[p] == 1:
                affine = ti.Matrix.zero(ti.f32, 3, 3)

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                node = base + offset
                if node.x >= 0 and node.x < self.n_grid and node.y >= 0 and node.y < self.n_grid and node.z >= 0 and node.z < self.n_grid:
                    dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                    weight = w[i].x * w[j].y * w[k].z
                    momentum = self.particles.mass[p] * self.particles.velocity[p] + affine @ dpos
                    if self.particles.pinned[p] == 1:
                        momentum = ti.Vector([0.0, 0.0, 0.0])
                    self.grid_v[node.x, node.y, node.z] += weight * momentum
                    self.grid_m[node.x, node.y, node.z] += weight * self.particles.mass[p]

    @ti.func
    def _grid_op(self, dt: ti.f32):
        for i, j, k in self.grid_m:
            mass = self.grid_m[i, j, k]
            if mass > 0.0:
                v = self.grid_v[i, j, k] / mass
                v.y += dt * self.config.gravity
                v *= self.config.grid_v_damping_scale

                bw = self.config.boundary_width
                if i < bw and v.x < 0.0:
                    v.x = 0.0
                if i > self.n_grid - bw and v.x > 0.0:
                    v.x = 0.0
                if j < bw and v.y < 0.0:
                    v.y = 0.0
                if j > self.n_grid - bw and v.y > 0.0:
                    v.y = 0.0
                if k < bw and v.z < 0.0:
                    v.z = 0.0
                if k > self.n_grid - bw and v.z > 0.0:
                    v.z = 0.0

                self.grid_v[i, j, k] = v

    @ti.func
    def _g2p(self, dt: ti.f32):
        for p in range(self.particles.count[None]):
            x = (self.particles.position[p] - ti.Vector([self.config.domain_min] * 3)) * self.inv_dx
            base = ti.cast(x - 0.5, ti.i32)
            fx = x - ti.cast(base, ti.f32)
            w = self._weights(fx)

            new_v = ti.Vector([0.0, 0.0, 0.0])
            new_c = ti.Matrix.zero(ti.f32, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                node = base + offset
                if node.x >= 0 and node.x < self.n_grid and node.y >= 0 and node.y < self.n_grid and node.z >= 0 and node.z < self.n_grid:
                    dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                    weight = w[i].x * w[j].y * w[k].z
                    g_v = self.grid_v[node.x, node.y, node.z]
                    new_v += weight * g_v
                    new_c += 4.0 * self.inv_dx * weight * g_v.outer_product(dpos)

            if self.particles.pinned[p] == 1:
                self.particles.position[p] = self.particles.rest_position[p]
                self.particles.velocity[p] = ti.Vector([0.0, 0.0, 0.0])
                self.particles.affine_C[p] = ti.Matrix.zero(ti.f32, 3, 3)
                self.particles.deformation_gradient[p] = ti.Matrix.identity(ti.f32, 3)
            else:
                self.particles.velocity[p] = new_v
                self.particles.affine_C[p] = (1.0 - self.config.rpic_damping) * new_c
                self.particles.position[p] += dt * new_v
                self.particles.deformation_gradient[p] = (ti.Matrix.identity(ti.f32, 3) + dt * new_c) @ self.particles.deformation_gradient[p]
                u, sig, v = ti.svd(self.particles.deformation_gradient[p])
                clamped = ti.Matrix.zero(ti.f32, 3, 3)
                for d in ti.static(range(3)):
                    clamped[d, d] = ti.min(1.8, ti.max(0.45, sig[d, d]))
                self.particles.deformation_gradient[p] = u @ clamped @ v.transpose()

            f = self.particles.deformation_gradient[p]
            self.particles.current_covariance[p] = f @ self.particles.base_covariance[p] @ f.transpose()
