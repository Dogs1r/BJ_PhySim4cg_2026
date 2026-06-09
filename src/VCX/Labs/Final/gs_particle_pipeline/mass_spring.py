from dataclasses import dataclass

import numpy as np
import taichi as ti

from .particles import GaussianParticleSet


@dataclass
class MassSpringConfig:
    """Mass-Spring 布料参数，结构参考 lab0 的 CaseMassSpring。"""

    nx: int = 24
    ny: int = 18
    stiffness: float = 80.0
    damping: float = 1.5
    gravity: float = 5.5
    velocity_damping: float = 0.985


@ti.data_oriented
class MassSpringSolver:
    """显式 Mass-Spring 布料求解器。

    弹簧连接方式对应 lab0：
    - 相邻结构弹簧；
    - 跨两格弯曲弹簧；
    - 两条对角弹簧。

    这不是 lab0 的隐式 Eigen 求解器，而是更适合当前 Taichi 工程的小步长显式版。
    """

    def __init__(self, particles: GaussianParticleSet, config: MassSpringConfig):
        self.particles = particles
        self.config = config
        self.max_springs = self._count_springs(config.nx, config.ny)
        self.spring_a = ti.field(dtype=ti.i32, shape=self.max_springs)
        self.spring_b = ti.field(dtype=ti.i32, shape=self.max_springs)
        self.rest_length = ti.field(dtype=ti.f32, shape=self.max_springs)
        self.spring_count = ti.field(dtype=ti.i32, shape=())
        self.force = ti.Vector.field(3, dtype=ti.f32, shape=particles.max_particles)
        self._build_springs()

    @staticmethod
    def _count_springs(nx: int, ny: int) -> int:
        count = 0
        count += max(nx - 1, 0) * ny
        count += max(nx - 2, 0) * ny
        count += nx * max(ny - 1, 0)
        count += nx * max(ny - 2, 0)
        count += max(nx - 1, 0) * max(ny - 1, 0) * 2
        return count

    def _build_springs(self) -> None:
        pairs: list[tuple[int, int]] = []

        def idx(i: int, j: int) -> int:
            return j * self.config.nx + i

        for j in range(self.config.ny):
            for i in range(self.config.nx):
                if i > 0:
                    pairs.append((idx(i, j), idx(i - 1, j)))
                if i > 1:
                    pairs.append((idx(i, j), idx(i - 2, j)))
                if j > 0:
                    pairs.append((idx(i, j), idx(i, j - 1)))
                if j > 1:
                    pairs.append((idx(i, j), idx(i, j - 2)))
                if i > 0 and j > 0:
                    pairs.append((idx(i, j), idx(i - 1, j - 1)))
                if i > 0 and j < self.config.ny - 1:
                    pairs.append((idx(i, j), idx(i - 1, j + 1)))

        if len(pairs) > self.max_springs:
            raise ValueError("spring buffer is too small")

        rest_pos = self.particles.rest_position.to_numpy()
        rest = np.zeros((self.max_springs,), dtype=np.float32)
        a = np.zeros((self.max_springs,), dtype=np.int32)
        b = np.zeros((self.max_springs,), dtype=np.int32)
        for s, (pa, pb) in enumerate(pairs):
            a[s] = pa
            b[s] = pb
            rest[s] = float(np.linalg.norm(rest_pos[pa] - rest_pos[pb]))

        self.spring_count[None] = len(pairs)
        self.spring_a.from_numpy(a)
        self.spring_b.from_numpy(b)
        self.rest_length.from_numpy(rest)

    @ti.kernel
    def substep(self, dt: ti.f32):
        for p in range(self.particles.count[None]):
            if self.particles.pinned[p] == 1:
                self.force[p] = ti.Vector([0.0, 0.0, 0.0])
            else:
                self.force[p] = ti.Vector([0.0, -self.config.gravity * self.particles.mass[p], 0.0])

        for s in range(self.spring_count[None]):
            a = self.spring_a[s]
            b = self.spring_b[s]
            x_ab = self.particles.position[b] - self.particles.position[a]
            length = ti.max(x_ab.norm(), 1e-6)
            direction = x_ab / length
            v_ab = self.particles.velocity[b] - self.particles.velocity[a]
            magnitude = self.config.stiffness * (length - self.rest_length[s]) + self.config.damping * v_ab.dot(direction)
            f = magnitude * direction
            if self.particles.pinned[a] == 0:
                self.force[a] += f
            if self.particles.pinned[b] == 0:
                self.force[b] -= f

        for p in range(self.particles.count[None]):
            if self.particles.pinned[p] == 1:
                self.particles.position[p] = self.particles.rest_position[p]
                self.particles.velocity[p] = ti.Vector([0.0, 0.0, 0.0])
                self.particles.deformation_gradient[p] = ti.Matrix.identity(ti.f32, 3)
            else:
                acceleration = self.force[p] / ti.max(self.particles.mass[p], 1e-6)
                self.particles.velocity[p] = (self.particles.velocity[p] + dt * acceleration) * self.config.velocity_damping
                self.particles.position[p] += dt * self.particles.velocity[p]

                # Mass-spring 没有连续介质 F，这里用局部各向同性缩放保持接口一致。
                self.particles.deformation_gradient[p] = ti.Matrix.identity(ti.f32, 3)

            f = self.particles.deformation_gradient[p]
            self.particles.current_covariance[p] = f @ self.particles.base_covariance[p] @ f.transpose()
