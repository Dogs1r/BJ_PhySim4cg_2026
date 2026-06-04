import taichi as ti

from .particles import GaussianParticleSet


@ti.data_oriented
class PhysicsBridge:
    """未来物理仿真的统一入口。

    当前版本是静态 no-op：它不改变粒子，只保证数据流已经按
    “粒子状态 -> 物理 step -> Gaussian 状态 -> 渲染” 的方式组织。
    后续接入 MPM 时，应优先替换 step() 内部实现，而不是改渲染器接口。
    """

    def __init__(self, particles: GaussianParticleSet):
        self.particles = particles

    def step(self, dt: float) -> None:
        self._static_step(float(dt))

    @ti.kernel
    def _static_step(self, dt: ti.f32):
        for p in range(self.particles.count[None]):
            # 静态管线中 F 保持单位阵，因此当前协方差等于初始协方差。
            # 未来物理接入后，这里会变成：
            #   position += dt * velocity
            #   deformation_gradient = update_F(...)
            #   current_covariance = F @ base_covariance @ F.transpose()
            self.particles.current_covariance[p] = self.particles.base_covariance[p]
