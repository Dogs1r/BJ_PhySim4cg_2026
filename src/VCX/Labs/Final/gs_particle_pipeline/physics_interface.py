import taichi as ti

from .elastic_mpm import ElasticMPMConfig, ElasticMPMSolver
from .mass_spring import MassSpringConfig, MassSpringSolver
from .particles import GaussianParticleSet


@ti.data_oriented
class PhysicsBridge:
    """物理仿真的统一入口。

    当前核心版本只保留弹性 MPM。渲染器不直接关心物理细节，只读取
    PhysicsBridge 更新后的 GaussianParticleSet。
    """

    def __init__(
        self,
        particles: GaussianParticleSet,
        mode: str = "elastic",
        elastic_config: ElasticMPMConfig | None = None,
        mass_spring_config: MassSpringConfig | None = None,
    ):
        self.particles = particles
        self.mode = mode
        self.time = 0.0
        if mode not in {"elastic", "mass-spring"}:
            raise ValueError("当前版本只支持 mode='elastic' 或 mode='mass-spring'")
        self.elastic_solver = None
        self.mass_spring_solver = None
        if mode == "mass-spring":
            self.mass_spring_solver = MassSpringSolver(particles, mass_spring_config or MassSpringConfig())
        else:
            self.elastic_solver = ElasticMPMSolver(particles, elastic_config or ElasticMPMConfig())

    def step(self, dt: float) -> None:
        dt = float(dt)
        self.time += dt
        if self.mode == "mass-spring":
            if self.mass_spring_solver is None:
                raise RuntimeError("Mass-Spring solver is not initialized")
            self.mass_spring_solver.substep(dt)
        else:
            if self.elastic_solver is None:
                raise RuntimeError("Elastic MPM solver is not initialized")
            self.elastic_solver.substep(dt)
