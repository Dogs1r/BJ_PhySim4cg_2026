from dataclasses import dataclass

import numpy as np
import taichi as ti


@dataclass
class GaussianHostData:
    """CPU 侧输入数据。

    这些数组用于初始化 Taichi field。后续如果接入 3DGS 训练结果，
    可以把 ply/npz 中的 Gaussian 参数整理成同样的结构。
    """

    position: np.ndarray
    base_covariance: np.ndarray
    opacity: np.ndarray
    sh_coefficients: np.ndarray
    mass: np.ndarray
    volume: np.ndarray
    material_id: np.ndarray | None = None
    density: np.ndarray | None = None
    youngs_modulus: np.ndarray | None = None
    poisson_ratio: np.ndarray | None = None
    pinned: np.ndarray | None = None


@ti.data_oriented
class GaussianParticleSet:
    """同时服务渲染和未来物理仿真的粒子容器。

    当前静态流水线只真正使用 position/current_covariance/opacity/SH。
    velocity/deformation_gradient/mass/volume 是为之后 MPM 或其它连续介质
    求解器预留的数据接口，避免后续重写 Gaussian 数据传输方式。
    """

    def __init__(self, max_particles: int):
        self.max_particles = max_particles
        self.count = ti.field(dtype=ti.i32, shape=())

        # 核心跨模块变量：位置会从物理模块传到渲染模块。
        self.rest_position = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.position = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)

        # 物理预留变量：速度、变形梯度、质量、体积。
        self.velocity = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.affine_C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)
        self.deformation_gradient = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)
        self.mass = ti.field(dtype=ti.f32, shape=max_particles)
        self.volume = ti.field(dtype=ti.f32, shape=max_particles)
        self.material_id = ti.field(dtype=ti.i32, shape=max_particles)
        self.pinned = ti.field(dtype=ti.i32, shape=max_particles)
        self.density = ti.field(dtype=ti.f32, shape=max_particles)
        self.youngs_modulus = ti.field(dtype=ti.f32, shape=max_particles)
        self.poisson_ratio = ti.field(dtype=ti.f32, shape=max_particles)

        # Gaussian 渲染变量：base_covariance 是初始形状，current_covariance 是当前形状。
        # 未来物理仿真接入后，典型更新为 current_covariance = F A F^T。
        self.base_covariance = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)
        self.current_covariance = ti.Matrix.field(3, 3, dtype=ti.f32, shape=max_particles)
        self.opacity = ti.field(dtype=ti.f32, shape=max_particles)

        # 简化 SH：保留 4 个三通道系数，分别对应常数项和一阶方向项。
        # sh_coefficients[p, k] 是第 p 个 Gaussian 的第 k 个 SH RGB 系数。
        self.sh_coefficients = ti.Vector.field(3, dtype=ti.f32, shape=(max_particles, 4))

    def load_from_host(self, data: GaussianHostData) -> None:
        n = int(data.position.shape[0])
        if n > self.max_particles:
            raise ValueError(f"输入粒子数 {n} 超过容量 {self.max_particles}")

        self.count[None] = n
        position = np.zeros((self.max_particles, 3), dtype=np.float32)
        covariance = np.zeros((self.max_particles, 3, 3), dtype=np.float32)
        opacity = np.zeros((self.max_particles,), dtype=np.float32)
        mass = np.ones((self.max_particles,), dtype=np.float32)
        volume = np.zeros((self.max_particles,), dtype=np.float32)
        material_id = np.zeros((self.max_particles,), dtype=np.int32)
        pinned = np.zeros((self.max_particles,), dtype=np.int32)
        density = np.full((self.max_particles,), 1000.0, dtype=np.float32)
        youngs_modulus = np.full((self.max_particles,), 2.0e4, dtype=np.float32)
        poisson_ratio = np.full((self.max_particles,), 0.30, dtype=np.float32)

        position[:n] = data.position.astype(np.float32)
        covariance[:n] = data.base_covariance.astype(np.float32)
        opacity[:n] = data.opacity.astype(np.float32)
        mass[:n] = data.mass.astype(np.float32)
        volume[:n] = data.volume.astype(np.float32)
        if data.material_id is not None:
            material_id[:n] = data.material_id.astype(np.int32)
        if data.pinned is not None:
            pinned[:n] = data.pinned.astype(np.int32)
        if data.density is not None:
            density[:n] = data.density.astype(np.float32)
        if data.youngs_modulus is not None:
            youngs_modulus[:n] = data.youngs_modulus.astype(np.float32)
        if data.poisson_ratio is not None:
            poisson_ratio[:n] = data.poisson_ratio.astype(np.float32)

        self.position.from_numpy(position)
        self.rest_position.from_numpy(position)
        self.base_covariance.from_numpy(covariance)
        self.current_covariance.from_numpy(covariance)
        self.opacity.from_numpy(opacity)
        self.mass.from_numpy(mass)
        self.volume.from_numpy(volume)
        self.material_id.from_numpy(material_id)
        self.pinned.from_numpy(pinned)
        self.density.from_numpy(density)
        self.youngs_modulus.from_numpy(youngs_modulus)
        self.poisson_ratio.from_numpy(poisson_ratio)

        zero_velocity = np.zeros((self.max_particles, 3), dtype=np.float32)
        identity_f = np.tile(np.eye(3, dtype=np.float32), (self.max_particles, 1, 1))
        self.velocity.from_numpy(zero_velocity)
        self.affine_C.from_numpy(np.zeros((self.max_particles, 3, 3), dtype=np.float32))
        self.deformation_gradient.from_numpy(identity_f)

        sh_full = np.zeros((self.max_particles, 4, 3), dtype=np.float32)
        sh_full[:n] = data.sh_coefficients.astype(np.float32)
        self.sh_coefficients.from_numpy(sh_full)

    @ti.kernel
    def reset_dynamic_state(self):
        """把动态状态重置为静态状态，便于调试和回退单帧结果。"""
        for p in range(self.max_particles):
            self.velocity[p] = ti.Vector([0.0, 0.0, 0.0])
            self.affine_C[p] = ti.Matrix.zero(ti.f32, 3, 3)
            self.position[p] = self.rest_position[p]
            self.deformation_gradient[p] = ti.Matrix.identity(ti.f32, 3)
            self.current_covariance[p] = self.base_covariance[p]

    def state_summary(self) -> dict[str, np.ndarray | float]:
        """返回当前粒子状态摘要，用于判断仿真是否真的在动。"""
        n = int(self.count[None])
        positions = self.position.to_numpy()[:n]
        velocities = self.velocity.to_numpy()[:n]
        speeds = np.linalg.norm(velocities, axis=1)
        return {
            "min": positions.min(axis=0),
            "max": positions.max(axis=0),
            "mean_speed": float(speeds.mean()),
            "max_speed": float(speeds.max()),
        }
