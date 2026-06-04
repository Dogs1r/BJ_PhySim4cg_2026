import math

import numpy as np

from .particles import GaussianHostData


def make_demo_scene(max_particles: int) -> GaussianHostData:
    """生成一个可直接渲染的静态 Gaussian 场景。

    这不是训练得到的真实 3DGS，而是用于验证数据传输和渲染管线的合成输入。
    每个点都按照 Gaussian/物理粒子的完整字段生成。
    """

    rings = 9
    per_ring = max(12, min(48, max_particles // rings))
    n = min(max_particles, rings * per_ring)

    position = np.zeros((max_particles, 3), dtype=np.float32)
    base_covariance = np.zeros((max_particles, 3, 3), dtype=np.float32)
    opacity = np.zeros((max_particles,), dtype=np.float32)
    sh_coefficients = np.zeros((max_particles, 4, 3), dtype=np.float32)
    mass = np.ones((max_particles,), dtype=np.float32)
    volume = np.full((max_particles,), 1.0 / max(1, n), dtype=np.float32)

    idx = 0
    for r in range(rings):
        y = (r - (rings - 1) * 0.5) * 0.13
        radius = 0.25 + 0.045 * r
        for k in range(per_ring):
            if idx >= n:
                break
            theta = 2.0 * math.pi * k / per_ring + r * 0.22
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            position[idx] = [x, y, z]

            # 各向异性协方差，后续可通过 F A F^T 接入物理形变。
            sx = 0.018 + 0.004 * (r % 3)
            sy = 0.026
            sz = 0.018 + 0.003 * ((k + r) % 4)
            base_covariance[idx] = np.diag([sx * sx, sy * sy, sz * sz]).astype(np.float32)

            opacity[idx] = 0.34
            base_color = np.array(
                [
                    0.35 + 0.45 * r / max(1, rings - 1),
                    0.52 + 0.18 * math.sin(theta),
                    0.85 - 0.35 * r / max(1, rings - 1),
                ],
                dtype=np.float32,
            )
            sh_coefficients[idx, 0] = base_color
            sh_coefficients[idx, 1] = [0.04, 0.0, -0.03]
            sh_coefficients[idx, 2] = [0.0, 0.03, 0.02]
            sh_coefficients[idx, 3] = [-0.02, 0.01, 0.04]
            mass[idx] = 1.0
            idx += 1

    return GaussianHostData(
        position=position,
        base_covariance=base_covariance,
        opacity=opacity,
        sh_coefficients=sh_coefficients,
        mass=mass,
        volume=volume,
    )
