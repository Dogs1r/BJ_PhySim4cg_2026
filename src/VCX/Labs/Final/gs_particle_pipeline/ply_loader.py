import math
from typing import Dict, List, Tuple

import numpy as np

from .particles import GaussianHostData


SH_C0 = 0.28209479177387814


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def read_ascii_ply(path: str) -> Tuple[List[str], np.ndarray]:
    """读取 ASCII PLY。

    这里先实现教学工程中最容易检查的 ASCII PLY。常见 3DGS 导出也可能是
    binary_little_endian；如果遇到二进制文件，后续可以扩展 struct/np.fromfile 读取。
    """

    with open(path, "rb") as f:
        header_lines: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY 文件缺少 end_header")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                break

        if "format ascii" not in header_lines[1]:
            raise ValueError("当前读取器只支持 ASCII PLY；请先转换或扩展 binary PLY 读取。")

        vertex_count = 0
        properties: List[str] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element" and parts[1] == "vertex":
                vertex_count = int(parts[2])
                in_vertex = True
                continue
            if len(parts) >= 2 and parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            if in_vertex and len(parts) >= 3 and parts[0] == "property":
                properties.append(parts[-1])

        values = []
        for _ in range(vertex_count):
            values.append([float(x) for x in f.readline().decode("ascii").split()])

    return properties, np.asarray(values, dtype=np.float32)


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """把 3DGS 中的四元数转换为旋转矩阵。

    常见字段顺序是 rot_0, rot_1, rot_2, rot_3，对应 w, x, y, z。
    """

    q = q.astype(np.float32)
    q = q / max(float(np.linalg.norm(q)), 1e-8)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float32,
    )


def load_3dgs_ply(path: str, max_particles: int | None = None) -> GaussianHostData:
    """把 3D Gaussian Splatting 的 PLY 参数转成工程内部粒子数据。

    预期字段包括：
    - x, y, z
    - opacity
    - scale_0, scale_1, scale_2
    - rot_0, rot_1, rot_2, rot_3
    - f_dc_0, f_dc_1, f_dc_2

    如果缺少某些字段，会使用保守默认值，让读取器更适合早期调试。
    """

    properties, table = read_ascii_ply(path)
    prop_index: Dict[str, int] = {name: i for i, name in enumerate(properties)}
    n = table.shape[0] if max_particles is None else min(table.shape[0], max_particles)
    table = table[:n]

    def column(name: str, default: float) -> np.ndarray:
        if name in prop_index:
            return table[:, prop_index[name]]
        return np.full((n,), default, dtype=np.float32)

    position = np.stack([column("x", 0.0), column("y", 0.0), column("z", 0.0)], axis=1).astype(np.float32)
    opacity = sigmoid(column("opacity", 0.0)).astype(np.float32)

    scales = np.stack(
        [np.exp(column("scale_0", math.log(0.01))), np.exp(column("scale_1", math.log(0.01))), np.exp(column("scale_2", math.log(0.01)))],
        axis=1,
    ).astype(np.float32)

    base_covariance = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        q = np.array(
            [
                column("rot_0", 1.0)[i],
                column("rot_1", 0.0)[i],
                column("rot_2", 0.0)[i],
                column("rot_3", 0.0)[i],
            ],
            dtype=np.float32,
        )
        r = quaternion_to_rotation(q)
        s2 = np.diag(scales[i] * scales[i]).astype(np.float32)
        base_covariance[i] = r @ s2 @ r.T

    sh_coefficients = np.zeros((n, 4, 3), dtype=np.float32)
    dc = np.stack([column("f_dc_0", 0.0), column("f_dc_1", 0.0), column("f_dc_2", 0.0)], axis=1)
    # 原始 3DGS 通常用 RGB = 0.5 + SH_C0 * f_dc 还原 DC 颜色。
    sh_coefficients[:, 0, :] = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0).astype(np.float32)

    for axis in range(3):
        for channel in range(3):
            name = f"f_rest_{axis * 3 + channel}"
            if name in prop_index:
                sh_coefficients[:, axis + 1, channel] = table[:, prop_index[name]] * SH_C0

    mass = np.ones((n,), dtype=np.float32)
    volume = np.maximum(np.linalg.det(base_covariance), 1e-12).astype(np.float32)
    material_id = column("material_id", 0.0).astype(np.int32)
    pinned = column("pinned", 0.0).astype(np.int32)
    density = column("density", 1000.0).astype(np.float32)
    youngs_modulus = column("youngs_modulus", 2.0e4).astype(np.float32)
    poisson_ratio = column("poisson_ratio", 0.30).astype(np.float32)
    if "mass" in prop_index:
        mass = column("mass", 1.0).astype(np.float32)
    if "volume" in prop_index:
        volume = column("volume", 1e-6).astype(np.float32)

    return GaussianHostData(
        position=position,
        base_covariance=base_covariance,
        opacity=opacity,
        sh_coefficients=sh_coefficients,
        mass=mass,
        volume=volume,
        material_id=material_id,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        pinned=pinned,
    )
