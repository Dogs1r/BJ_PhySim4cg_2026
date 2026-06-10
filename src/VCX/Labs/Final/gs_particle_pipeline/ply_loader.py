import math
from typing import Dict, List, Tuple

import numpy as np

from .particles import GaussianHostData


SH_C0 = 0.28209479177387814
PLY_DTYPE_MAP = {
    "char": "i1",
    "uchar": "u1",
    "int8": "i1",
    "uint8": "u1",
    "short": "<i2",
    "ushort": "<u2",
    "int16": "<i2",
    "uint16": "<u2",
    "int": "<i4",
    "uint": "<u4",
    "int32": "<i4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _parse_ply_header(f) -> tuple[list[str], str, int, list[tuple[str, str]]]:
    header_lines: List[str] = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("PLY 文件缺少 end_header")
        text = line.decode("ascii").strip()
        header_lines.append(text)
        if text == "end_header":
            break

    if len(header_lines) < 2 or not header_lines[1].startswith("format "):
        raise ValueError("PLY header 缺少 format 行")
    ply_format = header_lines[1].split()[1]
    vertex_count = 0
    properties: list[tuple[str, str]] = []
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
            if parts[1] == "list":
                raise ValueError("当前 3DGS 顶点读取器不支持 list property")
            properties.append((parts[-1], parts[1]))
    return header_lines, ply_format, vertex_count, properties


def read_ply_vertex_table(path: str) -> Tuple[List[str], np.ndarray]:
    """读取 3DGS PLY 顶点表，支持 ascii 和 binary_little_endian。"""

    with open(path, "rb") as f:
        _, ply_format, vertex_count, typed_properties = _parse_ply_header(f)

        properties = [name for name, _ in typed_properties]
        if ply_format == "ascii":
            values = []
            for _ in range(vertex_count):
                values.append([float(x) for x in f.readline().decode("ascii").split()])
            return properties, np.asarray(values, dtype=np.float32)

        if ply_format == "binary_little_endian":
            dtype_fields = []
            for name, type_name in typed_properties:
                if type_name not in PLY_DTYPE_MAP:
                    raise ValueError(f"不支持的 PLY property 类型: {type_name}")
                dtype_fields.append((name, PLY_DTYPE_MAP[type_name]))
            structured = np.fromfile(f, dtype=np.dtype(dtype_fields), count=vertex_count)
            table = np.stack([structured[name].astype(np.float32) for name in properties], axis=1)
            return properties, table.astype(np.float32)

    raise ValueError(f"不支持的 PLY format: {ply_format}")


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

    properties, table = read_ply_vertex_table(path)
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

    rotations = np.stack(
        [column("rot_0", 1.0), column("rot_1", 0.0), column("rot_2", 0.0), column("rot_3", 0.0)],
        axis=1,
    ).astype(np.float32)

    base_covariance = np.zeros((n, 3, 3), dtype=np.float32)
    for i in range(n):
        q = rotations[i]
        r = quaternion_to_rotation(q)
        s2 = np.diag(scales[i] * scales[i]).astype(np.float32)
        base_covariance[i] = r @ s2 @ r.T

    sh_coefficients = np.zeros((n, 16, 3), dtype=np.float32)
    dc = np.stack([column("f_dc_0", 0.0), column("f_dc_1", 0.0), column("f_dc_2", 0.0)], axis=1)
    sh_coefficients[:, 0, :] = dc.astype(np.float32)

    # 官方 3DGS PLY 的 f_rest 通常按 coefficient-major、RGB-channel-minor 存储：
    # f_rest_0..2 是第 1 个非 DC SH 系数的 RGB，直到 15 个非 DC 系数。
    for coeff in range(1, 16):
        for channel in range(3):
            name = f"f_rest_{(coeff - 1) * 3 + channel}"
            if name in prop_index:
                sh_coefficients[:, coeff, channel] = table[:, prop_index[name]]

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
        scale=scales,
        rotation=rotations,
        mass=mass,
        volume=volume,
        material_id=material_id,
        density=density,
        youngs_modulus=youngs_modulus,
        poisson_ratio=poisson_ratio,
        pinned=pinned,
    )
