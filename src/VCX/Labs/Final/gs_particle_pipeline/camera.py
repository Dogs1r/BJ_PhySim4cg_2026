import math
from dataclasses import dataclass

import numpy as np

from .config import CameraConfig


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def get_projection_matrix(znear: float, zfar: float, fov_x: float, fov_y: float) -> np.ndarray:
    """返回与官方 gaussian-splatting `getProjectionMatrix` 对齐的 4x4 矩阵。"""

    tan_half_y = math.tan(fov_y * 0.5)
    tan_half_x = math.tan(fov_x * 0.5)
    top = tan_half_y * znear
    bottom = -top
    right = tan_half_x * znear
    left = -right

    p = np.zeros((4, 4), dtype=np.float32)
    z_sign = 1.0
    p[0, 0] = 2.0 * znear / (right - left)
    p[1, 1] = 2.0 * znear / (top - bottom)
    p[0, 2] = (right + left) / (right - left)
    p[1, 2] = (top + bottom) / (top - bottom)
    p[3, 2] = z_sign
    p[2, 2] = z_sign * zfar / (zfar - znear)
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p


def world_view_from_simple_camera(config: CameraConfig) -> np.ndarray:
    """把当前简化相机参数转成官方 renderer 使用的 world_view_transform 形态。

    当前教学 renderer 的相机约定是：先绕 x 轴 pitch，再把 z 平移 `camera_z`。
    这里保留同一约定，便于两个 renderer 在接入初期做结果对照。
    """

    pitch = math.radians(config.pitch_degrees)
    c = math.cos(pitch)
    s = math.sin(pitch)
    column_world_view = np.eye(4, dtype=np.float32)
    column_world_view[:3, :3] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )
    column_world_view[2, 3] = config.camera_z
    return column_world_view.transpose()


@dataclass
class GaussianCamera:
    """CUDA 3DGS renderer 需要的相机数据边界。

    字段名刻意贴近官方 `Camera/MiniCam`，后续接入
    `diff_gaussian_rasterization` 时可以直接转换为 torch tensor。
    """

    image_width: int
    image_height: int
    fov_x: float
    fov_y: float
    znear: float
    zfar: float
    world_view_transform: np.ndarray
    projection_matrix: np.ndarray
    full_proj_transform: np.ndarray
    camera_center: np.ndarray

    @classmethod
    def from_config(cls, config: CameraConfig, flip_y: bool = False) -> "GaussianCamera":
        fov_x = focal_to_fov(config.focal_length, config.width)
        fov_y = focal_to_fov(config.focal_length, config.height)
        world_view = world_view_from_simple_camera(config)
        # 官方 Camera 中 world_view_transform 和 projection_matrix 都是
        # getWorld2View2/getProjectionMatrix 后再 transpose，供 row-vector
        # 风格的 CUDA rasterizer 使用。
        projection = get_projection_matrix(config.near, config.far, fov_x, fov_y).transpose()
        if flip_y:
            projection[1, 1] *= -1.0
        full_proj = world_view @ projection
        camera_center = np.linalg.inv(world_view)[3, :3].astype(np.float32)
        return cls(
            image_width=config.width,
            image_height=config.height,
            fov_x=fov_x,
            fov_y=fov_y,
            znear=config.near,
            zfar=config.far,
            world_view_transform=world_view.astype(np.float32),
            projection_matrix=projection.astype(np.float32),
            full_proj_transform=full_proj.astype(np.float32),
            camera_center=camera_center,
        )
