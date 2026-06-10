import numpy as np

from .camera import GaussianCamera
from .particles import GaussianParticleSet


def compact_symmetric_covariance(covariance: np.ndarray) -> np.ndarray:
    """把 Nx3x3 对称 covariance 压成官方 rasterizer 常用的 Nx6 格式。"""

    compact = np.empty((covariance.shape[0], 6), dtype=np.float32)
    compact[:, 0] = covariance[:, 0, 0]
    compact[:, 1] = covariance[:, 0, 1]
    compact[:, 2] = covariance[:, 0, 2]
    compact[:, 3] = covariance[:, 1, 1]
    compact[:, 4] = covariance[:, 1, 2]
    compact[:, 5] = covariance[:, 2, 2]
    return compact


def build_cuda_rasterizer_payload(
    particles: GaussianParticleSet,
    camera: GaussianCamera,
    use_covariance: bool = False,
) -> dict[str, np.ndarray | int | float]:
    """导出与官方 CUDA 3DGS rasterizer 对齐的数据边界。

    本函数刻意不 import torch，也不调用 `diff_gaussian_rasterization`。
    后续真正接入官方实现时，只需要把这些 numpy array 转成 CUDA tensor，
    再喂给 `GaussianRasterizer`。
    """

    n = int(particles.count[None])
    payload: dict[str, np.ndarray | int | float] = {
        "image_width": camera.image_width,
        "image_height": camera.image_height,
        "fov_x": camera.fov_x,
        "fov_y": camera.fov_y,
        "znear": camera.znear,
        "zfar": camera.zfar,
        "world_view_transform": camera.world_view_transform.astype(np.float32),
        "projection_matrix": camera.projection_matrix.astype(np.float32),
        "full_proj_transform": camera.full_proj_transform.astype(np.float32),
        "camera_center": camera.camera_center.astype(np.float32),
        "means3D": particles.position.to_numpy()[:n].astype(np.float32),
        "opacities": particles.opacity.to_numpy()[:n, None].astype(np.float32),
        "shs": particles.sh_coefficients.to_numpy()[:n].astype(np.float32),
    }

    if use_covariance:
        covariance = particles.current_covariance.to_numpy()[:n].astype(np.float32)
        payload["cov3D_precomp"] = compact_symmetric_covariance(covariance)
    else:
        payload["scales"] = particles.scale.to_numpy()[:n].astype(np.float32)
        payload["rotations"] = particles.rotation.to_numpy()[:n].astype(np.float32)

    return payload
