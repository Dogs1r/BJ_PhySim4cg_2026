from .camera import GaussianCamera
from .config import CameraConfig, RenderConfig
from .cuda_renderer import CudaGaussianRenderer
from .particles import GaussianParticleSet
from .renderer import GaussianRenderer


def create_renderer(
    backend: str,
    particles: GaussianParticleSet,
    reference_camera: CameraConfig,
    gaussian_camera: GaussianCamera,
    render: RenderConfig,
    use_covariance: bool = False,
    device: str = "cuda",
    diagnose_render: bool = False,
    scale_modifier: float = 1.0,
    color_mode: str = "sh",
):
    """按名称创建渲染后端。

    `reference` 保留当前 Taichi 教学 renderer；
    `gs` 使用官方 CUDA 3DGS rasterizer 接口。
    """

    if backend == "reference":
        return GaussianRenderer(particles, reference_camera, render)
    if backend == "gs":
        return CudaGaussianRenderer(
            particles,
            gaussian_camera,
            render,
            use_covariance=use_covariance,
            device=device,
            diagnose_render=diagnose_render,
            scale_modifier=scale_modifier,
            color_mode=color_mode,
        )
    raise ValueError(f"未知渲染后端: {backend}")
