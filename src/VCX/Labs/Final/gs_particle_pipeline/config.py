from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """相机参数；后续可以替换为 COLMAP/真实数据读取结果。"""

    width: int = 800
    height: int = 600
    focal_length: float = 620.0
    camera_z: float = 3.2
    pitch_degrees: float = 0.0
    near: float = 0.05
    far: float = 20.0


@dataclass
class RenderConfig:
    """渲染参数；当前是教学版 CPU Taichi splatting。"""

    max_particles: int = 4096
    tile_radius_scale: float = 3.0
    low_pass_variance: float = 0.005
    alpha_threshold: float = 0.001
    background_r: float = 0.02
    background_g: float = 0.025
    background_b: float = 0.03


@dataclass
class PipelineConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
