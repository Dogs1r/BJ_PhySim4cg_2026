import math

import numpy as np

from .camera import GaussianCamera
from .config import RenderConfig
from .cuda_rasterizer_adapter import build_cuda_rasterizer_payload
from .particles import GaussianParticleSet


SH_C0 = 0.28209479177387814


class CudaRendererUnavailable(RuntimeError):
    pass


class CudaGaussianRenderer:
    """官方 CUDA 3DGS rasterizer 后端。

    该类只负责把当前工程的 GaussianParticleSet/GaussianCamera 适配到
    diff_gaussian_rasterization。训练、优化和 COLMAP 数据集逻辑不进入本工程。
    """

    backend_name = "gs"

    def __init__(
        self,
        particles: GaussianParticleSet,
        camera: GaussianCamera,
        render: RenderConfig,
        use_covariance: bool = False,
        device: str = "cuda",
        diagnose_render: bool = False,
        scale_modifier: float = 1.0,
        color_mode: str = "sh",
    ):
        self.particles = particles
        self.camera = camera
        self.render = render
        self.use_covariance = use_covariance
        self.device_name = device
        self.diagnose_render = diagnose_render
        self.scale_modifier = scale_modifier
        self.color_mode = color_mode
        self._diagnosed = False

        try:
            import torch
            from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        except Exception as exc:
            raise CudaRendererUnavailable(
                "CUDA GS renderer 不可用：需要安装 PyTorch CUDA 和官方 "
                "diff_gaussian_rasterization。当前可以使用 --renderer reference "
                "运行 Taichi 教学渲染器；后续可在单独虚拟环境或云服务器安装 CUDA 后端。"
            ) from exc

        if device == "cuda" and not torch.cuda.is_available():
            raise CudaRendererUnavailable(
                "CUDA GS renderer 不可用：torch.cuda.is_available() 为 False。"
                "请使用 --renderer reference，或在有 CUDA 的环境中运行。"
            )

        self.torch = torch
        self.settings_cls = GaussianRasterizationSettings
        self.rasterizer_cls = GaussianRasterizer
        self.device = torch.device(device)
        self.background = torch.tensor(
            [render.background_r, render.background_g, render.background_b],
            dtype=torch.float32,
            device=self.device,
        )

    def _tensor(self, value: np.ndarray):
        return self.torch.as_tensor(value, dtype=self.torch.float32, device=self.device)

    def _make_raster_settings(self, payload: dict[str, np.ndarray | int | float]):
        tanfovx = math.tan(self.camera.fov_x * 0.5)
        tanfovy = math.tan(self.camera.fov_y * 0.5)
        kwargs = dict(
            image_height=int(self.camera.image_height),
            image_width=int(self.camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background,
            scale_modifier=self.scale_modifier,
            viewmatrix=self._tensor(payload["world_view_transform"]),
            projmatrix=self._tensor(payload["full_proj_transform"]),
            sh_degree=3,
            campos=self._tensor(payload["camera_center"]),
            prefiltered=False,
            debug=False,
        )
        try:
            return self.settings_cls(**kwargs, antialiasing=False)
        except TypeError:
            return self.settings_cls(**kwargs)

    def _print_diagnostics(self, payload: dict[str, np.ndarray | int | float]) -> None:
        means = payload["means3D"]
        opacities = payload["opacities"]
        scales = payload.get("scales")
        world_view = payload["world_view_transform"]
        full_proj = payload["full_proj_transform"]

        homogeneous = np.concatenate([means, np.ones((means.shape[0], 1), dtype=np.float32)], axis=1)
        clip = homogeneous @ full_proj
        w = clip[:, 3]
        valid_w = np.abs(w) > 1e-6
        ndc = np.zeros((means.shape[0], 3), dtype=np.float32)
        ndc[valid_w] = clip[valid_w, :3] / w[valid_w, None]
        screen_x = (ndc[:, 0] * 0.5 + 0.5) * float(self.camera.image_width)
        screen_y = (ndc[:, 1] * 0.5 + 0.5) * float(self.camera.image_height)
        on_screen = (
            valid_w
            & (ndc[:, 0] >= -1.0)
            & (ndc[:, 0] <= 1.0)
            & (ndc[:, 1] >= -1.0)
            & (ndc[:, 1] <= 1.0)
            & (ndc[:, 2] >= 0.0)
            & (ndc[:, 2] <= 1.0)
        )

        camera_space = homogeneous @ world_view
        text = [
            "[diagnose-render]",
            f"n={means.shape[0]}",
            f"world_bbox_min={means.min(axis=0)}",
            f"world_bbox_max={means.max(axis=0)}",
            f"camera_z_minmax=({camera_space[:, 2].min():.4f},{camera_space[:, 2].max():.4f})",
            f"screen_bbox_x=({screen_x.min():.1f},{screen_x.max():.1f})",
            f"screen_bbox_y=({screen_y.min():.1f},{screen_y.max():.1f})",
            f"visible_ndc={int(on_screen.sum())}/{means.shape[0]}",
            f"opacity_minmax=({opacities.min():.4f},{opacities.max():.4f})",
        ]
        if scales is not None:
            text.append(f"scale_minmax=({scales.min():.6f},{scales.max():.6f})")
            z = np.maximum(camera_space[:, 2], 1e-6)
            focal_x = float(self.camera.image_width) / (2.0 * math.tan(float(payload["fov_x"]) * 0.5))
            sigma_px = focal_x * scales[:, 0] * self.scale_modifier / z
            text.append(f"approx_sigma_px=({sigma_px.min():.3f},{sigma_px.max():.3f})")
        text.append(f"scale_modifier={self.scale_modifier:.3f}")
        text.append(f"color_mode={self.color_mode}")
        print(" ".join(text), flush=True)

    def _precompute_rgb(self, sh_coefficients: np.ndarray) -> np.ndarray:
        rgb = 0.5 + SH_C0 * sh_coefficients[:, 0, :]
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    def render_frame(self) -> np.ndarray:
        payload = build_cuda_rasterizer_payload(
            self.particles,
            self.camera,
            use_covariance=self.use_covariance,
        )
        if self.diagnose_render and not self._diagnosed:
            self._print_diagnostics(payload)
            self._diagnosed = True

        raster_settings = self._make_raster_settings(payload)
        rasterizer = self.rasterizer_cls(raster_settings=raster_settings)

        means3d = self._tensor(payload["means3D"])
        means2d = self.torch.zeros_like(means3d, dtype=self.torch.float32, device=self.device)
        opacities = self._tensor(payload["opacities"])
        shs = None
        colors_precomp = None
        if self.color_mode == "rgb":
            colors_precomp = self._tensor(self._precompute_rgb(payload["shs"]))
        else:
            shs = self._tensor(payload["shs"])

        scales = None
        rotations = None
        cov3d_precomp = None
        if self.use_covariance:
            cov3d_precomp = self._tensor(payload["cov3D_precomp"])
        else:
            scales = self._tensor(payload["scales"])
            rotations = self._tensor(payload["rotations"])

        with self.torch.no_grad():
            result = rasterizer(
                means3D=means3d,
                means2D=means2d,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacities,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3d_precomp,
            )
            rendered = result[0] if isinstance(result, tuple) else result

        image = rendered.clamp(0.0, 1.0).detach().cpu().numpy()
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        return (image * 255.0).astype(np.uint8)
