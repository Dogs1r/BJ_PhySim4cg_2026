# Taichi 静态 3DGS 基础流水线

本目录从头搭建一个用于后续 PhysGaussian/PhySplatting 复刻的最小工程。当前版本先实现静态 Gaussian 粒子到图像的渲染流程，同时把粒子字段、物理 step 入口和 Gaussian covariance 更新位置预留好。

## 运行

```powershell
pip install -r requirements.txt
python main.py --cpu --output outputs/static_demo.ppm
```

当前输出为 PPM 图片，Windows 下可用支持 PPM 的图像查看器打开，也可以后续改成 PNG 输出。

导入 3DGS 导出的 ASCII PLY：

```powershell
python main.py --cpu --ply path\to\point_cloud.ply --particles 20000 --output outputs/ply_demo.ppm
```

当前 PLY 读取器支持常见字段：

```text
x, y, z
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
f_dc_0, f_dc_1, f_dc_2
f_rest_*
```

如果你的 3DGS 文件是 `binary_little_endian` PLY，需要后续扩展二进制读取，或先转换为 ASCII PLY。

## 当前范围

- 已有：合成 Gaussian 粒子、ASCII PLY 导入、Taichi field 数据容器、静态物理接口、投影协方差、conic 权重、深度排序 alpha 合成。
- 未做：真实 3DGS 训练、COLMAP 相机读取、CUDA tile rasterizer、MPM 物理仿真。
