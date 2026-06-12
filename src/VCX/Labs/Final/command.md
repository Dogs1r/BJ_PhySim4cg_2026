目录结构建议：

~/autodl-tmp/data/my_scene/
  input/
    0001.jpg
    0002.jpg
    0003.jpg
    ...
要求：

同一个物体/场景
多视角
图片清晰
尽量避免动态物体和运动模糊
几十张可测试，100-300 张更稳
4. COLMAP 转换

安装 COLMAP：

sudo apt update
sudo apt install colmap -y
运行转换：

cd ~/autodl-tmp/gaussian-splatting

python convert.py -s ~/autodl-tmp/data/my_scene
成功后应出现：

~/autodl-tmp/data/my_scene/images/
~/autodl-tmp/data/my_scene/sparse/
5. 训练 GS

标准训练：

python train.py \
  -s ~/autodl-tmp/data/my_scene \
  -m ~/autodl-tmp/outputs/my_scene \
  --iterations 30000 \
  --disable_viewer
如果只是快速测试流程，可以先少训一点：

python train.py \
  -s ~/autodl-tmp/data/my_scene \
  -m ~/autodl-tmp/outputs/my_scene_test \
  --iterations 7000 \
  --disable_viewer
训练完成后，PLY 在：

~/autodl-tmp/outputs/my_scene/point_cloud/iteration_30000/point_cloud.ply
这就是训练得到的真实 GS 模型。

6. 官方渲染图片

渲染训练/测试视角：

python render.py \
  -m ~/autodl-tmp/outputs/my_scene \
  --iteration 30000
只渲染训练视角：

python render.py \
  -m ~/autodl-tmp/outputs/my_scene \
  --iteration 30000 \
  --skip_test
输出通常在：

~/autodl-tmp/outputs/my_scene/train/ours_30000/renders/
~/autodl-tmp/outputs/my_scene/test/ours_30000/renders/
7. 合成视频

训练视角视频：

ffmpeg -y \
  -framerate 30 \
  -i ~/autodl-tmp/outputs/my_scene/train/ours_30000/renders/%05d.png \
  -pix_fmt yuv420p \
  ~/autodl-tmp/outputs/my_scene/train_render.mp4
测试视角视频：

ffmpeg -y \
  -framerate 30 \
  -i ~/autodl-tmp/outputs/my_scene/test/ours_30000/renders/%05d.png \
  -pix_fmt yuv420p \
  ~/autodl-tmp/outputs/my_scene/test_render.mp4
8. 接入我们当前工程

训练出来的 PLY：

~/autodl-tmp/outputs/my_scene/point_cloud/iteration_30000/point_cloud.ply
后续要给我们的 Final/simulate.py 用。当前最合理的下一步是给我们的工程补一个：

--mode static
然后才能这样渲染真实 GS：

python simulate.py \
  --renderer gs \
  --mode static \
  --ply ~/autodl-tmp/outputs/my_scene/point_cloud/iteration_30000/point_cloud.ply \
  --particles 100000 \
  --frames 1 \
  --no-window
现在不建议直接把真实训练 PLY 塞进 mass-spring 或 elastic，因为真实 GS 点云不是规则布料网格，直接物理仿真会不合理。

推荐执行顺序

1. 验证 CUDA 扩展 import 成功
2. 准备 input 图片
3. python convert.py -s 场景目录
4. python train.py -s 场景目录 -m 输出目录 --iterations 30000 --disable_viewer
5. python render.py -m 输出目录 --iteration 30000
6. ffmpeg 合成视频
7. 再把 point_cloud.ply 接入我们的 Final 工程
如果你现在只是想确认流程，先用 --iterations 7000。如果效果可以，再跑 30000。