import argparse
import math
import os


def parse_args():
    parser = argparse.ArgumentParser(description="生成用于弹性 MPM 测试的长方条 ASCII PLY")
    parser.add_argument("--output", default="assets/elastic_bar.ply", help="输出 PLY 路径")
    parser.add_argument("--nx", type=int, default=24, help="长方条 x 方向粒子数")
    parser.add_argument("--ny", type=int, default=5, help="长方条 y 方向粒子数")
    parser.add_argument("--nz", type=int, default=5, help="长方条 z 方向粒子数")
    parser.add_argument("--length", type=float, default=1.15, help="长方条长度")
    parser.add_argument("--thickness", type=float, default=0.26, help="长方条横截面尺寸")
    parser.add_argument("--youngs", type=float, default=2.0e4, help="杨氏模量")
    parser.add_argument("--poisson", type=float, default=0.30, help="泊松比")
    parser.add_argument("--density", type=float, default=1000.0, help="密度")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    count = args.nx * args.ny * args.nz
    dx = args.length / max(args.nx - 1, 1)
    dy = args.thickness / max(args.ny - 1, 1)
    dz = args.thickness / max(args.nz - 1, 1)
    volume = dx * dy * dz
    mass = args.density * volume
    sigma = min(dx, dy, dz) * 0.38
    log_scale = math.log(sigma)

    with open(args.output, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {count}\n")
        properties = [
            ("float", "x"),
            ("float", "y"),
            ("float", "z"),
            ("float", "opacity"),
            ("float", "scale_0"),
            ("float", "scale_1"),
            ("float", "scale_2"),
            ("float", "rot_0"),
            ("float", "rot_1"),
            ("float", "rot_2"),
            ("float", "rot_3"),
            ("float", "f_dc_0"),
            ("float", "f_dc_1"),
            ("float", "f_dc_2"),
            ("float", "mass"),
            ("float", "volume"),
            ("float", "density"),
            ("float", "youngs_modulus"),
            ("float", "poisson_ratio"),
            ("int", "material_id"),
            ("int", "pinned"),
        ]
        for kind, name in properties:
            f.write(f"property {kind} {name}\n")
        f.write("end_header\n")

        for ix in range(args.nx):
            for iy in range(args.ny):
                for iz in range(args.nz):
                    x = -0.62 + ix * dx
                    y = 0.28 + (iy - (args.ny - 1) * 0.5) * dy
                    z = (iz - (args.nz - 1) * 0.5) * dz
                    pinned = 1 if ix <= 1 else 0
                    # f_dc 按 3DGS 近似反推，让渲染颜色接近蓝橙渐变。
                    r = 0.25 + 0.65 * ix / max(args.nx - 1, 1)
                    g = 0.55
                    b = 0.90 - 0.35 * ix / max(args.nx - 1, 1)
                    sh_c0 = 0.28209479177387814
                    fdc0 = (r - 0.5) / sh_c0
                    fdc1 = (g - 0.5) / sh_c0
                    fdc2 = (b - 0.5) / sh_c0
                    row = [
                        x,
                        y,
                        z,
                        2.2,  # sigmoid 后约 0.9
                        log_scale,
                        log_scale,
                        log_scale,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        fdc0,
                        fdc1,
                        fdc2,
                        mass,
                        volume,
                        args.density,
                        args.youngs,
                        args.poisson,
                        0,
                        pinned,
                    ]
                    f.write(" ".join(str(v) for v in row) + "\n")

    print(f"已生成长方条 PLY: {args.output} ({count} particles)")


if __name__ == "__main__":
    main()
