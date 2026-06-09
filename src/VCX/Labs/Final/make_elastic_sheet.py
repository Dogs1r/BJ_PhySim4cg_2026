import argparse
import math
import os


SH_C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser(description="生成 400-600 粒子的弹性布片/薄板 ASCII PLY")
    parser.add_argument("--output", default="assets/elastic_sheet.ply", help="输出 PLY 路径")
    parser.add_argument("--nx", type=int, default=22, help="水平方向粒子数")
    parser.add_argument("--ny", type=int, default=22, help="竖直方向粒子数")
    parser.add_argument("--width", type=float, default=1.35, help="布片宽度")
    parser.add_argument("--height", type=float, default=1.35, help="布片高度")
    parser.add_argument("--z-thickness", type=float, default=0.012, help="用于体积估计的厚度")
    parser.add_argument("--youngs", type=float, default=1.2e4, help="杨氏模量")
    parser.add_argument("--poisson", type=float, default=0.35, help="泊松比")
    parser.add_argument("--density", type=float, default=650.0, help="密度")
    parser.add_argument("--pin-mode", choices=["top-row", "top-corners"], default="top-corners", help="固定方式")
    parser.add_argument("--pin-top-rows", type=int, default=1, help="pin-mode=top-row 时固定顶部几行")
    parser.add_argument("--splat-scale", type=float, default=0.18, help="Gaussian 尺寸相对网格间距的比例，越小越锐利")
    return parser.parse_args()


def rgb_to_fdc(r: float, g: float, b: float):
    return (r - 0.5) / SH_C0, (g - 0.5) / SH_C0, (b - 0.5) / SH_C0


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    count = args.nx * args.ny
    dx = args.width / max(args.nx - 1, 1)
    dy = args.height / max(args.ny - 1, 1)
    volume = dx * dy * args.z_thickness
    mass = args.density * volume
    log_sx = math.log(dx * args.splat_scale)
    log_sy = math.log(dy * args.splat_scale)
    log_sz = math.log(args.z_thickness * 0.40)

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

    with open(args.output, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {count}\n")
        for kind, name in properties:
            f.write(f"property {kind} {name}\n")
        f.write("end_header\n")

        for iy in range(args.ny):
            for ix in range(args.nx):
                # 与 lab0 MassSpring 类似：初始为水平布片，两个角固定。
                x = -args.width * 0.5 + ix * dx
                y = 0.48
                z = -args.height * 0.5 + iy * dy
                if args.pin_mode == "top-row":
                    pinned = 1 if iy >= args.ny - args.pin_top_rows else 0
                else:
                    pinned = 1 if iy == args.ny - 1 and (ix == 0 or ix == args.nx - 1) else 0

                u = ix / max(args.nx - 1, 1)
                v = iy / max(args.ny - 1, 1)
                fdc0, fdc1, fdc2 = rgb_to_fdc(
                    0.20 + 0.55 * u,
                    0.45 + 0.35 * (1.0 - v),
                    0.85 - 0.30 * u,
                )

                row = [
                    x,
                    y,
                    z,
                    1.6,
                    log_sx,
                    log_sy,
                    log_sz,
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
                f.write(" ".join(str(value) for value in row) + "\n")

    print(f"已生成弹性布片 PLY: {args.output} ({count} particles)")


if __name__ == "__main__":
    main()
