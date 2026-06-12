import argparse
import math
import os


SH_C0 = 0.28209479177387814


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a shell-only elastic jelly ball PLY for MPM filling tests")
    parser.add_argument("--output", default="assets/elastic_ball_shell.ply", help="Output ASCII PLY path")
    parser.add_argument("--points", type=int, default=1200, help="Number of shell Gaussians")
    parser.add_argument("--radius", type=float, default=0.42, help="Ball radius")
    parser.add_argument("--center-x", type=float, default=0.0, help="Ball center x")
    parser.add_argument("--center-y", type=float, default=0.45, help="Ball center y")
    parser.add_argument("--center-z", type=float, default=0.0, help="Ball center z")
    parser.add_argument("--jitter", type=float, default=0.0, help="Optional radial jitter as a fraction of radius")
    parser.add_argument("--opacity", type=float, default=2.2, help="Raw 3DGS opacity logit")
    parser.add_argument("--splat-scale", type=float, default=1.15, help="Gaussian sigma relative to shell spacing")
    parser.add_argument("--density", type=float, default=200.0, help="Jelly density")
    parser.add_argument("--youngs", type=float, default=1.0e4, help="Jelly Young's modulus")
    parser.add_argument("--poisson", type=float, default=0.40, help="Jelly Poisson ratio")
    parser.add_argument("--pin-bottom", action="store_true", help="Write pinned=1 on the bottom shell band")
    parser.add_argument("--pin-thickness", type=float, default=0.08, help="Bottom pinned band thickness as a fraction of diameter")
    return parser.parse_args()


def rgb_to_fdc(r: float, g: float, b: float):
    return (r - 0.5) / SH_C0, (g - 0.5) / SH_C0, (b - 0.5) / SH_C0


def fibonacci_sphere(count: int):
    if count <= 0:
        return []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    points = []
    for i in range(count):
        y = 1.0 - (2.0 * i + 1.0) / count
        r = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        points.append((x, y, z))
    return points


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    count = max(1, int(args.points))
    center = (args.center_x, args.center_y, args.center_z)
    shell_area = 4.0 * math.pi * args.radius * args.radius
    area_per_point = shell_area / count
    shell_spacing = math.sqrt(area_per_point)
    sigma = max(shell_spacing * args.splat_scale, 1e-5)
    log_scale = math.log(sigma)
    # This shell-only PLY carries approximate attributes; the elastic builder
    # recomputes physical volume after filling, as in the PhysGaussian workflow.
    volume = max(area_per_point * sigma, 1e-9)
    mass = args.density * volume
    bottom_limit = center[1] - args.radius + 2.0 * args.radius * max(args.pin_thickness, 0.0)

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

        for i, (nx, ny, nz) in enumerate(fibonacci_sphere(count)):
            radial = args.radius * (1.0 + args.jitter * math.sin(i * 12.9898) * 0.5)
            x = center[0] + radial * nx
            y = center[1] + radial * ny
            z = center[2] + radial * nz
            pinned = 1 if args.pin_bottom and y <= bottom_limit else 0

            height_t = (ny + 1.0) * 0.5
            side_t = (math.atan2(nz, nx) / (2.0 * math.pi)) + 0.5
            fdc0, fdc1, fdc2 = rgb_to_fdc(
                0.25 + 0.45 * height_t,
                0.45 + 0.30 * (1.0 - height_t),
                0.85 - 0.20 * side_t,
            )

            row = [
                x,
                y,
                z,
                args.opacity,
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
            f.write(" ".join(str(value) for value in row) + "\n")

    print(f"Generated shell elastic ball PLY: {args.output} ({count} shell Gaussians)")


if __name__ == "__main__":
    main()
