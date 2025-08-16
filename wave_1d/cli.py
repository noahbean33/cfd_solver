"""
CLI for 1D linear advection solvers (FTBS, MacCormack).
"""
from __future__ import annotations

import argparse
import numpy as np
from .core import SimConfig, run, make_grid
from .ic import gaussian, sinusoid, multi_sin, step as step_ic, poly3
from .plotting import plot_snapshot


IC_MAP = {
    "gaussian": gaussian,
    "sin": sinusoid,
    "multi_sin": multi_sin,
    "step": step_ic,
    "poly3": poly3,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="1D linear advection (u_t + c u_x=0)")
    p.add_argument("--scheme", choices=["ftbs", "maccormack"], default="maccormack")
    p.add_argument("--dx", type=float, default=0.004)
    p.add_argument("--dt", type=float, default=0.002)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--domain", nargs=2, type=float, default=[-0.5, 1.5])
    p.add_argument("--bc", choices=["periodic", "outflow"], default="periodic")
    p.add_argument("--ic", choices=list(IC_MAP.keys()), default="step")
    p.add_argument("--x0", type=float, default=0.0)
    p.add_argument("--wavelength", type=float, default=0.4)
    p.add_argument("--width", type=float, default=0.4)
    p.add_argument("--plot", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    x_start, x_end = args.domain
    xs = make_grid(x_start, x_end, args.dx)

    # Build IC
    if args.ic == "gaussian":
        u0 = gaussian(xs, args.x0, width=args.width)
    elif args.ic == "sin":
        u0 = sinusoid(xs, args.x0, wavelength=args.wavelength)
    elif args.ic == "multi_sin":
        u0 = multi_sin(xs, args.x0, wavelength=args.wavelength)
    elif args.ic == "step":
        u0 = step_ic(xs, args.x0, width=args.width)
    else:
        u0 = poly3(xs, args.x0)

    cfg = SimConfig(dx=args.dx, dt=args.dt, c=args.c, steps=args.steps, x_start=x_start, x_end=x_end, bc=args.bc, scheme=args.scheme)

    uT, xs, _ = run(u0, cfg, keep_trajectory=False)

    if args.plot:
        plot_snapshot(xs, {"Initial": u0, "Final": uT}, title=f"{args.scheme.upper()} t={args.steps*args.dt:.3f}")
        import matplotlib.pyplot as plt
        plt.show()

    # Print simple metrics
    shift = args.c * args.steps * args.dt
    if args.bc == "periodic":
        # Exact shift by rolling grid cells
        cells = int(round(shift / args.dx))
        u_exact = np.roll(u0, cells)
        l2 = float(np.linalg.norm(uT - u_exact) / np.sqrt(uT.size))
        print(f"L2 error vs. exact shift: {l2:.3e}")
    else:
        print("Completed run (outflow BC, no exact periodic shift).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
