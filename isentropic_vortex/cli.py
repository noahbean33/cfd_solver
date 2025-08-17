from __future__ import annotations

import argparse

import numpy as np

from .core import GridConfig, GasConfig, FlowConfig, TimeConfig, run
from .plotting import plot_vorticity


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Isentropic vortex (2D Euler)")
    p.add_argument("--nx", type=int, default=201)
    p.add_argument("--ny", type=int, default=201)
    p.add_argument("--Lx", type=float, default=10.0)
    p.add_argument("--Ly", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--u_inf", type=float, default=1.0)
    p.add_argument("--v_inf", type=float, default=0.0)
    p.add_argument("--vortex_gamma", type=float, default=0.5)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--CFL", type=float, default=0.5)
    p.add_argument("--plot", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    grid = GridConfig(nx=args.nx, ny=args.ny, Lx=args.Lx, Ly=args.Ly)
    gas = GasConfig(gamma=args.gamma)
    flow = FlowConfig(u_inf=args.u_inf, v_inf=args.v_inf, vortex_gamma=args.vortex_gamma, cx=args.cx, cy=args.cy)
    time = TimeConfig(steps=args.steps, dt=args.dt, CFL=args.CFL)

    res = run(grid, gas, flow, time)

    print(f"Ran {len(res.dt_hist)} steps. Mean dt={np.mean(res.dt_hist):.4e}")
    if res.err_hist.size:
        print(f"Last residual: {res.err_hist[-1]:.3e}")

    if args.plot:
        plot_vorticity(res.X, res.Y, res.u, res.v)
        import matplotlib.pyplot as plt
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
