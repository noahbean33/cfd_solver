"""
CLI for lid-driven cavity simulation (streamfunction-vorticity).
"""
from __future__ import annotations

import argparse
import numpy as np

from .core import GridConfig, FluidConfig, TimeConfig, PoissonConfig, run
from .plotting import plot_streamlines_and_vorticity, plot_convergence


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lid-driven cavity (psi-omega)")
    p.add_argument("--nx", type=int, default=129)
    p.add_argument("--ny", type=int, default=129)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--Ly", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--Re", type=float, default=1000.0)
    p.add_argument("--nu", type=float, default=None)
    p.add_argument("--Uwall", type=float, default=1.0)
    p.add_argument("--poisson", choices=["jacobi", "sor"], default="jacobi")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--omega", type=float, default=1.7)
    p.add_argument("--plot", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    grid = GridConfig(nx=args.nx, ny=args.ny, Lx=args.Lx, Ly=args.Ly)
    fluid = FluidConfig(Re=args.Re if args.nu is None else None, nu=args.nu, Uwall=args.Uwall)
    time = TimeConfig(dt=args.dt, steps=args.steps)
    poisson = PoissonConfig(method=args.poisson, max_iters=args.max_iters, tol=args.tol, omega=args.omega)

    res = run(grid, fluid, time, poisson)

    print(f"Completed {res.err_psi.size} steps; final ||Δψ||={res.err_psi[-1]:.3e}, ||Δω||={res.err_omega[-1]:.3e}")

    if args.plot:
        plot_streamlines_and_vorticity(res.X, res.Y, res.U, res.V, res.omega)
        plot_convergence(res.err_psi, res.err_omega)
        import matplotlib.pyplot as plt
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
