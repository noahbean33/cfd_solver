"""
CLI for 2D potential flow around rectangular obstacles solved via Laplace (Jacobi/SOR).
"""
from __future__ import annotations

import argparse
import numpy as np

from .core import GridConfig, SolverConfig, FlowConfig, run
from .plotting import plot_potential_surface, plot_streamlines_cp, plot_convergence


def parse_obstacles(arg_list: list[str] | None):
    if not arg_list:
        return []
    # Expect tuples imin,imax,jmin,jmax repeated
    vals = list(map(int, arg_list))
    if len(vals) % 4 != 0:
        raise SystemExit("--obstacles must be groups of 4 integers: imin imax jmin jmax ...")
    return [tuple(vals[i:i+4]) for i in range(0, len(vals), 4)]


esschemes = ["jacobi", "sor"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="2D Laplace potential flow solver (Jacobi/SOR)")
    p.add_argument("--nx", type=int, default=201)
    p.add_argument("--ny", type=int, default=111)
    p.add_argument("--Lx", type=float, default=20.0)
    p.add_argument("--Ly", type=float, default=11.0)
    p.add_argument("--Uinf", type=float, default=1.0)
    p.add_argument("--method", choices=esschemes, default="jacobi")
    p.add_argument("--max-iters", type=int, default=40000)
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--omega", type=float, default=1.7)
    p.add_argument("--obstacles", nargs="*", help="List of obstacle rectangles as imin imax jmin jmax ...")
    p.add_argument("--plot", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    grid = GridConfig(nx=args.nx, ny=args.ny, Lx=args.Lx, Ly=args.Ly)
    solver = SolverConfig(method=args.method, max_iters=args.max_iters, tol=args.tol, omega=args.omega)
    flow = FlowConfig(Uinf=args.Uinf)

    obstacles = parse_obstacles(args.obstacles)

    res = run(obstacles, grid, flow, solver)

    print(f"Completed in {res.err.size} iterations; final residual {res.err[-1]:.3e}")

    if args.plot:
        plot_potential_surface(res.X, res.Y, res.phi)
        plot_streamlines_cp(res.X, res.Y, res.U, res.V, res.Cp, obstacles=obstacles, Lx=args.Lx, Ly=args.Ly, nx=args.nx, ny=args.ny)
        plot_convergence(res.err)
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
