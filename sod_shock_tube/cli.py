from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np

from .core import initialize_grid, initialize_state, march
from .plotting import plot_profiles


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sod shock tube (1D MacCormack)")
    p.add_argument("--nx", type=int, default=401)
    p.add_argument("--Lx", type=float, default=1.0)
    p.add_argument("--nt", type=int, default=20000)
    p.add_argument("--dt", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--diaphragm", type=float, default=0.5, help="Location as fraction of Lx or absolute if >1")
    p.add_argument("--Cx", type=float, default=0.4)
    p.add_argument("--save", action="store_true")
    p.add_argument("--outdir", type=str, default="images/sod")
    p.add_argument("--pause", type=int, default=500, help="Snapshot every N steps")
    p.add_argument("--no-plot", action="store_true")
    return p


def main():
    args = build_parser().parse_args()

    grid = initialize_grid(args.Lx, args.nx)
    diaphragm_x = args.diaphragm * args.Lx if args.diaphragm <= 1.0 else args.diaphragm
    rho0, u0, e0 = initialize_state(grid.x, args.gamma, diaphragm_x)

    # snapshot steps
    steps: List[int] = list(range(args.pause, args.nt + 1, args.pause)) if args.pause > 0 else []

    res = march(
        x=grid.x,
        rho0=rho0,
        u0=u0,
        e0=e0,
        gamma=args.gamma,
        dx=grid.dx,
        dt=args.dt,
        nt=args.nt,
        Cx=args.Cx,
        snapshot_steps=steps,
    )

    if not args.no_plot and steps:
        profiles = {
            "rho": [s["rho"] for s in res["snapshots"]],
            "u": [s["u"] for s in res["snapshots"]],
            "p": [s["p"] for s in res["snapshots"]],
        }
        outdir = args.outdir if args.save else None
        plot_profiles(grid.x, profiles, steps, outdir=outdir, show=not args.no_plot)


if __name__ == "__main__":
    main()
