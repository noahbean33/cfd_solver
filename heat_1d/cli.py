from __future__ import annotations

import argparse

from .core import solve
from .plotting import plot_profiles


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="1D Heat Equation solver")
    p.add_argument("--numX", type=int, default=101)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--tmax", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=10.0)
    p.add_argument("--scheme", choices=["explicit", "implicit"], default="implicit")
    p.add_argument("--temp1", type=float, default=20.0)
    p.add_argument("--temp2", type=float, default=100.0)
    p.add_argument("--save", action="store_true")
    p.add_argument("--out", type=str, default="images/heat_1d/profiles.png")
    p.add_argument("--no-show", action="store_true")
    p.add_argument("--checkpoints", type=str, default="1,4,10,20,100")
    return p


def main():
    args = build_parser().parse_args()
    checkpoints = [float(x) for x in args.checkpoints.split(",") if x]
    res = solve(
        numX=args.numX,
        alpha=args.alpha,
        t_max=args.tmax,
        dt=args.dt,
        temp1=args.temp1,
        temp2=args.temp2,
        scheme=args.scheme,
        checkpoints=checkpoints,
    )

    out_path = args.out if args.save else None
    plot_profiles(
        x=res["x"],
        snapshots=res["snapshots"],
        labels=res["labels"],
        temp1=args.temp1,
        temp2=args.temp2,
        out_path=out_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
