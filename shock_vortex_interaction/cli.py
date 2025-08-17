from __future__ import annotations

import argparse

from .core import (
    initialize_grid,
    initialize_shock,
    inject_vortex,
    march,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Shock–Vortex Interaction simulator")
    p.add_argument("--nx", type=int, default=321)
    p.add_argument("--ny", type=int, default=321)
    p.add_argument("--Lx", type=float, default=40.0)
    p.add_argument("--Ly", type=float, default=40.0)
    p.add_argument("--nt", type=int, default=30001)
    p.add_argument("--dt", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=1.4)
    p.add_argument("--vortex-gamma", dest="vortex_gamma", type=float, default=0.125)
    p.add_argument("--save", action="store_true", help="Save frames to images/shockVortex/")
    p.add_argument("--outdir", type=str, default="images/shockVortex/", help="Output directory for frames")
    p.add_argument("--pause", type=int, default=200)
    p.add_argument("--no-plot", action="store_true", help="Disable plotting during march")
    p.add_argument("--Cx", type=float, default=0.3)
    p.add_argument("--Cy", type=float, default=0.3)
    return p


def main():
    args = build_parser().parse_args()

    X, Y, dx, dy = initialize_grid(args.Lx, args.Ly, args.nx, args.ny)
    rho, u, v, e = initialize_shock(X, Y, args.nx, args.ny, args.gamma, 0.0, args.Lx, args.Ly)
    # Shock-only convergence
    rho, u, v, e, mean_pressure = march(
        X,
        Y,
        rho,
        u,
        v,
        e,
        rho * e * (args.gamma - 1.0),
        args.gamma,
        dx,
        dy,
        args.dt,
        args.nt,
        plot=0 if args.no_plot else 1,
        save=0,
        pause=max(args.nt - 1, 1),
        Cx=args.Cx,
        Cy=args.Cy,
    )

    rho, u, v, e = inject_vortex(
        X, Y, args.nx, args.ny, args.gamma, args.vortex_gamma, rho, u, v, e, args.Lx, args.Ly
    )

    # Full interaction
    rho, u, v, e, p = march(
        X,
        Y,
        rho,
        u,
        v,
        e,
        mean_pressure,
        args.gamma,
        dx,
        dy,
        args.dt,
        args.nt,
        plot=0 if args.no_plot else 1,
        save=1 if args.save else 0,
        pause=args.pause if args.save else max(args.nt - 1, 1),
        Cx=args.Cx,
        Cy=args.Cy,
    )

    # Keep plot window open if plotting enabled and not saving
    if not args.no_plot:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
