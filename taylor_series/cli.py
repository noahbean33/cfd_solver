"""
Command-line interface for Taylor series approximations.
"""
from __future__ import annotations

import argparse
import numpy as np
from .core import get_function, approximate, error_metrics
from .plotting import plot_comparison


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Taylor series approximation CLI")
    p.add_argument("--func", type=str, default="exp", choices=["sin", "cos", "exp"], help="Target function")
    p.add_argument("--a", type=float, default=0.0, help="Expansion center a")
    p.add_argument("--degree", type=int, default=6, help="Polynomial degree n >= 0")
    p.add_argument("--x-min", type=float, default=-2.0, help="Domain min")
    p.add_argument("--x-max", type=float, default=2.0, help="Domain max")
    p.add_argument("--num", type=int, default=400, help="Number of points")
    p.add_argument("--save", type=str, default=None, help="Path to save the plot (optional)")
    p.add_argument("--metrics", action="store_true", help="Print error metrics (L2, Linf)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    x = np.linspace(args.x_min, args.x_max, args.num)
    f = get_function(args.func)
    y_true = f(x)
    y_approx = approximate(args.func, args.a, args.degree, x)
    if args.metrics:
        m = error_metrics(y_true, y_approx)
        print(f"L2: {m['l2']:.6e}  Linf: {m['linf']:.6e}")
    title = f"{args.func}(x) ~ Taylor n={args.degree} about a={args.a}"
    plot_comparison(x, y_true, y_approx, title=title, save_path=args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
