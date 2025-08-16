"""
CLI for generating finite-difference stencils.
"""
from __future__ import annotations

import argparse
import numpy as np
from .core import fd_coefficients, build_offsets, scale_to_integers
from .formatting import format_latex, format_readable


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finite-difference stencil generator")
    p.add_argument("--derivative", type=int, required=True, help="Derivative order m >= 1")
    p.add_argument("--scheme", type=str, choices=["centered", "forward", "backward"], default="centered")
    p.add_argument("--accuracy", type=int, default=2, help="Desired accuracy order (>=1)")
    p.add_argument("--h", type=float, default=1.0, help="Grid spacing h > 0")
    p.add_argument("--latex", action="store_true", help="Output LaTeX expression")
    p.add_argument("--scale-integers", action="store_true", help="Scale to near-integers when possible")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    offsets, coeffs = fd_coefficients(args.derivative, args.scheme, args.accuracy, h=args.h)

    if args.scale_integers:
        scaled, mult = scale_to_integers(coeffs)
        # Adjust by h^m scaling factor already applied; scale_to_integers treats values as-is
        coeffs_out = scaled
    else:
        coeffs_out = coeffs

    if args.latex:
        s = format_latex(offsets, coeffs_out, derivative_order=args.derivative, h_symbol="\\Delta x")
    else:
        s = format_readable(offsets, coeffs_out)
    print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
