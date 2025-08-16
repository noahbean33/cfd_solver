"""
Formatting helpers for finite-difference stencils.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np


def format_readable(offsets: Sequence[int], coeffs: Sequence[float]) -> str:
    """
    Produce a human-readable string like: -1*u_{i-1} + 0*u_i + 1*u_{i+1}
    """
    parts = []
    for k, c in zip(offsets, coeffs):
        if abs(c) < 1e-14:
            continue
        if k == 0:
            ref = "u_{i}"
        elif k > 0:
            ref = f"u_{{i+{k}}}"
        else:
            ref = f"u_{{i{'' if k==0 else k}}}"  # k negative prints like i-1
        # Coefficient formatting
        if np.isclose(c, 1.0):
            term = ref
        elif np.isclose(c, -1.0):
            term = f"-{ref}"
        else:
            term = f"{c}*{ref}"
        parts.append(term)
    if not parts:
        return "0"
    # Join with plus/minus signs correctly
    s = parts[0]
    for term in parts[1:]:
        if term.startswith('-'):
            s += f" {term}"
        else:
            s += f" + {term}"
    return s


def format_latex(
    offsets: Sequence[int],
    coeffs: Sequence[float],
    derivative_order: int,
    h_symbol: str = "\\Delta x",
) -> str:
    """
    Return a LaTeX fraction string representing the stencil over h^m.
    Example: \frac{-u_{i-1} + 0u_i + u_{i+1}}{2 \\Delta x}
    """
    # Build numerator string
    terms = []
    for k, c in zip(offsets, coeffs):
        if abs(c) < 1e-14:
            continue
        if k == 0:
            ref = "u_{i}"
        elif k > 0:
            ref = f"u_{{i+{k}}}"
        else:
            ref = f"u_{{i{k}}}"  # e.g., i-1
        if np.isclose(c, 1.0):
            terms.append(ref)
        elif np.isclose(c, -1.0):
            terms.append(f"-{ref}")
        else:
            # Prefer integers when close
            if np.isclose(c, round(c)):
                coeff_str = str(int(round(c)))
            else:
                coeff_str = f"{c}"
            terms.append(f"{coeff_str}{ref}")
    num = ' + '.join(terms).replace('+ -', '- ')

    # Denominator
    denom = h_symbol if derivative_order == 1 else f"{h_symbol}^{{{derivative_order}}}"
    return f"\\frac{{{num}}}{{{denom}}}"
