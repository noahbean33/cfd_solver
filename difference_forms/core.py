"""
Core utilities to generate finite-difference stencils.

This module computes coefficients for approximating the m-th derivative using
uniform-grid samples f(x + k h) over integer offsets k.

Definition:
    sum_i c_i f(x + k_i h) ≈ f^{(m)}(x)
with conditions on moments:
    sum_i c_i k_i^p = 0 for p != m, and sum_i c_i k_i^m = m!
The user applies the coefficients as (c_i / h^m).
"""
from __future__ import annotations

from typing import Literal, Tuple
import numpy as np

Scheme = Literal["centered", "forward", "backward"]


def build_offsets(derivative_order: int, scheme: Scheme, accuracy_order: int) -> np.ndarray:
    """
    Choose integer offsets for a stencil with desired derivative and accuracy.

    Heuristic: number of points N = derivative_order + accuracy_order.
    - centered: ensure N is odd, use symmetric offsets [-r..r]
    - forward: use [0..N-1]
    - backward: use [-(N-1)..0]
    """
    if derivative_order < 1:
        raise ValueError("derivative_order must be >= 1")
    if accuracy_order < 1:
        raise ValueError("accuracy_order must be >= 1")

    N = derivative_order + accuracy_order
    scheme = scheme.lower()

    if scheme == "centered":
        # Prefer classic minimal stencils for O(Δx^2) first/second derivatives
        if accuracy_order == 2 and derivative_order in (1, 2):
            offsets = np.array([-1, 0, 1], dtype=int)
        else:
            if N % 2 == 0:
                N += 1  # make odd
            r = (N - 1) // 2
            offsets = np.arange(-r, r + 1, dtype=int)
    elif scheme == "forward":
        offsets = np.arange(0, N, dtype=int)
    elif scheme == "backward":
        offsets = -np.arange(0, N, dtype=int)[::-1]
    else:
        raise ValueError("scheme must be one of: centered, forward, backward")

    return offsets


def moment_matrix(offsets: np.ndarray) -> np.ndarray:
    """
    Build the square moment matrix A with A[p, i] = k_i^p for p=0..N-1 and i over offsets.
    """
    k = np.asarray(offsets, dtype=float)
    N = k.size
    A = np.vstack([k ** p for p in range(N)])
    return A


def fd_coefficients(
    derivative_order: int,
    scheme: Scheme,
    accuracy_order: int,
    h: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute finite-difference coefficients for m-th derivative.

    Returns (offsets, coeffs_scaled) where coeffs_scaled = c / h^m.
    """
    if h <= 0:
        raise ValueError("h must be > 0")

    offsets = build_offsets(derivative_order, scheme, accuracy_order)
    A = moment_matrix(offsets)
    N = A.shape[0]

    # RHS b: b[m] = m!, others 0
    b = np.zeros(N, dtype=float)
    m = derivative_order
    if m >= N:
        # This shouldn't happen with our N rule, but guard regardless
        raise ValueError("Stencil too small for requested derivative order")
    # factorial
    fact = 1.0
    for i in range(2, m + 1):
        fact *= i
    b[m] = fact

    # Solve A c = b
    c = np.linalg.solve(A, b)
    # Scale for spacing h: apply as c / h^m
    coeffs_scaled = c / (h ** m)
    return offsets.astype(int), coeffs_scaled


def scale_to_integers(coeffs: np.ndarray, tol: float = 1e-8, max_mult: int = 2000) -> tuple[np.ndarray, int]:
    """
    Try to scale coefficients by a small integer to make them near-integers.
    Returns (rounded_coeffs, multiplier). If not found, returns original and 1.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    for mult in range(1, max_mult + 1):
        scaled = coeffs * mult
        rounded = np.rint(scaled)
        if np.max(np.abs(rounded - scaled)) < tol:
            return rounded.astype(int), mult
    return coeffs.copy(), 1
