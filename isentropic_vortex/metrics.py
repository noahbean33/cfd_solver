"""
Metric utilities and variable conversions for compressible Euler.
"""
from __future__ import annotations

import numpy as np


def primitives_from_conservative(Q: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert conservative Q=[rho, rho u, rho v, E] to (rho, u, v, p)."""
    rho = Q[0]
    inv_rho = 1.0 / np.maximum(rho, 1e-12)
    u = Q[1] * inv_rho
    v = Q[2] * inv_rho
    kinetic = 0.5 * rho * (u * u + v * v)
    p = (Q[3] - kinetic) * (gamma - 1.0)
    p = np.maximum(p, 1e-12)
    return rho, u, v, p


def vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    dudY = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy)
    dvdX = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dx)
    return dvdX - dudY
