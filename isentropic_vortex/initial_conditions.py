"""
Initial conditions for the isentropic vortex problem.
"""
from __future__ import annotations

import numpy as np


def isentropic_vortex_ic(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float,
    vortex_gamma: float,
    u_inf: float,
    v_inf: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Return conservative state Q with shape (4, ny, nx).

    The free-stream is (u_inf, v_inf), and a Gaussian vortex is imposed about (cx, cy).
    """
    ny, nx = X.shape
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    expfac = np.exp(0.5 * (1.0 - r2))

    du = -(vortex_gamma / (2.0 * np.pi)) * (Y - cy) * expfac
    dv = +(vortex_gamma / (2.0 * np.pi)) * (X - cx) * expfac

    u = u_inf + du
    v = v_inf + dv

    tempVar = 0.125 * vortex_gamma * vortex_gamma * (gamma - 1.0) / (gamma * np.pi * np.pi)
    rho = (1.0 - tempVar * np.exp(1.0 * (1.0 - r2))) ** (1.0 / (gamma - 1.0))
    p = rho ** gamma

    Q = np.zeros((4, ny, nx), dtype=float)
    Q[0] = rho
    Q[1] = rho * u
    Q[2] = rho * v
    E_internal = p / (gamma - 1.0)
    kinetic = 0.5 * rho * (u * u + v * v)
    Q[3] = E_internal + kinetic
    return Q
