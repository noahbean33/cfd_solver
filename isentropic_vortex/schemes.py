"""
Numerical schemes for 2D Euler: MacCormack predictor–corrector (conservative form).
"""
from __future__ import annotations

import numpy as np

from .metrics import primitives_from_conservative
from .boundaries import apply_periodic


def fluxes(Q: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    rho, u, v, p = primitives_from_conservative(Q, gamma)
    E = Q[3]
    F = np.empty_like(Q)
    G = np.empty_like(Q)
    F[0] = rho * u
    F[1] = rho * u * u + p
    F[2] = rho * u * v
    F[3] = (E + p) * u

    G[0] = rho * v
    G[1] = rho * u * v
    G[2] = rho * v * v + p
    G[3] = (E + p) * v
    return F, G


def max_wave_speed(Q: np.ndarray, gamma: float) -> float:
    rho, u, v, p = primitives_from_conservative(Q, gamma)
    a = np.sqrt(gamma * p / np.maximum(rho, 1e-12))
    return float(np.max(np.abs(u) + a) + np.max(np.abs(v) + a))


def mac_cormack_step(Q: np.ndarray, dx: float, dy: float, gamma: float, CFL: float = 0.5, dt: float | None = None):
    """Advance conservative state one step using MacCormack PC with periodic BCs.

    Returns (Q_next, dt_used, residual_norm).
    """
    # Determine dt
    if dt is None:
        # Estimate wave speeds in x and y separately
        rho, u, v, p = primitives_from_conservative(Q, gamma)
        a = np.sqrt(gamma * p / np.maximum(rho, 1e-12))
        sx = np.max(np.abs(u) + a)
        sy = np.max(np.abs(v) + a)
        smax = max(sx / dx if sx > 1e-12 else 0.0, sy / dy if sy > 1e-12 else 0.0)
        dt_used = CFL / (smax + 1e-14)
    else:
        dt_used = float(dt)

    # Predictor using forward differences with periodic wrap
    F, G = fluxes(Q, gamma)
    dFdx_f = (np.roll(F, -1, axis=2) - F) / dx
    dGdy_f = (np.roll(G, -1, axis=1) - G) / dy
    Qp = Q - dt_used * (dFdx_f + dGdy_f)
    apply_periodic(Qp)

    # Corrector using backward differences on predicted fluxes
    Fp, Gp = fluxes(Qp, gamma)
    dFdx_b = (Fp - np.roll(Fp, 1, axis=2)) / dx
    dGdy_b = (Gp - np.roll(Gp, 1, axis=1)) / dy
    Qn1 = 0.5 * (Q + Qp - dt_used * (dFdx_b + dGdy_b))
    apply_periodic(Qn1)

    res = float(np.linalg.norm(Qn1 - Q))
    return Qn1, dt_used, res
