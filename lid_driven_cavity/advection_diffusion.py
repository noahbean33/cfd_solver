"""
Vorticity advection-diffusion update for lid-driven cavity.
"""
from __future__ import annotations

import numpy as np


def step_vorticity(omega: np.ndarray, psi: np.ndarray, dx: float, dy: float, dt: float, nu: float) -> np.ndarray:
    ny, nx = omega.shape
    w = omega.copy()

    # velocities from psi: u = dpsi/dy, v = -dpsi/dx
    u = np.zeros_like(psi)
    v = np.zeros_like(psi)
    u[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dy)
    u[0, :] = (psi[1, :] - psi[0, :]) / dy
    u[-1, :] = (psi[-1, :] - psi[-2, :]) / dy
    v[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dx)
    v[:, 0] = -(psi[:, 1] - psi[:, 0]) / dx
    v[:, -1] = -(psi[:, -1] - psi[:, -2]) / dx

    # derivatives of omega (centered in interior)
    wx = np.zeros_like(omega)
    wy = np.zeros_like(omega)
    wxx = np.zeros_like(omega)
    wyy = np.zeros_like(omega)

    wx[:, 1:-1] = (omega[:, 2:] - omega[:, :-2]) / (2.0 * dx)
    wy[1:-1, :] = (omega[2:, :] - omega[:-2, :]) / (2.0 * dy)
    wxx[:, 1:-1] = (omega[:, 2:] - 2.0 * omega[:, 1:-1] + omega[:, :-2]) / (dx * dx)
    wyy[1:-1, :] = (omega[2:, :] - 2.0 * omega[1:-1, :] + omega[:-2, :]) / (dy * dy)

    adv = u * wx + v * wy
    diff = nu * (wxx + wyy)

    w[1:-1, 1:-1] = omega[1:-1, 1:-1] + dt * (-(adv[1:-1, 1:-1]) + diff[1:-1, 1:-1])

    return w
