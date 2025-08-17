"""
Boundary condition helpers for potential_flow.
"""
from __future__ import annotations

import numpy as np


def apply_farfield(phi: np.ndarray, Uinf: float, Lx: float, Ly: float, nx: int, ny: int) -> None:
    # Dirichlet on left/right consistent with uniform flow phi = Uinf * x
    phi[:, 0] = 0.0
    phi[:, -1] = Uinf * Lx
    # Neumann (zero-normal-flux) on bottom/top by copying interior rows
    phi[0, :] = phi[1, :]
    phi[-1, :] = phi[-2, :]


def apply_obstacle_bc(phi: np.ndarray, mask: np.ndarray) -> None:
    # Enforce zero-normal-flux on obstacle boundaries by copying neighbor values
    ny, nx = phi.shape
    ys, xs = np.where(mask)
    for i, j in zip(ys, xs):
        # Copy from nearest fluid neighbor (simple heuristic matching original code)
        if j - 1 >= 0 and not mask[i, j - 1]:
            phi[i, j] = phi[i, j - 1]
        elif j + 1 < nx and not mask[i, j + 1]:
            phi[i, j] = phi[i, j + 1]
        elif i - 1 >= 0 and not mask[i - 1, j]:
            phi[i, j] = phi[i - 1, j]
        elif i + 1 < ny and not mask[i + 1, j]:
            phi[i, j] = phi[i + 1, j]
        # else: isolated cell; leave as is

