"""
Boundary conditions for streamfunction-vorticity lid-driven cavity.
"""
from __future__ import annotations

import numpy as np


def apply_wall_bcs(psi: np.ndarray, omega: np.ndarray, Uwall: float, dx: float, dy: float) -> None:
    """Apply no-slip, no-penetration BCs and moving lid to psi/omega.

    Conventions:
    - Walls have psi = 0 (no penetration)
    - Vorticity at walls via ghost-cell elimination
      bottom (i=0):    ω_0,j = 2(ψ_0,j - ψ_1,j)/dy^2
      top (i=ny-1):    ω_-1,j = 2(ψ_-1,j - ψ_-2,j)/dy^2 - 2 Uwall/dy
      left (j=0):      ω_i,0 = 2(ψ_i,0 - ψ_i,1)/dx^2
      right (j=nx-1):  ω_i,-1 = 2(ψ_i,-1 - ψ_i,-2)/dx^2
    """
    ny, nx = psi.shape
    # Streamfunction on walls: set to zero
    psi[0, :] = 0.0
    psi[-1, :] = 0.0
    psi[:, 0] = 0.0
    psi[:, -1] = 0.0

    # Vorticity from streamfunction and lid speed
    # interior x-range and y-range
    j_inner = slice(1, nx - 1)
    i_inner = slice(1, ny - 1)

    # bottom wall i=0
    omega[0, j_inner] = 2.0 * (psi[0, j_inner] - psi[1, j_inner]) / (dy * dy)
    # top wall i=ny-1 with moving lid (u=Uwall)
    omega[-1, j_inner] = 2.0 * (psi[-1, j_inner] - psi[-2, j_inner]) / (dy * dy) - 2.0 * Uwall / dy
    # left wall j=0
    omega[i_inner, 0] = 2.0 * (psi[i_inner, 0] - psi[i_inner, 1]) / (dx * dx)
    # right wall j=nx-1
    omega[i_inner, -1] = 2.0 * (psi[i_inner, -1] - psi[i_inner, -2]) / (dx * dx)
