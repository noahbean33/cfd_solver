"""
Poisson solver for streamfunction: ∇²ψ = rhs.
Supports Jacobi and SOR iterations with tolerance stopping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:  # avoid runtime circular import
    from .core import PoissonConfig


def residual_laplacian(psi: np.ndarray, rhs: np.ndarray, dx: float, dy: float) -> float:
    lap = (
        (psi[2:, 1:-1] - 2 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dy**2
        + (psi[1:-1, 2:] - 2 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dx**2
    )
    r = lap - rhs[1:-1, 1:-1]
    return float(np.linalg.norm(r))


def solve_poisson(
    psi: np.ndarray,
    rhs: np.ndarray,
    dx: float,
    dy: float,
    cfg: 'PoissonConfig',
) -> Tuple[np.ndarray, int, np.ndarray]:
    ny, nx = psi.shape
    res_hist = []
    if cfg.method == "jacobi":
        for it in range(cfg.max_iters):
            psi_old = psi.copy()
            psi[1:-1, 1:-1] = (
                -rhs[1:-1, 1:-1]
                - (psi_old[2:, 1:-1] + psi_old[:-2, 1:-1]) / dy**2
                - (psi_old[1:-1, 2:] + psi_old[1:-1, :-2]) / dx**2
            ) / (-2.0 / dx**2 - 2.0 / dy**2)
            res = residual_laplacian(psi, rhs, dx, dy)
            res_hist.append(res)
            if res < cfg.tol:
                break
        return psi, it + 1, np.asarray(res_hist)
    elif cfg.method == "sor":
        omega = cfg.omega
        for it in range(cfg.max_iters):
            for i in range(1, ny - 1):
                for j in range(1, nx - 1):
                    a = -2.0 / dx**2 - 2.0 / dy**2
                    b = (
                        -rhs[i, j]
                        - (psi[i + 1, j] + psi[i - 1, j]) / dy**2
                        - (psi[i, j + 1] + psi[i, j - 1]) / dx**2
                    )
                    psi[i, j] = (1 - omega) * psi[i, j] + omega * (b / a)
            res = residual_laplacian(psi, rhs, dx, dy)
            res_hist.append(res)
            if res < cfg.tol:
                break
        return psi, it + 1, np.asarray(res_hist)
    else:
        raise ValueError("Unknown Poisson method")
