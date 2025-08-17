"""
Core solver, configs, and run orchestration for 2D Laplace potential flow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np

from .boundaries import apply_farfield, apply_obstacle_bc

MethodType = Literal["jacobi", "sor"]


@dataclass
class GridConfig:
    nx: int
    ny: int
    Lx: float
    Ly: float

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must be >= 3")
        if self.Lx <= 0.0 or self.Ly <= 0.0:
            raise ValueError("Lx and Ly must be positive")

    def spacing(self) -> tuple[float, float]:
        dx = self.Lx / (self.nx - 1)
        dy = self.Ly / (self.ny - 1)
        return dx, dy


@dataclass
class SolverConfig:
    method: MethodType = "jacobi"
    max_iters: int = 20000
    tol: float = 1e-6
    omega: float = 1.7  # for SOR

    def __post_init__(self) -> None:
        if self.max_iters <= 0:
            raise ValueError("max_iters must be > 0")
        if self.tol < 0.0:
            raise ValueError("tol must be >= 0")
        if self.method not in ("jacobi", "sor"):
            raise ValueError("method must be 'jacobi' or 'sor'")
        if self.method == "sor" and not (0.0 < self.omega < 2.0):
            raise ValueError("omega must be in (0,2) for SOR")


@dataclass
class FlowConfig:
    Uinf: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.Uinf):
            raise ValueError("Uinf must be finite")


@dataclass
class Result:
    phi: np.ndarray
    U: np.ndarray
    V: np.ndarray
    Cp: np.ndarray
    err: np.ndarray
    X: np.ndarray
    Y: np.ndarray


def make_grid(cfg: GridConfig) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Create meshgrid and spacings.

    Returns X, Y arrays with shape (ny,nx) and spacings dx, dy.
    """
    dx, dy = cfg.spacing()
    X, Y = np.meshgrid(np.linspace(0.0, cfg.Lx, cfg.nx), np.linspace(0.0, cfg.Ly, cfg.ny))
    return X, Y, dx, dy


def iterate_laplace(phi: np.ndarray, mask: np.ndarray, grid: GridConfig, flow: FlowConfig, solver: SolverConfig) -> np.ndarray:
    """Iteratively solve Laplace equation with Jacobi or SOR until tol or max_iters.

    Returns residual history (L2 norm of update).
    """
    ny, nx = phi.shape

    err = np.zeros((solver.max_iters,), dtype=float)
    for t in range(solver.max_iters):
        phi_old = phi.copy()

        # Jacobi/SOR update on fluid interior (exclude mask and domain boundary)
        phi_new_center = 0.25 * (
            phi_old[2:, 1:-1]   # i+1
            + phi_old[:-2, 1:-1]  # i-1
            + phi_old[1:-1, 2:]   # j+1
            + phi_old[1:-1, :-2]  # j-1
        )
        update_region = (~mask)[1:-1, 1:-1]
        if solver.method == "jacobi":
            phi[1:-1, 1:-1][update_region] = phi_new_center[update_region]
        elif solver.method == "sor":
            phi[1:-1, 1:-1][update_region] = (
                (1.0 - solver.omega) * phi_old[1:-1, 1:-1][update_region]
                + solver.omega * phi_new_center[update_region]
            )
        else:
            raise ValueError("Unknown method")

        # Boundary conditions and obstacle enforcement
        apply_farfield(phi, flow.Uinf, grid.Lx, grid.Ly, grid.nx, grid.ny)
        apply_obstacle_bc(phi, mask)

        # Convergence metric
        err[t] = np.linalg.norm(phi - phi_old)
        if err[t] < solver.tol:
            return err[: t + 1]

    return err


def velocities(phi: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute velocity components from potential via backward differences."""
    U = np.zeros_like(phi)
    V = np.zeros_like(phi)
    # Central differences in the interior
    U[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
    V[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dy)
    # One-sided at edges
    U[:, 0] = (phi[:, 1] - phi[:, 0]) / dx
    U[:, -1] = (phi[:, -1] - phi[:, -2]) / dx
    V[0, :] = (phi[1, :] - phi[0, :]) / dy
    V[-1, :] = (phi[-1, :] - phi[-2, :]) / dy
    return U, V


def pressure_coefficient(U: np.ndarray, V: np.ndarray, Uinf: float, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute pressure coefficient on interior cells from velocity magnitude.

    If a mask is provided, Cp at masked interior cells is set to 0.0 to avoid
    spurious values inside solid obstacles.
    """
    vel_sq = U[1:-1, 1:-1] ** 2 + V[1:-1, 1:-1] ** 2
    Cp = 1.0 - vel_sq / (Uinf * Uinf + 1e-12)
    if mask is not None:
        interior_mask = mask[1:-1, 1:-1]
        Cp[interior_mask] = 0.0
    return Cp


def run(obstacles: list[tuple[int, int, int, int]] | None, grid: GridConfig, flow: FlowConfig, solver: SolverConfig) -> Result:
    """High-level driver for potential flow around rectangular obstacles.

    obstacles: list of (imin, imax, jmin, jmax) index rectangles in (i=row, j=col) order.
    """
    X, Y, dx, dy = make_grid(grid)
    phi = np.zeros((grid.ny, grid.nx), dtype=float)
    # Linear initial guess consistent with uniform flow
    phi[:, :] = flow.Uinf * X
    # Build mask
    mask = np.zeros_like(phi, dtype=bool)
    if obstacles:
        for imin, imax, jmin, jmax in obstacles:
            if not (0 <= imin < imax <= grid.ny - 1) or not (0 <= jmin < jmax <= grid.nx - 1):
                raise ValueError("Obstacle indices out of bounds or not ordered")
            mask[imin:imax, jmin:jmax] = True

    apply_farfield(phi, flow.Uinf, grid.Lx, grid.Ly, grid.nx, grid.ny)
    apply_obstacle_bc(phi, mask)

    err = iterate_laplace(phi, mask, grid, flow, solver)

    U, V = velocities(phi, dx, dy)
    Cp = pressure_coefficient(U, V, flow.Uinf, mask)
    return Result(phi=phi, U=U, V=V, Cp=Cp, err=err, X=X, Y=Y)
