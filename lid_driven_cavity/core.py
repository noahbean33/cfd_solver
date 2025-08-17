"""
Core driver for the lid-driven cavity using the streamfunction-vorticity formulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import numpy as np

from .poisson import solve_poisson
from .advection_diffusion import step_vorticity
from .boundaries import apply_wall_bcs


@dataclass
class GridConfig:
    nx: int
    ny: int
    Lx: float = 1.0
    Ly: float = 1.0

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx, ny must be >= 3")
        if self.Lx <= 0 or self.Ly <= 0:
            raise ValueError("Lx, Ly must be positive")

    def spacing(self) -> tuple[float, float]:
        dx = self.Lx / (self.nx - 1)
        dy = self.Ly / (self.ny - 1)
        return dx, dy


@dataclass
class FluidConfig:
    Re: float | None = 1000.0
    nu: float | None = None
    Uwall: float = 1.0

    def __post_init__(self) -> None:
        if self.nu is None and self.Re is None:
            raise ValueError("Provide either Re or nu")
        if self.nu is None:
            self.nu = 1.0 / float(self.Re)
        if self.nu <= 0:
            raise ValueError("nu must be positive")


@dataclass
class TimeConfig:
    dt: float
    steps: int
    save_every: int = 0

    def __post_init__(self) -> None:
        if self.dt <= 0 or self.steps <= 0:
            raise ValueError("dt > 0 and steps > 0 required")


@dataclass
class PoissonConfig:
    method: Literal["jacobi", "sor"] = "jacobi"
    max_iters: int = 2000
    tol: float = 1e-6
    omega: float = 1.7


@dataclass
class Result:
    psi: np.ndarray
    omega: np.ndarray
    U: np.ndarray
    V: np.ndarray
    err_psi: np.ndarray
    err_omega: np.ndarray
    X: np.ndarray
    Y: np.ndarray


def make_grid(grid: GridConfig) -> tuple[np.ndarray, np.ndarray, float, float]:
    dx, dy = grid.spacing()
    X, Y = np.meshgrid(np.linspace(0.0, grid.Lx, grid.nx), np.linspace(0.0, grid.Ly, grid.ny))
    return X, Y, dx, dy


def velocities_from_psi(psi: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    # u = dpsi/dy, v = -dpsi/dx (streamfunction definition)
    U = np.zeros_like(psi)
    V = np.zeros_like(psi)
    U[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dy)
    U[0, :] = (psi[1, :] - psi[0, :]) / dy
    U[-1, :] = (psi[-1, :] - psi[-2, :]) / dy
    V[:, 1:-1] = -(psi[:, 2:] - psi[:, :-2]) / (2.0 * dx)
    V[:, 0] = -(psi[:, 1] - psi[:, 0]) / dx
    V[:, -1] = -(psi[:, -1] - psi[:, -2]) / dx
    return U, V


def stable_dt(U: np.ndarray, V: np.ndarray, dx: float, dy: float, nu: float, cfl: float = 0.5) -> float:
    """Compute a conservative stable timestep bound for explicit advection-diffusion.

    Uses advective CFL and diffusive limit: dt <= cfl * min(dx/|u|_max, dy/|v|_max, 0.5*min(dx,dy)^2/nu).
    If velocities are zero, falls back to diffusion limit.
    """
    umax = float(np.nanmax(np.abs(U))) if np.isfinite(U).any() else 0.0
    vmax = float(np.nanmax(np.abs(V))) if np.isfinite(V).any() else 0.0
    adv_dt_x = dx / umax if umax > 1e-14 else np.inf
    adv_dt_y = dy / vmax if vmax > 1e-14 else np.inf
    diff_dt = 0.5 * min(dx, dy) ** 2 / max(nu, 1e-14)
    dt_stable = cfl * min(adv_dt_x, adv_dt_y, diff_dt)
    if not np.isfinite(dt_stable) or dt_stable <= 0:
        dt_stable = cfl * diff_dt
    return dt_stable


def run(grid: GridConfig, fluid: FluidConfig, time: TimeConfig, poisson: PoissonConfig) -> Result:
    X, Y, dx, dy = make_grid(grid)
    psi = np.zeros((grid.ny, grid.nx), dtype=float)
    omega = np.zeros((grid.ny, grid.nx), dtype=float)

    err_psi_hist: list[float] = []
    err_omega_hist: list[float] = []

    # initial BCs
    apply_wall_bcs(psi, omega, fluid.Uwall, dx, dy)

    for k in range(time.steps):
        psi_old_outer = psi.copy()
        omega_old_outer = omega.copy()

        # velocities for CFL estimate from current psi
        U_est, V_est = velocities_from_psi(psi, dx, dy)
        dt_bound = stable_dt(U_est, V_est, dx, dy, fluid.nu, cfl=0.2)
        nsub = max(1, int(np.ceil(time.dt / dt_bound)))
        dt_sub = time.dt / nsub

        for _ in range(nsub):
            # Enforce boundary conditions from current fields (includes moving lid)
            apply_wall_bcs(psi, omega, fluid.Uwall, dx, dy)

            # Vorticity advection-diffusion explicit substep
            omega = step_vorticity(omega, psi, dx, dy, dt_sub, fluid.nu)
            # Re-enforce wall BCs on updated vorticity
            apply_wall_bcs(psi, omega, fluid.Uwall, dx, dy)

            # Solve Poisson for streamfunction each substep to keep coupling tight
            psi, _, _ = solve_poisson(psi, -omega, dx, dy, poisson)

        # Track errors
        err_psi = np.linalg.norm(psi - psi_old_outer)
        err_omega = np.linalg.norm(omega - omega_old_outer)
        err_psi_hist.append(err_psi)
        err_omega_hist.append(err_omega)

        # Early exit if both small
        if err_psi < poisson.tol and err_omega < poisson.tol:
            break

    U, V = velocities_from_psi(psi, dx, dy)
    return Result(
        psi=psi,
        omega=omega,
        U=U,
        V=V,
        err_psi=np.asarray(err_psi_hist),
        err_omega=np.asarray(err_omega_hist),
        X=X,
        Y=Y,
    )
