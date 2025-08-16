"""
Core solvers and run loop for the 1D linear advection (wave) equation:
    u_t + c u_x = 0
on a uniform grid with simple boundary conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, List
import numpy as np

BCType = Literal["periodic", "outflow"]
SchemeType = Literal["ftbs", "maccormack"]


@dataclass
class SimConfig:
    dx: float
    dt: float
    c: float
    steps: int
    x_start: float
    x_end: float
    bc: BCType = "periodic"
    scheme: SchemeType = "maccormack"

    def cfl(self) -> float:
        return abs(self.c) * self.dt / self.dx


def apply_bc(u: np.ndarray, bc: BCType) -> np.ndarray:
    if bc == "periodic":
        # handled implicitly by wraparound indexing when needed
        return u
    elif bc == "outflow":
        # replicate boundary values to avoid inflow
        u[0] = u[1]
        u[-1] = u[-2]
        return u
    else:
        raise ValueError("Unsupported BC type")


def step_ftbs(u: np.ndarray, c: float, dt: float, dx: float, bc: BCType) -> np.ndarray:
    """Forward time, backward space (upwind for c>0)."""
    un = u.copy()
    sigma = c * dt / dx
    # upwind (assuming c>=0; if c<0, would use forward difference)
    if c >= 0:
        un[1:] = u[1:] - sigma * (u[1:] - u[:-1])
        # left boundary
        if bc == "periodic":
            un[0] = u[0] - sigma * (u[0] - u[-1])
        else:
            un[0] = u[0] - sigma * (u[0] - u[0])
    else:
        # c<0: use FTFS (mirror) to remain upwind
        un[:-1] = u[:-1] - sigma * (u[1:] - u[:-1])
        if bc == "periodic":
            un[-1] = u[-1] - sigma * (u[0] - u[-1])
        else:
            un[-1] = u[-1] - sigma * (u[-1] - u[-1])
    return apply_bc(un, bc)


def step_maccormack(u: np.ndarray, c: float, dt: float, dx: float, bc: BCType) -> np.ndarray:
    """MacCormack predictor-corrector for linear advection."""
    sigma = c * dt / dx
    # Predictor (use upwind/backward if c>=0, else forward)
    if c >= 0:
        up = u.copy()
        up[1:] = u[1:] - sigma * (u[1:] - u[:-1])
        if bc == "periodic":
            up[0] = u[0] - sigma * (u[0] - u[-1])
        else:
            up[0] = u[0]
        # Corrector (use forward diff of predictor)
        uc = u.copy()
        uc[:-1] = 0.5 * (u[:-1] + up[:-1] - sigma * (up[1:] - up[:-1]))
        if bc == "periodic":
            uc[-1] = 0.5 * (u[-1] + up[-1] - sigma * (up[0] - up[-1]))
        else:
            uc[-1] = up[-1]
    else:
        # Mirror for c<0
        up = u.copy()
        up[:-1] = u[:-1] - sigma * (u[1:] - u[:-1])
        if bc == "periodic":
            up[-1] = u[-1] - sigma * (u[0] - u[-1])
        else:
            up[-1] = u[-1]
        uc = u.copy()
        uc[1:] = 0.5 * (u[1:] + up[1:] - sigma * (up[1:] - up[:-1]))
        if bc == "periodic":
            uc[0] = 0.5 * (u[0] + up[0] - sigma * (up[0] - up[-1]))
        else:
            uc[0] = up[0]
    return apply_bc(uc, bc)


def make_grid(x_start: float, x_end: float, dx: float) -> np.ndarray:
    # Ensure last point is excluded like original arange usage (open interval)
    n = int(np.floor((x_end - x_start) / dx))
    return x_start + dx * np.arange(0, n)


def run(u0: np.ndarray, cfg: SimConfig, keep_trajectory: bool = False) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray] | None]:
    if cfg.dx <= 0 or cfg.dt <= 0:
        raise ValueError("dx and dt must be positive")
    if cfg.steps < 1:
        raise ValueError("steps must be >= 1")
    if cfg.cfl() > 1.0 and cfg.scheme == "ftbs":
        raise ValueError("FTBS unstable: CFL = |c| dt/dx must be <= 1")

    u = u0.copy()
    frames: List[np.ndarray] | None = [] if keep_trajectory else None

    for _ in range(cfg.steps):
        if cfg.scheme == "ftbs":
            u = step_ftbs(u, cfg.c, cfg.dt, cfg.dx, cfg.bc)
        elif cfg.scheme == "maccormack":
            u = step_maccormack(u, cfg.c, cfg.dt, cfg.dx, cfg.bc)
        else:
            raise ValueError("Unknown scheme")
        if keep_trajectory and frames is not None:
            frames.append(u.copy())

    xs = make_grid(cfg.x_start, cfg.x_end, cfg.dx)
    return u, xs, frames
