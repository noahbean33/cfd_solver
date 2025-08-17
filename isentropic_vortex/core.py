"""
Core driver for the isentropic vortex using the 2D compressible Euler equations.
Conservative variables with MacCormack predictor–corrector and periodic BCs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .initial_conditions import isentropic_vortex_ic
from .schemes import mac_cormack_step
from .metrics import primitives_from_conservative


@dataclass
class GridConfig:
    nx: int
    ny: int
    Lx: float = 10.0
    Ly: float = 10.0

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
class GasConfig:
    gamma: float = 1.4

    def __post_init__(self) -> None:
        if not (1.0 < self.gamma < 2.0):
            raise ValueError("gamma must be in (1,2)")


@dataclass
class FlowConfig:
    u_inf: float = 1.0
    v_inf: float = 0.0
    vortex_gamma: float = 0.5
    cx: Optional[float] = None
    cy: Optional[float] = None


@dataclass
class TimeConfig:
    steps: int = 10000
    dt: Optional[float] = None
    CFL: float = 0.5

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be > 0")
        if self.dt is not None and self.dt <= 0:
            raise ValueError("dt must be positive if provided")
        if not (0 < self.CFL <= 0.9):
            raise ValueError("CFL must be in (0,0.9]")


@dataclass
class Result:
    Q: np.ndarray  # shape (4, ny, nx)
    rho: np.ndarray
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    err_hist: np.ndarray
    dt_hist: np.ndarray


def make_grid(grid: GridConfig) -> tuple[np.ndarray, np.ndarray, float, float]:
    dx, dy = grid.spacing()
    X, Y = np.meshgrid(np.linspace(0.0, grid.Lx, grid.nx), np.linspace(0.0, grid.Ly, grid.ny))
    return X, Y, dx, dy


def run(grid: GridConfig, gas: GasConfig, flow: FlowConfig, time: TimeConfig) -> Result:
    X, Y, dx, dy = make_grid(grid)
    cx = flow.cx if flow.cx is not None else 0.5 * grid.Lx
    cy = flow.cy if flow.cy is not None else 0.5 * grid.Ly

    Q = isentropic_vortex_ic(X, Y, gas.gamma, flow.vortex_gamma, flow.u_inf, flow.v_inf, cx, cy)

    err_hist: list[float] = []
    dt_hist: list[float] = []

    for _ in range(time.steps):
        Q_old = Q.copy()
        Q, dt_used, res = mac_cormack_step(Q, dx, dy, gas.gamma, CFL=time.CFL, dt=time.dt)
        err_hist.append(float(res))
        dt_hist.append(float(dt_used))

    rho, u, v, p = primitives_from_conservative(Q, gas.gamma)
    return Result(
        Q=Q,
        rho=rho,
        u=u,
        v=v,
        p=p,
        X=X,
        Y=Y,
        err_hist=np.asarray(err_hist),
        dt_hist=np.asarray(dt_hist),
    )
