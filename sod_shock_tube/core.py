from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class Grid1D:
    x: Array
    dx: float


def initialize_grid(Lx: float, nx: int) -> Grid1D:
    x = np.linspace(0.0, Lx, nx)
    dx = Lx / (nx - 1)
    return Grid1D(x=x, dx=dx)


def initialize_state(x: Array, gamma: float, diaphragm_x: float) -> Tuple[Array, Array, Array]:
    # Left and right states
    rho = np.where(x <= diaphragm_x, 1.0, 0.125)
    p = np.where(x <= diaphragm_x, 1.0, 0.1)
    u = np.zeros_like(x)
    e = p / (rho * (gamma - 1.0))
    return rho, u, e


def pressure(rho: Array, e: Array, gamma: float) -> Array:
    return rho * e * (gamma - 1.0)


def macCormack_step(rho: Array, u: Array, e: Array, gamma: float, dx: float, dt: float,
                    Cx: float = 0.4) -> Tuple[Array, Array, Array]:
    # Compute fluxes in 1D Euler: U=[rho, rho u, rho E]
    p = pressure(rho, e, gamma)
    E = e + 0.5 * (u ** 2)

    # Forward differences (predictor)
    drho_dx = np.zeros_like(rho)
    du_dx = np.zeros_like(u)
    de_dx = np.zeros_like(e)
    dp_dx = np.zeros_like(p)

    drho_dx[:-1] = (rho[1:] - rho[:-1]) / dx
    du_dx[:-1] = (u[1:] - u[:-1]) / dx
    de_dx[:-1] = (e[1:] - e[:-1]) / dx
    dp_dx[:-1] = (p[1:] - p[:-1]) / dx

    drho_dt = -(rho * du_dx + u * drho_dx)
    du_dt = -(u * du_dx + dp_dx / rho)
    de_dt = -(u * de_dx + (p / rho) * du_dx)

    rho_p = rho.copy()
    u_p = u.copy()
    e_p = e.copy()

    # Artificial viscosity term (x-only, second-difference on p)
    if rho.size >= 3:
        press_term = np.zeros_like(rho)
        press_term[1:-1] = np.abs(p[2:] - 2 * p[1:-1] + p[:-2]) / (
            p[2:] + 2 * p[1:-1] + p[:-2] + 1e-14
        )
        rho_p[1:-1] += drho_dt[1:-1] * dt + Cx * press_term[1:-1] * (
            rho[2:] - 2 * rho[1:-1] + rho[:-2]
        )
        u_p[1:-1] += du_dt[1:-1] * dt + Cx * press_term[1:-1] * (
            u[2:] - 2 * u[1:-1] + u[:-2]
        )
        e_p[1:-1] += de_dt[1:-1] * dt + Cx * press_term[1:-1] * (
            e[2:] - 2 * e[1:-1] + e[:-2]
        )

    # Apply transmissive BCs on predictor
    rho_p[0] = rho_p[1]
    rho_p[-1] = rho_p[-2]
    u_p[0] = u_p[1]
    u_p[-1] = u_p[-2]
    e_p[0] = e_p[1]
    e_p[-1] = e_p[-2]

    # Recompute with backward differences (corrector)
    p_p = pressure(rho_p, e_p, gamma)
    drho_dx_b = np.zeros_like(rho)
    du_dx_b = np.zeros_like(u)
    de_dx_b = np.zeros_like(e)
    dp_dx_b = np.zeros_like(p)

    drho_dx_b[1:] = (rho_p[1:] - rho_p[:-1]) / dx
    du_dx_b[1:] = (u_p[1:] - u_p[:-1]) / dx
    de_dx_b[1:] = (e_p[1:] - e_p[:-1]) / dx
    dp_dx_b[1:] = (p_p[1:] - p_p[:-1]) / dx

    drho_dt_p = -(rho_p * du_dx_b + u_p * drho_dx_b)
    du_dt_p = -(u_p * du_dx_b + dp_dx_b / rho_p)
    de_dt_p = -(u_p * de_dx_b + (p_p / rho_p) * du_dx_b)

    rho_new = rho.copy()
    u_new = u.copy()
    e_new = e.copy()

    rho_new[1:-1] = rho[1:-1] + 0.5 * (drho_dt[1:-1] + drho_dt_p[1:-1]) * dt
    u_new[1:-1] = u[1:-1] + 0.5 * (du_dt[1:-1] + du_dt_p[1:-1]) * dt
    e_new[1:-1] = e[1:-1] + 0.5 * (de_dt[1:-1] + de_dt_p[1:-1]) * dt

    # Transmissive BCs on corrector
    rho_new[0] = rho_new[1]
    rho_new[-1] = rho_new[-2]
    u_new[0] = u_new[1]
    u_new[-1] = u_new[-2]
    e_new[0] = e_new[1]
    e_new[-1] = e_new[-2]

    return rho_new, u_new, e_new


def march(x: Array, rho0: Array, u0: Array, e0: Array, gamma: float, dx: float, dt: float, nt: int,
          Cx: float = 0.4, snapshot_steps: Sequence[int] | None = None) -> Dict[str, object]:
    if snapshot_steps is None:
        snapshot_steps = []

    rho = rho0.copy()
    u = u0.copy()
    e = e0.copy()

    snapshots: List[Dict[str, Array]] = []

    for step in range(1, nt + 1):
        rho, u, e = macCormack_step(rho, u, e, gamma, dx, dt, Cx=Cx)
        if step in snapshot_steps:
            snapshots.append({
                "step": step,
                "rho": rho.copy(),
                "u": u.copy(),
                "p": pressure(rho, e, gamma).copy(),
            })

    return {
        "rho": rho,
        "u": u,
        "e": e,
        "p": pressure(rho, e, gamma),
        "snapshots": snapshots,
    }
