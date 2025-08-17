"""
Core API for the Shock–Vortex Interaction example.

This module wraps the existing implementation in `shockVortexInteraction.py`
with a stable, importable interface so other code and tests can call it
without relying on `__main__`.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .shockVortexInteraction import (
    initializeGrid as _initializeGrid,
    initializeSolution as _initializeSolution,
    initializeVortex as _initializeVortex,
    marchSolution as _marchSolution,
)

Array = np.ndarray


def initialize_grid(Lx: float, Ly: float, nx: int, ny: int) -> Tuple[Array, Array, float, float]:
    """Create structured Cartesian grid and spacings.

    Returns: (X, Y, dx, dy)
    """
    return _initializeGrid(Lx, Ly, nx, ny)


def initialize_shock(
    X: Array,
    Y: Array,
    nx: int,
    ny: int,
    gamma: float,
    vortex_gamma: float,
    Lx: float,
    Ly: float,
) -> Tuple[Array, Array, Array, Array]:
    """Initialize a steady (approximate) shock state.

    Returns: (rho, u, v, e)
    """
    return _initializeSolution(X, Y, nx, ny, gamma, vortex_gamma, Lx, Ly)


def inject_vortex(
    X: Array,
    Y: Array,
    nx: int,
    ny: int,
    gamma: float,
    vortex_gamma: float,
    rho: Array,
    u: Array,
    v: Array,
    e: Array,
    Lx: float,
    Ly: float,
) -> Tuple[Array, Array, Array, Array]:
    """Inject the isentropic vortex perturbation into the existing flow field."""
    return _initializeVortex(X, Y, nx, ny, gamma, vortex_gamma, rho, u, v, e, Lx, Ly)


def march(
    X: Array,
    Y: Array,
    rho: Array,
    u: Array,
    v: Array,
    e: Array,
    mean_pressure: Array,
    gamma: float,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    plot: int = 1,
    save: int = 0,
    pause: int = 1,
    Cx: float = 0.3,
    Cy: float = 0.3,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Advance the solution using predictor–corrector with artificial viscosity.

    Returns: (rho, u, v, e, p)
    """
    return _marchSolution(
        X,
        Y,
        rho,
        u,
        v,
        e,
        mean_pressure,
        gamma,
        dx,
        dy,
        dt,
        nt,
        plot,
        save,
        pause,
        Cx,
        Cy,
    )
