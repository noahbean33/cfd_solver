from __future__ import annotations

import numpy as np

from lid_driven_cavity.core import GridConfig, FluidConfig, TimeConfig, PoissonConfig, run


def small_run(nx=33, ny=33, steps=400, dt=1e-4, Re=100.0):
    grid = GridConfig(nx=nx, ny=ny, Lx=1.0, Ly=1.0)
    fluid = FluidConfig(Re=Re, Uwall=1.0)
    time = TimeConfig(dt=dt, steps=steps)
    poisson = PoissonConfig(method="sor", max_iters=4000, tol=1e-6, omega=1.7)
    return run(grid, fluid, time, poisson)


def test_residual_decreases_basic():
    res = small_run(steps=200, nx=33, ny=33)
    assert res.err_psi.size > 0 and res.err_omega.size > 0
    # Trend should generally decrease; allow noise
    dpsi = res.err_psi
    # Check last value smaller than first by a factor
    assert dpsi[-1] < 0.9 * dpsi[0]


def test_lid_velocity_enforced():
    res = small_run(steps=600, nx=41, ny=41, Re=100.0)
    # velocity just below top wall should approach Uwall
    U_top_inner = res.U[-2, 1:-1]
    assert np.isfinite(U_top_inner).all()
    mean_u = float(np.mean(U_top_inner))
    # Loose tolerance; transient
    assert abs(mean_u - 1.0) < 0.3


def test_incompressibility_divergence_small():
    res = small_run(steps=300, nx=41, ny=41, Re=100.0)
    U, V = res.U, res.V
    dx = res.X[0, 1] - res.X[0, 0]
    dy = res.Y[1, 0] - res.Y[0, 0]
    dudx = (U[1:-1, 2:] - U[1:-1, :-2]) / (2.0 * dx)
    dvdy = (V[2:, 1:-1] - V[:-2, 1:-1]) / (2.0 * dy)
    div = dudx + dvdy
    assert np.mean(np.abs(div)) < 5e-2
