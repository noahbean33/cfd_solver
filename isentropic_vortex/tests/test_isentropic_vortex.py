from __future__ import annotations

import numpy as np

from isentropic_vortex.core import GridConfig, GasConfig, FlowConfig, TimeConfig, run, make_grid
from isentropic_vortex.initial_conditions import isentropic_vortex_ic
from isentropic_vortex.metrics import primitives_from_conservative


def test_periodicity_one_period():
    # One convective period T = Lx/u_inf
    grid = GridConfig(nx=81, ny=81, Lx=10.0, Ly=10.0)
    gas = GasConfig(gamma=1.4)
    flow = FlowConfig(u_inf=1.0, v_inf=0.0, vortex_gamma=0.5)

    # Choose dt respecting CFL roughly and integer number of steps per period
    X, Y, dx, dy = make_grid(grid)
    # rough max wave speed ~ |u|+a ~ 2
    dt = 0.03
    T = grid.Lx / flow.u_inf
    steps = int(T / dt)
    time = TimeConfig(steps=steps, dt=dt, CFL=0.5)

    # initial reference
    Q0 = isentropic_vortex_ic(X, Y, gas.gamma, flow.vortex_gamma, flow.u_inf, flow.v_inf, grid.Lx * 0.5, grid.Ly * 0.5)
    rho0, _, _, _ = primitives_from_conservative(Q0, gas.gamma)

    res = run(grid, gas, flow, time)

    # Compare final density to initial (should be close due to periodic translation by exactly one period)
    num = np.linalg.norm(res.rho - rho0)
    den = np.linalg.norm(rho0)
    rel = num / (den + 1e-16)
    assert rel < 5e-2


def test_mass_conservation():
    grid = GridConfig(nx=81, ny=81, Lx=10.0, Ly=10.0)
    gas = GasConfig(gamma=1.4)
    flow = FlowConfig(u_inf=1.0, v_inf=0.0, vortex_gamma=0.5)
    dt = 0.02
    steps = 500
    time = TimeConfig(steps=steps, dt=dt, CFL=0.5)

    X, Y, dx, dy = make_grid(grid)
    Q0 = isentropic_vortex_ic(X, Y, gas.gamma, flow.vortex_gamma, flow.u_inf, flow.v_inf, grid.Lx * 0.5, grid.Ly * 0.5)
    rho0, _, _, _ = primitives_from_conservative(Q0, gas.gamma)
    mass0 = float(np.sum(rho0) * dx * dy)

    res = run(grid, gas, flow, time)
    mass1 = float(np.sum(res.rho) * dx * dy)

    rel_err = abs(mass1 - mass0) / mass0
    assert rel_err < 1e-3
