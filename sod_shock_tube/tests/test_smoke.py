import numpy as np

from sod_shock_tube.core import initialize_grid, initialize_state, march


def test_smoke_small_run():
    Lx = 1.0
    nx = 101
    grid = initialize_grid(Lx, nx)
    gamma = 1.4
    diaphragm_x = 0.5 * Lx
    rho0, u0, e0 = initialize_state(grid.x, gamma, diaphragm_x)

    res = march(
        x=grid.x,
        rho0=rho0,
        u0=u0,
        e0=e0,
        gamma=gamma,
        dx=grid.dx,
        dt=2e-4,
        nt=200,
        Cx=0.4,
        snapshot_steps=[50, 100, 200],
    )

    rho = res["rho"]
    u = res["u"]
    p = res["p"]

    assert rho.shape == (nx,)
    assert u.shape == (nx,)
    assert p.shape == (nx,)
    assert np.isfinite(rho).all()
    assert np.isfinite(u).all()
    assert np.isfinite(p).all()
