import numpy as np

from shock_vortex_interaction.core import (
    initialize_grid,
    initialize_shock,
    inject_vortex,
    march,
)


def test_smoke_small_run():
    # Small grid, tiny steps to ensure it runs end-to-end quickly
    nx = ny = 51
    Lx = Ly = 4.0
    nt = 5
    dt = 1e-3
    gamma = 1.4

    X, Y, dx, dy = initialize_grid(Lx, Ly, nx, ny)
    rho, u, v, e = initialize_shock(X, Y, nx, ny, gamma, 0.0, Lx, Ly)

    # Shock-only convergence step (no plotting/saving)
    rho, u, v, e, mean_pressure = march(
        X, Y, rho, u, v, e, rho * e * (gamma - 1.0), gamma, dx, dy, dt, nt, plot=0, save=0
    )

    # Inject a small vortex and march a bit
    rho, u, v, e = inject_vortex(X, Y, nx, ny, gamma, 0.05, rho, u, v, e, Lx, Ly)
    rho2, u2, v2, e2, p2 = march(
        X, Y, rho, u, v, e, mean_pressure, gamma, dx, dy, dt, nt, plot=0, save=0
    )

    # Basic sanity checks
    assert rho2.shape == (ny, nx)
    assert u2.shape == (ny, nx)
    assert v2.shape == (ny, nx)
    assert e2.shape == (ny, nx)
    assert p2.shape == (ny, nx)
    assert np.isfinite(rho2).all()
    assert np.isfinite(p2).all()
