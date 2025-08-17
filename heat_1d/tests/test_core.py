import numpy as np

from heat_1d.core import build_grid, explicit_step, implicit_step_thomas, solve, _thomas_tridiagonal


def test_build_grid_and_boundary_conditions():
    x, dx = build_grid(21)
    assert x.shape == (21,)
    res = solve(numX=21, alpha=0.2, t_max=1.0, dt=0.1, temp1=20.0, temp2=100.0, scheme="explicit")
    y = res["y"]
    assert y.shape == (21,)
    # Dirichlet BCs preserved
    assert np.isclose(y[0], 20.0)
    assert np.isclose(y[-1], 100.0)


def test_explicit_stability_smoke():
    # C <= 0.5 for stability
    numX = 51
    x, dx = build_grid(numX)
    alpha = 0.2
    dt = 0.5 * dx * dx / alpha  # C = 0.5
    res = solve(numX=numX, alpha=alpha, t_max=0.5, dt=dt, temp1=0.0, temp2=1.0, scheme="explicit")
    y = res["y"]
    assert np.isfinite(y).all()
    # monotone between BCs for this setup
    assert y.min() >= 0.0 - 1e-12 and y.max() <= 1.0 + 1e-12


def test_thomas_matches_solve_for_random_tridiagonal():
    rng = np.random.default_rng(0)
    n = 20
    a = rng.random(n - 1)
    b = 2.0 + rng.random(n)
    c = rng.random(n - 1)
    d = rng.random(n)

    # Build full matrix for reference solve
    A = np.zeros((n, n))
    A[np.arange(n), np.arange(n)] = b
    A[np.arange(n - 1) + 1, np.arange(n - 1)] = a
    A[np.arange(n - 1), np.arange(n - 1) + 1] = c

    x_ref = np.linalg.solve(A, d)
    x_thomas = _thomas_tridiagonal(a, b, c, d)
    assert np.allclose(x_ref, x_thomas, atol=1e-10, rtol=1e-10)


def test_implicit_solver_runs_and_preserves_bcs():
    res = solve(numX=33, alpha=0.2, t_max=1.0, dt=0.2, temp1=10.0, temp2=50.0, scheme="implicit")
    y = res["y"]
    assert np.isclose(y[0], 10.0)
    assert np.isclose(y[-1], 50.0)
