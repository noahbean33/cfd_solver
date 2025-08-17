import numpy as np

from potential_flow.core import GridConfig, SolverConfig, FlowConfig, run, velocities


def test_residual_decreases_jacobi_small_grid():
    grid = GridConfig(nx=61, ny=33, Lx=2.0, Ly=1.0)
    solver = SolverConfig(method="jacobi", max_iters=2000, tol=0.0)
    flow = FlowConfig(Uinf=1.0)

    res = run(obstacles=None, grid=grid, flow=flow, solver=solver)
    # Residual should decrease; at least final < initial
    assert res.err.size > 2
    assert res.err[-1] < res.err[0]
    # Over the last half, residual roughly non-increasing
    mid = res.err.size // 2
    assert (np.diff(res.err[mid:]) <= 1e-12).sum() >= (res.err.size - mid - 2)


def test_empty_domain_linear_potential_velocity():
    grid = GridConfig(nx=81, ny=41, Lx=2.0, Ly=1.0)
    solver = SolverConfig(method="jacobi", max_iters=5000, tol=1e-8)
    flow = FlowConfig(Uinf=1.0)

    res = run(obstacles=None, grid=grid, flow=flow, solver=solver)
    # In empty domain with farfield BCs, solution should be approximately linear in x,
    # so U ≈ Uinf and V ≈ 0 in the interior.
    U, V = res.U[1:-1, 1:-1], res.V[1:-1, 1:-1]
    mae_U = float(np.mean(np.abs(U - flow.Uinf)))
    mae_V = float(np.mean(np.abs(V)))

    assert mae_U < 5e-2
    assert mae_V < 5e-2
