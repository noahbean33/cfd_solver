import numpy as np

from potential_flow.core import GridConfig, SolverConfig, FlowConfig, run


def test_obstacle_zero_normal_flux_edges():
    # Grid and single obstacle in the middle
    grid = GridConfig(nx=81, ny=61, Lx=4.0, Ly=3.0)
    flow = FlowConfig(Uinf=1.0)
    solver = SolverConfig(method="jacobi", max_iters=8000, tol=1e-8)

    # Define a rectangle away from boundaries
    obstacles = [(25, 35, 30, 50)]  # (imin, imax, jmin, jmax)
    res = run(obstacles=obstacles, grid=grid, flow=flow, solver=solver)

    phi = res.phi
    imin, imax, jmin, jmax = obstacles[0]

    # Check zero normal gradient on each face by comparing adjacent cells
    # Left/right faces -> normal is +/- x direction
    left_face_grad = phi[imin:imax, jmin] - phi[imin:imax, jmin - 1]
    right_face_grad = phi[imin:imax, jmax] - phi[imin:imax, jmax + 1]
    # Bottom/top faces -> normal is +/- y direction
    bottom_face_grad = phi[imin, jmin:jmax] - phi[imin - 1, jmin:jmax]
    top_face_grad = phi[imax, jmin:jmax] - phi[imax + 1, jmin:jmax]

    # Mean absolute gradients should be small
    mae = np.mean(np.abs(np.concatenate([
        left_face_grad.ravel(), right_face_grad.ravel(),
        bottom_face_grad.ravel(), top_face_grad.ravel()
    ])))

    assert mae < 1e-2


def test_cp_finite_and_reasonable():
    grid = GridConfig(nx=101, ny=71, Lx=5.0, Ly=3.0)
    flow = FlowConfig(Uinf=1.5)
    solver = SolverConfig(method="jacobi", max_iters=12000, tol=1e-8)
    obstacles = [(30, 45, 40, 60)]

    res = run(obstacles=obstacles, grid=grid, flow=flow, solver=solver)

    Cp = res.Cp
    assert np.all(np.isfinite(Cp))
    # Cp bounds are problem-dependent; check they're not absurd
    assert Cp.min() > -10.0 and Cp.max() < 10.0
