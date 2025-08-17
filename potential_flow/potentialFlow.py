# Script to compute the numerical solution to the 2D Laplace equation

import numpy as np
import matplotlib.pyplot as plt

from potential_flow.core import GridConfig, SolverConfig, FlowConfig, run
from potential_flow.plotting import plot_potential_surface, plot_streamlines_cp, plot_convergence


def main():
    # Geometry and grid
    Lx, Ly = 20.0, 11.0
    nx, ny = 201, 111

    # Obstacles as (imin, imax, jmin, jmax)
    obstacles = [(30, 50, 60, 80), (60, 80, 90, 120)]

    # Solver and flow
    solver = SolverConfig(method="jacobi", max_iters=40000, tol=1e-6, omega=1.7)
    flow = FlowConfig(Uinf=1.0)
    grid = GridConfig(nx=nx, ny=ny, Lx=Lx, Ly=Ly)

    # Run
    res = run(obstacles, grid, flow, solver)

    # Plots similar to original
    plot_potential_surface(res.X, res.Y, res.phi)
    plot_streamlines_cp(res.X, res.Y, res.U, res.V, res.Cp, obstacles=obstacles, Lx=Lx, Ly=Ly, nx=nx, ny=ny)
    plot_convergence(res.err)
    plt.show()


if __name__ == "__main__":
    main()
