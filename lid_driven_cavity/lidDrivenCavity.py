"""
Thin runner for the modular lid-driven cavity solver.
Use the CLI (python -m lid_driven_cavity.cli) for full control.
"""
from __future__ import annotations

from .core import GridConfig, FluidConfig, TimeConfig, PoissonConfig, run
from .plotting import plot_streamlines_and_vorticity, plot_convergence


def main() -> None:
    grid = GridConfig(nx=129, ny=129, Lx=1.0, Ly=1.0)
    fluid = FluidConfig(Re=1000.0, Uwall=1.0)
    time = TimeConfig(dt=3e-4, steps=2000)
    poisson = PoissonConfig(method="sor", tol=1e-6, max_iters=5000, omega=1.7)

    res = run(grid, fluid, time, poisson)
    print(f"Steps: {res.err_psi.size}, final residuals: dpsi={res.err_psi[-1]:.3e}, domega={res.err_omega[-1]:.3e}")

    try:
        import matplotlib.pyplot as plt
        plot_streamlines_and_vorticity(res.X, res.Y, res.U, res.V, res.omega)
        plot_convergence(res.err_psi, res.err_omega)
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
