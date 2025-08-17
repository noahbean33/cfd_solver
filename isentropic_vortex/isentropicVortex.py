"""
Thin runner delegating to the modular API.
"""
from __future__ import annotations

from .core import GridConfig, GasConfig, FlowConfig, TimeConfig, run
from .plotting import plot_vorticity


def main() -> int:
    grid = GridConfig(nx=201, ny=201, Lx=10.0, Ly=10.0)
    gas = GasConfig(gamma=1.4)
    flow = FlowConfig(u_inf=1.0, v_inf=0.0, vortex_gamma=0.5)
    time = TimeConfig(steps=1000, dt=None, CFL=0.5)
    res = run(grid, gas, flow, time)
    plot_vorticity(res.X, res.Y, res.u, res.v)
    import matplotlib.pyplot as plt
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
