# Potential Flow (Laplace)

This module solves the 2D Laplace equation for potential flow around rectangular obstacles on a uniform grid using Jacobi or SOR iteration.

## Background

For incompressible, irrotational flow, the velocity field u = ∇ϕ derives from a scalar potential ϕ that satisfies Laplace's equation ∇²ϕ = 0. With farfield conditions corresponding to a uniform free stream of speed U∞ in the x-direction, a suitable boundary condition is ϕ = U∞ x on the left and right domain boundaries; zero-normal-flux (Neumann) is enforced on the top/bottom and across solid obstacle boundaries.

## Module Structure

- `potential_flow/core.py`: dataclasses, grid utilities, Laplace iterator (Jacobi/SOR), velocities, Cp, and `run()` driver.
- `potential_flow/boundaries.py`: farfield and obstacle boundary conditions.
- `potential_flow/geometry.py`: rectangle representation and mask utilities.
- `potential_flow/plotting.py`: potential surface, streamlines/Cp, and convergence plots.
- `potential_flow/cli.py`: command-line interface.
- `potential_flow/potentialFlow.py`: thin runner preserving the original script behavior.

## Public API

- `GridConfig(nx, ny, Lx, Ly)`
- `SolverConfig(method={'jacobi','sor'}, max_iters, tol, omega=1.7)`
- `FlowConfig(Uinf)`
- `run(obstacles, grid, flow, solver) -> Result(phi, U, V, Cp, err, X, Y)`

Obstacles are given as index rectangles `(imin, imax, jmin, jmax)` in i=row (y), j=col (x) order.

Example:

```python
from potential_flow.core import GridConfig, SolverConfig, FlowConfig, run
grid = GridConfig(nx=201, ny=111, Lx=20.0, Ly=11.0)
solver = SolverConfig(method='jacobi', max_iters=40000, tol=1e-6)
flow = FlowConfig(Uinf=1.0)
obstacles = [(30, 50, 60, 80), (60, 80, 90, 120)]
res = run(obstacles, grid, flow, solver)
```

## CLI

Run from the repository root:

```bash
# Thin runner (mirrors original script defaults)
python .\potential_flow\potentialFlow.py

# General CLI
python -m potential_flow.cli --nx 201 --ny 111 --Lx 20 --Ly 11 \
  --Uinf 1.0 --method jacobi --max-iters 40000 --tol 1e-6 \
  --obstacles 30 50 60 80 60 80 90 120 --plot
```

CLI options:

- `--nx, --ny, --Lx, --Ly`: grid resolution and physical size
- `--Uinf`: free-stream speed
- `--method`: `jacobi` or `sor`
- `--max-iters, --tol, --omega`: solver controls
- `--obstacles`: one or more rectangles: `imin imax jmin jmax ...`
- `--plot`: show potential surface, streamlines/Cp, and convergence

## Testing

Pytest-based tests live in `potential_flow/tests/`.

```bash
pytest potential_flow -q
```

Current tests:

- residual decreases for Jacobi on a small grid
- empty-domain solution yields uniform U ≈ U∞ and V ≈ 0 in the interior

## Notes

- Dirichlet conditions are imposed at left/right to match `ϕ = U∞ x`; top/bottom are Neumann.
- SOR is supported via `SolverConfig(method='sor', omega=1.7)`.
