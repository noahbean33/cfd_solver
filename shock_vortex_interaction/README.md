# Shock–Vortex Interaction

Two-phase run: converge a steady shock, inject an isentropic vortex, then simulate the interaction.

## Quickstart (CLI)

Run from the repository root:

```bash
python -m shock_vortex_interaction.cli --nx 161 --ny 161 --Lx 40 --Ly 40 --nt 2001 --dt 1e-3 --gamma 1.4 --vortex-gamma 0.125 --save --pause 200
```

- Add `--no-plot` to disable plotting.
- Frames are written under `images/shockVortex/` (directories are created automatically).

## Legacy script

You can still run the original script (kept for reference):

```bash
python .\shock_vortex_interaction\shockVortexInteraction.py
```

## Library API

Programmatic usage is available via `shock_vortex_interaction/core.py`:

- `initialize_grid(Lx, Ly, nx, ny)` -> `X, Y, dx, dy`
- `initialize_shock(X, Y, nx, ny, gamma, vortex_gamma, Lx, Ly)` -> `rho, u, v, e`
- `inject_vortex(X, Y, nx, ny, gamma, vortex_gamma, rho, u, v, e, Lx, Ly)` -> `rho, u, v, e`
- `march(X, Y, rho, u, v, e, mean_pressure, gamma, dx, dy, dt, nt, plot=1, save=0, pause=1, Cx=0.3, Cy=0.3)` -> `rho, u, v, e, p`

## Notes

- Periodic in y; outlet uses linear extrapolation at the right boundary.
- For development or CI, prefer smaller grids, e.g., `--nx 51 --ny 51 --nt 2001`.
