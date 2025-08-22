# cfd_solver

Educational CFD/examples repository with multiple small, self-contained projects. Each project lives in its own subfolder and can be run from the repository root.
 
## Layout
- `heat_1d/` — 1D Heat Equation
- `wave_1d/` — 1D Wave Equation (FTBS vs MacCormack vs analytical)
- `lid_driven_cavity/` — 2D streamfunction–vorticity lid-driven cavity
- `potential_flow/` — 2D Laplace potential flow with rectangular obstacles
- `isentropic_vortex/` — Isentropic vortex convection (Euler)
- `sod_shock_tube/` — Sod shock tube (Euler) with experimental comparison
- `shock_vortex_interaction/` — Shock–vortex interaction (Euler)
- `difference_forms/` — Finite-difference stencil generator
- `taylor_series/` — Taylor series approximations for sin/exp
- `docs/` — Reference PDFs
- `docs_txt/` — Extracted text versions of PDFs
- `images/` — Output figures (if scripts are configured to save)
- `tools/` — Utility scripts (e.g., PDF text extraction)
- `notes.md` — Code quality review and recommendations
 
## Quick Start
1) Create and activate a Python environment (Python 3.10+ recommended), then install dependencies:
 
```bash
pip install -r requirements.txt
```
 
2) Run an example from the repository root (to keep relative paths working):
 
```bash
# Heat equation
python .\heat_1d\1dHeatEquation.py
 
# Wave equation
python .\wave_1d\1dWaveEquation.py
 
# Lid-driven cavity
python .\lid_driven_cavity\lidDrivenCavity.py
 
# Potential flow (Laplace)
python .\potential_flow\potentialFlow.py
 
# Isentropic vortex
python .\isentropic_vortex\isentropicVortex.py
 
# Sod shock tube
python .\sod_shock_tube\sodShockTube.py
 
# Shock–vortex interaction
python .\shock_vortex_interaction\shockVortexInteraction.py
 
# Difference forms generator
python .\difference_forms\differenceForms.py
 
# Taylor series approximations
python .\taylor_series\taylorSeriesApproximation.py
```
 
Parameters for legacy scripts are set in the `__main__` blocks. See each submodule's `README.md` for notes.

### Modern CLI entry points (recommended)
Some modules provide a CLI with configurable arguments and directory-safe saving:

```bash
# 1D Heat Equation
python -m heat_1d.cli --numX 101 --alpha 0.2 --tmax 100 --dt 10 --scheme implicit --save --out images/heat_1d/profiles.png --no-show

# Sod Shock Tube
python -m sod_shock_tube.cli --nx 401 --Lx 1 --nt 20000 --dt 1e-5 --gamma 1.4 --diaphragm 0.5 --Cx 0.4 --pause 500 --save --outdir images/sod

# Shock–Vortex Interaction
python -m shock_vortex_interaction.cli --nx 161 --ny 161 --Lx 40 --Ly 40 --nt 2001 --dt 1e-3 --gamma 1.4 --vortex-gamma 0.125 --save --pause 200 --no-plot
```

Images are saved under `images/...` with directories created automatically.
 
## Testing and Notes
- Run tests: `python -m pytest -q`
- See `notes.md` for a detailed code quality assessment and a roadmap to further modularization, testing, and CI.