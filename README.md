# cfd_solver
 
Educational CFD/examples repository organized as multiple small projects. Each project lives in its own subfolder and can be run directly with Python from the repository root.
 
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
1) Create and activate a Python environment (3.10+ recommended), then install dependencies:
 
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
 
Parameters for each example are currently set in the script `__main__` blocks. See per-project `README.md` for brief notes.
 
## Utilities
- `tools/extract_pdf_texts.py`: Extract text from all PDFs under the repo root into `docs_txt/` (already run once). Requires `pdfminer.six`.
 
```bash
python .\tools\extract_pdf_texts.py
```
 
## Notes
See `notes.md` for a detailed code quality assessment and a roadmap to further modularization, testing, and CI.