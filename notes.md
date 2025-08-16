# Code Quality and Architecture Notes for `cfd_solver`

## Summary
Educational, script-style CFD prototypes that demonstrate core methods (FTBS, MacCormack, predictor–corrector, artificial viscosity, Laplace solver). Numerics are mostly vectorized and readable, but the repo lacks structure, documentation, tests, and robustness around I/O and parameterization. A few correctness/performance issues exist.

## Strengths
- **Pedagogical clarity**: Each script targets a single phenomenon with compact implementations.
- **Vectorized updates**: Wide use of NumPy slicing over Python loops (e.g., `sodShockTube.py` 57–67; `isentropicVortex.py` 54–64).
- **Visualization**: Sensible plots and labels show outcomes clearly (e.g., `sodShockTube.py` 174–199; `potentialFlow.py` 70–105).

## Issues and Risks
- **Script-only design, no package structure**
  - Standalone scripts with globals and hardcoded parameters. Hard to reuse/import/test.
  - No CLI/config; parameter changes require editing files.

- **Hidden global dependencies**
  - Several functions implicitly depend on globals like `nx`, `ny`, `Lx`, `Ly` rather than arguments.
  - Example: `shockVortexInteraction.py` `initializeSolution()` uses `Lx`, `Ly` (37–45) but does not accept them as parameters. Similar patterns in `isentropicVortex.py`, `sodShockTube.py` `marchSolution()`.

- **Boundary conditions and I/O robustness**
  - Image saving assumes directories exist (e.g., `sodShockTube.py` 173; `isentropicVortex.py` 180; `potentialFlow.py` 83). No `os.makedirs(..., exist_ok=True)` or error handling.
  - Minor inconsistency: `lidDrivenCavity.py` sets `U[ny-1,0:] = 1` while `Uwall = 1` is defined; better to use the variable.

- **Potential bugs / correctness concerns**
  - `lidDrivenCavity.py`: `omega = np.zeros([ny,ny])` (34) likely should be `[ny,nx]`; shape mismatch vs. `psi` and later indexing.
  - `1dHeatEquation.py`: Implicit step uses `np.linalg.inv(triDiag)` per step (52). Should use `np.linalg.solve` (or `scipy.linalg.solve`) and pre-factorize; current approach is slow/inaccurate.
  - `differenceForms.py`: Uses matrix inverse (47) similarly; prefer `np.linalg.solve`.
  - `shockVortexInteraction.py`: `marchSolution()` returns 5 arrays (236) as `[rho,u,v,e,p]`, but the “shock-only” phase assigns to `[rho,u,v,e,meanP]` (261). The naming `meanP` is misleading; also `initializeSolution()` uses globals `Lx/Ly`.
  - `1dHeatEquation.py`: Legend label uses `pausePercentages[index][0]` (58); works but is brittle if multiple matches.

- **Performance scalability**
  - Repeated dense inverses and lack of banded/iterative solvers (heat equation tridiagonal).
  - Fixed large iteration counts (e.g., `potentialFlow.py` `numSteps = 40000`) without early stopping based on tolerance, despite computing `err`.

- **Documentation & reproducibility**
  - `README.md` only contains a title; no setup or run instructions, expected outputs, or reference figures.
  - `requirements.txt` pins older versions; no Python version or environment guidance. No `pyproject.toml`/`setup.cfg`.
  - No tests; only one CSV data file present for validation (`sod_pressure_experimental.csv`).

- **Style & maintainability**
  - No docstrings or type hints. Array shapes/units undocumented.
  - Duplicate logic across scripts (grid init, BCs, loops) could be shared utilities.
  - Some unused imports (`matplotlib.animation`, `Rectangle`) in places.

## File-by-File Highlights
- `1dHeatEquation.py`
  - Clear explicit/implicit schemes; good vectorization.
  - Issues: `np.linalg.inv` in loop (52); solver logic mixed with plotting; hardcoded `__main__` params.

- `1dWaveEquation.py`
  - Good comparison: MacCormack vs FTBS vs analytical (`analyticalFunction()`).
  - Hardcoded function choice (47) and image path; no directory checks.

- `differenceForms.py`
  - Useful finite-difference stencil generator with LaTeX output.
  - Uses `np.linalg.inv` (47); should switch to `np.linalg.solve`.

- `isentropicVortex.py`
  - Predictor–corrector with periodic BCs; artificial viscosity off (appropriate for smooth flow).
  - `marchSolution()` implicitly uses globals `nx`, `ny`; pass dimensions explicitly.

- `lidDrivenCavity.py`
  - Compact streamfunction–vorticity scheme with streamlines/vorticity contours.
  - Likely bug: `omega` allocated as `[ny,ny]` (34) instead of `[ny,nx]`.
  - Fixed iteration count; computes an error but only prints intermittently.

- `potentialFlow.py`
  - Laplace iteration with obstacles and Cp visualization; nice masking for boxes.
  - Long fixed iteration count; has `err` array but lacks tolerance-based termination.

- `sodShockTube.py`
  - 2D grid for 1D shock tube; predictor–corrector with AV (`Cx=0.4`, `Cy=0`).
  - Depends on globals in functions; saves images without ensuring directories; reads `sod_pressure_experimental.csv` (191) assuming CWD.

- `shockVortexInteraction.py`
  - Two-phase: converge shock, inject vortex, then simulate; good pedagogical structure.
  - Globals used in initialization; outlet BC uses linear extrapolation; plotting levels relative to mean pressure is thoughtful.

## Recommended Actions (Priority)
- **[critical] Fix correctness bugs**
  - `lidDrivenCavity.py`: set `omega = np.zeros([ny,nx])`.
  - Replace all `np.linalg.inv(A) @ b` with `np.linalg.solve(A, b)` (`1dHeatEquation.py`, `differenceForms.py`).
  - Ensure output dirs exist before `plt.savefig` across scripts.

- **[high] Remove global coupling**
  - Pass `nx`, `ny`, `Lx`, `Ly` into functions using them.
  - Add docstrings describing inputs/outputs and array shapes.

- **[high] Introduce basic structure**
  - Create a package `cfd_solver/` with modules: `grid.py`, `bc.py`, `schemes.py`, `problems/`, `viz.py`.
  - Move current scripts into `examples/` using small CLIs (`argparse`) that call library functions.

- **[medium] Reproducibility & docs**
  - Expand `README.md` with setup, run commands, and sample figures.
  - Add Python version; consider `pyproject.toml` and looser pins or exact env via `conda`/`uv` lock.

- **[medium] Testing**
  - Add `pytest` tests: conservation checks, MMS for Laplace, wave advection vs analytical shift.

- **[medium] Performance/UX**
  - Heat equation: use Thomas algorithm/banded solver; pre-factorize if using dense.
  - Add tolerance-based early stopping in iterative solvers.
  - Replace print with logging; add `--save` flag to control file output and create dirs.

- **[nice-to-have] Style & CI**
  - Adopt `black`/`ruff`, optional type hints and `mypy`.
  - Add pre-commit hooks and a simple CI to run tests.

## Quick Wins (low risk)
- Fix `omega` shape in `lidDrivenCavity.py`.
- Replace `np.linalg.inv` with `np.linalg.solve` where present.
- Add a small helper to `os.makedirs` image directories and use it.
- Add minimal docstrings for `initializeGrid()`, `initializeSolution()`, `marchSolution()` in each solver.
- Expand `README.md` with instructions per script and expected outputs.

## Possible Roadmap
1. Patch correctness/performance quick wins.
2. Add README + directory handling + basic docstrings.
3. Extract shared utilities (grid, BCs, plotting) into `cfd_solver/`.
4. Introduce CLIs for examples and add basic tests.
5. Add formatting, type hints, CI, and more validation cases.
