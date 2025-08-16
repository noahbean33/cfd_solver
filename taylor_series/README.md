# Taylor Series Approximation

This module provides clean, testable utilities for building Taylor polynomial approximations
of common functions and visualizing their accuracy.

## Quick Start (from repository root)

```bash
# Original example runner (plots multiple degrees)
python .\taylor_series\taylorSeriesApproximation.py

# CLI: single approximation with options and optional plot save
python -m taylor_series.cli --func exp --a 0 --degree 6 --x-min -2 --x-max 2 --num 400 --metrics --save taylor_exp.png
```

## API (taylor_series.core)
- `get_function(name: str)` → vectorized function (`sin`, `cos`, `exp`)
- `series_coefficients(func: str, a: float, n: int) -> ndarray` — coefficients for `sum c[k](x-a)^k`
- `evaluate_polynomial(coeffs: ndarray, x: ndarray, a: float) -> ndarray` — Horner evaluation
- `approximate(func: str, a: float, n: int, x: ndarray) -> ndarray` — convenience wrapper
- `error_metrics(y_true: ndarray, y_pred: ndarray) -> dict` — L2 and Linf errors

## CLI
`python -m taylor_series.cli --help`

Options include:
- `--func {sin,cos,exp}`
- `--a` (expansion center)
- `--degree` (polynomial degree)
- `--x-min --x-max --num` (domain)
- `--metrics` (print L2/Linf)
- `--save path.png` (save plot instead of showing)

## Testing
Run pytest from repo root:

```bash
pytest taylor_series -q
```

## Notes
- The legacy script `taylorSeriesApproximation.py` now delegates to the new core API while preserving its plotting behavior.
