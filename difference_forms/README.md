# Difference Forms Generator

Generate finite-difference stencils (coefficients) for derivatives on a uniform grid.

## Quick Start (from repository root)

```bash
# Legacy runner (shows LaTeX of a chosen stencil)
python .\difference_forms\differenceForms.py

# CLI: compute a stencil and print LaTeX/plain text
python -m difference_forms.cli --derivative 1 --scheme centered --accuracy 2 --h 1 --latex
```

## API (difference_forms.core)
- `build_offsets(derivative_order, scheme, accuracy_order) -> ndarray`
- `fd_coefficients(derivative_order, scheme, accuracy_order, h=1.0) -> (offsets, coeffs)`
  - Returns coefficients already scaled by `h^m` (apply directly to samples f(x + k h))
- `scale_to_integers(coeffs, tol=1e-8) -> (rounded_coeffs, multiplier)`

## Formatting (difference_forms.formatting)
- `format_readable(offsets, coeffs) -> str`
- `format_latex(offsets, coeffs, derivative_order, h_symbol='\\Delta x') -> str`

## Examples
- Centered first derivative, O(Δx^2): offsets `[-1,0,1]`, coeffs `[-1/2, 0, 1/2] / h`
- Centered second derivative, O(Δx^2): offsets `[-1,0,1]`, coeffs `[1, -2, 1] / h^2`

## Testing
Run pytest from the repository root:

```bash
pytest difference_forms -q
```

## Notes
- The original script `differenceForms.py` now delegates to the new API and just renders a LaTeX expression for quick visualization.
