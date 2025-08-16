import numpy as np
from taylor_series.core import evaluate_polynomial


def test_evaluate_polynomial_matches_naive_poly():
    # P(x) = 1 + 2*(x-a) + 3*(x-a)^2
    a = 0.5
    coeffs = np.array([1.0, 2.0, 3.0])
    x = np.linspace(-1.0, 1.0, 21)
    y_horner = evaluate_polynomial(coeffs, x, a)
    y_naive = coeffs[0] + coeffs[1] * (x - a) + coeffs[2] * (x - a) ** 2
    assert np.allclose(y_horner, y_naive)
