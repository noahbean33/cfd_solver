import numpy as np
import math
from taylor_series.core import series_coefficients


def test_exp_coefficients_a0():
    n = 6
    coeffs = series_coefficients("exp", 0.0, n)
    expected = np.array([1 / math.factorial(k) for k in range(n + 1)], dtype=float)
    assert np.allclose(coeffs, expected)


def test_sin_coefficients_a0():
    n = 7
    coeffs = series_coefficients("sin", 0.0, n)
    expected = np.zeros(n + 1, dtype=float)
    # sin series: x - x^3/3! + x^5/5! - x^7/7!
    for k in [1, 3, 5, 7]:
        expected[k] = ((-1) ** ((k - 1) // 2)) / math.factorial(k)
    assert np.allclose(coeffs, expected)


def test_cos_coefficients_a0():
    n = 6
    coeffs = series_coefficients("cos", 0.0, n)
    expected = np.zeros(n + 1, dtype=float)
    # cos series: 1 - x^2/2! + x^4/4! - x^6/6!
    expected[0] = 1.0
    for k in [2, 4, 6]:
        expected[k] = ((-1) ** (k // 2)) / math.factorial(k)
    assert np.allclose(coeffs, expected)
