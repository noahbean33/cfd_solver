"""
Core functionality for Taylor series approximations.

Provides pure functions with type hints and docstrings.
"""
from __future__ import annotations

from typing import Callable, Dict
import numpy as np


def get_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a vectorized numpy function by name.

    Supported: 'sin', 'cos', 'exp'.
    """
    name = name.lower()
    if name == "sin":
        return np.sin
    if name == "cos":
        return np.cos
    if name == "exp":
        return np.exp
    raise ValueError(f"Unsupported function '{name}'. Supported: sin, cos, exp.")


def _kth_derivative_value(name: str, a: float, k: int) -> float:
    """
    Closed-form kth derivative value at point a for supported functions.
    """
    name = name.lower()
    if name == "exp":
        # d^k/dx^k exp(x) = exp(x)
        return float(np.exp(a))
    if name == "sin":
        # cycle: sin, cos, -sin, -cos
        r = k % 4
        if r == 0:
            return float(np.sin(a))
        if r == 1:
            return float(np.cos(a))
        if r == 2:
            return float(-np.sin(a))
        return float(-np.cos(a))
    if name == "cos":
        # cycle: cos, -sin, -cos, sin
        r = k % 4
        if r == 0:
            return float(np.cos(a))
        if r == 1:
            return float(-np.sin(a))
        if r == 2:
            return float(-np.cos(a))
        return float(np.sin(a))
    raise ValueError(f"Unsupported function '{name}'. Supported: sin, cos, exp.")


def series_coefficients(func: str, a: float, n: int) -> np.ndarray:
    """
    Compute Taylor series coefficients up to degree n around center a.

    Returns coefficients c where P_n(x) = sum_{k=0..n} c[k] * (x - a)^k,
    with c[k] = f^{(k)}(a)/k!
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    coeffs = np.empty(n + 1, dtype=float)
    fact = 1.0
    for k in range(0, n + 1):
        if k > 0:
            fact *= k
        coeffs[k] = _kth_derivative_value(func, a, k) / fact
    return coeffs


def evaluate_polynomial(coeffs: np.ndarray, x: np.ndarray, a: float) -> np.ndarray:
    """
    Evaluate P(x) = sum_{k=0..n} coeffs[k] * (x - a)^k using Horner's rule.
    """
    # Horner in terms of (x - a)
    t = x - a
    y = np.zeros_like(x, dtype=float)
    for c in coeffs[::-1]:
        y = y * t + c
    return y


def approximate(func: str, a: float, n: int, x: np.ndarray) -> np.ndarray:
    """
    Build degree-n Taylor polynomial of func around a and evaluate on x.
    """
    coeffs = series_coefficients(func, a, n)
    return evaluate_polynomial(coeffs, x, a)


essential_metrics = ("l2", "linf")


def error_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute L2 and Linf error metrics between arrays.
    """
    diff = np.asarray(y_true) - np.asarray(y_pred)
    l2 = float(np.linalg.norm(diff) / np.sqrt(diff.size))
    linf = float(np.max(np.abs(diff)))
    return {"l2": l2, "linf": linf}
