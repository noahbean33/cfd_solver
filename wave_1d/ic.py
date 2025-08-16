"""
Initial conditions for 1D advection.
"""
from __future__ import annotations

import numpy as np


def gaussian(x: np.ndarray, x0: float, width: float = 0.1) -> np.ndarray:
    return np.exp(-((x - x0) ** 2) / (2 * width ** 2))


def sinusoid(x: np.ndarray, x0: float, wavelength: float) -> np.ndarray:
    return np.sin(2 * np.pi * (x - x0) / wavelength)


def multi_sin(x: np.ndarray, x0: float, wavelength: float) -> np.ndarray:
    return (
        (1.0 / 3.0)
        * (
            np.sin(2 * np.pi * (x - x0) / wavelength)
            + np.sin(4 * np.pi * (x - x0) / wavelength)
            + np.sin(8 * np.pi * (x - x0) / wavelength)
        )
    )


def step(x: np.ndarray, x0: float, width: float) -> np.ndarray:
    y = np.zeros_like(x)
    mask = (x >= x0) & (x <= x0 + width)
    y[mask] = 1.0
    return y


def poly3(x: np.ndarray, x0: float) -> np.ndarray:
    z = x - x0
    return 0.5 + z + z * z + z * z * z
