"""
Periodic boundary utilities.
"""
from __future__ import annotations

import numpy as np


def apply_periodic(Q: np.ndarray) -> None:
    """Apply 2D periodic BCs in-place to state Q with shape (..., ny, nx)."""
    if Q.ndim == 3:
        _, ny, nx = Q.shape
        Q[:, 0, :] = Q[:, -2, :]
        Q[:, -1, :] = Q[:, 1, :]
        Q[:, :, 0] = Q[:, :, -2]
        Q[:, :, -1] = Q[:, :, 1]
    elif Q.ndim == 2:
        ny, nx = Q.shape
        Q[0, :] = Q[-2, :]
        Q[-1, :] = Q[1, :]
        Q[:, 0] = Q[:, -2]
        Q[:, -1] = Q[:, 1]
    else:
        raise ValueError("Q must be 2D or 3D array")
