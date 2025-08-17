from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .metrics import vorticity


def plot_vorticity(X: np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray, levels: int | None = 20):
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    vort = vorticity(u, v, dx, dy)
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, vort, levels=levels)
    plt.colorbar(cs, ax=ax)
    ax.set_title("Isentropic Vortex: Vorticity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax
