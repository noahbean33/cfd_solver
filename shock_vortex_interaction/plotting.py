from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

Array = np.ndarray


def plot_pressure_contours(
    X: Array,
    Y: Array,
    p: Array,
    mean_pressure: Array,
    midx: int,
    out_path: Optional[str] = None,
    close: bool = True,
):
    """Plot pressure contours centered around mean pressure at mid x-index.

    If out_path is provided, ensure its directory exists and save to file.
    """
    fig, ax = plt.subplots()
    cs = ax.contourf(
        X[1:-1, 1:-1],
        Y[1:-1, 1:-1],
        p[1:-1, 1:-1],
        np.linspace(mean_pressure[1, midx] - 0.001, mean_pressure[1, midx] + 0.001, 400),
        cmap="jet",
        extend="both",
    )
    cb = fig.colorbar(cs, ax=ax, shrink=0.9)
    cb.set_label("Non-dimensional Pressure")
    ax.axis("equal")
    ax.set_aspect("equal", "box")
    plt.suptitle("Shock Vortex Interaction", fontsize=20)

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=200)
    if close:
        plt.close(fig)
    return fig, ax
