"""
Plotting utilities for Taylor series approximations.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from matplotlib import pyplot as plt


def plot_comparison(
    x: np.ndarray,
    y_true: np.ndarray,
    y_approx: np.ndarray,
    title: str = "Taylor Series Approximation",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot exact vs approximation. If save_path is provided, save instead of blocking.
    """
    fig, ax = plt.subplots()
    ax.plot(x, y_true, "k-", label="Exact", linewidth=2.5)
    ax.plot(x, y_approx, "--", label="Approx", linewidth=2.0)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
    else:
        plt.show(block=True)
