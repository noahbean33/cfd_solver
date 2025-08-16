"""
Plotting utilities for wave_1d.
"""
from __future__ import annotations

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def plot_snapshot(xs: np.ndarray, curves: Dict[str, np.ndarray], ylim=None, title: str = "", save_path: str | None = None):
    fig, ax = plt.subplots()
    for label, y in curves.items():
        ax.plot(xs, y, label=label)
    ax.legend()
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlim(xs.min(), xs.max())
    if title:
        ax.set_title(title)
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        return fig, ax
