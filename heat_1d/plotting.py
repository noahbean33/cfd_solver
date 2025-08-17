from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt

Array = np.ndarray


def plot_profiles(
    x: Array,
    snapshots: List[Array],
    labels: List[str],
    temp1: float,
    temp2: float,
    out_path: Optional[str] = None,
    show: bool = True,
):
    fig, ax = plt.subplots()
    ax.set_title("Temperature Distribution across Time", fontsize=18)
    ax.set_xlim(0, 1)
    ax.set_ylim(temp1, temp2)
    ax.set_xlabel("X (m)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.grid(True)

    for y, label in zip(snapshots, labels):
        ax.plot(x, y, label=label, linewidth=2)
    if labels:
        ax.legend(prop={"size": 10})

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
