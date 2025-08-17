from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt

Array = np.ndarray


def plot_line(x: Array, y: Array, ylabel: str, title: str, out_path: Optional[str] = None, show: bool = True):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.grid(True)
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax


essential_profiles = ("rho", "u", "p")


def plot_profiles(x: Array, profiles: Dict[str, List[Array]], steps: List[int], outdir: Optional[str] = None, show: bool = True):
    # Plot each quantity in separate axes for the provided steps
    out_paths = []
    for name in essential_profiles:
        if name not in profiles:
            continue
        fig, ax = plt.subplots()
        for arr, step in zip(profiles[name], steps):
            ax.plot(x, arr, label=f"step {step}")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel(name)
        ax.grid(True)
        ax.legend()
        if outdir:
            os.makedirs(outdir, exist_ok=True)
            path = os.path.join(outdir, f"{name}_profiles.png")
            fig.savefig(path, dpi=150)
            out_paths.append(path)
        if show:
            plt.show()
        else:
            plt.close(fig)
    return out_paths
