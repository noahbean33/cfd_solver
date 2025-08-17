"""
Plotting utilities for potential_flow.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_potential_surface(X: np.ndarray, Y: np.ndarray, phi: np.ndarray):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cs = ax.plot_surface(X, Y, -phi, cmap='RdYlGn_r')
    cb = fig.colorbar(cs, ax=ax, shrink=0.9, location="right")
    cb.set_label('Negative Scalar Potential', fontsize=12)
    return fig, ax


def plot_streamlines_cp(X: np.ndarray, Y: np.ndarray, U: np.ndarray, V: np.ndarray, Cp: np.ndarray, obstacles: list[tuple[int,int,int,int]] | None = None, Lx: float | None = None, Ly: float | None = None, nx: int | None = None, ny: int | None = None):
    fig, ax = plt.subplots()
    ax.streamplot(X[1:-2,1:-2], Y[1:-2,1:-2], U[1:-2,1:-2], V[1:-2,1:-2], density=2.0, color='k', linewidth=0.5, arrowstyle='-', broken_streamlines=False)
    cs = ax.contourf(X[1:-1,1:-1], Y[1:-1,1:-1], Cp, np.linspace(-1, 1, 41), cmap='RdYlGn_r', extend='both')
    cb = fig.colorbar(cs, ax=ax, shrink=0.9, location="bottom")
    cb.set_label('Pressure Coefficient', fontsize=12)
    if obstacles and Lx is not None and Ly is not None and nx is not None and ny is not None:
        for (imin, imax, jmin, jmax) in obstacles:
            ax.add_patch(Rectangle((Lx*(jmin)/(nx-1), Ly*(imin)/(ny-1)), Lx*(jmax-jmin)/(nx-1), Ly*(imax-imin)/(ny-1), zorder=10, edgecolor='k', facecolor='gray', fill=True, lw=1))
    ax.set_aspect('equal', adjustable='box')
    return fig, ax


def plot_convergence(err: np.ndarray):
    fig, ax = plt.subplots()
    ax.plot(err)
    ax.set_yscale('log')
    ax.set_ylabel('Norm of Update Delta', fontsize=12)
    ax.set_xlabel('Iteration Count', fontsize=12)
    ax.set_title('Convergence Trend', fontsize=14)
    return fig, ax
