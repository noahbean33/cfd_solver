"""
Plotting utilities for lid-driven cavity results.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_streamlines_and_vorticity(X: np.ndarray, Y: np.ndarray, U: np.ndarray, V: np.ndarray, omega: np.ndarray):
    fig, ax = plt.subplots()
    ax.streamplot(X, Y, U, V, density=2.0, color='k', linewidth=0.6, arrowstyle='-')
    cs = ax.contourf(X, Y, omega, levels=41, cmap='RdYlGn', extend='both')
    cb = fig.colorbar(cs, ax=ax, shrink=0.9, location="right")
    cb.set_label('Vorticity', fontsize=12)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Lid-Driven Cavity: Streamlines and Vorticity')
    return fig, ax


def plot_convergence(err_psi: np.ndarray, err_omega: np.ndarray):
    fig, ax = plt.subplots()
    ax.semilogy(err_psi, label='||Δψ||')
    ax.semilogy(err_omega, label='||Δω||')
    ax.set_xlabel('Step')
    ax.set_ylabel('Residual (L2)')
    ax.legend()
    ax.set_title('Convergence History')
    return fig, ax
