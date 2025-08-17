from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

Array = np.ndarray


def build_grid(numX: int) -> Tuple[Array, float]:
    dx = 1.0 / (numX - 1)
    x = np.linspace(0.0, 1.0, numX)
    return x, dx


def explicit_step(y: Array, C: float) -> Array:
    y_new = y.copy()
    y_old = y
    y_new[1:-1] = y_old[1:-1] + C * (y_old[2:] - 2.0 * y_old[1:-1] + y_old[:-2])
    return y_new


def _thomas_tridiagonal(a: Array, b: Array, c: Array, d: Array) -> Array:
    """Thomas algorithm for tridiagonal Ax=d.
    a: sub-diagonal (len n-1), b: diagonal (len n), c: super-diagonal (len n-1), d: RHS (len n)
    Returns x (len n).
    """
    n = len(b)
    ac, bc, cc, dc = map(np.array, (a.copy(), b.copy(), c.copy(), d.copy()))
    for i in range(1, n):
        mc = ac[i - 1] / bc[i - 1]
        bc[i] = bc[i] - mc * cc[i - 1]
        dc[i] = dc[i] - mc * dc[i - 1]
    x = np.zeros(n, dtype=dc.dtype)
    x[-1] = dc[-1] / bc[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
    return x


def implicit_step_thomas(y: Array, C: float) -> Array:
    n = y.size
    # Build tridiagonal for interior points only
    m = n - 2
    a = -C * np.ones(m - 1)
    b = (1.0 + 2.0 * C) * np.ones(m)
    c = -C * np.ones(m - 1)

    rhs = y[1:-1].copy()
    rhs[0] += C * y[0]
    rhs[-1] += C * y[-1]

    sol = _thomas_tridiagonal(a, b, c, rhs)
    y_new = y.copy()
    y_new[1:-1] = sol
    return y_new


def solve(
    numX: int,
    alpha: float,
    t_max: float,
    dt: float,
    temp1: float,
    temp2: float,
    scheme: str,
    checkpoints: Sequence[float] | None = None,
) -> Dict[str, object]:
    x, dx = build_grid(numX)
    C = alpha * dt / (dx * dx)
    y = np.full_like(x, fill_value=temp1, dtype=float)
    y[-1] = temp2

    T = 0.0
    snapshots: List[Array] = []
    labels: List[str] = []

    if checkpoints is None:
        checkpoints = [1, 4, 10, 20, 100]
    steps_total = int(np.rint(t_max / dt))
    pause_steps = np.rint(np.array(checkpoints) * 0.01 * steps_total).astype(int).tolist()

    step = 0
    while T < t_max - 1e-12:
        if scheme == "explicit":
            y = explicit_step(y, C)
        else:
            y = implicit_step_thomas(y, C)
        T += dt
        step += 1
        if step in pause_steps:
            snapshots.append(y.copy())
            labels.append(f"{checkpoints[pause_steps.index(step)]}% of tMax")

    return {
        "x": x,
        "dx": dx,
        "y": y,
        "snapshots": snapshots,
        "labels": labels,
        "C": C,
    }
