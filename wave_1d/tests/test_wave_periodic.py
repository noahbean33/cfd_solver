import numpy as np
import pytest

from wave_1d.core import SimConfig, run, make_grid, step_ftbs, step_maccormack
from wave_1d.ic import sinusoid


def l2(a: np.ndarray) -> float:
    return float(np.linalg.norm(a) / np.sqrt(a.size))


def test_periodic_translation_ftbs():
    dx = 0.01
    dt = 0.005
    c = 1.0
    steps = 100
    x_start, x_end = 0.0, 1.0

    xs = make_grid(x_start, x_end, dx)
    u0 = sinusoid(xs, x0=0.0, wavelength=1.0)

    cfg = SimConfig(dx=dx, dt=dt, c=c, steps=steps, x_start=x_start, x_end=x_end, bc="periodic", scheme="ftbs")
    uT, xs2, _ = run(u0, cfg, keep_trajectory=False)
    assert np.allclose(xs, xs2)

    shift = c * dt * steps
    cells = int(round(shift / dx))
    u_exact = np.roll(u0, cells)
    err = l2(uT - u_exact)
    assert err < 5e-2  # first-order scheme


def test_periodic_translation_maccormack():
    dx = 0.01
    dt = 0.005
    c = 1.0
    steps = 100
    x_start, x_end = 0.0, 1.0

    xs = make_grid(x_start, x_end, dx)
    u0 = sinusoid(xs, x0=0.0, wavelength=1.0)

    cfg = SimConfig(dx=dx, dt=dt, c=c, steps=steps, x_start=x_start, x_end=x_end, bc="periodic", scheme="maccormack")
    uT, xs2, _ = run(u0, cfg, keep_trajectory=False)
    assert np.allclose(xs, xs2)

    shift = c * dt * steps
    cells = int(round(shift / dx))
    u_exact = np.roll(u0, cells)
    err = l2(uT - u_exact)
    assert err < 1e-2  # MacCormack is higher accuracy


def test_cfl_validation_ftbs():
    dx = 0.01
    dt = 0.02  # CFL=2 > 1
    c = 1.0
    steps = 1
    x_start, x_end = 0.0, 1.0
    xs = make_grid(x_start, x_end, dx)
    u0 = sinusoid(xs, x0=0.0, wavelength=1.0)

    cfg = SimConfig(dx=dx, dt=dt, c=c, steps=steps, x_start=x_start, x_end=x_end, bc="periodic", scheme="ftbs")
    with pytest.raises(ValueError):
        run(u0, cfg)
