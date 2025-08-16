# Script to compute the numerical solution of the 1D (uni-directional) wave equation

import numpy as np
import matplotlib.pyplot as plt

from wave_1d.core import SimConfig, run, make_grid
from wave_1d.ic import step as step_ic, multi_sin, sinusoid, gaussian, poly3
from wave_1d.plotting import plot_snapshot


def analytical_function(x: np.ndarray, x_ref: float, wavelength: float, kind: str) -> np.ndarray:
    if kind == 'exponential':
        # Map to gaussian IC
        return gaussian(x, x_ref, width=np.sqrt(1/200))
    elif kind == 'single sinusoid':
        return sinusoid(x, x_ref, wavelength)
    elif kind == 'multiple sinusoids':
        return multi_sin(x, x_ref, wavelength)
    elif kind == 'step function':
        return step_ic(x, x_ref, width=wavelength)
    elif kind == 'polynomial':
        return poly3(x, x_ref)
    else:
        raise ValueError('Function type not recognized')


def main():
    # Parameters (kept similar to original defaults)
    dx = 0.004
    dt = 0.002
    numSteps = 101
    c = 1.0
    waveLength = 0.4
    x_start, x_end = -0.5, 1.5

    saveFile = 0
    pauseCount = 10 if saveFile == 1 else max(numSteps - 1, 1)

    x = make_grid(x_start, x_end, dx)
    x_ref_init = 0.0
    functionType = 'step function'
    y_init = analytical_function(x, x_ref_init, waveLength, functionType)

    # Build config and run both schemes
    cfg_mc = SimConfig(dx=dx, dt=dt, c=c, steps=numSteps, x_start=x_start, x_end=x_end, bc='periodic', scheme='maccormack')
    cfg_ft = SimConfig(dx=dx, dt=dt, c=c, steps=numSteps, x_start=x_start, x_end=x_end, bc='periodic', scheme='ftbs')

    y_MacCormack, xs, _ = run(y_init, cfg_mc, keep_trajectory=False)
    y_FTBS, _, _ = run(y_init, cfg_ft, keep_trajectory=False)
    y_analytical = analytical_function(x, x_ref_init + c * numSteps * dt, waveLength, functionType)

    # Plot a single snapshot like before
    ylim = (-1.2, 1.2) if functionType in ('single sinusoid', 'multiple sinusoids') else (-0.2, 1.2)
    fig, ax = plot_snapshot(xs, {"MacCormack": y_MacCormack, "FTBS": y_FTBS, "Analytical": y_analytical}, ylim=ylim, title="Wave Equation")
    plt.show()


if __name__ == '__main__':
    main()
