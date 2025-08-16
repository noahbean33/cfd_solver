import numpy as np
from taylor_series.core import approximate, get_function


def test_exp_approximation_converges_with_degree():
    x = np.linspace(-1.0, 1.0, 201)
    f = get_function("exp")
    y_true = f(x)
    err_prev = None
    for n in [0, 2, 4, 6, 8]:
        y_approx = approximate("exp", a=0.0, n=n, x=x)
        err = float(np.max(np.abs(y_true - y_approx)))
        if err_prev is not None:
            assert err <= err_prev + 1e-10
        err_prev = err
