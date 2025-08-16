import numpy as np
from difference_forms.core import fd_coefficients


def test_centered_first_derivative_o2():
    offsets, coeffs = fd_coefficients(derivative_order=1, scheme="centered", accuracy_order=2, h=1.0)
    # Expect offsets [-1,0,1] and coeffs [-1/2, 0, 1/2]
    assert np.array_equal(offsets, np.array([-1, 0, 1]))
    assert np.allclose(coeffs, np.array([-0.5, 0.0, 0.5]))


def test_centered_second_derivative_o2():
    offsets, coeffs = fd_coefficients(derivative_order=2, scheme="centered", accuracy_order=2, h=1.0)
    # Expect offsets [-1,0,1] and coeffs [1, -2, 1]
    assert np.array_equal(offsets, np.array([-1, 0, 1]))
    assert np.allclose(coeffs, np.array([1.0, -2.0, 1.0]))


def test_forward_first_derivative_o1():
    offsets, coeffs = fd_coefficients(derivative_order=1, scheme="forward", accuracy_order=1, h=1.0)
    # Expect offsets [0,1] and coeffs [-1, 1]
    assert np.array_equal(offsets, np.array([0, 1]))
    assert np.allclose(coeffs, np.array([-1.0, 1.0]))
