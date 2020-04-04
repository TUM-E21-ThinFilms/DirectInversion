import numpy as np
from dinv.function import Function, Derivative
from ..helper import assert_equal


def test_derivative_linear_function():
    x_domain = np.linspace(0, 10, 100)
    f = Function(x_domain, lambda x: 5 * x)
    df = Derivative.to_function(x_domain, f)
    assert_equal(df, Function(x_domain, lambda x: 5))


def test_derivative_sin():
    x_domain = np.linspace(0, 1, 10000)
    f = Function(x_domain, lambda x: np.sin(x))
    df = Derivative.to_function(x_domain, f)

    assert_equal(df, Function(x_domain, lambda x: np.cos(x)), TOL=1e-8)


def test_derivative_exp():
    x_domain = np.linspace(0, 1, 10000)
    f = Function(x_domain, lambda x: np.exp(x))
    df = Derivative.to_function(x_domain, f)

    assert_equal(df, Function(x_domain, lambda x: np.exp(x)), TOL=1e-8)
