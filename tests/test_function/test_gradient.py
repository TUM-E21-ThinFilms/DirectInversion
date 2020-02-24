import pytest
import numpy as np
from dinv.function import Function, Derivative



def test_derivative_linear_function():
    x_domain = np.linspace(0, 10, 100)
    f = Function(x_domain, lambda x: 5 * x)
    df = Derivative.to_function(x_domain, f)
    TOL = 1e-11
    for x in x_domain:
        assert abs(df(x) - 5) < TOL

def test_derivative_sin():
    x_domain = np.linspace(0, 10, 10000)
    f = Function(x_domain, lambda x: np.sin(x))
    df = Derivative.to_function(x_domain, f)

    TOL = 1e-6
    for x in x_domain:
        assert abs(df(x) - np.cos(x)) < TOL

def test_derivative_exp():
    x_domain = np.linspace(0, 10, 10000)
    f = Function(x_domain, lambda x: np.exp(x))
    df = Derivative.to_function(x_domain, f)

    TOL = 1e-6
    for x in x_domain:
        assert abs(df(x) - f(x))/f(x) < TOL