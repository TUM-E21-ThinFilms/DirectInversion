import numpy as np

from dinv.function import Function, FourierTransform
from ..helper import assert_equal


def rect(x):
    if abs(x) > 0.5:
        return 0
    if abs(x) == 0.5:
        return 0.5
    if abs(x) < 0.5:
        return 1


def sinc(x):
    if x == 0:
        return 1.0

    return np.sin(np.pi * x) / (np.pi * x)


def test_gauss():
    domain = np.linspace(-1, 1, 10000)
    f = Function.to_function(domain, lambda x: rect(1 / (2 * np.pi) * x))

    freq_dom = np.linspace(-20, 20, 1000)

    F = FourierTransform.from_function(freq_dom, f)

    fourier_trafo = Function(domain, lambda x: 2 * sinc(x / np.pi))

    assert_equal(F, fourier_trafo, TOL=1e-6)


def test_exp():
    domain = np.linspace(-5, 5, 100)
    w_space = np.linspace(-10, 10, 1000)
    a = 2

    f = Function.to_function(domain, lambda x: np.exp(-a * x ** 2))
    F_analytical = Function.to_function(w_space, lambda x: np.sqrt(np.pi / a) * np.exp(-x ** 2 / (4 * a)))
    F = FourierTransform.from_function(w_space, f)

    assert_equal(F, F_analytical)


def test_exp_random_domain():
    np.random.seed(10)
    domain = np.unique(10 * np.random.random(10000) - 5)
    w_space = np.linspace(-10, 10, 1000)
    a = 2
    f = Function.to_function(domain, lambda x: np.exp(-a * x ** 2))
    F_analytical = Function.to_function(w_space, lambda x: np.sqrt(np.pi / a) * np.exp(-x ** 2 / (4 * a)))
    F = FourierTransform.from_function(w_space, f)

    assert_equal(F, F_analytical, TOL=7e-6)
