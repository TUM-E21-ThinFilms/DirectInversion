import numpy as np

from dinv.function import Function, FourierTransform, InverseFourierTransform, GaussianSmoothing
from ..helper import assert_equal, indicator


def test_linearity():
    x_space = np.linspace(0, 10, 10)
    w_space = np.linspace(-5, 5, 1000)

    f1 = Function(x_space, lambda x: 3 * x)
    f2 = Function(x_space, lambda x: -2 * x)
    f3 = Function(x_space, lambda x: x)

    F1 = FourierTransform.from_function(w_space, f1)
    F2 = FourierTransform.from_function(w_space, f2)
    F3 = FourierTransform.from_function(w_space, f3)
    F3p = FourierTransform.from_function(w_space, (f1 + f2))

    assert_equal(F1 + F2, F3)
    assert_equal(F3, F3p)


def test_linearity_in_domain_for_real_part():
    x_space = np.linspace(-100, 300, 1000)
    w_space = np.linspace(0, 5, 1000)

    i1 = indicator(0, 200)
    f = Function(x_space, lambda x: i1(x) * 5)
    f = GaussianSmoothing.from_function(f, sigma=10)

    F = FourierTransform.from_function(w_space, f)
    fp = InverseFourierTransform.from_function(x_space, F).real * 2

    assert_equal(fp, f, TOL=1e-7)

    stop = 200  # i.e. np.linspace(0, 0.2, ...)

    ind1 = slice(0, stop + 1)
    ind2 = slice(stop, len(w_space))

    f1 = InverseFourierTransform.to_function(w_space[ind1], F, x_space).real * 2
    f2 = InverseFourierTransform.to_function(w_space[ind2], F, x_space).real * 2
    f3 = InverseFourierTransform.to_function(w_space, F, x_space).real * 2

    assert_equal(f1 + f2, f3)


def test_linearity_in_domain():
    x_space = np.linspace(-100, 300, 1000)
    w_space = np.linspace(-5, 5, 1000)

    i1 = indicator(0, 200)
    f = Function(x_space, lambda x: i1(x) * 5)
    f = GaussianSmoothing.from_function(f, sigma=10)

    F = FourierTransform.from_function(w_space, f)
    fp = InverseFourierTransform.from_function(x_space, F)

    assert_equal(fp, f, TOL=1e-7)

    stop = 100  # i.e. np.linspace(0, 0.2, ...)

    ind1 = slice(0, stop + 1)
    ind2 = slice(stop, len(w_space))

    f1 = InverseFourierTransform.to_function(w_space[ind1], F, x_space)
    f2 = InverseFourierTransform.to_function(w_space[ind2], F, x_space)
    f3 = InverseFourierTransform.to_function(w_space, F, x_space)

    assert_equal(f1 + f2, f3)