import numpy as np

from dinv.function import Function, FourierTransform, InverseFourierTransform, GaussianSmoothing


from ..helper import assert_equal

def test_splitting():

    x_space = np.linspace(0, 10, 100)
    w_space = np.linspace(-5, 5, 1000)

    stop = int(len(x_space) / 2)
    ind1 = slice(0, stop)
    ind2 = slice(stop-1, len(x_space))

    f = Function(x_space, lambda x: x)
    F1 = FourierTransform.to_function(x_space[ind1], f, w_space)
    F2 = FourierTransform.to_function(x_space[ind2], f, w_space)
    F3 = F1 + F2
    F3p = FourierTransform.from_function(w_space, f)

    assert_equal(F3, F3p)

def test_splitting_inverse_fourier():
    x_space = np.linspace(-20, 120, 1000)
    w_space = np.linspace(-5, 5, 1000)

    stop = int(len(w_space) / 2)
    ind1 = slice(0, stop)
    ind2 = slice(stop - 1, len(w_space))

    f = Function(x_space, lambda x: np.heaviside(x, 0) * x * np.heaviside(-x+100, 0))
    f = GaussianSmoothing.from_function(f, sigma=4)

    F = FourierTransform.from_function(w_space, f)

    F1 = F.remesh(w_space[ind1])
    F2 = F.remesh(w_space[ind2])

    f1 = InverseFourierTransform.from_function(x_space, F1)
    f2 = InverseFourierTransform.from_function(x_space, F2)

    f3 = InverseFourierTransform.to_function(w_space, F, x_space)

    assert_equal(f1 + f2, f3)
    assert_equal(f3, f, TOL=1e-3)
