import numpy as np

from dinv.function import Function, FourierTransform, fourier_matrix, InverseFourierTransform, \
    invfourier_matrix

from ..helper import assert_equal


def test_identity_Fourier():
    x_space = np.linspace(-10, 10, 1000)
    fun = Function(x_space, lambda x: np.exp(-5*x ** 2 / 2))
    w_space = np.linspace(-50, 50, 1000)

    f1 = FourierTransform.from_function(w_space, fun)
    fun2 = InverseFourierTransform.from_function(x_space, f1)

    assert_equal(fun2, fun, 1e-14)


def test_identity_matrix():
    x_space = np.linspace(-10, 10, 1000)
    w_space = np.linspace(-10, 10, 1000)

    fun = Function(x_space, lambda x: np.exp(-x ** 2 / 2))

    trafo = fourier_matrix(x_space, w_space)
    f = Function.to_function(w_space, np.dot(trafo, fun(x_space)))
    trafo = invfourier_matrix(w_space, x_space)
    fun2 = Function.to_function(x_space, np.dot(trafo, f(w_space)))

    assert_equal(fun, fun2)
