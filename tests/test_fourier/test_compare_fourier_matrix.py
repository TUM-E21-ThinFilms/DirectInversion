import numpy as np

from dinv.function import Function, FourierTransform, fourier_matrix, InverseFourierTransform, \
    invfourier_matrix

from ..helper import assert_equal


def test_compare_matrix_with_fourier_function():
    x_space = np.linspace(-10, 10, 1000)
    fun = Function(x_space, lambda x: np.exp(-x ** 2 / 2))

    w_space = np.linspace(-10, 10, 1000)
    f1 = FourierTransform.from_function(w_space, fun)

    trafo = fourier_matrix(x_space, w_space)
    f2 = Function.to_function(w_space, np.dot(trafo, fun(x_space)))

    assert_equal(f1, f2, 1e-14)


def test_compare_matrix_with_invfourier_function():
    x_space = np.linspace(-10, 10, 1000)
    fun = Function(x_space, lambda x: np.exp(-x ** 2 / 2))

    w_space = np.linspace(-10, 10, 1000)
    f1 = InverseFourierTransform.from_function(w_space, fun)

    trafo = invfourier_matrix(x_space, w_space)
    f2 = Function.to_function(w_space, np.dot(trafo, fun(x_space)))

    assert_equal(f1, f2, 1e-15)

def test_compare_matrix_InverseFourier():
    x_space = np.linspace(-10, 10, 1000)
    w_space = np.linspace(-10, 10, 1000)
    fun = Function(x_space, lambda x: np.exp(-x ** 2 / 2))

    trafo = fourier_matrix(x_space, w_space)
    f = Function.to_function(w_space, np.dot(trafo, fun(x_space)))

    fun2 = InverseFourierTransform.from_function(x_space, f)

    assert_equal(fun, fun2, 1e-15)


def test_compare_InverseFourier_matrix():
    x_space = np.linspace(-10, 10, 1000)
    w_space = np.linspace(-10, 10, 1000)
    fun = Function(x_space, lambda x: np.exp(-x ** 2 / 2))

    f = FourierTransform.from_function(w_space, fun)

    trafo = invfourier_matrix(x_space, w_space)
    fun2 = Function.to_function(w_space, np.dot(trafo, f(x_space)))
    
    assert_equal(fun, fun2, 1e-15)
