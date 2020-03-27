import numpy as np

from dinv.function import Function, FourierTransform



def test_linearity():

    x_space = np.linspace(0, 10, 10)
    w_space = np.linspace(-5, 5, 1000)

    f1 = Function(x_space, lambda x: 3*x)
    f2 = Function(x_space, lambda x: -2*x)
    f3 = Function(x_space, lambda x: x)

    F1 = FourierTransform.from_function(w_space, f1)
    F2 = FourierTransform.from_function(w_space, f2)
    F3 = FourierTransform.from_function(w_space, f3)
    F3p = FourierTransform.from_function(w_space, (f1 + f2))


    TOL = 1e-5
    for w in w_space:
        assert abs(F1(w) + F2(w) - F3(w)) <= TOL
        assert abs(F3(w) - F3p(w)) <= TOL