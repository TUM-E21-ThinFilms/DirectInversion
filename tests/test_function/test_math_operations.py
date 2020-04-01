import numpy as np
from dinv.function import Function

from ..helper import assert_equal


def test_complex_conjugate():
    f1 = Function(np.linspace(0, 10, 100), lambda x: np.exp(1j*x))
    f2 = Function(np.linspace(0, 10, 100), lambda x: np.exp(-1j*x))

    assert_equal(f1.conj(), f2)

def test_absolute_value():
    f1 = Function(np.linspace(0, 10, 100), lambda x: np.exp(1j * x))
    f2 = Function(np.linspace(0, 10, 100), lambda x: 1)

    f3 = Function(f1.get_domain(), lambda x: -x)
    f4 = Function(f1.get_domain(), lambda x: x)

    assert_equal(f1.abs(), f2)
    assert_equal(f3.abs(), f4)