import numpy as np
from dinv.function import Function

from ..helper import assert_equal



def test_addition():
    f1 = Function(np.linspace(0, 10, 100), lambda x: np.sin(x) ** 2)
    f2 = Function(np.linspace(0, 20, 100), lambda x: np.cos(x) ** 2)
    f3 = Function(np.linspace(0, 10, 100), lambda x: 1)

    f = f1 + f2

    assert np.all(f.get_domain() == f1.get_domain())

    assert_equal(f, f3)


def test_addition_with_constant():
    f1 = Function(np.linspace(0, 10, 100), lambda x: np.sin(x))
    f2 = Function(np.linspace(0, 10, 100), lambda x: np.sin(x) + 5)

    f = f1 + 5

    assert_equal(f, f2)


def test_subtraction():
    f1 = Function(np.linspace(0, 10, 100), lambda x: 5 * x + 1)
    f2 = Function(np.linspace(0, 20, 100), lambda x: 4 * x)
    f3 = Function(np.linspace(0, 10, 100), lambda x: x + 1)

    f = f1 - f2

    assert np.all(f.get_domain() == f1.get_domain())
    assert_equal(f, f3)


def test_subtraction_with_constant():
    f1 = Function(np.linspace(0, 10, 100), lambda x: np.sin(x))
    f2 = Function(np.linspace(0, 10, 100), lambda x: np.sin(x) - 5)

    assert_equal(f1 - 5, f2)


def test_multiplication():
    f1 = Function(np.linspace(0, 10, 100), lambda x: np.sin(x))
    f2 = Function(np.linspace(0, 10, 100), lambda x: np.sin(x) ** 2)

    assert_equal(f1 * f1, f2)


def test_multiplication_with_constant():
    f1 = Function(np.linspace(0, 10, 100), lambda x: np.cos(x))
    f2 = Function(np.linspace(0, 10, 100), lambda x: 4 * np.cos(x))

    assert_equal(f1 * 4, f2)


def test_division():
    f1 = Function(np.linspace(1, 11, 100), lambda x: 4 * x ** 2)
    f2 = Function(np.linspace(1, 11, 100), lambda x: 2 * x)
    f3 = Function(np.linspace(1, 11, 100), lambda x: 2 * x)

    assert_equal(f1 / f2, f3)


def test_division_with_constant():
    f1 = Function(np.linspace(1, 11, 100), lambda x: 4 * x ** 2)
    f3 = Function(np.linspace(1, 11, 100), lambda x: x ** 2)

    assert_equal(f1 / 4, f3)
