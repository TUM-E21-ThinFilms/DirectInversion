import numpy as np
from dinv.function import Function


def test_shift_without_domain_shift():
    f = Function(np.linspace(0, 10, 100), lambda x: 5 * x)
    shift = 2
    fshifted = Function(f.get_domain(), lambda x: 5 * (x - shift))

    f.shift(shift)

    for x in fshifted.get_domain():
        assert abs(fshifted(x) - f(x)) == 0


def test_shift_with_domain_shift():
    f = Function(np.linspace(0, 10, 100), lambda x: 5 * x)
    shift = 10
    fshifted = Function(np.linspace(10, 20, 100), lambda x: 5 * (x - shift))

    f.shift(shift, domain=True)

    for x in fshifted.get_domain():
        assert abs(fshifted(x) - f(x)) == 0
