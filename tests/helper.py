from numpy import heaviside


def assert_equal(f1, f2=None, TOL=1e-10, domain=None, do_print=False):
    if domain is None:
        domain = f1.get_domain()

    if f2 is None:
        f2 = lambda x: 0

    for x in domain:
        cond = abs(f1(x) - f2(x))
        if do_print:
            print(cond)
        assert cond <= TOL

def assert_equal_relative(f1, f2, TOL=1e-10, domain=None):
    if domain is None:
        domain = f1.get_domain()

    for x in domain:
        _f1 = f1(x)
        _f2 = f2(x)

        if abs(_f1) <= TOL:
            pass
        else:
            print(abs(_f2/_f1 - 1))
            assert abs(_f2/_f1 - 1) <= TOL

def indicator(x_min, x_max):
    return lambda x: heaviside(x - x_min, 0) * heaviside(-x + x_max, 0)
