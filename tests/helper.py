from numpy import heaviside


def assert_equal(f1, f2, TOL=1e-10, domain=None):
    if domain is None:
        domain = f1.get_domain()

    for x in domain:
        assert abs(f1(x) - f2(x)) <= TOL


def indicator(x_min, x_max):
    return lambda x: heaviside(x - x_min, 0) * heaviside(-x + x_max, 0)
