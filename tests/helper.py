

def assert_equal(f1, f2, TOL=1e-10):
    for x in f1.get_domain():
        assert abs(f1(x) - f2(x)) <= TOL