import numpy as np

from skipi.function import Function, UnevenlySpacedFunction
from skipi.fourier import FourierTransform, InverseFourierTransform, invfourier_matrix
from skipi.convolution import GaussianSmoothing
from skipi.util import vslice
from skipi.collection import FunctionCollection

from ..helper import assert_equal, indicator


def test_fourier_extrapolation():
    x_space = np.linspace(-50, 250, 1000)
    w_space = np.linspace(-0.5, 0.5, 20000)

    i1 = indicator(0, 100)
    i2 = indicator(100, 200)
    f = Function(x_space, lambda x: 5 * i1(x) + 10 * i2(x))
    f = GaussianSmoothing.from_function(f, sigma=4)

    F = FourierTransform.from_function(w_space, f)
    # f = InverseFourierTransform.from_function(x_space, F)

    # Assume now, F is only known on a subset of w_space. How can we extrapolate this now?
    kc = 0.1
    F_measured = F.vremesh((-kc, kc), dstop=1, dstart=-1)
    # F_measured.plot(show=True)
    # assert_equal(F, F_measured, domain=F_measured.get_domain())

    # Note here the dstop=+1 (and dstart=-1). This means that we include also the next (and prev) value in
    # w_space which is bigger than kc (smaller than -kc). This is _crucial_ for the integration. The
    # intervals have to be overlapping, otherwise the resulting numerical integration fails and the
    # extrapolation cannot be done correctly.
    w1 = vslice(w_space, (-kc, kc), dstop=+1, dstart=-1)
    w2l = vslice(w_space, (None, -kc))
    w2u = vslice(w_space, (kc, None))
    w = np.concatenate([w2l, w2u])

    c1 = Function(np.linspace(-1000, -25, 651), lambda x: 0)
    c2 = Function(np.linspace(225, 900, 301), lambda x: 0)
    constraint = FunctionCollection([c1, c2])
    # Note here, we have to use the to_function wrapper to create the function. Otherwise, numpy thinks
    # we're passing the null-function to it, which means f(domain) = 0 but what we expect is
    # f(domain) = 0, ..., 0
    # constraint = Function.to_function(constraint_space, lambda x: 0)
    constraint_space = constraint.get_domain()
    #constraint = UnevenlySpacedFunction(constraint_space, np.vectorize(lambda x: 0))

    f1_full = InverseFourierTransform.to_function(w1, F, np.linspace(-1000, 900, 2000))
    f1 = InverseFourierTransform.from_function(constraint_space, F_measured)
    # f1 = f1_full.remesh(constraint_space)
    # f1_full = InverseFourierTransform.to_function(w1, F, np.linspace(-1000, 900, 2000))
    # Note: This has to be done, since we assume in the Fourier Transform an equidistantly spaced domain.
    # Since we're stitching the upper and lower part together, it's not equidistant anymore, and we have to
    # split up the computation into two parts. Annoying...
    f2l = InverseFourierTransform.to_function(w2l, F, constraint_space)
    f2u = InverseFourierTransform.to_function(w2u, F, constraint_space)
    f2 = f2l + f2u

    actual = f1
    target = constraint

    # (f1 + f2).plot(show=True)

    finversionl = invfourier_matrix(w2l, constraint_space)
    finversionu = invfourier_matrix(w2u, constraint_space)

    finversion = np.concatenate([finversionl, finversionu], axis=1)

    b = (- actual + target)(constraint_space)

    F_approx_val, residuals, rank, s = np.linalg.lstsq(finversion, b, rcond=1e-5)

    F_approx = Function.to_function(w, F_approx_val)

    # f_exactl = InverseFourierTransform.from_function(f1_full.get_domain(), F.vremesh((None, -kc)))
    # f_exactu = InverseFourierTransform.from_function(f1_full.get_domain(), F.vremesh((kc, None)))
    # f_exact = f_exactl + f_exactu

    K = 0.11
    K0 = 0.1005
    f_approxl = InverseFourierTransform.from_function(f1_full.get_domain(), F_approx.vremesh((-K, 0)))
    f_approxu = InverseFourierTransform.from_function(f1_full.get_domain(), F_approx.vremesh((0, K)))
    f_approx = f_approxu + f_approxl

    ideal = InverseFourierTransform.from_function(f1_full.get_domain(), F.vremesh((-K, K)))

    F.remesh(w).plot()
    F_approx.plot(show=True)

    Kt = 0.13
    assert_equal((F - F_approx).vremesh((-Kt, -K0), (K0, Kt)).real, TOL=6.5)
    # f_exact.plot()
    # (f_approxl+f_approxu).plot()
    # f_approx.plot(show=True)

    f.plot(label='exact')
    ideal.plot(label='ideal')
    f1_full.plot(label='no improvement')

    (f1_full + f_approx).plot(show=True, label='improved')

    (ideal - (f1_full + f_approx)).plot(show=True)
    assert_equal(ideal, f1_full + f_approx, TOL=5e-3)
