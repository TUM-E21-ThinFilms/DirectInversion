import numpy

from dinv.function import Function, FourierTransform


def rect(x):
    if abs(x) > 0.5:
        return 0
    if abs(x) == 0.5:
        return 0.5
    if abs(x) < 0.5:
        return 1


def sinc(x):
    if x == 0:
        return 1.0

    return numpy.sin(numpy.pi * x) / (numpy.pi * x)


def test_gauss():
    domain = numpy.linspace(-1, 1, 1000)
    f = Function.to_function(domain, lambda x: rect(1 / (2 * numpy.pi) * x))

    freq_dom = numpy.linspace(-20, 20, 1000)

    F = FourierTransform.from_function(freq_dom, f)

    fourier_trafo = lambda x: 2 * sinc(x / numpy.pi)

    """
    import pylab
    pylab.plot(freq_dom, F(freq_dom))
    pylab.plot(freq_dom, [fourier_trafo(w) for w in freq_dom])
    pylab.show()
    """

    TOL = 1e-2

    for w in freq_dom:
        assert abs(fourier_trafo(w) - F(w)) <= TOL
