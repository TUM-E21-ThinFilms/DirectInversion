import numpy
import scipy

from numpy import array

from dinv.glm import ReflectionCalculation
from dinv.fourier import smooth, GeneralFourierTransform
from dinv.helper import shift_potential

"""
import numpy as np
import pylab


sup = (-5, 5)
a = AutoCorrelation(smooth(df, sup, 1e-3, sigma=0.05), (-5, 5), 1e-4)
b = AutoCorrelation(smooth(df2, sup, 1e-3, sigma=0.05), (-5, 5), 1e-4)
# b = AutoCorrelation(smooth(f2, sup, 1e-2, sigma=0.1), (-10, 10), 1e-4)
#
#a.plot_f()
b.plot_f()


trange = numpy.linspace(-10, 10, 10000)
# pylab.plot(trange, [a.calculate(t) for t in trange])
# pylab.plot(trange, [b.calculate(t) for t in trange])
# space, autocor = b.calc()

# pylab.plot(space, autocor)

a.plot_correlation(trange)
b.plot_correlation(trange)
# b.plot_correlation(trange)


# pylab.plot(trange, smooth(f, sup, 1e-2, sigma=0.05)(trange))
# pylab.plot(trange, smooth(f2, sup, 1e-2, sigma=0.05)(trange))

pylab.show()
"""


import pylab
def f(x):
    if x < 0:
        return 0.0

    if x < 40:
        return 4.662

    if x < 70:
        return 6.554

    if x < 135:
        return 8.024

    if x < 175:
        return 6.554

    return 0.0

support = numpy.linspace(-300, 500, 5000)
V = smooth(f, (-20, 200), 1e-2, sigma=3)
F = GeneralFourierTransform(support, V)
w_space = numpy.linspace(-5, 5, 5000)

import cmath, math

numpy.random.seed(2)

def sgn(x):
    #return cmath.exp(1j * 2.0/x*(math.sin(3.0*x)**2)**2)
    return cmath.exp(1j * (2.0/x*(math.sin(4.0*x)**2)**2 + x + x**3))

F2 = GeneralFourierTransform(w_space, scipy.interpolate.interp1d(w_space, [sgn(w)*F.fourier(w) for w in w_space]))
V2 = scipy.interpolate.interp1d(support, [F2.fourier_inverse(x).real for x in support], bounds_error=False, fill_value=0)

pylab.plot(support, V(support))
pylab.plot(support, V2(support))
#pylab.plot(support, V2(support).imag)
pylab.show()


F3 = GeneralFourierTransform(support, V2)
pylab.plot(w_space[len(w_space)/2:], abs(array([1/w*F.fourier(w) for w in w_space if w > 0]))**2)
pylab.plot(w_space[len(w_space)/2:], abs(array([1/w*F3.fourier(w) for w in w_space if w > 0]))**2)
pylab.yscale('log')
#pylab.ylim(-1, 5)
pylab.show()


reflcalc = ReflectionCalculation(None, -250, 300, 0.05)

def scale(f):
    return lambda x: f(x) * 1e-6

reflcalc.set_potential(scale(V))

reflcalc.plot_refl(numpy.linspace(0, 2, 1000))
reflcalc.set_potential(scale(V2))
#reflcalc.plot_potential()
#pylab.show()
reflcalc.plot_refl(numpy.linspace(0, 2, 1000))
pylab.show()

q_space = numpy.linspace(0, 2, 1000)

numpy.savetxt('initial.dat', zip(support, scale(V)(support)))
numpy.savetxt('equivalent.dat', zip(support, scale(V2)(support)))
reflcalc.set_potential(scale(V))
numpy.savetxt('reflectivtiy_initial.dat', zip(q_space, reflcalc.reflectivity(q_space)))
reflcalc.set_potential(scale(V2))
numpy.savetxt('reflectivtiy_equivalent.dat', zip(q_space, reflcalc.reflectivity(q_space)))


