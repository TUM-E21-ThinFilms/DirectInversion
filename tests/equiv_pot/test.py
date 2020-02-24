import cmath
import math
import numpy
import scipy
import pylab

from numpy import array

from dinv.glm import ReflectionCalculation
from dinv.fourier import smooth, GeneralFourierTransform
from dinv.helper import shift_potential
from dinv.ba import BornApproximationReflectionCalculation

print("Starting")

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
numpy.random.seed(2)

def scale(f):
    return lambda x: f(x) * 1e-6


def to_str(fl):
    fl = round(fl, 4)
    return str(fl).replace('.', 'd')

def f(x):
    if x < 0:
        return 0.0

    if x < 40:
        return 4.662

    if x < 70:
        return 6.554

    if x < 135:
        return 8.024

    if x < 200:
        return 6.554

    return 0.0


support = numpy.arange(-100, 300, 1)
V = smooth(f, (support[0], support[-1]), 1e-2, sigma=3)
F = GeneralFourierTransform(support, V)
#w_space = numpy.hstack((numpy.linspace(-1, -1e-8, 1000), numpy.linspace(1e-8, 1, 1000)))
w_space = numpy.linspace(-1, 1, 1000)
q_space = numpy.linspace(0, 1, 500)
a = 2.0



def sgn(x):
    #f = a / x * (math.sin(3.0 * x) ** 2) ** 2


    f = 10*math.sin(2*x)

    return cmath.exp(1j * f)
    # return cmath.exp(1j * 50*math.sqrt(abs(x))*x)
    # return cmath.exp(-1j * x)

    # return cmath.exp(1j * 20.0/x*(math.sin(3.0*x)**2)**2)
    # return cmath.exp(-1j * a*(1/x*(math.sin(4.0*x)**2)**2 + x + x**3))
    # return cmath.exp(1j * math.pi / 4)
    # return cmath.exp(1j*x*10)  # shifts potential 10 points to the left
    # return -1   # mirrors the potential at the y-axis
    # return 1j   # switches real and imaginary part of the potential
    # return cmath.exp(-1j*x**3*50)


interpolation = scipy.interpolate.interp1d(w_space, [sgn(w) * F.fourier(w) for w in w_space])
F2 = GeneralFourierTransform(w_space, interpolation)
V2 = scipy.interpolate.interp1d(support, [F2.fourier_inverse(x).real for x in support], bounds_error=False,
                                fill_value=0)
V2imag = scipy.interpolate.interp1d(support, [F2.fourier_inverse(x).imag for x in support], bounds_error=False,
                                    fill_value=0)

pylab.subplot(211)
pylab.plot(support, V(support))
pylab.plot(support, V2(support))
pylab.plot(support, V2imag(support))
#pylab.show()


"""
F.plot(w_space)
pylab.plot(w_space, [interpolation(w).real for w in w_space])
pylab.plot(w_space, [interpolation(w).imag for w in w_space])
pylab.legend()
pylab.show()

pylab.plot(support, V(support))
pylab.plot(support, V2(support))
pylab.plot(support, V2imag(support))
#pylab.plot(support, V2(support).imag)
pylab.show()
"""

# exit(1)

F3 = GeneralFourierTransform(support, V2)
"""
barefl = BornApproximationReflectionCalculation(V, support[0], support[1], support[1]-support[0])
barefl._fourier = F
barefl.plot_refl(q_space)
barefl._fourier = F3
barefl.plot_refl(q_space)


#pylab.plot(w_space[len(w_space)//2:], abs(array([1/w*F.fourier(w) for w in w_space if w > 0]))**2)
#pylab.plot(w_space[len(w_space)//2:], abs(array([1/w*F3.fourier(w) for w in w_space if w > 0]))**2)
pylab.yscale('log')
#pylab.ylim(-1, 5)
pylab.show()
"""

reflcalc = ReflectionCalculation(None, support[0], support[-1], support[1] - support[0])

"""
reflcalc.set_potential(scale(V))
reflcalc.plot_refl(q_space)

refl = [reflcalc.reflectivity(q) for q in q_space]

"""
# exit(0)

pylab.subplot(212)
reflcalc.set_potential(scale(V))
reflcalc.plot_refl(q_space)
reflcalc.set_potential(scale(V2))
reflcalc.plot_refl(q_space)
pylab.show()


#refl = [abs(reflcalc.refl(q)) ** 2 for q in q_space]
#numpy.savetxt("calc/new_refl_{}.dat".format(to_str(a)), list(zip(q_space, refl)))
# pylab.show()

#numpy.savetxt("calc/new_pot_{}.dat".format(to_str(a)), list(zip(support, V2(support))))


"""
numpy.savetxt('initial.dat', list(zip(support, V(support))))
reflcalc.set_potential(scale(V))
numpy.savetxt('reflectivtiy_initial.dat', zip(q_space, reflcalc.reflectivity(q_space)))
reflcalc.set_potential(scale(V2))
numpy.savetxt('reflectivtiy_equivalent.dat', zip(q_space, reflcalc.reflectivity(q_space)))
"""
