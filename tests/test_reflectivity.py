import numpy
import pylab
import scipy.interpolate

from dinv.glm import FourierTransform, PotentialReconstruction, ReflectionCalculation
from numpy import array


exact_phase = numpy.loadtxt("data/amplitude.real").T
exact_potential = numpy.loadtxt("data/simulation-1-profile.dat").T
exact_potential[0] = exact_potential[0]+10
exact_potential[1] *= 1e-6
#exact_potential[1] = numpy.flip(exact_potential[1])

# total film thickness
end = 50


k_space = exact_phase[0] / 2

real = exact_phase[1]
imag = exact_phase[2]

pylab.plot(exact_potential[0], exact_potential[1])
pylab.show()

potential = scipy.interpolate.interp1d(exact_potential[0], exact_potential[1], fill_value=(0, 0), bounds_error=False)
reflcalc = ReflectionCalculation(potential, 0, end, 0.01)

#reflcalc.plot_potential()
#pylab.show()
refl = reflcalc.refl(2*k_space)

pylab.plot(k_space, refl.real**2 + refl.imag**2)
pylab.plot(k_space, (exact_phase[1]**2 + exact_phase[2]**2)*10)

pylab.yscale('log')
pylab.show()

pylab.plot(k_space, refl.real*k_space**2)
pylab.plot(k_space, real*k_space**2)

pylab.legend(['calculated real', 'loaded'])
pylab.show()

pylab.plot(k_space, -refl.imag*k_space**2)
pylab.plot(k_space, imag*k_space**2)
pylab.show()