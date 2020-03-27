import numpy
import pylab
import scipy.interpolate

from dinv.glm import FourierTransform, PotentialReconstruction, ReflectionCalculation


from dinv.helper import load_potential


exact_phase = numpy.loadtxt("data/amplitude.real").T
exact_potential = numpy.loadtxt("data/simulation-1-profile.dat").T
exact_potential[0] = exact_potential[0]+10
exact_potential[1] *= 1e-6


potential = load_potential("data/simulation-1-profile.dat", as_function=True)

k_space = exact_phase[0] / 2

real = exact_phase[1]
imag = exact_phase[2]


#potential.shift(abs(potential.get_domain().min()), domain=True)
potential.plot()
pylab.show()


reflcalc = ReflectionCalculation(potential, potential.get_domain().min(), potential.get_domain().max(), 0.1)

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