import numpy
import pylab
import scipy.interpolate

from dinv.glm import FourierTransform, PotentialReconstruction, ReflectionCalculation
from numpy import array

numpy.random.seed(1)

# file structure:
# q     R(q).real   R(q).imag
# imaginary part might be equal to zero
exact_phase = numpy.loadtxt("data/amplitude.real").T


# Sets R(q) = uniform(-1, 1) for q < cutoff
cutoff = 0.01

# adds gaussian noise to R(q)
noise = 5e-2

# It
iterations = 50

# total film thickness
end = 280
# int, 1/precision is the discretization distance
precision = 1

# used if the potential reconstruction behaves strange at the left boundary. This will set values close to the boundary to 0
pot_cutoff = 2


plot_potential = True
plot_phase = False
plot_reflectivity = False

plot_potential_space = numpy.linspace(-5, end, 1000)


def shake(var, start_end):
    var[start_end[0]:start_end[1]] = numpy.random.uniform(-1, 1, start_end[1]-start_end[0])
    var *= numpy.random.normal(loc=1.0, scale=noise, size=len(var))
    return var

k_space = exact_phase[0] / 2
x_space = numpy.linspace(0, end, precision * (end + 1))
start_end = (1, numpy.argmax(k_space > cutoff))

Refl = (array(exact_phase[1]), array(exact_phase[2]))
real = shake(exact_phase[1], start_end)
imag = shake(exact_phase[2], start_end)

eps = 1.0 / precision
offset = 0

def constrain(potential, x_space):
    data = potential(x_space)

    # doesnt really help ...
    # data[(x_space >= 200) & (x_space <= 450)] = 2e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


legends = []

if plot_potential:
    transform = FourierTransform(k_space, Refl[0], imag, offset)

    # cosine transform doesnt use imaginary part of the reflectivity amplitude
    transform.method = transform.cosine_transform

    rec = PotentialReconstruction(end, precision, pot_cutoff)

    potential = rec.reconstruct(transform)
    reference_potential = potential

    pylab.plot(plot_potential_space, potential(plot_potential_space), '-', color='black')
    #pylab.axvline(x_space[pot_cutoff])
    legends.append('Exact SLD')

if plot_reflectivity:
    pylab.plot(2 * k_space, real ** 2 + imag ** 2)
    pylab.yscale('log')

# The actual logic
for iter in range(0, iterations + 1):
    transform = FourierTransform(k_space, real, imag, offset)
    transform.method = transform.cosine_transform

    rec = PotentialReconstruction(end, precision, pot_cutoff)

    potential = rec.reconstruct(transform)
    potential = constrain(potential, rec._xspace)
    reflcalc = ReflectionCalculation(potential, 0, end, .1)

    # Use the new reflection coefficient for small k-values and re-do the inversion ...
    R = reflcalc.refl(2 * k_space[start_end[0]:start_end[1] + 1])
    R = array(map(lambda x: x.real, R))

    real[start_end[0]:start_end[1]] = R[0:-1]

    if plot_potential and iter % 5 == 0:
        pylab.plot(plot_potential_space, (potential(plot_potential_space)), '--')
        legends.append("Iteration {}".format(iter))

    if plot_phase and iter % 5 == 0:
        pylab.plot(k_space[0:start_end[1]], (real * k_space ** 2)[0:start_end[1]], '.')

    if plot_reflectivity:
        R = reflcalc.refl(2 * k_space)
        pylab.plot(2 * k_space, R.real**2 + R.imag**2)

    print("Step {} done".format(iter))

if plot_phase:
    pylab.plot(k_space, (Refl[0] * k_space ** 2))

if len(legends) > 0:
    pylab.legend(legends)

pylab.show()
