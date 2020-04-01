from numpy import linspace, angle, sqrt, exp
from dinv.fourier import FourierTransform, UpdateableFourierTransform
from dinv.helper import load_potential, shift_potential
from dinv.glm import PotentialReconstruction, ReflectionCalculation

import scipy
import pylab

shift = 40
thickness = 100
precision = 4
cutoff = 2
q_max = 0.5
q_extrapolation = 0.6


x_space = linspace(0, thickness + shift, 10*(thickness + shift) + 1)
k_space = linspace(0, q_max / 2, int(1000 * q_max) + 1)
k_extrapolation_space = linspace(q_max / 2, q_extrapolation, int((q_extrapolation - q_max) * 1000 + 1))


potential = load_potential("simulation.profile")
potential = shift_potential(potential, shift)


reconstruction = PotentialReconstruction(shift + thickness, precision, cutoff=cutoff)
reflection_calc = ReflectionCalculation(lambda x: 0, 0, shift + thickness)

reflection_calc.set_potential(potential)

refl = reflection_calc.refl(2 * k_space)
refl_extrapolation = reflection_calc.refl(2*k_extrapolation_space)
#refl_extrapolation = (refl_extrapolation * exp(-1j * angle(refl_extrapolation))).real


transform_left = FourierTransform(k_space, refl.real, refl.imag)
transform_extrapolation = FourierTransform(k_extrapolation_space, refl_extrapolation.real, refl_extrapolation.imag)

transform = UpdateableFourierTransform(transform_extrapolation, transform_left)

pylab.plot(x_space, potential(x_space))

def constrain(potential, x_space):
    data = potential(x_space)

    #data[(x_space >= 80)] = 0e-6
    #data[(x_space <= 30)] = 0e-6
    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation

diff = 0

for i in range(0, 10):
    pot = reconstruction.reconstruct(transform)
    pot = constrain(pot, x_space)
    reflection_calc.set_potential(pot)
    reflection = reflection_calc.refl(2*k_extrapolation_space)

    #phase = angle(reflection)
    #update = refl_extrapolation * exp(1j * phase)
    update = reflection
    print(update)
    diff = transform.update(k_extrapolation_space, update)
    #print(abs(update - refl_extrapolation))

    pylab.plot(x_space, pot(x_space))
    print(i, diff)

    if diff < 1e-8:
        break




pylab.show()
