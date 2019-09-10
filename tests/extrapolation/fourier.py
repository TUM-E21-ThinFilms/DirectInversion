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
q_max = 1.0



x_space = linspace(0, thickness + shift, 10*(thickness + shift) + 1)
for q_max in [0.5, 1.0, 5.0]:
    q_extrapolation = q_max + 0.1
    k_space = linspace(0, q_max / 2, int(1000 * q_max) + 1)
    k_extrapolation_space = linspace(q_max / 2, q_extrapolation, (q_extrapolation - q_max) * 1000 + 1)


    potential = load_potential("profile.dat")
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

    w_space = linspace(-40, 60, 1000)


    transform_left.method = transform_left.cosine_transform
    transform_extrapolation.method = transform_extrapolation.cosine_transform

    pylab.plot(w_space, [transform(w) for w in w_space])

pylab.ylim(-0.4e-7, 0.4e-7)
pylab.show()