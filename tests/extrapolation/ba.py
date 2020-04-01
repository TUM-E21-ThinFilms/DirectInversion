import pylab
import numpy as np

from dinv.ba import ExtrapolationReflection, BornApproximationReflectionCalculation, \
    BornApproximationPotentialReconstruction
from dinv.fourier import GeneralFourierTransform
from dinv.helper import load_potential, shift_potential

from numpy import pi

import time

print(time.ctime())

potential = shift_potential(load_potential("simulation.profile"), 40)

length = 3000
thickness = 400


k_range_full = np.linspace(1e-10, 1, 1000)
idx_upper = np.argmin(k_range_full <= 0.1)

k_range = k_range_full[0:idx_upper+1]
k_range_upper = k_range_full[idx_upper:]

#k_range = np.linspace(1e-20, 0.1, 1000)
#k_range_upper = np.linspace(0.1, 1, 1000)
#k_range_full = np.linspace(1e-20, 0.1, 1000), np.linspace(0.1, 1, 1000)
x_space = np.linspace(0, length, 2 * length)

refl_calc = BornApproximationReflectionCalculation(potential, 0, thickness, dz=1)
reflection = refl_calc.reflection(k_range)
refl_upper = refl_calc.reflection(k_range_upper)
refl_full = refl_calc.reflection(k_range_full)

extr = ExtrapolationReflection(k_range, reflection)

# refl_full = np.array([extr.approx_refl(x_space, potential, 2*k) for k in k_range_full])


fourier = GeneralFourierTransform(k_range[0:idx_upper+5], refl_calc.to_function(k_range, reflection))
fourier_upper = GeneralFourierTransform(k_range_upper, refl_calc.to_function(k_range_upper, refl_upper))
fourier_full = GeneralFourierTransform(k_range_full, refl_calc.to_function(k_range_full, refl_full))

recr = BornApproximationPotentialReconstruction(length, 2)
pot = recr.reconstruct(fourier)
pot2 = recr.reconstruct(fourier_upper)
pot_full = recr.reconstruct(fourier_full)

# plot the fourier transforms
if False:
    x_space = np.linspace(0, 0.2, 50000)

    pylab.plot(x_space, [fourier._f(x) * x**2 for x in x_space])
    pylab.plot(x_space, [fourier_upper._f(x) *x**2 for x in x_space])
    pylab.show()

    freq_space = np.linspace(0, 0.1, 1000)
    fourier.plot(np.linspace(0, 1, 1000))
    fourier_upper.plot(np.linspace(0, 1, 1000))
    pylab.show()
    exit(1)

# Reconstruct the potential using the matrix formalism
if False:
    x_range = np.linspace(500, 3000, 3000)
    potential_vector = extr.potential(x_range, k_range, reflection)
    potential_vector_full = extr.potential(x_range, k_range_full, refl_full)
    # pylab.plot(x_range, potential(x_range))
    pylab.plot(x_range, potential_vector)
    pylab.plot(x_range, potential_vector_full)
    pylab.plot(x_space, potential(x_space))
    pylab.show()
    # exit(1)

# Test whether the exact reflection solves the linear system itself ...
if True:
    x_range = np.linspace(500, 3000, 8000)
    k_extrapolation = k_range_upper

    #pylab.plot(x_range, pot(x_range))
    #pylab.plot(x_range, pot2(x_range))
    pylab.plot(x_range, pot(x_range) + pot2(x_range))

    """
    potential_vector = extr.potential(x_range, k_range, reflection)
    matrix = extr.fourier_matrix(x_range, k_range)
    Vapprox = np.dot(matrix, reflection)
    matrix = extr.fourier_matrix(x_range, k_extrapolation)
    Vdiff = np.dot(matrix, refl_upper)
    matrix = extr.fourier_matrix(x_range, k_range_full)
    V = np.dot(matrix, refl_full)
    pylab.plot(x_range, Vapprox)
    pylab.plot(x_range, Vdiff)
    """
    #pylab.plot(x_range, pot2(x_range))
    #pylab.plot(x_range, pot_full(x_range))

    #pylab.plot(x_range, Vapprox + Vdiff)
    #pylab.plot(x_range, Vapprox)
    #pylab.plot(x_range, -Vdiff)
    #pylab.plot(x_range, V)
    #pylab.plot(x_range, -potential_vector + pot(x_range))
    pylab.show()

if False:
    x_range = np.linspace(500, 3000, 1000)
    k_extrapolation = k_range_upper
    potential_vector = extr.potential(x_range, k_range, reflection)
    #potential_vector = np.array(len(potential_vector) * [0])
    R = extr.reconstruct(x_range, k_extrapolation, potential_vector)

    pylab.plot(k_range_upper, refl_upper.real * k_range_upper ** 2, label="target")
    pylab.plot(k_extrapolation, R.real * k_extrapolation ** 2, label="actual")
    pylab.legend()
    pylab.show()

    exit(1)

# Test that we can reconstruct the potential in the BA and it is the same as the original potential.
# i.e. we test for consistency
if False:
    x_range = np.linspace(0, 800, 2000)

    pylab.plot(x_range, [potential(x) for x in x_range])
    # pylab.plot(x_range, [pot2(x) for x in x_range])
    pylab.plot(x_range, [pot_full(x).real for x in x_range])
    pylab.plot(x_range, [pot_full(x).imag for x in x_range])
    pylab.show()
    exit(1)

# Check that F^-1[R^approx] =  - F^-1[R^diff] on x > L, i.e. the reflection should "sum" up to zero where
# the potential is zero
if False:
    x_range, dx = np.linspace(400, 800, 1000, retstep=True)

    prefactor = - 2 / pi
    Vapprox = prefactor * np.gradient([fourier.fourier(2 * x).real for x in x_range], dx)
    Vdiff = prefactor * np.gradient([fourier_upper.fourier(2 * x).real for x in x_range], dx)

    # pylab.plot(x_range, potential(x_range))
    pylab.plot(x_range, Vapprox)
    pylab.plot(x_range, Vapprox + Vdiff)
    pylab.plot(x_range, Vdiff)
    pylab.show()
    exit(1)

# Check that F^-1[R^approx] =  - F^-1[R^diff] on x > L, i.e. the reflection should "sum" up to zero where
# the potential is zero, using not the "differentiation method", but the plain sum method ...
# and it works :)
if False:
    x_range, dx = np.linspace(400, 800, 1000, retstep=True)
    Vapprox = np.array([extr.approx(k_range, reflection, x) for x in x_range])
    Vdiff = [extr.approx(k_range_upper, refl_upper, x) for x in x_range]

    pylab.plot(x_range, Vapprox + Vdiff)
    pylab.plot(x_range, Vdiff)
    pylab.show()
    exit(1)

# Plot the reflection in the BA
if False:
    refl_neg = refl_calc.reflection(-k_range)

    pylab.plot(-k_range, refl_neg.real, label='real')
    pylab.plot(-k_range, refl_neg.imag, label='imag')

    pylab.plot(k_range, reflection.real, label='real')
    pylab.plot(k_range, reflection.imag, label='imag')
    pylab.legend()
    pylab.show()
    exit(1)


exit(1)

def neg_real(as_list):
    as_list.real = - as_list.real


# Plot the potential using the reflection data
if False:
    Va = []
    Vd = []
    Vf = []
    x_range = np.linspace(100, 800, 2000)

    neg_real(reflection)
    neg_real(refl_upper)
    neg_real(refl_full)
    for x in x_range:
        Vapprox = extr.approx(k_range, reflection, x)
        Vdiff = extr.approx(k_range_upper, refl_upper, x)
        Vfull = extr.approx(k_range_full, refl_full, x)
        # print((Vapprox.real, Vdiff.real))

        Va.append(Vapprox)
        Vd.append(Vdiff)
        Vf.append(Vfull)

    # pylab.plot(x_range, np.array(Va).real)
    # pylab.plot(x_range, np.array(Vd).real)
    pylab.plot(x_range, np.array(Vf).real)

    # pylab.plot(x_range, np.array(Vd).real + np.array(Va).real)
    pylab.show()

    exit(1)

recr = BornApproximationPotentialReconstruction(length, 2)
pot = recr.reconstruct(fourier)
pot2 = recr.reconstruct(fourier_upper)

x_space_plot = np.linspace(0, length, 2 * length)
pylab.plot(x_space_plot, pot(x_space_plot), label='reconstruction')
pylab.plot(x_space_plot, pot2(x_space_plot), label='reconstruction error')
pylab.plot(x_space_plot, pot_full(x_space_plot), label='exact')
pylab.legend()
pylab.show()

extr = ExtrapolationReflection(k_range, reflection)

k_range_extr = k_range_upper  # np.linspace(0.1, 0.2, 500)
x_points = np.linspace(thickness, length, len(k_range_extr))
# Rextr = extr.extrapolate(k_range_extr, x_points, pot)
extr.test(k_range_extr, refl_upper, x_points, pot)

pylab.plot(k_range, reflection.real * k_range ** 2)
# pylab.plot(k_range, reflection.imag*k_range**2)
pylab.plot(k_range_extr, Rextr.real * k_range_extr ** 2)
pylab.plot(k_range_upper, refl_upper.real * k_range_upper ** 2)
pylab.show()

exit(0)


def constrain(fun):
    def f(x):
        if 90 < x < 340:
            return 0
        return fun(x)

    def wrap(xval):
        if isinstance(xval, np.ndarray):
            xval = np.array(xval)

            return np.array([f(x) for x in xval])
        return f(xval)

    return wrap


def identity(fun):
    def wrap(x):
        return fun(x)

    return wrap


# Check now the fourier trafo of the upper potential
refl_calc = BornApproximationReflectionCalculation(constrain(pot), 0, length, dz=1)
pylab.plot(x_space, identity(pot2)(x_space))
pylab.plot(x_space, constrain(pot)(x_space))
pylab.show()

# refl_calc.plot_potential()
# pylab.show()
k_range_full = np.linspace(1e-6, 1, 5000)
reflection_new = refl_calc.reflection(k_range_full)
pylab.plot(k_range, reflection.real, label='exact', color='blue')
pylab.plot(k_range_upper, refl_upper.real, color='blue')
pylab.plot(k_range_full, reflection_new.real, label='approximation', color='red')

pylab.legend()
pylab.show()
