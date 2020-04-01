import numpy as np
import scipy.interpolate
import pylab
import math

from numpy import array
from dinv.ba import BornApproximationPotentialReconstruction, BornApproximationReflectionCalculation
from dinv.fourier import UpdateableFourierTransform, GeneralFourierTransform
from dinv.helper import load_potential, shift_potential

np.set_printoptions(linewidth=210, precision=4)

thickness = 250

k_start = 1e-7
k_split = 0.005
k_end = 1

k_space_full = np.linspace(k_start, k_end, 5001)
k_space = np.linspace(k_split, k_end, 5001)
k_rec_space = np.linspace(k_start, k_split, 100)

k_plot_space = np.linspace(k_start, 0.2, 1000)
x_space = np.linspace(-10, thickness, 10000)


potential = load_potential("profile.dat", as_function=True)
potential = potential.shift(25, True)
#potential = shift_potential(potential, 50)

# potential = numpy.loadtxt("profile.dat").T
# potential = scipy.interpolate.interp1d(potential[0], potential[1], kind='nearest', bounds_error=False, fill_value=(0,0))


np.random.seed(1)


# pylab.plot(x_space, potential(x_space))
# pylab.show()

# R = ReflectionCalculation(potential, 0, 200, 0.1)
# R.plot_ampl(2*k_space, scale=True)


# Rba.plot_ampl(2*k_space, scale=True)
# fourier = GeneralFourierTransform(x_space, p)
# fourier_eval = array([fourier(k) for k in 2*k_space])
# pylab.plot(2*k_space, fourier_eval.real*4*math.pi/(2*k_space), label='BA real')
# pylab.plot(2*k_space, -fourier_eval.imag*4*math.pi/(2*k_space), label='BA imag')
# pylab.show()


def constrain(potential, x_space):
    data = potential(x_space).real

    #data[(x_space <= 35)] = 0e-6
    #data[(x_space >= 210)] = 0e-6
    # data[(x_space >= 125) & (x_space <= 450)] = 4.662e-6
    # data[(x_space <= 690)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


rec = BornApproximationPotentialReconstruction(thickness, 2)
Rba = BornApproximationReflectionCalculation(potential, 0, thickness, 0.1)

Rba.plot_potential(style='-', label='Input potential')
exact_amplitude = [Rba(k) for k in k_rec_space]

f = GeneralFourierTransform(k_space_full, Rba)
pot = rec.reconstruct(f)
pylab.plot(x_space, pot(x_space), label='Ideal reconstructed potential')




# Split the fourier transform into two parts, first part is for the changing reflectivity amplitude
# second part stays constant -> Saves computation time
f1 = GeneralFourierTransform(k_rec_space, lambda x: 0)
f2 = GeneralFourierTransform(k_space, Rba)
f = UpdateableFourierTransform(f1, f2)

max_iter = 200
Rba = BornApproximationReflectionCalculation(None, 0, thickness, 0.1)

for iter in range(1, max_iter + 1):

    rec_potential = rec.reconstruct(f)
    rec_potential = constrain(rec_potential, x_space)

    Rba.set_potential(rec_potential)

    if iter % 10 == 0:
        Rba.plot_potential(label="Iteration {}".format(iter))

    update = array([Rba(k) for k in k_rec_space])
    diff = f.update(k_rec_space, update)

    ampl_err = abs((update - exact_amplitude) / exact_amplitude)
    err_percent = np.mean(ampl_err) * 100
    print(f"{iter:3d} diff: {diff:.5f} rel. err: {err_percent:.2f}%")

    if diff < 1e-8:
        break

pylab.legend()
pylab.show()
