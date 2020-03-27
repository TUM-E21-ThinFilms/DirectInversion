import numpy
import scipy.interpolate

from dinv.helper import TestRun

numpy.random.seed(1)
numpy.set_printoptions(precision=2, linewidth=220)


def constrain(potential, x_space):
    data = potential(x_space)

    data[(x_space >= 480)] = 0e-6
    data[(x_space >= 105) & (x_space <= 205)] = 4.662e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


test = TestRun("50nm.profile")

test.cutoff = 0.01
test.noise = 5e-2
test.iterations = 500
test.tolerance = 1e-5
test.offset = 20
test.thickness = 520
test.precision = 1
test.pot_cutoff = 2
test.q_max = 0.5

test.plot_potential = True
test.plot_phase = False
test.plot_reflectivity = False

test.run(constrain)
