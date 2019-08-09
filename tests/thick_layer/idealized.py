import numpy
import scipy.interpolate

from dinv.helper import TestRun

numpy.random.seed(1)
numpy.set_printoptions(precision=2, linewidth=220)


def constrain(potential, x_space):
    data = potential(x_space)

    data[(x_space >= 1550)] = 0e-6
    data[(x_space <= 690)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


print("might take some time to calculate the initial reflectivity ...")
test = TestRun("profile.dat")

test.cutoff = 0.009
test.noise = 0
test.iterations = 1000
test.tolerance = 1e-10
test.offset = 700
test.thickness = 900
test.precision = 0.25
test.pot_cutoff = 2
test.plot_every_nth = 50
test.q_max = 5
test.q_precision = 5

test.plot_potential = True
test.plot_phase = False
test.plot_reflectivity = False

test.run(constrain)
