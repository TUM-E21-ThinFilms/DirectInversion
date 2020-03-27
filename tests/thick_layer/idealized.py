import numpy
import scipy.interpolate

from dinv.helper import TestRun

numpy.random.seed(1)
numpy.set_printoptions(precision=2, linewidth=220)


def constrain(potential, x_space):
    data = potential(x_space)

    data[(x_space >= 1500)] = 0e-6
    data[(x_space <= 500)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


print("might take some time to calculate the initial reflectivity ...")
test = TestRun("simulation.profile")

test.cutoff = 0.007
test.noise = 0
test.iterations = 200
test.tolerance = 1e-8
test.offset = 600#700
test.thickness = 1000
test.precision = 0.25
test.pot_cutoff = 2
test.plot_every_nth = 10
test.q_max = 0.50
test.q_precision = 1

test.plot_potential = True
test.plot_phase = False
test.plot_reflectivity = False

test.run(constrain)
