import numpy
import scipy.interpolate
import os
from dinv.helper import TestRun

numpy.random.seed(1)

numpy.set_printoptions(precision=2, linewidth=220)


def constrain(potential, x_space):
    data = potential(x_space)

    #data[(x_space >= 200)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


test = TestRun("profile.dat")

test.cutoff = 0.005
test.noise = 0
test.iterations = 1000
test.tolerance = 1e-8
test.offset = 50
test.thickness = 160
test.precision = 1
test.pot_cutoff = 2
test.plot_every_nth = 100
test.q_max = 0.5

test.plot_potential = True
test.plot_phase = False
test.plot_phase_angle = True
test.plot_reflectivity = False

test.run(constrain)
