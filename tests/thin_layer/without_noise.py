import numpy
import scipy.interpolate

from dinv.helper import TestRun

numpy.random.seed(1)

numpy.set_printoptions(precision=2, linewidth=220)


def constrain(potential, x_space):
    data = potential(x_space)

    #data[(x_space >= 200)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


test = TestRun("profile.dat")

test.cutoff = 0.01
test.noise = 0
test.iterations = 100
test.tolerance = 1e-8
test.offset = 20
test.thickness = 180
test.precision = 2
test.pot_cutoff = 2
test.plot_every_nth = 10
test.use_only_real_part = True
test.q_max = 0.25

test.store_path = 'store/'

test.plot_potential = False
test.plot_phase = True
test.plot_reflectivity = False

test.run(constrain)
