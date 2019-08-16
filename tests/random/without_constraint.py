import numpy
import scipy.interpolate

from dinv.helper import TestRun

numpy.random.seed(1)
numpy.set_printoptions(precision=2, linewidth=210)


def constrain(potential, x_space):
    data = potential(x_space)

    data[(x_space >= 400)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


test = TestRun("simulation-1-profile.dat")

test.cutoff = 0.010
test.noise = 0
test.iterations = 10000
test.tolerance = 1e-8
test.offset = 20
test.thickness = 350
test.precision = 1
test.pot_cutoff = 2
test.use_only_real_part = False
test.q_max = 0.5
test.plot_every_nth = 100
test.store_path = 'data/'

test.start = 0

test.plot_potential = True
test.plot_phase = False
test.plot_reflectivity = False

test.run(constrain)
