import numpy
import scipy.interpolate
import os

from dinv.helper import TestRun, load_potential_bumps

numpy.random.seed(1)
numpy.set_printoptions(precision=2, linewidth=210)


def constrain(potential, x_space):
    data = potential(x_space)

    #data[(x_space >= 670)] = 0e-6
    #data[(x_space > 270) & (x_space < 295)] = 4.77e-6
    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation

potential = load_potential_bumps('../../../profile/EG435.dat', 'up')

test = TestRun(potential)

test.cutoff = 0.006
test.noise = 0
test.iterations = 100
test.tolerance = 1e-8
test.offset = 30
test.thickness = 670
test.precision = 0.25
test.pot_cutoff = 2
test.use_only_real_part = False
test.q_max = 0.25
test.q_precision = 1
test.plot_every_nth = 10

test.store_path = 'data/up/'

test.start = 0

test.plot_potential = True
test.plot_phase = False
test.plot_reflectivity = False
test.show_plot = True

test.run(constrain)
