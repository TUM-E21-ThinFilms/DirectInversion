import numpy
import scipy.interpolate

from dinv.helper import TestRun

numpy.random.seed(1)

numpy.set_printoptions(precision=2, linewidth=220)

def range(a, b):
    def f(x):
        x[x < a] = a
        x[x > b] = b
        return x

    return f


def constrain(potential, x_space):
    data = potential(x_space)

    data[(x_space >= 155)] = 0e-6
    #data[(x_space >= 51) & (x_space <= 82)] = range(7.9e-6, 8.1e-6)(data[(x_space >= 51) & (x_space <= 82)])
    #data[(x_space >= 117) & (x_space <= 149)] = range(7.9e-6, 8.1e-6)(data[(x_space >= 117) & (x_space <= 149)])
    data[(x_space <= 45)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


test = TestRun("simulation.profile")

test.cutoff = 0.01
test.noise = 0
test.iterations = 50
test.tolerance = 1e-10
test.offset = 50
test.thickness = 150
test.precision = 4
test.pot_cutoff = 2
test.plot_every_nth = 10
test.q_max = 5.0

#test.store_path = 'store/'

test.plot_potential = True
test.plot_phase = False
test.plot_reflectivity = False

test.run(constrain)
