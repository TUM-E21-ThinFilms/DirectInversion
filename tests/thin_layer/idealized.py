import numpy
import scipy.interpolate
import os
from dinv.helper import TestRun

numpy.random.seed(1)

numpy.set_printoptions(precision=2, linewidth=220)


def constrain(potential, x_space):
    data = potential(x_space)

    data[(x_space >= 200)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation


q_test_space = list(map(lambda x: round(x, 8), numpy.linspace(0.01, 0.02, 21)))
q_test_space = 0.0005 * numpy.array(range(1, 25))

for q in q_test_space:
    test = TestRun("profile.dat")

    q_as_string = str(q).replace(".", 'd')

    test.cutoff = q
    test.noise = 0
    test.iterations = 1000
    test.tolerance = 1e-8
    test.offset = 10
    test.thickness = 160
    test.precision = 2
    test.pot_cutoff = 2
    test.plot_every_nth = 10
    test.q_max = 0.25
    test.store_path = 'store/test/kc_250/' + q_as_string + '/'

    test.show_plot = True
    test.plot_potential = True
    test.plot_phase = False
    test.plot_reflectivity = False

    try:
        os.mkdir(os.getcwd() + "/" + test.store_path)
    except:
        pass

    test.run(constrain)
