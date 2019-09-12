import numpy
import scipy.interpolate
import os

from dinv.helper import TestRun

numpy.random.seed(1)
#numpy.set_printoptions(precision=2, linewidth=210)

def constrain(potential, x_space):
    data = potential(x_space)

    data[(x_space >= 400)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation

rng = numpy.linspace(0.0, 0.0125, 0.0125 * 2000 + 1)
rng = list(map(lambda x: round(x, 5), rng))
print(rng)
#exit(1)

for var in rng:
    print(var)
    print("\n\n\n")

    test = TestRun("simulation-1-profile.dat")
    q_as_string = str(var).replace(".", 'd')

    test.cutoff = var
    test.noise = 0
    test.iterations = 10000
    test.tolerance = 1e-8
    test.offset = 20
    test.thickness = 340
    test.precision = 1
    test.pot_cutoff = 2
    test.use_only_real_part = False
    test.q_max = 0.5
    test.plot_every_nth = 100
    test.store_path = 'store/test/kc/' + q_as_string + "/"
    test.q_precision = 1

    test.start = 0

    test.plot_potential = True
    test.plot_phase = False
    test.plot_reflectivity = False
    test.show_plot = False


    try:
        os.mkdir(os.getcwd() + "/" + test.store_path)
    except:
        pass


    test.run(constrain)
