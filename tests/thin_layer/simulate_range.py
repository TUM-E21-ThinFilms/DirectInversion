import numpy
import scipy.interpolate
import os

from dinv.helper import TestRun

numpy.random.seed(1)

#numpy.set_printoptions(precision=2, linewidth=220)


def constrain(potential, x_space):
    data = potential(x_space)

    #data[(x_space >= 200)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation

#rng = numpy.linspace(180, 350, 11)
#rng = numpy.linspace(0, 0.0175, 36)
rng = numpy.linspace(0.2, 10, 50)
rng = list(map(lambda x: round(x, 4), rng))
print(rng)


#exit(1)
for var in rng:

    print("\n\n\n")
    print(var)

    test = TestRun("simulation-1-profile.dat")
    path = str(var).replace('.', 'd')

    test.cutoff = 0.01
    test.noise = 0
    test.iterations = 5000
    test.tolerance = 1e-8
    test.offset = 20
    test.thickness = 180
    test.precision = 1
    test.pot_cutoff = 2
    test.plot_every_nth = 100
    test.use_only_real_part = False
    test.q_max = var

    test.store_path = 'store/test/K/' + path + '/'
    try:
        os.mkdir(test.store_path)
    except:
        pass

    test.plot_potential = True
    test.plot_phase = False
    test.plot_reflectivity = False
    test.show_plot = False

    test.run(constrain)
