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

K_rng = numpy.arange(0.05, 5.05, 0.05)
kc_rng = numpy.arange(0, 1.80, 0.05001) * 1e-2

#rng = K_rng
rng = kc_rng
rng = list(map(lambda x: round(x, 6), rng))
print(rng)


#exit(1)
def simulate(var):

    print("\n\n\n")
    print(var)

    test = TestRun("simulation-1-profile.dat")
    path = str(var).replace('.', 'd')

    test.cutoff = var #0.01
    test.noise = 0
    test.iterations = 5000
    test.tolerance = 1e-8
    test.offset = 20
    test.thickness = 180
    test.precision = 1
    test.pot_cutoff = 2
    test.plot_every_nth = 2
    test.use_only_real_part = False
    test.q_max = 0.2

    """
    test.store_path = 'store/test/K/' + path + '/'
    try:
        os.mkdir(test.store_path)
    except:
        pass
    """

    test.plot_potential = False
    test.plot_phase = False
    test.plot_reflectivity = False
    test.show_plot = False

    test.diagnostic_session = True

    solution = test.run(constrain)
    print(solution)
    print(test.diagnosis()['iteration'][-1])
    return (var, test.diagnosis())

    #result.append((var, test.diagnosis()))

result = []
for var in rng:
    result.append(simulate(var))

for res in result:
    var = res[0]
    data = res[1]

    diff = data['iteration'][-1][1]
    rel_err = data['iteration'][-1][2]
    print(var, data['iteration'][-1])
