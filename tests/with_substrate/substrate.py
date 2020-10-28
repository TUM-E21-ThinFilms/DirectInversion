import numpy
import scipy.interpolate

from skipi.function import FunctionFileLoader, Function

from dinv.helper import DataRun

numpy.random.seed(1)
#numpy.set_printoptions(precision=2, linewidth=220)



def constrain(potential, x_space):
    data = potential(x_space)

    data[(x_space >= 220)] = 2.1e-6
    #data[(x_space <= 500)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation

reflection = FunctionFileLoader("reflection.dat").from_file()
reflection = reflection.scale_domain(0.5)
reflection = reflection.vremesh((0.01, None))
import pylab

reflection.plot(real=False)
reflection.show()


data = DataRun(reflection)

data.iterations = 200
data.tolerance = 1e-8
data.thickness = 230
#test.precision = 0.25
data.pot_cutoff = 2
data.plot_every_nth = 10
data.q_max = 0.50
data.q_precision = 1

data.plot_potential = True
data.plot_phase = False
data.plot_reflectivity = False

data.run(constrain)