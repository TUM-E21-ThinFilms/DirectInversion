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

potential = load_potential_bumps('../../../profile/EG438.dat', 'up')

test = TestRun(potential)

test.cutoff = 0.0055
test.noise = 0
test.iterations = 100
test.tolerance = 1e-8
test.offset = 30
test.thickness = 750
test.precision = 0.25
test.pot_cutoff = 2
test.use_only_real_part = False
test.q_max = 5
test.q_precision = 1
test.plot_every_nth = 10


#test.start = [(-1-0j), (-0.9839419303191327-0.1784701651711336j), (-0.936071420411944-0.35177020393383585j), (-0.8573034052807786-0.5147451451323483j), (-0.7491763110947222-0.6622679566576577j), (-0.6138752007003537-0.7892470515393012j), (-0.45427087848516984-0.8906243732077943j), (-0.27398460434827937-0.9613575902580922j), (-0.0774970106618031-0.9963758109957849j), (0.12966133310252037-0.9904912641096525j), (0.3405577017479486-0.9382388873484099j)]


test.plot_potential = True
test.plot_phase = False
test.plot_reflectivity = False
test.show_plot = True

test.run(constrain)
