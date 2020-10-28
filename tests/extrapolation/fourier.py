from numpy import linspace, angle, sqrt, exp
from dinv.fourier import (
    FourierExtrapolation
)
from dinv.helper import load_potential, shift_potential
from dinv.glm import PotentialReconstruction, ReflectionCalculation

from skipi.function import Function
from skipi.fourier import FourierTransform, InverseFourierTransform, InverseCosineTransform, fourier_matrix

import numpy as np
import time
import pylab

print(time.ctime())

potential = load_potential("simulation.profile", as_function=True)
# potential = load_potential("../random/profile.dat", as_function=True)
potential = Function(potential.get_domain(), potential)
potential = potential.shift(-25)

# c1 = Function.to_function(np.append(np.linspace(-1000, -30, 1000), np.linspace(350, 1000, 1000)), lambda x: 0)
# c1 = Function.to_function(np.append(np.linspace(-1000, 50, 500), np.linspace(350, 1000, 500)), lambda x: 0)
c2 = Function.to_function(np.append(np.linspace(-1000, 50, 350), np.linspace(350, 1000, 700)), lambda x: 0)
#c2 = Function.to_function(np.linspace(-1000, 50, 500), lambda x: 0)

# c2 = Function.to_function(np.append(np.linspace(-1000, 50, 1000), np.linspace(330, 1000, 1000)), lambda x: 0)


# c2 = Function.to_function(np.linspace(-1000, 50, 100), lambda x: 0)

# potential.plot()
# constraint.plot(marker='.')
# pylab.show()

raise NotImplemented("remesh was re-programmed!")

def to_potential(fourier_transform):
    # return InverseCosineTransform.from_function(np.linspace(-500, 500, 1000), fourier_transform)
    # return InverseFourierTransform.from_function(np.linspace(-500, 500, 1000), fourier_transform)
    w_space = fourier_transform.get_domain()
    x_space = np.linspace(-500, 500, 1000)
    transform_matrix = invfourier_matrix(w_space, x_space)

    pot = np.dot(transform_matrix, fourier_transform(w_space))
    return Function.to_function(x_space, pot)


def to_potential2(fourier_transform):
    return InverseFourierTransform.from_function(np.linspace(-500, 500, 1000), fourier_transform)


def to_reflection(potential):
    return FourierTransform.from_function(np.linspace(-1, 1, 4000), potential)


def to_file(f: Function, file):
    x = f.get_domain()
    feval = f(x)
    np.savetxt(file, np.column_stack([x, feval.real, feval.imag]), header='x f(x).real f(x).imag')
    print("saved to file")


def from_file(file):
    x, freal, fimag = np.loadtxt(file).T
    return Function.to_function(x, freal + 1j * fimag)


f = FourierTransform.from_function(np.linspace(-2, 2, 5000), potential)
f.remesh(np.linspace(-2, 2, 5000))

scaling = Function(f.get_domain(), lambda x: x ** 2 * 100)

rows = 3
i = 4
barriers = [0.1, 0.12, 0.13, 0.13, 0.14, 0.2]
# barriers = [0.15, 0.2, 0.25]
constraints = [c2, c2, c2, c2, c2, c2, c2, c2]
w_lengths = [None, 2000, 500]
w_space = np.linspace(-barriers[0], barriers[0], 1000)
k_max = 2

try:
    f_measured = from_file("measured_fourier.dat")
    raise RuntimeError()
    save = False
    it = 2
except Exception as e:
    save = True
    f_measured = Function.to_function(w_space, f(w_space))
    f_measured.remesh(np.linspace(-barriers[0], barriers[0], 2000))

    print("Using calculated f")
    it = 1

pylab.subplot(rows, 4, 1)
(scaling * f).plot(real=False)
pylab.subplot(rows, 4, 2)
to_potential(f).plot()
pylab.subplot(rows, 4, 3)

(f * scaling).plot(f_measured.get_domain(), real=False)
(f_measured * scaling).plot(real=False)

pylab.subplot(rows, 4, 4)
pot = to_potential(f_measured)
pot.plot()

# for it in range(1, rows):

extr_algorithm = FourierExtrapolation(f_measured, constraints[it])

f_measured = extr_algorithm.extrapolate_function(np.linspace(f_measured.get_domain().max(), k_max, 10000), f_exact=f)
f_measured.remesh(np.linspace(-barriers[it], barriers[it], int(100 * barriers[it] * 1000)))

w_space_test = np.linspace(barriers[it], k_max, 2000)





extr_algorithm.init(w_space_test)

print(
    f"""Errors:
        exact:          {extr_algorithm.minimize_func(f):e} 
        reconstruction: {extr_algorithm.minimize_func(f_measured):e}""")

i = i + 1
pylab.subplot(rows, 4, i)
(f * scaling).plot(f_measured.get_domain(), label="True", real=False)
(f_measured * scaling).plot(real=False)
# pylab.ylim(-1.5e-3, 1.5e-3)


i = i + 1
pylab.subplot(rows, 4, i)

f_plot = Function.from_function(f)
f_plot.remesh(np.linspace(-barriers[it], barriers[it], 5000))
to_potential(f_plot).plot(label="exact")
to_potential(f_measured).plot(label="reconstructed")

# print(Lp(to_potential(f_plot), potential, domain=constraints[it].get_domain()))
# print(Lp(to_potential(f_measured), potential, domain=constraints[it].get_domain()))

# constraints[it].plot(marker='.')

i = i + 1
pylab.subplot(rows, 4, i)
pot = to_potential(f_measured)
pot.remesh(np.linspace(-25, 350, 1000))
f_new = to_reflection(pot)
f_new.remesh(f_measured.get_domain())
(f * scaling).plot(f_measured.get_domain(), real=False)
((f_new) * scaling).plot(f_measured.get_domain(), real=False)

i = i + 1
pylab.subplot(rows, 4, i)
f_new.remesh(f_measured.get_domain())
# f_new.remesh(np.linspace(-f_measured.get_domain().max(), f_measured.get_domain().max(), 2000))
to_potential(f_plot).plot(label="exact")
to_potential(f_new).plot(np.linspace(-25, 350, 1000), label="reconstructed")

f_measured = f_new

if True:
    it = 2
    extr_algorithm = FourierExtrapolation(f_measured, constraints[it])

    f_measured = extr_algorithm.extrapolate_function(np.linspace(f_measured.get_domain().max(), k_max, 10000))
    f_measured.remesh(np.linspace(-barriers[it], barriers[it], int(100 * barriers[it] * 1000)))

    extr_algorithm.init(w_space_test)

    print(
        f"""Errors:
            exact:          {extr_algorithm.minimize_func(f):e} 
            reconstruction: {extr_algorithm.minimize_func(f_measured):e}""")

    i = i + 1
    pylab.subplot(rows, 4, i)
    (f * scaling).plot(f_measured.get_domain(), label="True", real=False)
    (f_measured * scaling).plot(real=False)
    # pylab.ylim(-1.5e-3, 1.5e-3)

    i = i + 1
    pylab.subplot(rows, 4, i)

    f_plot = Function.from_function(f)
    f_plot.remesh(np.linspace(-barriers[it], barriers[it], 5000))
    to_potential(f_plot).plot(label="exact")
    to_potential(f_measured).plot(label="reconstructed")

    i = i + 1
    pylab.subplot(rows, 4, i)
    pot = to_potential(f_measured)
    pot.remesh(np.linspace(-25, 350, 1000))
    f_new = to_reflection(pot)
    f_new.remesh(f_measured.get_domain())
    (f * scaling).plot(f_measured.get_domain(), real=False)
    ((f_new) * scaling).plot(f_measured.get_domain(), real=False)

    i = i + 1
    pylab.subplot(rows, 4, i)
    f_new.remesh(f_measured.get_domain())
    # f_new.remesh(np.linspace(-f_measured.get_domain().max(), f_measured.get_domain().max(), 2000))
    to_potential(f_plot).plot(label="exact")
    to_potential(f_new).plot(np.linspace(-25, 350, 1000), label="reconstructed")

    f_measured = f_new

if save:
    to_file(f_measured, "measured_fourier.dat")

pylab.legend()
pylab.show()

