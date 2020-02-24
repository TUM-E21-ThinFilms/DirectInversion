from numpy import linspace, angle, sqrt, exp
from dinv.fourier import (
    FourierExtrapolation
)
from dinv.helper import load_potential, shift_potential
from dinv.glm import PotentialReconstruction, ReflectionCalculation
from dinv.function import (
    Function,
    FourierTransform,
    InverseFourierTransform,
    fourier_matrix
)

import numpy as np
import time
import pylab

print(time.ctime())

potential = load_potential("initial.dat", as_function=True)
potential = Function(potential.get_domain(), potential)

c1 = Function.to_function(np.append(np.linspace(-1000, 50, 1000), np.linspace(330, 1000, 1000)), lambda x: 0)
c2 = Function.to_function(np.append(np.linspace(-1000, 50, 1000), np.linspace(330, 1000, 1000)), lambda x: 0)


# c2 = Function.to_function(np.linspace(-1000, 50, 100), lambda x: 0)

# potential.plot()
# constraint.plot(marker='.')
# pylab.show()


def to_potential(fourier_transform):
    return InverseFourierTransform.from_function(np.linspace(-500, 500, 1000), fourier_transform)


def to_file(f: Function, file):
    x = f.get_domain()
    feval = f(x)
    np.savetxt(file, np.column_stack([x, feval.real, feval.imag]), header='x f(x).real f(x).imag')


def from_file(file):
    x, freal, fimag = np.loadtxt(file).T
    return Function.to_function(x, freal + 1j * fimag)

def new_space(barrier):
    w_extr_spaces = [np.linspace(-1, -barrier, l, endpoint=False),
                     # just so that barriers[it-1] is removed as an endpoint ...
                     - np.flip(np.linspace(-1, -barrier, l, endpoint=False))]
    return w_extr_spaces, np.concatenate(w_extr_spaces)



f = FourierTransform.from_function(np.linspace(-1, 1, 5000), potential)

rows = 4
i = 4
barriers = [0.2, 0.25, 0.3]
constraints = [None, c1, c2]
w_lengths = [None, 2000, 500]
w_space = np.linspace(-barriers[0], barriers[0], 1000)

try:
    f_measured = from_file("measured_fourier.dat")
    feval = f_measured(f_measured.get_domain())
    save = False
    w_space = f_measured.get_domain()
except Exception as e:
    print(e)
    save = True

    feval = f(w_space)
    f_measured = Function.to_function(w_space, feval)
    print("Calculated f measured")

pylab.subplot(rows, 2, 1)
f.plot(np.linspace(-0.2, 0.2, 1000), real=False)
pylab.subplot(rows, 2, 2)
to_potential(f).plot()

pylab.subplot(rows, 2, 3)
f.plot(f_measured.get_domain(), real=False)
f_measured.plot(real=False)

pylab.subplot(rows, 2, 4)
to_potential(f_measured).plot()


for it in range(1, 2):
    barrier = barriers[it]
    constraint = constraints[it]

    extr_algorithm = FourierExtrapolation(f_measured, constraint)
    l = w_lengths[it]

    w_extr_spaces, w_extr_space = new_space(barriers[it-1])

    extrapolation = extr_algorithm.extrapolate(w_extr_spaces)

    idx_lower = np.bitwise_and(0 >= w_extr_space, w_extr_space > - barrier)
    idx_upper = np.bitwise_and(0 <= w_extr_space, w_extr_space < barrier)

    f_add_lower = extrapolation[idx_lower]
    w_space_lower = w_extr_space[idx_lower]
    f_add_upper = extrapolation[idx_upper]
    w_space_upper = w_extr_space[idx_upper]

    w_space = np.concatenate([w_space_lower, w_space, w_space_upper])
    feval = np.concatenate([f_add_lower, feval, f_add_upper])
    f_measured = Function.to_function(w_space, feval)
    w_interpolated = np.linspace(min(w_space), max(w_space), 2000)
    f_measured = Function.to_function(w_interpolated, f_measured(w_interpolated))

    i = i + 1
    pylab.subplot(rows, 2, i)
    f.plot(f_measured.get_domain(), real=False)
    f_measured.plot(real=False)
    pylab.ylim(-1.5e-3, 1.5e-3)
    i = i + 1
    pylab.subplot(rows, 2, i)
    to_potential(f_measured).plot()
    # constraints[it].plot(marker='.')

if save:
    to_file(f_measured, "measured_fourier.dat")

pylab.legend()
pylab.show()
