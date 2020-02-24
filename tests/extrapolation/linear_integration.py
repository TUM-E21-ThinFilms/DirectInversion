from scipy.integrate import trapz
from math import pi

import numpy as np
import pylab

from dinv.function import Function


def gauss(x):
    x0 = 0
    sigma = 1
    return 1 / np.sqrt(2 * pi * sigma ** 2) * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def antiderivative_function(x_space, function, dx):
    return Function.to_function(x_space, np.array([trapz(function(x_space[0:idx]), dx=dx) for idx in range(0, len(x_space))]))


x_space, dx = np.linspace(-10, 10, 1000, retstep=True)
l = np.argmin(x_space < 0)

x_space_lower = x_space[0:l]
x_space_upper = x_space[l:]

g = Function(x_space, gauss)

g_lower = Function.to_function(x_space_lower, gauss)
g_upper = Function.to_function(x_space_upper, gauss)

print(l)

pylab.plot(x_space, g(x_space))
pylab.plot(x_space, g_lower(x_space))
pylab.plot(x_space, g_upper(x_space))
pylab.show()

G = antiderivative_function(x_space, g, dx)
Glower = antiderivative_function(x_space, g_lower, dx)
Gupper = antiderivative_function(x_space, g_upper, dx)

Gconc = Function(x_space, Glower + Gupper)

pylab.plot(x_space, (G - Gconc)(x_space))
#pylab.plot(x_space, Glower(x_space))
#pylab.plot(x_space, Gupper(x_space))
#pylab.plot(x_space, Gconc(x_space))
pylab.show()
