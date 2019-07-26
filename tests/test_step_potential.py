import numpy
import pylab
import scipy

from dinv.glm import GLMSolver, FourierTransform
from math import sqrt

# We're trying to reconstruct the step-like potential from
#
# V. P. Revenko: Nonlinear Oscillations, Vol. 6, No. 1, 2003
# Determination of an Exact Solution of the Integral Gelfand-Levitan-Marchenko Equation
#           for the Sturm-Liouville Operators with Step-Type Potential
#
# The potential V looks like
#   V(x) = U >= 0 for x in [0, a]
#   V(x) = 0 otherwise
# a > 0
#
# In this case, we don't use the reflection amplitudes (R(k)) directly, but instead use the Fourier-Transform of it
# since the analytical form is known. However, one can simply use refl1d with the given potential and retrieve the
# complex reflection amplitudes.
#
# The Fourier transform has the analytical form
#
#   F(R)(x) = 2 J_2(b(2a-x))/(2a-x) for x in (0, 2a)
#   F(R)(x) = 0 otherwise
#
# where J_2 is the second Bessel function, b = sqrt(U)
#
# Note that we omitted the negative sign for the Fourier transform, since the cited paper deals with a
# different reference frame.
#

# These values can be adjusted
U = 1e-6
a = 500



# This comes from the paper cited above
b = sqrt(U)
J2 = lambda x: scipy.special.jv(2, x)
potential = lambda x: U * numpy.heaviside(x, 1) * numpy.heaviside(-(x-a), 1)

I1 = lambda x: scipy.special.iv(1, x)

z = lambda x, y: numpy.sqrt((2*a-x-y)*(y-x))
z_diag = lambda x: z(x, x)


def K(x, y):
    if x == y:
        if 0 <= x <= a:
            return U/2 * (a-x)
        else:
            return 0
    return b * (2 * a - x - y) / (2 * z(x, y)) * I1(b * z(x, y))


def exact_fourier_transform(x):

    if x < a:
        return 2.0 * J2(b * (a - x)) / (a - x) * 2 * 2
    return 0.0

if False:
    x_space = numpy.linspace(-1, 25, 10000)
    pylab.plot(x_space, [exact_fourier_transform(x) for x in x_space])
    pylab.show()


transform = FourierTransform(range(0, 2), range(0, 2))
transform.method = exact_fourier_transform


solver = GLMSolver(transform)


end = 300
x_space = numpy.linspace(0, 510, 1*510 + 1)
B = []

for x in x_space:
    B.append(solver.solve(x, end, 0.5)[0])
    print("{} done".format(x))


pylab.plot(x_space[1:], numpy.diff(B))
pylab.ylim(0)
pylab.show()
