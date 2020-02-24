import numpy

from dinv.helper import load_potential
from dinv.glm import ReflectionCalculation

pot = load_potential("initial.dat")

support = (0, 400)
qspace = numpy.linspace(0, 1, num=1000)



refl = ReflectionCalculation(pot, support[0], support[1])
refl.plot_potential(show=True)
refl.plot_reflection(qspace, scale=True, show=True)
