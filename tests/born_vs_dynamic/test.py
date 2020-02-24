import numpy as np

from dinv.helper import load_potential, shift_potential
from dinv.ba import BornApproximationReflectionCalculation
from dinv.glm import ReflectionCalculation


potential = shift_potential(load_potential("profile.dat"), 20)

pot_range = (0, 200)
k_range = np.linspace(1e-10, 1, 1000)

ba = BornApproximationReflectionCalculation(potential, pot_range[0], pot_range[1], 0.1)
glm = ReflectionCalculation(potential, pot_range[0], pot_range[1], 0.1)


r_ba = ba.reflection(k_range)
r_glm = glm.reflection(k_range)

glm.plot_potential(show=True)

ba.plot_refl(2*k_range)
glm.plot_refl(2*k_range, show=True)

ba.plot_ampl(2*k_range)
glm.plot_ampl(2*k_range, show=True)

