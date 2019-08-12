import numpy
import scipy.interpolate

from dinv.fourier import GeneralFourierTransform
from dinv.glm import PotentialReconstruction, ReflectionCalculation


class BornApproximationPotentialReconstruction(PotentialReconstruction):
    def reconstruct(self, fourier_transform):
        # assert isinstance(fourier_transform, GeneralFourierTransform)

        # The fourier transform should contain the reflection amplitude ...
        #
        # we're dealing here with some inconsistency:
        # The reflection amplitude is calculated in q, rather than in k
        # hence, for the inverse operation, we have to deal with this annoying fact
        # and "walk" twice as fast in the fourier inverse
        # thus, we also need to scale the potential accordingly.
        inv = [fourier_transform.fourier_inverse(2 * x).real for x in self._xspace]
        pot = 4 * numpy.gradient(inv, self._dx, edge_order=2)

        if self._cut > 0:
            pot[0:self._cut] = 0
            pot[-self._cut:] = 0

        potential = scipy.interpolate.interp1d(self._xspace, pot, fill_value=(0, 0), bounds_error=False, kind='nearest')

        return potential


class BornApproximationReflectionCalculation(ReflectionCalculation):

    def __init__(self, potential_function, z_min, z_max, dz):
        super(BornApproximationReflectionCalculation, self).__init__(potential_function, z_min, z_max, dz)
        self._fourier = None

    def set_potential(self, potential_function):
        super(BornApproximationReflectionCalculation, self).set_potential(potential_function)
        self._fourier = None

    def refl(self, q):
        if self._fourier is None:
            z_space = numpy.linspace(self._z0, self._z1, (self._z1 - self._z0) / self._dz + 1)
            self._fourier = GeneralFourierTransform(z_space, self._pot)

        # In fact, this here is the integral of the fourier transform (except a minus sign, but who cares about that)
        # Hence, the inverse operator, will do a differentiation operation
        R = 1 / (1j * q) * self._fourier(q)

        return R

    def reflection(self, k_range):
        refl = [self.refl(2 * k) for k in k_range]
        return scipy.interpolate.interp1d(k_range, refl, fill_value=(0, 0), bounds_error=False, kind='nearest')
