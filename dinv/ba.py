import numpy
import scipy.interpolate

from numpy import pi
from dinv.fourier import GeneralFourierTransform, FourierTransform
from dinv.glm import PotentialReconstruction, ReflectionCalculation
from dinv.function import to_function

from scipy.interpolate import interp1d


class BornApproximationPotentialReconstruction(PotentialReconstruction):
    """
    The class reconstructs a potential based on a given fourier transform (only in the
    Born-Approximation)

    This is supposed to be the "inverse" operation to
    BornApproximationReflectionCalculation. Hence, the computation
    is done simply by evaluating:

    R(k) being the reflection amplitude, then:
        V(x) = i/2pi \int_{R} k R(k) e^{-ikx} dk

    Note that we use it slightly different,  by observing that this can be expressed in
    terms of a derivative:
        V(x) = - 1/2pi d/dx \int_{R} R(k) e^{-ikx} dk = - d/dx F[R](x)

    """

    def reconstruct(self, transform):
        # assert isinstance(fourier_transform, GeneralFourierTransform)

        inv = numpy.array([transform.fourier(2 * x) for x in self._xspace])

        # Note: the prefactor should be 1/(2 pi). BUT since we're computing it using the gradient
        # we get a minus sign (e^-iqx) -> -i. AND since the formula is using q as the integration variable (q = 2k)
        # we get by change of variables a 2 * 2 (q -> 2k, and dq = 2 dk).
        pot = - 2 / (pi) * numpy.gradient(inv, self._dx, edge_order=2)

        # if self._cut > 0:
        #    pot[0:self._cut] = 0
        #    pot[-self._cut:] = 0

        pot_real = interp1d(self._xspace, pot.real, fill_value=(0, 0), bounds_error=False, kind='linear')
        pot_imag = interp1d(self._xspace, pot.imag, fill_value=(0, 0), bounds_error=False, kind='linear')

        return lambda x: pot_real(x) + 1j * pot_imag(x)

    def reconstruct_at(self, transform, x_point):
        x_space = numpy.array([x_point - self._dx, x_point, x_point + self._dx])
        inv = [transform.fourier(2 * x) for x in x_space]
        pot = - 2 / (pi) * numpy.gradient(inv, self._dx, edge_order=2)

        return pot[1]


class BornApproximationReflectionCalculation(ReflectionCalculation):
    """
    This calculates the reflection amplitude using a potential.

    The calculation is simply a fourier transform. Then we divide by ik.
    """

    def __init__(self, potential_function, z_min, z_max, dz):
        super(BornApproximationReflectionCalculation, self).__init__(potential_function, z_min,
                                                                     z_max, dz)
        self._fourier = None

    def set_potential(self, potential_function):
        super(BornApproximationReflectionCalculation, self).set_potential(potential_function)
        self._fourier = None

    def refl(self, q):
        if self._fourier is None:
            z_space = numpy.linspace(self._z0, self._z1, int((self._z1 - self._z0) / self._dz + 1))
            self._fourier = GeneralFourierTransform(z_space, self._pot)

        if abs(q) < 1e-30:
            return 0

        # Multiply by 2 * pi, since the inverse fourier transform has a 1/(2pi) as a prefactor.
        R = 2 * pi * numpy.array(1 / (1j * q) * self._fourier.fourier_inverse(q))
        return R

    def reflectivity(self, q_space):
        return numpy.array([abs(self.refl(q)) ** 2 for q in q_space])

    def reflection(self, k_range):
        refl = numpy.array([self.refl(2 * k) for k in k_range])
        return refl
        # return scipy.interpolate.interp1d(k_range, refl, fill_value=(0, 0), bounds_error=False,
        #                                  kind='nearest')

    def to_function(self, k_range, reflection):
        return to_function(k_range, reflection, interpolation='linear')

    def reflection_function(self, k_range):
        refl = self.reflection(k_range)
        return self.to_function(k_range, refl)


class ExtrapolationReflection(object):
    def __init__(self, k_range, reflection):
        self._k = k_range
        self._r = numpy.array(reflection)

    def test(self, k_extrapolation_range, exact_reflection, x_space, potential_approximation):
        assert len(k_extrapolation_range) == len(x_space)
        assert len(k_extrapolation_range) == len(exact_reflection)

        A = numpy.zeros((len(k_extrapolation_range), len(k_extrapolation_range)), dtype=complex)
        weight = numpy.ones((len(k_extrapolation_range)))
        weight[0], weight[-1] = 0.5, 0.5

        k = k_extrapolation_range
        dk = k_extrapolation_range[1] - k_extrapolation_range[0]

        # x_points = numpy.linspace(x_length, x_length ** 2 + len(k_extrapolation_range), len(k_extrapolation_range))

        for idx, x in enumerate(x_space):
            e = numpy.exp(-2j * k_extrapolation_range * x)
            A[idx] = numpy.multiply(k, e)
            # A[idx] = [weight[i] * k[i] * numpy.exp(-2j * k[i] * x) for i in range(0, len(k))]
            # A[idx] = [k[i] * numpy.exp(-2j * k[i] * x) for i in range(0, len(k))]
            A[idx][-1] *= 0.5
            A[idx][0] *= 0.5

        A = dk * A
        # print(A)
        # print(numpy.linalg.cond(A))
        b = [potential_approximation(x) for x in x_space]
        # print(b)
        print(numpy.dot(A, exact_reflection) - b)
        # print(numpy.dot(A, exact_reflection)-b)

    def fourier_matrix(self, x_space, k_space):
        # Important, otherwise k_space changes outside the function
        k_space = numpy.array(k_space)

        dq = k_space[1] - k_space[0]

        q = numpy.array(k_space).reshape((1, len(k_space)))
        x = numpy.array(x_space).reshape((len(x_space), 1))

        x_q_matrix = numpy.dot(x, q)
        e_x_q_matrix = numpy.exp(-2j * x_q_matrix)

        # this is kinda the weighting of the trapezoidal integration rule
        k_space[0] *= 0.5
        k_space[-1] *= 0.5

        matrix = numpy.multiply(k_space, e_x_q_matrix)
        return matrix * 4j / (numpy.pi) * dq

    def potential(self, x_space, k_space, reflection):
        return numpy.dot(self.fourier_matrix(x_space, k_space), reflection)

    def reconstruct(self, x_space, k_space, potential_approximation):

        assert len(x_space) >= len(k_space)

        matrix = self.fourier_matrix(x_space, k_space)

        if isinstance(potential_approximation, numpy.ndarray):
            if not len(potential_approximation) == len(x_space):
                raise RuntimeError("Given potential approximation must be evaluated at x_space")
            Vapprox = potential_approximation
        else:
            Vapprox = numpy.array([potential_approximation(x) for x in x_space])

        print(numpy.linalg.cond(matrix))

        if len(x_space) == len(k_space):
            R = - numpy.linalg.solve(matrix, Vapprox)
        else:
            R, _, _, _ = numpy.linalg.lstsq(matrix, Vapprox, rcond=None)
            R = -R
        return R

    # def extrapolate(self, length, extrapolation_range, potential_approximation, fourier_transform):
    def extrapolate(self, k_extrapolation_range, x_space, potential_approximation):

        assert len(k_extrapolation_range) == len(x_space)

        A = numpy.zeros((len(k_extrapolation_range), len(k_extrapolation_range)), dtype=complex)
        weight = numpy.ones((len(k_extrapolation_range)))
        weight[0], weight[-1] = 0.5, 0.5

        k = k_extrapolation_range
        dk = k_extrapolation_range[1] - k_extrapolation_range[0]

        # x_points = numpy.linspace(x_length, x_length ** 2 + len(k_extrapolation_range), len(k_extrapolation_range))

        for idx, x in enumerate(x_space):
            A[idx] = [weight[i] * k[i] * numpy.exp(-2j * k[i] * x) for i in range(0, len(k))]

        A = dk * A
        b = [potential_approximation(x) for x in x_space]
        print(b)
        solv = - numpy.linalg.solve(A, b)

        return solv

        """
        assert isinstance(potential_approximation, BornApproximationPotentialReconstruction)

        fourier = FourierTransform(self._k, self._r.real, self._r.imag)
        Vapprox = potential_approximation

        A = numpy.zeros((len(extrapolation_range), len(extrapolation_range)), dtype=complex)
        # assuming the extrapolation range is evenly spaced (equidistant points)
        dk = extrapolation_range[1] - extrapolation_range[0]

        # To define
        # length = len(extrapolation_range)
        x_points = numpy.linspace(length, length ** 2 + len(extrapolation_range), len(extrapolation_range))

        for idx, k in enumerate(extrapolation_range):
            A[idx] = k * numpy.exp(-1j * k * x_points) * dk * 1j / pi

        b = [Vapprox.reconstruct_at(fourier_transform, x) for x in x_points]

        # print(b)
        # print(A)
        print(dk)
        print(numpy.linalg.cond(A))

        solv = - numpy.linalg.solve(A, b)

        # print(numpy.dot(A, solv) + b)

        return solv
        """

    def approx_refl(self, x_range, potential, q):

        e = numpy.exp(1j * q * x_range)
        dx = x_range[1] - x_range[0]
        V = numpy.array([potential(x) for x in x_range])
        return 1 / (1j * q) * sum(numpy.dot(V, e)) * dx

    def approx(self, k_range, reflection, x):
        assert len(k_range) == len(reflection)

        k = numpy.array(k_range)
        R = numpy.array(reflection)
        dk = k_range[1] - k_range[0]
        dx = 0.01

        if False:
            f = []
            for xval in [x - dx, x, x + dx]:
                e = numpy.exp(-2j * k * xval)
                f.append(sum([R[i] * e[i] for i in range(0, len(k_range))]))

            return -dk * numpy.gradient(f, edge_order=2)[1] * 4
        elif False:

            from scipy.integrate import trapz

            e = numpy.exp(-2j * k * x)
            return 2 * 2 * 1j / pi * trapz([k[i] * R[i] * e[i] for i in range(0, len(k_range))], dx=dk)
        else:
            # Note that, this is in principle just the trapz rule for integration in python
            e = numpy.exp(-2j * k * x)
            weight = numpy.ones((len(e)))
            weight[0], weight[-1] = 0.5, 0.5
            return 4 * 1j / pi * dk * sum([weight[i] * k[i] * R[i] * e[i] for i in range(0, len(k_range))])

    def approx_refl(self, x_range, pot, k):
        e = numpy.exp(1j * k * x_range)
        V = [pot(x) for x in x_range]

        dx = x_range[1] - x_range[0]

        return 4 * pi / (1j * k) * sum([e[i] * V[i] for i in range(0, len(x_range))]) * dx


