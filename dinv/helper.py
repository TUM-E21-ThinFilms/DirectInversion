import numpy
import scipy.interpolate
import pylab

from numpy import array
from dinv.glm import ReflectivityAmplitudeInterpolation, ReflectionCalculation, PotentialReconstruction
from dinv.fourier import UpdateableFourierTransform, FourierTransform


def load_potential(file):
    potential = numpy.loadtxt(file).T
    potential[1] = 1e-6 * numpy.flip(potential[1])
    # potential[1] *= 1e-6
    return scipy.interpolate.interp1d(potential[0], potential[1], fill_value=(0, 0), bounds_error=False, kind='linear')


def shift_potential(potential, offset):
    return lambda x: potential(x - offset)


def shake(var, start_end, noise=0.0):
    # make a copy
    var = array(list(var))

    var[start_end[0]:start_end[1]] = 0
    var *= numpy.random.normal(loc=1.0, scale=noise, size=len(var))

    return var


class TestRun(object):
    def __init__(self, file):
        # Sets R(k) = uniform(-1, 1) for k < cutoff
        # 0.1 is approximately the critical edge for Si
        self.cutoff = 0.1
        # adds gaussian noise to R(q)
        self.noise = 5e-2

        self.iterations = 30

        self.tolerance = 1e-8

        self.offset = 20  # -500

        self.thickness = 520

        # int, 1/precision is the discretization distance
        self.precision = 1

        # This will set values close to the boundary to 0
        self.pot_cutoff = 2

        self.q_max = 0.5
        self.q_precision = 1

        self.plot_potential = True
        self.plot_phase = False
        self.plot_reflectivity = False

        self.plot_every_nth = 10

        self.legends = []

        self.file = file

    def setup(self):

        self.plot_potential_space = numpy.linspace(0, self.thickness + self.offset,
                                                   10 * (self.thickness + self.offset) + 1)
        self.k_space = numpy.linspace(0, self.q_max / 2.0, int(self.q_precision * self.q_max * 1000) + 1)

        self.start_end = (0, numpy.argmax(self.k_space > self.cutoff))
        self.k_interpolation_range = self.k_space[self.start_end[0]:self.start_end[1]]

        self.ideal_potential = load_potential(self.file)
        self.ideal_potential = shift_potential(self.ideal_potential, self.offset)
        self.reflcalc = ReflectionCalculation(self.ideal_potential, 0, self.thickness + self.offset, 0.1)

        self.reflectivity = self.reflcalc.refl(2 * self.k_space)

        self.real = shake(self.reflectivity.real, self.start_end, self.noise)
        self.imag = shake(self.reflectivity.imag, self.start_end, self.noise)

    def _plot_exact(self):

        if self.plot_potential:
            transform = FourierTransform(self.k_space, self.reflectivity.real, self.reflectivity.imag)
            # cosine transform doesnt use imaginary part of the reflectivity amplitude
            # transform.method = transform.cosine_transform

            rec = PotentialReconstruction(self.thickness + self.offset, self.precision, cutoff=self.pot_cutoff)

            potential = rec.reconstruct(transform)
            self.reference_potential = potential

            pylab.plot(self.plot_potential_space, self.ideal_potential(self.plot_potential_space))
            pylab.plot(self.plot_potential_space, potential(self.plot_potential_space), '-', color='black')
            pylab.ylim(-1e-6, 1e-5)
            self.legends.append('Exact SLD')
            self.legends.append("Reconstructed SLD (exact)")

        if self.plot_phase:
            pylab.plot(self.k_space, (self.reflectivity.real * self.k_space ** 2))
            self.legends.append("Exact reflectivity amplitude")

        if self.plot_reflectivity:
            pylab.plot(2 * self.k_space, self.reflectivity.real ** 2 + self.reflectivity.imag ** 2)
            pylab.plot(2 * self.k_space, self.real ** 2 + self.imag ** 2)

            pylab.yscale('log')
            self.legends.append('Exact Reflectivity (w/o noise)')
            self.legends.append('Exact Reflectivity (w/ noise)')

    def _plot_hook(self, interpolator):
        iteration = interpolator.iteration
        potential = interpolator.potential
        interpolated_reflectivity = interpolator.reflectivity.real
        is_last = interpolator.is_last_iteration

        if iteration % self.plot_every_nth == 0 or is_last is True or iteration == 1:

            if self.plot_potential:
                pylab.plot(self.plot_potential_space, potential(self.plot_potential_space))

            if self.plot_phase:
                pylab.plot(self.k_space[0:self.start_end[1]],
                           interpolated_reflectivity * self.k_space[0:self.start_end[1]] ** 2, '.')

            if self.plot_reflectivity:
                R = interpolator.reflcalc.refl(2 * self.k_space)
                pylab.plot(2 * self.k_space, R.real ** 2 + R.imag ** 2)

            self.legends.append("Iteration {}".format(iteration))

        exact_real = (self.reflectivity[self.start_end[0]:self.start_end[1]]).real
        # relative error
        print(iteration, (interpolated_reflectivity - exact_real) / exact_real * 100)

    def run(self, constrain):

        self.setup()
        self._plot_exact()

        rec = PotentialReconstruction(self.thickness + self.offset, self.precision, cutoff=self.pot_cutoff)
        # split the fourier transform up into two parts
        # f1 has the changing input
        # f2 has the non-changing input
        # since f2 contains much more data in general, we can save alot of computation by caching f2 and just
        # computing f1 each time.
        f1 = FourierTransform(self.k_space[:self.start_end[1] + 1], self.real[:self.start_end[1] + 1],
                              self.imag[:self.start_end[1] + 1])
        f2 = FourierTransform(self.k_space[self.start_end[1]:], self.real[self.start_end[1]:],
                              self.imag[self.start_end[1]:])
        transform = UpdateableFourierTransform(f1, f2)

        reflcalc = ReflectionCalculation(None, 0, self.thickness + self.offset, 0.1)

        interpolation = ReflectivityAmplitudeInterpolation(transform, self.k_interpolation_range, rec, reflcalc,
                                                           constrain)
        interpolation.set_hook(self._plot_hook)

        interpolation.interpolate(self.iterations, tolerance=self.tolerance)

        pylab.legend(self.legends)
        pylab.show()
