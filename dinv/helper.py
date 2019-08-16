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


def shake(var, start_end, noise=0.0, start=0):
    # make a copy
    var = array(list(var))

    var[start_end[0]:start_end[1]] = start
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

        self.use_only_real_part = False

        self.file = file
        self.store_path = None

        self.start = [0]

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

        if self.start == 'exact':
            self.start = self.reflectivity[self.start_end[0]:self.start_end[1]]

        self.start = array(self.start)

        self.real = shake(self.reflectivity.real, self.start_end, self.noise, self.start.real)
        self.imag = shake(self.reflectivity.imag, self.start_end, self.noise, self.start.imag)

    def _plot_exact(self):

        rec = PotentialReconstruction(self.thickness + self.offset, self.precision, cutoff=self.pot_cutoff)

        transform = FourierTransform(self.k_space, self.reflectivity.real, self.reflectivity.imag)
        potential = rec.reconstruct(transform)

        if self.plot_potential:

            # cosine transform doesnt use imaginary part of the reflectivity amplitude
            # transform.method = transform.cosine_transform

            self.reference_potential = potential

            self.store_data(zip(self.plot_potential_space, self.ideal_potential(self.plot_potential_space)), 'pot_exact', 'potential')
            self.store_data(zip(self.plot_potential_space, potential(self.plot_potential_space)), 'pot_ideal', 'potential')

            pylab.plot(self.plot_potential_space, self.ideal_potential(self.plot_potential_space))
            pylab.plot(self.plot_potential_space, potential(self.plot_potential_space), '-', color='black')
            pylab.ylim(-1e-6, 1e-5)
            self.legends.append('Exact SLD')
            self.legends.append("Reconstructed SLD (exact)")

        if self.plot_phase:
            pylab.plot(self.k_space, (self.reflectivity.real * self.k_space ** 2))
            self.store_data(zip(self.k_space, self.reflectivity.real, self.reflectivity.imag, self.reflectivity.real * self.k_space ** 2, self.reflectivity.imag * self.k_space ** 2), 'phase_exact', 'phase')
            self.legends.append("Exact reflectivity amplitude")

        if self.plot_reflectivity:
            refl_ideal = self.reflcalc.refl(2 * self.k_space)

            self.store_data(zip(2*self.k_space, self.reflectivity.real, self.reflectivity.imag, abs(self.reflectivity)**2), 'refl_exact', 'reflectivity')
            self.store_data(zip(2*self.k_space, refl_ideal.real, refl_ideal.imag, abs(refl_ideal)**2), 'refl_ideal', 'reflectivity')

            pylab.plot(2 * self.k_space, self.reflectivity.real ** 2 + self.reflectivity.imag ** 2)
            pylab.plot(2 * self.k_space, self.real ** 2 + self.imag ** 2)
            pylab.plot(2 * self.k_space, refl_ideal.real ** 2 + refl_ideal.imag ** 2)

            pylab.yscale('log')
            self.legends.append('Exact Reflectivity (w/o noise)')
            self.legends.append('Exact Reflectivity (w/ noise)')
            self.legends.append('Ideal Reflectivity')

    def _plot_hook(self, interpolator):
        iteration = interpolator.iteration
        potential = interpolator.potential
        interpolated_reflectivity = interpolator.reflectivity.real
        is_last = interpolator.is_last_iteration

        if iteration % self.plot_every_nth == 0 or is_last is True or iteration == 1:

            if self.plot_potential:
                x_space = self.plot_potential_space
                pot = potential(x_space)

                self.store_data(zip(x_space, pot), 'pot_it_{}'.format(iteration), 'potential')
                pylab.plot(x_space, pot, '--')

            if self.plot_phase:
                k_subspace = self.k_space[0:self.start_end[1]]
                refl = interpolator.reflectivity
                self.store_data(zip(k_subspace,  refl.real, refl.imag, refl.real * k_subspace ** 2, refl.imag * k_subspace ** 2), 'phase_it_{}'.format(iteration), 'phase')
                pylab.plot(k_subspace, interpolated_reflectivity * k_subspace ** 2, '.')

            if self.plot_reflectivity:
                R = interpolator.reflcalc.refl(2 * self.k_space)
                self.store_data(zip(2 * self.k_space, abs(R)**2), 'refl_it_{}'.format(iteration), 'reflectivity')
                pylab.plot(2 * self.k_space, R.real ** 2 + R.imag ** 2, '--')

            self.legends.append("Iteration {}".format(iteration))

        exact_real = (self.reflectivity[self.start_end[0]:self.start_end[1]]).real
        # relative error
        print(iteration, (interpolated_reflectivity - exact_real) / exact_real * 100)
        print(sum(abs((interpolated_reflectivity - exact_real) / exact_real * 100)))

    def store_data(self, X, filename, header=[]):
        if self.store_path is None:
            return

        if header == 'potential':
            header = ["x [Ang]", "SLD [1/Ang^2]"]
        elif header == 'phase':
            header = ["k [1/Ang]", "Re R [1]", "Im R [1]", "k^2 * Re R [1/Ang^2]", "k^2 * Im R [1/Ang^2]"]
        elif header == 'reflectivity':
            header = ["q [1/Ang]", "Re R [1]", "Im R [1]", "|R|^2 [1]"]

        if len(header) > 0:
            header = "\t".join(header)

        numpy.savetxt(self.store_path + filename + ".dat", X, header=header, delimiter="\t")

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

        if self.use_only_real_part:
            f1.method = f1.cosine_transform
            f2.method = f2.cosine_transform

        transform = UpdateableFourierTransform(f1, f2)
        reflcalc = ReflectionCalculation(None, 0, self.thickness + self.offset, 0.1)

        interpolation = ReflectivityAmplitudeInterpolation(transform, self.k_interpolation_range, rec, reflcalc, constrain)
        interpolation.set_hook(self._plot_hook)

        solution = interpolation.interpolate(self.iterations, tolerance=self.tolerance)

        print("Algorithm terminated with solution:")
        print(list(solution))

        pylab.legend(self.legends)
        pylab.show()


