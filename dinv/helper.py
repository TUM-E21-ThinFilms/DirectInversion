import numpy
import scipy.interpolate
import pylab

from skipi.function import Function, NullFunction

from numpy import array
from dinv.glm import ReflectivityAmplitudeInterpolation, ReflectionCalculation, \
    PotentialReconstruction
from dinv.fourier import UpdateableFourierTransform, FourierTransform


def load_potential(file, as_function=False):
    """
    Loads a generic potential created by refl1d from a file.
    The first column are the x-values
    The second column are the V(x) values

    :param file:
    :return: Callable potential function
    """
    potential = numpy.loadtxt(file).T
    potential[1] = 1e-6 * numpy.flip(potential[1])
    # potential[1] *= 1e-6
    fun = scipy.interpolate.interp1d(potential[0], potential[1], fill_value=(0, 0),
                                     bounds_error=False, kind='linear')
    if as_function:
        return Function(potential[0], fun)
    else:
        return fun


def load_potential_bumps(file, spin_state='up'):
    """
    Loads a generic potential created by the fitting of refl1d from a file.

    :param file:
    :param spin_state: If the potential is magnetic, the potential changes based on the
    spin type of the neutrons. Select either "up"/"+" or "down"/"-".
    :return: Callable potential function
    """
    potential = numpy.loadtxt(file).T

    if spin_state == 'up' or spin_state == '+':
        sgn = 1
    else:
        sgn = -1

    print("Selected spin-state {}".format(sgn))

    z = potential[0]
    # might help for figuring out the correct offset ;)
    # print(numpy.min(z))

    rho, rhoM = potential[1], potential[3]

    V = 1e-6 * numpy.flip(rho + sgn * rhoM)

    return scipy.interpolate.interp1d(z, V, fill_value=(0, 0), bounds_error=False,
                                      kind='linear')


def shift_potential(potential, offset):
    """
    Shifts a potential by offset to the right
    """
    return lambda x: potential(x - offset)


def shake(var, start_end, noise=0.0, start=0):
    """
    "Shakes" a reflectivity amplitude, i.e. setting it to zero in a given range  and adding
    noise.
    """
    # make a copy
    var = array(list(var))

    var[start_end[0]:start_end[1]] = start
    var *= numpy.random.normal(loc=1.0, scale=noise, size=len(var))

    return var


class TestRun(object):
    """
    This class is used for some numerical simulations of given potentials.
    This sets-up the required classes to do a reflectivity amplitude interpolation.

    On top of that, it has some nice plotting features, to see how the
    potential/reflection amplitzde/reflectivity is evolving for each iteration.

    The algorithm can be adjusted by changing the attributes defined in __init__.
    """

    def __init__(self, file_or_potential):

        # Starting value for the iteration. 'exact' is possible.
        # This will start the iteration at the exact reflectivity amplitude.
        # None/0 will start with a reflectivity amplitude being 0 below the
        # critical edge, see self.cutoff
        self.start = [0]

        # Sets R(k) = self.start for k < cutoff
        # 0.01 is approximately the critical edge for Si
        self.cutoff = 0.01

        # adds gaussian noise to R(q)
        self.noise = 5e-2

        # Max number of iteration till abortion
        self.iterations = 30

        # Algorithm terminates after R(k) is changing less than the given tolerance
        self.tolerance = 1e-8

        # Shifts the potential to the right, usefull when dealing with potential
        # having a rough surface
        self.offset = 20  # -500

        # Film thickness
        self.thickness = 520

        # int, 1/precision is the discretization step
        self.precision = 1

        # This will set values close to the boundary to 0
        self.pot_cutoff = 2

        # simulated R(k) up to k = q_max / 2
        self.q_max = 0.5
        # Simulation precision. Higher values simulates R(k) more densely.
        self.q_precision = 1

        # For the algorithm, use only the real part of R(k)?
        # TODO: also only use the imaginary part?
        self.use_only_real_part = False
        self.use_only_imag_part = False

        # Plot anything? If None, this will be True if any of the plot_* attributes is True
        self.plot = True
        # Plot the potential?
        self.plot_potential = True
        # Plot the reflectivity amplitude?
        self.plot_phase = False
        # Plot the phase as: arg(r)
        self.plot_phase_angle = False
        # Plot the reflectivity?
        self.plot_reflectivity = False
        # Plot the reflection
        self.plot_reflection = False
        # Plot not every iteration, but every n-th
        self.plot_every_nth = 10
        # Shows the plot after the algorithm has finished
        self.show_plot = True

        # Store the results into this given path. If None, do not save to file.
        # Will only store every n-th iteration
        self.store_path = None

        # user defined hook
        self.hooks = []

        self.ideal_potential = None
        self.file = None
        # Load potential from a file or accept a given potential function
        if isinstance(file_or_potential, str):
            self.file = file_or_potential
        else:
            self.ideal_potential = file_or_potential

        # if set to true, the script will gather diagnostic information
        # and store it into _diagnostics, which can be accessed via
        # self.diagnosis()
        self.diagnostic_session = False

        # Diagnosis object, stores some useful information
        self._diagnosis = {
            'iteration': [],
            'last_iteration': 0
        }

    def setup(self):
        self.legends = []

        self.plot_potential_space = numpy.linspace(0, self.thickness + self.offset,
                                                   10 * self.precision * (self.thickness + self.offset) + 1)

        self.k_space = numpy.linspace(0, self.q_max / 2.0,
                                      int(self.q_precision * self.q_max * 1000) + 1)

        self.start_end = (0, numpy.argmax(self.k_space > self.cutoff))
        self.k_interpolation_range = self.k_space[self.start_end[0]:self.start_end[1]]

        if self.ideal_potential is None:
            self.ideal_potential = load_potential(self.file)

        self.ideal_potential = shift_potential(self.ideal_potential, self.offset)

        self.reflcalc = ReflectionCalculation(self.ideal_potential, 0,
                                              self.thickness + self.offset, 0.1)

        self.reflectivity = self.reflcalc.refl(2 * self.k_space)

        if self.start == 'exact':
            self.start = self.reflectivity[self.start_end[0]:self.start_end[1]]

        if self.start is None:
            self.start = 0

        self.start = array(self.start)

        self.real = shake(self.reflectivity.real, self.start_end, self.noise, self.start.real)
        self.imag = shake(self.reflectivity.imag, self.start_end, self.noise, self.start.imag)

    def _plot_exact(self):

        rec = PotentialReconstruction(self.thickness + self.offset, self.precision, cutoff=self.pot_cutoff)

        transform = FourierTransform(self.k_space, self.reflectivity.real, self.reflectivity.imag)
        potential = rec.reconstruct(transform)

        ind = slice(self.start_end[1], -1)
        i1_real = scipy.interpolate.interp1d([0, self.k_space[self.start_end[1]]], [0, 0], bounds_error=False,
                                             fill_value=(0, 0))
        i1_imag = scipy.interpolate.interp1d([0, self.k_space[self.start_end[1]]],
                                             [0, 0],
                                             bounds_error=False, fill_value=(0, 0))
        i2_real = scipy.interpolate.interp1d(self.k_space[ind], self.reflectivity.real[ind],
                                             bounds_error=False, fill_value=(0, 0))
        i2_imag = scipy.interpolate.interp1d(self.k_space[ind], self.reflectivity.imag[ind],
                                             bounds_error=False, fill_value=(0, 0))

        refl_ipd_real = lambda x: i2_real(x) + i1_real(x)
        refl_ipd_imag = lambda x: i2_imag(x) + i1_imag(x)

        trans = FourierTransform(self.k_space, refl_ipd_real(self.k_space), refl_ipd_imag(self.k_space))
        init_potential = rec.reconstruct(trans)

        if self.plot_potential:
            # cosine transform doesnt use imaginary part of the reflectivity amplitude
            # transform.method = transform.cosine_transform

            self.reference_potential = potential

            self.store_data(zip(self.plot_potential_space, self.ideal_potential(self.plot_potential_space)),
                            'pot_exact', 'potential')

            self.store_data(zip(self.plot_potential_space, potential(self.plot_potential_space)), 'pot_ideal',
                            'potential')

            pylab.plot(self.plot_potential_space, init_potential(self.plot_potential_space), '.')

            pylab.plot(self.plot_potential_space,
                       self.ideal_potential(self.plot_potential_space))
            pylab.plot(self.plot_potential_space, potential(self.plot_potential_space), '-',
                       color='black')
            # pylab.ylim(-1e-6, 1e-5)
            self.legends.append("Initial potential")
            self.legends.append("Exact SLD")
            self.legends.append("Reconstructed SLD (exact)")

        if self.plot_phase:
            if self.plot_phase_angle:
                pylab.plot(self.k_space, numpy.angle(self.reflectivity))
            else:
                pylab.plot(self.k_space, (self.reflectivity.real * self.k_space ** 2))

            self.store_data(zip(self.k_space, self.reflectivity.real, self.reflectivity.imag,
                                self.reflectivity.real * self.k_space ** 2,
                                self.reflectivity.imag * self.k_space ** 2),
                            'phase_exact', 'phase')

            self.legends.append("Exact reflectivity amplitude")

        if self.plot_reflectivity:
            refl_ideal = self.reflcalc.refl(2 * self.k_space)

            self.store_data(zip(2 * self.k_space, abs(self.reflectivity) ** 2), 'refl_exact', 'reflectivity')
            self.store_data(zip(2 * self.k_space, abs(refl_ideal) ** 2), 'refl_ideal', 'reflectivity')

            pylab.plot(2 * self.k_space, self.reflectivity.real ** 2 + self.reflectivity.imag ** 2)
            pylab.plot(2 * self.k_space, self.real ** 2 + self.imag ** 2)
            pylab.plot(2 * self.k_space, refl_ideal.real ** 2 + refl_ideal.imag ** 2)

            pylab.yscale('log')
            self.legends.append('Exact Reflectivity (w/o noise)')
            self.legends.append('Exact Reflectivity (w/ noise)')
            self.legends.append('Ideal Reflectivity')

        if self.plot_reflection:
            pylab.plot(2 * self.k_space, self.k_space ** 2 * self.reflectivity.real, label='Re R exact')
            # pylab.plot(2 * self.k_space, self.k_space**2 * self.reflectivity.imag, label='Im R exact')
            self.legends.append('Re R exact')

    def diagnosis(self):
        return self._diagnosis

    def _diagnostic_hook(self, interpolator):
        assert isinstance(interpolator, ReflectivityAmplitudeInterpolation)

        refl = interpolator.reflectivity
        exact = (self.reflectivity[self.start_end[0]:self.start_end[1]])
        # total relative error in percentage
        relative_error = 1.0 / len(refl) * sum(abs((refl - exact) / exact * 100))

        self._diagnosis['iteration'].append((interpolator.iteration, interpolator.diff, relative_error))
        self._diagnosis['last_iteration'] = interpolator.iteration

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
                if is_last:
                    pylab.plot(x_space, pot, '-')
                else:
                    pylab.plot(x_space, pot, '--')

            if self.plot_phase:
                k_subspace = self.k_space[0:self.start_end[1]]
                refl = interpolator.reflectivity

                self.store_data(
                    zip(k_subspace, refl.real, refl.imag, refl.real * k_subspace ** 2,
                        refl.imag * k_subspace ** 2),
                    'phase_it_{}'.format(iteration), 'phase')

                if self.plot_phase_angle:
                    # refl = interpolator.reflcalc.refl(2 * self.k_space)
                    qmax = 1.0
                    refl = interpolator.reflcalc.refl(
                        2 * numpy.linspace(0, qmax / 2, int(self.q_precision * qmax * 1000) + 1))
                    pylab.plot(
                        numpy.linspace(0, qmax / 2, int(self.q_precision * qmax * 1000) + 1),
                        numpy.angle(refl))
                else:
                    # That's just the real part of the reflectivity amplitude
                    pylab.plot(k_subspace, interpolated_reflectivity * k_subspace ** 2, '.')

            if self.plot_reflectivity:
                pylab.yscale('log')
                R = interpolator.reflcalc.refl(2 * self.k_space)
                self.store_data(zip(2 * self.k_space, abs(R) ** 2),
                                'refl_it_{}'.format(iteration), 'reflectivity')
                pylab.plot(2 * self.k_space, R.real ** 2 + R.imag ** 2, '--')

            if self.plot_reflection:
                R = interpolator.reflcalc.refl(2 * self.k_space)
                pylab.plot(2 * self.k_space, self.k_space ** 2 * R.real, label='Re R')
                # pylab.plot(2 * self.k_space, self.k_space**2 * R.imag, label='Im R')

            self.legends.append("Iteration {}".format(iteration))

        if False:
            exact_real = (self.reflectivity[self.start_end[0]:self.start_end[1]]).real
            # relative error
            print(iteration, (interpolated_reflectivity - exact_real) / exact_real * 100)
            refl = interpolator.reflectivity
            exact = (self.reflectivity[self.start_end[0]:self.start_end[1]])
            print(1.0 / len(refl) * sum(abs((refl - exact) / exact * 100)),
                  interpolator.diff)

    def store_data(self, X, filename, header=[]):
        if self.store_path is None:
            return

        if header == 'potential':
            header = ["x [Ang]", "SLD [1/Ang^2]"]
        elif header == 'phase':
            header = ["k [1/Ang]", "Re R [1]", "Im R [1]", "k^2 * Re R [1/Ang^2]",
                      "k^2 * Im R [1/Ang^2]"]
        elif header == 'reflectivity':
            header = ["q [1/Ang]", "|R|^2 [1]"]

        if len(header) > 0:
            header = "\t".join(header)

        numpy.savetxt(self.store_path + filename + ".dat", list(X), header=header, delimiter="\t")

    def _store_before_manipulation(self):

        # Zeroth iteration reconstruction, just for having an "initial" potential and phase
        # Note that the phase is just completely zero below the critical edge (or whatever you set it to)
        if self.plot_potential:
            rec = PotentialReconstruction(self.thickness + self.offset, self.precision,
                                          cutoff=self.pot_cutoff)

            transform = FourierTransform(self.k_space, self.real, self.imag)
            potential = rec.reconstruct(transform)
            self.store_data(zip(self.plot_potential_space, potential(self.plot_potential_space)), 'pot_it_0',
                            'potential')

        if self.plot_phase:
            k_subspace = self.k_space[0:self.start_end[1]]

            self.store_data(
                zip(k_subspace, self.real[0:self.start_end[1]], self.imag[0:self.start_end[1]],
                    self.real[0:self.start_end[1]] * k_subspace ** 2,
                    self.imag[0:self.start_end[1]] * k_subspace ** 2), 'phase_it_0', 'phase')

    def run(self, constrain):

        self.setup()
        # plot the exact potential/refl/phase for comparison
        self._plot_exact()
        self._store_before_manipulation()

        rec = PotentialReconstruction(self.thickness + self.offset, self.precision,
                                      cutoff=self.pot_cutoff)
        # split the fourier transform up into two parts
        # f1 has the changing input
        # f2 has the non-changing input
        # since f2 contains much more data in general, we can save a lot of computation by
        # caching f2 and just
        # computing f1 each time.

        # self.real, self.imag contain the reflection coefficient
        f1 = FourierTransform(self.k_space[:self.start_end[1] + 1],
                              self.real[:self.start_end[1] + 1],
                              self.imag[:self.start_end[1] + 1])
        f2 = FourierTransform(self.k_space[self.start_end[1]:], self.real[self.start_end[1]:],
                              self.imag[self.start_end[1]:])

        if self.use_only_real_part:
            f1.method = f1.cosine_transform
            f2.method = f2.cosine_transform

        if self.use_only_imag_part:
            f1.method = f1.sine_transform
            f2.method = f2.sine_transform

        if self.use_only_imag_part and self.use_only_real_part:
            raise RuntimeError("Only imaginary and only real part?")


        self.real[0] = -1
        transform = FourierTransform(self.k_space, self.real, self.imag)
        transform.method = transform.cosine_transform
        #transform = UpdateableFourierTransform(f1, f2)
        reflcalc = ReflectionCalculation(None, 0, self.thickness + self.offset, 0.1)

        # TODO: change the class name
        # Reconstruct the reflection using the fourier transform at k =
        # k_interpolation_range (the low k range)
        # rec is used to reconstruct the potential
        # reflcalc is used to calculate the reflection coefficient
        # the constrain function constrains the potential
        interpolation = ReflectivityAmplitudeInterpolation(transform,
                                                           self.k_interpolation_range, rec,
                                                           reflcalc, constrain)
        if self.plot:
            interpolation.add_hook(self._plot_hook)
        if self.diagnostic_session:
            interpolation.add_hook(self._diagnostic_hook)
        if self.hook is not None:
            interpolation.add_hook(self.hook)

        solution = interpolation.interpolate(self.iterations, tolerance=self.tolerance)

        if self.show_plot:
            pylab.legend(self.legends)
            pylab.show()

        return solution


class DataRun(TestRun):
    """
    This class is a convenient interface for the retrieval of reflection data below the critical edge.

    We use a similar interface as TestRun, so that you first can test whether you can reconstruct the
    reflection with simulated data, and after that, you can easily migrate to the case with real data.

    You can use some of the parameters as in TestRun, however a few are not possible anymore:
        - q_max / k_max
        - cutoff
        - noise
        - offset
    """

    def __init__(self, reflection: Function, is_k_domain=True):
        r"""
        Tries to retrieve the reflection R(k) below the critical edge.

        Given a reflection function k -> R(k) for :math:`k \in [k_c, k_max]`, we try to compute the reflection
        R(k) for :math:`k \in [0, k_c]` using a fixed point iteration.

        The given reflection has to be equidistantly spaced (dk being constant). This will be enforced by a
        linear interpolation anyways.

        If the reflection is rather given as q -> R(q), we transform the function to k -> R(K); set
        is_k_domain to False then.

        :param reflection: Reflection function k -> R(k)
        :param is_k_domain: If the reflection is given as q -> R(q), set this to False
        """
        super(DataRun, self).__init__("")
        self.ideal_potential = NullFunction(reflection.get_domain())
        if not is_k_domain:
            reflection = reflection.scale_domain(0.5)

        self._refl_function = reflection
        self.noise = 0

    def setup(self):
        self.legends = []

        self.plot_potential_space = numpy.linspace(0, self.thickness + self.offset,
                                                   int(10 * self.precision * (self.thickness + self.offset)) + 1)

        self.k_space = self._refl_function.get_domain()
        self.k_max = self._refl_function.get_domain().max()
        self.cutoff = self._refl_function.get_domain().min()

        dk = self._refl_function.get_dx(self.k_space)
        # Extend the k_space to k=0 with dk being constant
        self.k_space = numpy.linspace(0, self.k_max, int(self.q_precision * self.k_max / dk) + 1)
        self.reflection_function = self._refl_function.remesh(self.k_space)

        self.start_end = (0, numpy.argmax(self.k_space > self.cutoff))

        self.k_interpolation_range = self.k_space[self.start_end[0]:self.start_end[1]]
        self.q_interpolation_range = 2*self.k_interpolation_range

        self.reflcalc = ReflectionCalculation(self.ideal_potential, 0, self.thickness + self.offset, 0.1)

        self.reflectivity = self.reflection_function(self.k_space)

        if self.start is None:
            self.start = 0

        self.start = array(self.start)

        self.real = self.reflectivity.real
        self.imag = self.reflectivity.imag

        try:
            self.real[self.start_end[0]:self.start_end[1]] = self.start.real
            self.imag[self.start_end[0]:self.start_end[1]] = self.start.imag
        except:
            pass

    def _plot_exact(self):
        """
        Since we do not know the exact potential/reflection, we cannot plot it ...
        :return: None
        """
        pass
