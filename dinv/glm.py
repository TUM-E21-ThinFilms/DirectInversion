import numpy
import pylab

import scipy.integrate
import scipy.interpolate

from numpy import array, pi

from dinv.fourier import FourierTransform
from dinv.util import benchmark_start, benchmark_stop


class GLMSolver(object):
    """
        This class tries to solve the GLM (Gelfand Levitan Marchenko) integral equation
        by naivly discretizing it. We discretize it for a fixed x and let t vary
        from t = -x ... x with discretization
        step eps. We get then 2*x/eps equations for the same amount of variables

        The problem is the following

        Solve for K(x, t):

            K(x,t) + g(x+t) + \int_{-t}^{x} K(x,y) g(y + t) dy = 0 for x > t

            where g(t) = \int_{-\infty}^{\infty} R(k) \exp(-ikt) dk
                is the Fourier Transform of the reflection coefficient.

        After computing K(x,t), the desired scattering potential can be retrieved  by the
        diagonal elements at K, i.e.
            d/dx K(x, x) = 2 V(x)
                where V is the desired scattering potential.

        Note that the fourier transform is equivalent
        to either the cosine (real part of R) or sine (imaginary part of R) transform
        of the reflection coefficient since g(t) = 0 for t < 0 and since
        R(k) = \overline R(-k), where \overline R means complex conjugation. Hence the
        cosine transformation is used instead, since this is sufficient to calculate g,
        and less information is required (i.e. only the real part of R). Of course this
        might introduce some problems on real data, if the errors are too big.

        Actually, the condition g(t) = 0 for t < 0 might not be valid. For potentials with
        roughness at the top layer surface, g(0) != 0. But there might
        be a t0 < 0 such that g(t) = 0 for all t < t0?

        The GLM solver works reliably for surfaces with roughness IF the fourier transform
        is exact.
    """

    def __init__(self, fourier_transform):
        assert isinstance(fourier_transform, FourierTransform)
        self._fourier = fourier_transform

        self._cached_G = None
        self._Lprev = None
        self._Uprev = None

    def solve_init(self, max_x, eps):
        """
            Initializes the GLM solver. This sets up a huge matrix and inverts it
            The call of solve_all will use this matrix and reconstruct from it the desired
            kernel K. The special structure of the matrix is used here, i.e.  the matrix
            contains a smaller matrix which is related to the same problem, just with a
            smaller max_x.

            Hence, solving a 'big' matrix once, and then reconstructing the smaller
            matrices inside is faster than setting up multiple matrices and solving
            them independently.

            The LU decomposition is used here, so that we only have to invert the L matrix
            and the U matrix doesnt need to be inversed. In fact, this can be speeded up
            even more, by looking more deeply into the LU decomposition algorithm,
            where L is the inverse of some constructed matrices. Hence, L^-1 is actually just
            a matrix product of some known matrices, i.e.
                L = (Ln * ... * L1)^{-1}
            and later we use L^-1

        :param max_x: film thickness
        :param eps: discretization step, should be 1/int
        """
        N = 2 * int(max_x / eps)

        g = self._fourier

        A = numpy.zeros((N + 1, N + 1))

        benchmark_start()
        cache = [eps * g(eps * i) for i in range(0, N + 1)]
        benchmark_stop("Calculating Fourier Transform: {}")

        G = 1 / eps * array(cache)

        """
            This is a simple discretization of the GLM integral equation:
                
                K(x, t) + g(x+t) + int_{-t}^{x} K(x, y) g(t+y) dy = 0      for x > t
                
            using the trapezoidal rule. 
            Re-arranging with a linear shift in the integral yields
                
                K(x, t) + \int_{0}^{x+t} K(x, y-t) g(y) dy = -g(x+t)
                
            For the discretization
                
                K(x, t) will be represented by the identity matrix.
                the integral will be represented by the A matrix
                the rhs will be a vector of g-values, denoted by G
                
            and thus we derive a linear system (1 + A) x = -G, where the last entry in x 
            will be the desired solution
            K(x, x) 
        
        """
        benchmark_start()
        for i in range(1, N + 1):
            A[i][N - i:N] = cache[0:i]
        benchmark_stop("Setting up matrix: {}")

        """
            Multiply the last column vector by 0.5
            
            Note that in the trapezoidal rule, the start and end point of integration will 
            be multiply by 0.5, which is
            embodied by this action.
            Also note that - in principal - we should multiply the start point, too, 
            but since g(0) is zero, this has no  effect. The start point is located at the 
            diagonal of the transpose.
        """
        A[:, -1] *= 0.5

        A = A + numpy.identity(N + 1)

        self._cached_G = G
        self._last_N = N

        benchmark_start()
        _, L, U = scipy.linalg.lu(A)
        benchmark_stop("Calculating  LU decomposition: {}")

        benchmark_start()
        self._Lprev = numpy.linalg.inv(L)
        benchmark_stop("Inverting L: {}")

        self._Uprev = U

    def solve_all(self):
        """

        This method solves for the kernel K(x, x) now.  It uses the generated matrix from
        solve_init and reconstructs several smaller matrices, which will be used to compute
        K(x, x). In principle the smaller matrices are sub-matrices of the original one,
        by 'cutting off' the outer border of the matrix, i.e. removing top, bottom,
        left and right vectors of the matrix. Just one small adjustment has to  be made,
        i.e. multiplication by 0.5 of the last column vector. This is necessary for a
        correct implementation of the trapezoidal integration rule.

        Since only the K(x, x) is needed, we can save a whole backwards-substitution using U.
        In fact, it is possible to construct K(x, t) for t in [-x, x] completely.

        :return: Array of K(x, x) values for x in [0, eps, 2*eps, ... max_x]
        """

        Linv = self._Lprev
        U = self._Uprev
        G = self._cached_G
        # invA = self._invA

        pot = []
        benchmark_start()
        for k in range(0, self._last_N + 1, 2):

            # x = -numpy.dot(invA[-1,], G)
            # pot.append(x)

            y = -numpy.dot(Linv[-1,], G)

            # For the first entry we don't have to scale the column by something since this
            # was done by the init method but all other column vectors have to
            # "receive" this treatment
            if k == 0:
                pot.append(y / (U[-1][-1]))
            else:
                pot.append(2 * y / (U[-1][-1] + 1))

            Linv = Linv[1:-1, 1:-1]
            U = U[1:-1, 1:-1]
            G = G[0:-2]

        benchmark_stop("Calculating Kernel: {}")
        # Since we reconstruct from the "outmost" to the inner matrix, we reverse the list.
        # usually, you start with the smallest matrix and construct from that a bigger
        # matrix. But that wouldn't save computation time.
        return array(list(reversed(pot)))


class PotentialReconstruction(object):
    """
    This class reconstructs a potential given the knowledge of the fourier
    transform of the reflectivity amplitude.

    This class is just a 'convenient' wrapper. The main calculation is done in the GLMSolver.
    After the GLMSolver returns the diagonal elements of K(x, x), this class just simply
    calculates V via

        V(x) = 2 d/dx K(x, x)

    :param potential_end: The potential is reconstructed for 0 <= x <= potential_end
    :param precision: Higher precision leads to more discretization steps for V(x),
    via delta = 1/precision. Don't use too high precisions, usually anything betweent 0.25
    and 4 is fine.
    :param shift: Shifts the x-space used for the potential (Has no meaning anymore)
    :param cutoff: Cuts-off the potential at the end/start of the x-space. The parameters
    defines how many discretization steps are cut-off. Since the start and end usually
    behaves not nicely, a cut-off of 1 is fine.
    """

    def __init__(self, potential_end, precision, shift=0, cutoff=1):
        self._end = potential_end
        self._prec = precision
        self._cut = cutoff

        self._xspace, self._dx = numpy.linspace(0, self._end, self._prec * self._end + 1,
                                                retstep=True)
        self._xspace += shift

    def reconstruct(self, fourier_transform):
        eps = 1.0 / self._prec

        solver = GLMSolver(fourier_transform)

        solver.solve_init(self._end, eps)
        # solver.solve_init_new(self._end, eps)

        # in principle we're solving the diff.eq.
        #   u'' = (4*pi*rho(z) - k^2) u = (V(z) - k^2) u
        # Hence rho(z) = 1/ 4 pi  V(z)
        #
        # The factor 2 comes from the fact that V = 2 d/dx K(x,x)
        # The factor self._prec scales the d/dx correctly, since self._prec increases the
        # number of point evaluated
        # pot = self._prec / (4 * pi) * 2 * numpy.diff(solver.solve_all())

        solution = solver.solve_all()
        # solution = solver.solve_all_new()

        pot = 2 / (4 * pi) * numpy.gradient(solution, self._dx, edge_order=2)

        if self._cut > 0:
            pot[0:self._cut] = 0
            pot[-self._cut:] = 0

        potential = scipy.interpolate.interp1d(self._xspace, pot, fill_value=(0, 0),
                                               bounds_error=False)

        return potential


class ReflectionCalculation(object):
    """
    Calculates the (exact) reflectivity for a given potential in the dynamical theory.

    This is just a wrapper class, the logic for the calculation is in the refl1d package.
    Plus some nice plotting features.

    :param potential_function: A potential function (callable), to evaluate at any z in [
    z_min, z_max].
    :param z_min: min range of function. Note that V(z_min - eps) = 0 for any eps > 0 is
    assumed.
    :param z_max: max range of function. Note that V(z_min + eps) = 0 for any eps > 0 is
    assumed.
    :param dz: the discretization in the potential, kind of like a "slab" model.
    """

    def __init__(self, potential_function, z_min, z_max, dz=0.1):
        self._pot = potential_function
        self._z0 = z_min
        self._z1 = z_max
        self._dz = dz

        self._rho = None

    def __call__(self, *args, **kwargs):
        k = args[0]
        return self.refl(2 * k)

    def plot_potential(self, style='--', label=''):
        z_space = numpy.linspace(self._z0, self._z1, (self._z1 - self._z0) / self._dz + 1)
        pylab.plot(z_space, self._pot(z_space), style, label=label)

    def set_potential(self, potential_function):
        self._pot = potential_function
        self._rho = None

    def reflectivity(self, q_space):
        return array([abs(self.refl(q)) ** 2 for q in q_space])

    def refl(self, q):
        from refl1d.reflectivity import reflectivity_amplitude

        z_space = numpy.linspace(self._z0, self._z1,
                                 int((self._z1 - self._z0) / float(self._dz)) + 1)
        dz = numpy.hstack((0, numpy.diff(z_space), 0))

        if self._rho is None:
            # refl wants to have SLD*1e6
            self._rho = numpy.hstack((0, array([self._pot(z) * 1e6 for z in z_space])))

        rho = self._rho

        R = reflectivity_amplitude(kz=q / 2, depth=dz, rho=rho)

        # TODO: fix the wrong sign. Note, the 'wrong' sign is actually correct. the sign
        #  should be changed
        # in the reflectivity calculation module. but not my problem ...
        R.imag = -R.imag

        return R

    def plot_ampl(self, q_space, scale=True, style='-'):
        if scale:
            refl = [self.refl(q) * q ** 2 for q in q_space]
        else:
            refl = [self.refl(q) for q in q_space]

        Rreal = array([r.real for r in refl])

        Rimag = array([r.imag for r in refl])

        pylab.plot(q_space, Rreal, style)
        pylab.plot(q_space, Rimag, style)

        pylab.legend(["Real R", "Imag R"])
        pylab.xlabel("q")
        pylab.ylabel("q^2 * R")

    def plot_refl(self, q_space):

        refl = [abs(self.refl(q)) ** 2 for q in q_space]

        pylab.plot(q_space, refl)

        pylab.legend()
        pylab.xlabel("q_space")
        pylab.ylabel("log R")
        pylab.yscale("log")


class ReflectivityAmplitudeInterpolation(object):
    """
    This here is now the 'core' of the algorithm described in the paper.

    Whats happening here?
    We take a R(k), which is unknown for k <= k_c and interpolate/extrapolate it so that we
    get a function R(k) for 0 <= k.

    How does it work?
    Take this R(k), compute a potential from this (this potential is totally off),
    compute a reflectivity from this potential, and then just take the new reflectivity
    and update R(k) for all 0 <= k <= k_c and repeat.

    In each step of iteration, a call to hook (see set_hook) is made. The calling program
    can then update graphs, calculate further things, whatever. This is mainly used to get
    the information in each iteration available to anyone.

    :param transform: A fourier transform class, containing the reflectivity amplitude
    :param k_interpolation_range: The range where to update the reflectivity amplitude,
    usually this is [0, k_c]
    :param potential_reconstruction: An instance to a potential reconstruction class (see
    above or ba.py)
    :param reflection_calculation: An instance to a reflection calculation class (see above
    or ba.py)
    :param constraint: A callable function to incorporate additional constraints onto the
    potential. See _example_constraint for more info.
    """

    def __init__(self, transform, k_interpolation_range, potential_reconstruction,
                 reflection_calculation, constraint):
        self._transform = transform
        self._rec = potential_reconstruction
        self._constraint = constraint
        self._range = k_interpolation_range
        self._hook = None

        self.iteration = 0
        self.potential = None
        self.reflectivity = []
        self.is_last_iteration = False
        self.reflcalc = reflection_calculation

    def set_hook(self, hook):
        """
        :param hook: Must be a callable function, accepting this class as its only parameter.
        :return: None
        """
        self._hook = hook

    @staticmethod
    def _example_constrain(potential, x_space):
        """
        Constraints the potential.

        :param potential: This is a reconstructed potential, callable. An instance of
        scipy.interpolate.interp1d
        :param x_space: a range object (numpy.linspace probably) This is the values
        at which V will be evaluated
        :return: instance of scipy.interpolate.interp1d or any callable function on x_space
        """

        data = potential(x_space)
        # here, you can constrain the potential as you like ...
        # data[(x_space >= 670)] = 0e-6
        # data[(x_space > 270) & (x_space < 295)] = 4.77e-6

        # Return a callable function
        return scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)

    def interpolate(self, max_iterations, tolerance=1e-8):
        for self.iteration in range(1, max_iterations + 1):

            benchmark_start()
            self.potential = self._rec.reconstruct(self._transform)
            self.potential = self._constraint(self.potential, self._rec._xspace)
            benchmark_stop("Reconstructed Potential: {}")

            benchmark_start()
            # Calculate the reflectivity amplitude for the potential
            self.reflcalc.set_potential(self.potential)
            self.reflectivity = self.reflcalc.refl(2 * self._range)
            # Update the reflectivity amplitude for the given range.
            self.diff = self._transform.update(self._range, self.reflectivity)
            benchmark_stop("Updated amplitudes: {}")

            # The stopping criteria, max |R_k - R_{k-1} | < eps
            if self.diff < tolerance or self.iteration == max_iterations:
                # set this before _hook, so that the hook knows its the last iteration
                self.is_last_iteration = True

            # This now calls the hook and the calling program can now update graphs and do
            # something.
            if self._hook is not None:
                self._hook(self)

            if self.diff < tolerance:
                break

        return self.reflectivity


class ReflectivityAmplitudeExtrapolation(object):
    def __init__(self, correct_transform, reflectivity, k_extrapolation_range,
                 potential_reconstruction,
                 reflection_calculation, constraint):
        self.transform = correct_transform
        self.refl = reflectivity
        self.k_range = k_extrapolation_range
        self.rec_calc = potential_reconstruction
        self.refl_calc = reflection_calculation
        self.constraint = constraint

    def extrapolate(self, max_iterations, tolerance=1e-8):
        pass
