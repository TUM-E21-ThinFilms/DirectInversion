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
        by naivly discretizing it. We discretize it for a fixed x and let t vary from t = -x ... x with discretization
        step eps. We get then 2*x/eps equations for the same amount of variables

        The problem is the following

        Solve for K(x, t):

            K(x,t) + g(x+t) + \int_{-t}^{x} K(x,y) g(y + t) dy = 0 for x > t

            where g(t) = \int_{-\infty}^{\infty} R(k) \exp(-ikt) dk
                is the Fourier Transform of the reflection coefficient.

        After computing K(x,t), the desired scattering potential can be retrieved by the diagonal elements at K, i.e.
            d/dx K(x, x) = 2 V(x)
                where V is the desired scattering potential.

        Note that the fourier transform is equivalent
        to either the cosine (real part of R) or sine (imaginary part of R) transform of the reflection coefficient
        since g(t) = 0 for t < 0 and since R(k) = \overline R(-k), where \overline R means complex conjugation.
        Hence the cosine transformation is used instead, since this is sufficient to calculate g, and less information
        is required (i.e. only the real part of R). Of course this might introduce some problems on real data, if
        the errors are too big.

        Actually, the condition g(t) = 0 for t < 0 might not be valid. For potentials with roughness at the top layer
        surface, g(0) != 0. But there might be a t0 < 0 such that g(t) = 0 for all t < t0?


        Hence, this GLM solver does not reliably work for films having a roughness at the top layer (film -> surrounding)
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
            kernel K. The special structure of the matrix is used here, i.e. the matrix contains a
            smaller matrix which is related to the same problem, just with a smaller max_x.

            Hence, solving a 'big' matrix once, and then reconstructing the smaller matrices inside is
            faster than setting up multiple matrices and solving them independently.

            The LU decomposition is used here, so that we only have to invert the L matrix and the U matrix doesnt
            need to be inversed. In fact, this can be speeded up even more, by looking more deeply into the LU
            decomposition algorithm, where L is the inverse of some constructed matrices. Hence, L^-1 is actually just
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
                
            using the trapezoidal rule. Re-arranging with a linear shift in the integral yields
                
                K(x, t) + \int_{0}^{x+t} K(x, y-t) g(y) dy = -g(x+t)
                
            For the discretization
                
                K(x, t) will be represented by the identity matrix.
                the integral will be represented by the A matrix
                the rhs will be a vector of g-values, denoted by G
                
            and thus we derive a linear system (1 + A) x = -G, where the last entry in x will be the desired solution
            K(x, x) 
        
        """
        benchmark_start()
        for i in range(1, N + 1):
            A[i][N - i:N] = cache[0:i]
        benchmark_stop("Setting up matrix: {}")

        """
            Multiply the last column vector by 0.5
            
            Note that in the trapezoidal rule, the start and end point of integration will be multiply by 0.5, which is
            embodied by this action.
            Also note that - in principal - we should multiply the start point, too, but since g(0) is zero, this has no
            effect. The start point is located at the diagonal of the transpose.
        """
        A[:, -1] *= 0.5

        A = A + numpy.identity(N + 1)

        self._cached_G = G
        self._last_N = N

        # self._invA = numpy.linalg.inv(A)
        benchmark_start()
        _, L, U = scipy.linalg.lu(A)
        benchmark_stop("Calculating  LU decomposition: {}")

        benchmark_start()
        self._Lprev = numpy.linalg.inv(L)
        benchmark_stop("Inverting L: {}")

        self._Uprev = U

    def solve_all(self):
        """

        This method solves for the kernel K(x, x) now. It uses the generated matrix from solve_init and reconstructs
        several smaller matrices, which will be used to compute K(x, x).
        In principle the smaller matrices are sub-matrices of the original one, by 'cutting off' the outer border of
        the matrix, i.e. removing top, bottom, left and right vectors of the matrix. Just one small adjustment has to
        be made, i.e. multiplication by 0.5 of the last column vector. This is necessary for a correct implementation
        of the trapezoidal integration rule.

        Since only the K(x, x) is needed, we can save a whole backwards-substitution using U. In fact, it is possible
        to construct K(x, t) for t in [-x, x] completely.

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

            # For the first entry we dont have to scale the column by something since this was done by the init method
            # but all other column vectors have to "receive" this treatment
            if k == 0:
                pot.append(y / (U[-1][-1]))
            else:
                pot.append(2 * y / (U[-1][-1] + 1))

            Linv = Linv[1:-1, 1:-1]
            U = U[1:-1, 1:-1]

            # invA = invA[1:-1, 1:-1]
            G = G[0:-2]

        benchmark_stop("Calculating Kernel: {}")
        # Since we reconstruct from the "outmost" to the inner matrix, we reverse the list.
        # usually, you start with the smallest matrix and construct from that a bigger matrix. But that wouldnt save
        # computation time.
        return array(list(reversed(pot)))

    def solve_new(self, x, precision, max_thickness):

        N = 2 * max_thickness * precision
        eps = 1.0 / precision

        g = self._fourier

        cache = array([g(2 * (x + i * eps)) for i in range(0, N + 1)])
        G = array(list(cache))
        cache *= eps

        A = numpy.zeros((N + 1, N + 1))

        for i in range(0, N + 1):
            A[i][0:N - i] = cache[i:N]

        A = numpy.identity(N + 1) + A

        return numpy.linalg.solve(A, -G)

    def solve_init_new(self, max_thickness, eps):
        self._L = max_thickness
        self._prec = int(1.0 / eps)

    def solve_all_new(self, x_range=None):

        if x_range is None:
            x_range = numpy.linspace(0, self._L, self._prec * self._L + 1)

        pot = []

        for x in x_range:
            sol = self.solve_new(x, precision=self._prec, max_thickness=self._L)
            pot.append(sol[0])

        return pot


class PotentialReconstruction(object):
    def __init__(self, potential_end, precision, shift=0, cutoff=1):
        self._end = potential_end
        self._prec = precision
        self._cut = cutoff

        self._xspace, self._dx = numpy.linspace(0, self._end, self._prec * self._end + 1, retstep=True)
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
        # The factor self._prec scales the d/dx correctly, since self._prec increases the number of point evaluated
        # pot = self._prec / (4 * pi) * 2 * numpy.diff(solver.solve_all())

        solution = solver.solve_all()
        # solution = solver.solve_all_new()

        pot = 2 / (4 * pi) * numpy.gradient(solution, self._dx, edge_order=2)

        if self._cut > 0:
            pot[0:self._cut] = 0
            pot[-self._cut:] = 0

        potential = scipy.interpolate.interp1d(self._xspace, pot, fill_value=(0, 0), bounds_error=False)

        return potential


class ReflectionCalculation(object):
    def __init__(self, potential_function, z_min, z_max, dz):
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

    def refl(self, q):

        from refl1d.reflectivity import reflectivity_amplitude

        z_space = numpy.linspace(self._z0, self._z1, (self._z1 - self._z0) / self._dz + 1)
        dz = numpy.hstack((0, numpy.diff(z_space), 0))

        if self._rho is None:
            # refl wants to have SLD*1e6
            self._rho = numpy.hstack((0, array([self._pot(z) * 1e6 for z in z_space])))

        rho = self._rho

        R = reflectivity_amplitude(kz=q / 2, depth=dz, rho=rho)

        # TODO: fix the wrong sign. Note, the 'wrong' sign is actually correct. the sign should be changed
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
    def __init__(self, transform, k_interpolation_range, potential_reconstruction, reflection_calculation, constraint):
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
        self._hook = hook

    def interpolate(self, max_iterations, tolerance=1e-8):
        for self.iteration in range(1, max_iterations + 1):
            benchmark_start()
            self.potential = self._rec.reconstruct(self._transform)
            self.potential = self._constraint(self.potential, self._rec._xspace)
            benchmark_stop("Reconstructed Potential: {}")

            benchmark_start()
            # Use the new reflection coefficient for small k-values and re-do the inversion ...
            self.reflcalc.set_potential(self.potential)
            self.reflectivity = self.reflcalc.refl(2 * self._range)
            diff = self._transform.update(self._range, self.reflectivity)
            benchmark_stop("Updated amplitudes: {}")

            if diff < tolerance or self.iteration == max_iterations:
                # set this before _hook, so that the hook knows its the last iteration
                self.is_last_iteration = True

            if self._hook is not None:
                self._hook(self)

            if diff < tolerance:
                break
