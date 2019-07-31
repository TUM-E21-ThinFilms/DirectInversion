import scipy.integrate
import numpy
import math
import pylab

from numpy import array, pi
from dinv.refl import refl


class FourierTransform(object):
    def __init__(self, k_range, real_part, imaginary_part=None, offset=0, cache=None):
        # TODO: is doc still ok?
        """
        Calculates the continuous fourier transform F(f)(w)


        The following fourier transform definition is used:

            F(f)(w) := \frac{1}{2\pi} \int_{-\infty}^{\infty}{f(k) \exp{-ikw} \mathrm d k}

        Or more readable:
                                    -- oo
                          1        /
           F(f)(w) :=  -------    /   f(k) exp(-ikw) dk
                        2 pi     /
                               --  -oo

        This calculation assumes the following:
            1) k_range is equidistantly spaced, i.e. k[i] - k[i-1] = const. for all i
            2) The resulting fourier transform F(f)(w) is zero for negative frequencies, i.e. F(f)(w) = 0 for w < 0.
            3) The real part (Re f) is an even function, the imaginary part (Im f) is an odd function.
                Or in short, f is hermitian

        Some math properties:
            3) implies that its sufficient to evaluate the function only on positive k values,
                hence the fourier transform is calculated as
                        F(f)(w) = \frac{1}{\pi} \int_{0}^{\infty}{Re(f(k)) cos(kw) + Im(f(k)) sin(kw) dk}
                                = \frac{1}{\pi} \int_{0}^{\infty}{f(k) \exp(-ikw) dk}

            2) now implies that
                        F(f)(w) = \frac{2}{\pi} \int_{0}^{\infty}{Re(f(k)) cos(kw) dk}
                        F(f)(w) = \frac{2}{\pi} \int_{0}^{\infty}{Im(f(k)) sin(kw) dk}

        Hence you don't need to supply negative k-values, since these will be covered by 3). Actually supplying a
        symmetric k_range, like (-10, 10) will result in the calculated fourier transform to be scaled by 2. Hence, be
        careful with supplying the correct k_range.

        Note that even if we assume F(f)(w) = 0 for w < 0, the methods might return a non-zero value for w < 0.
        In particular, use these methods only if you are interested for F(f)(w) for w >= 0.

        The default fourier transform evaluation is {fourier_transform}. You can change this behaviour by setting the
        self.method variable to any other function. This will only affect the call to __call__. Direct calls to the
        methods will not change this behavior.

        For the numerical integration the trapezoidal integration rule is used.


        :param k_range: The given points where the function f is evaluated. Has to be equidistantly spaced. Must not
                        contain a sign change, i.e. positive and negative values. Either only positive or only negative
                        k-values. Example: numpy.linspace(0, 10, 100)
        :param real_part: The real part Re f evaluated at the given k points, i.e. f(k_range).real
        :param imaginary_part: The imaginary part Im f evaluated at the given k points, i.e. f(k_range).imag
        """

        self._k = array(k_range)
        self._k_spacing = k_range[1] - k_range[0]

        self._offset = offset

        self._r = array(real_part[0:len(self._k)])
        self._i = None

        if imaginary_part is not None:
            self._i = array(imaginary_part[0:len(self._k)])

        self.method = self.fourier_transform

        if cache is None:
            cache = {}

        self._cache = cache

    def __call__(self, *args, **kwargs):
        w = args[0] + self._offset

        # if args[0] <= 0:
        #    return 0.0

        if not w in self._cache:
            # print("calculating at {}".format(w))
            self._cache[w] = self.method(w)

        return self._cache[w]

    def fourier_transform(self, w):
        # Note here, since k_space is positive (see 2)), the factor reduces to 2/2pi.
        return 1 / pi * scipy.integrate.trapz((self._r + 1j * self._i) * numpy.exp(-1j * self._k * w),
                                              dx=self._k_spacing).real

    def cosine_transform(self, w):
        # And again, since we have to multiply the factor with 2 again, hence 2/pi is the result
        return 2 / pi * scipy.integrate.trapz(numpy.cos(self._k * w) * self._r, dx=self._k_spacing)

    def sine_transform(self, w):
        return 2 / pi * scipy.integrate.trapz(numpy.sin(self._k * w) * self._i, dx=self._k_spacing)

    def plot(self, w_range, show=True, offset=0):
        pylab.plot(w_range, [self.method(w) + offset for w in w_range])
        if show:
            pylab.show()

    def plot_data(self, show=True):
        pylab.plot(self._k, self._r)
        if self._i is not None:
            pylab.plot(self._k, self._i)
            pylab.legend(['Real', 'Imaginary'])
        if show:
            pylab.show()


# can be removed
def reconstruct_imaginary_reflection(real_reflection, first_imaginary):
    abs_real = abs(real_reflection)
    imaginary = numpy.sqrt(1 - real_reflection ** 2)
    sign = -1

    imaginary[0] *= sign

    for k in range(1, len(real_reflection) - 1):
        # local min/max and close to 1
        if (abs_real[k - 1] < abs_real[k] and abs_real[k] > abs_real[k + 1]) and abs(abs_real[k] - 1) < 0.1:
            sign = -1 * sign

        imaginary[k] *= sign

    imaginary[-1] *= sign

    if first_imaginary / imaginary[-1] < 0:
        imaginary *= -1

    return imaginary


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

        cache = [eps * g(eps * i) for i in range(0, N + 1)]

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
        for i in range(1, N + 1):
            A[i][N - i:N] = cache[0:i]

        """
            Multiply the last column vector by 0.5
            
            Note that in the trapezoidal rule, the start and end point of integration will be multiply by 0.5, which is
            embodied by this action.
            Also note that - in principal - we should multiply the start point, too, but since g(0) is zero, this has no
            effect. The start point is located at the diagonal of the transpose.
        """
        A[:, -1] *= 0.5

        A = A + numpy.identity(N + 1)

        _, L, U = scipy.linalg.lu(A)

        self._cached_G = G
        self._last_N = N

        self._Lprev = numpy.linalg.inv(L)
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

        pot = []

        for k in range(0, self._last_N + 1, 2):
            y = -numpy.dot(Linv[-1,], G)

            # For the first entry we dont have to scale the column by something since this was done by the init method
            # but all other column vectors have to "receive" this treatment
            if k == 0:
                pot.append(y / (U[-1][-1]))
            else:
                pot.append(2 * y / (U[-1][-1] + 1))

            Linv = Linv[1:-1, 1:-1]
            U = U[1:-1, 1:-1]
            G = G[0:-2]

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
    def __init__(self, potential_end, precision, cutoff=1):
        self._end = potential_end
        self._prec = precision
        self._cut = cutoff

        self._xspace, self._dx = numpy.linspace(0, self._end, self._prec * self._end + 1, retstep=True)

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
        # pot = self._prec / (4 * math.pi) * 2 * numpy.diff(solver.solve_all())

        solution = solver.solve_all()
        # solution = solver.solve_all_new()

        pot = 2 / (4 * math.pi) * numpy.gradient(solution, self._dx, edge_order=2)


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

    def plot_potential(self):
        z_space = numpy.linspace(self._z0, self._z1, (self._z1 - self._z0) / self._dz + 1)
        pylab.plot(z_space, self._pot(z_space))

    def refl(self, Q):

        from refl1d.reflectivity import reflectivity_amplitude

        z_space = numpy.linspace(self._z0, self._z1, (self._z1 - self._z0) / self._dz + 1)
        dz = numpy.hstack((0, numpy.diff(z_space), 0))

        # refl wants to have SLD*1e6
        rho = numpy.hstack((0, array([self._pot(z) * 1e6 for z in z_space])))

        return reflectivity_amplitude(kz=Q / 2, depth=dz, rho=rho)

        """
        z_space = numpy.linspace(self._z0, self._z1, (self._z1 - self._z0) / self._dz + 1)
        dz = numpy.hstack((0, numpy.diff(z_space), 0))

        # refl wants to have SLD*1e6
        rho = numpy.hstack((0, array([self._pot(z) * 1e6 for z in z_space]), 0))

        return refl(Q, dz, rho)
        """

    def plot_ampl(self, q_space, scale=True):
        if scale:
            refl = [self.refl(q) * q ** 2 for q in q_space]
        else:
            refl = [self.refl(q) for q in q_space]

        Rreal = array([r.real for r in refl])

        # TODO: imag has a sign error
        Rimag = array([-r.imag for r in refl])

        pylab.plot(q_space, Rreal)
        pylab.plot(q_space, Rimag)

        pylab.legend(["Real R", "Imag R"])
        pylab.xlabel("q_space")
        pylab.ylabel("q^2 R")

    def plot_refl(self, q_space):

        refl = [abs(self.refl(q)) ** 2 for q in q_space]

        pylab.plot(q_space, refl)

        pylab.legend()
        pylab.xlabel("q_space")
        pylab.ylabel("log R")
        pylab.yscale('log')
