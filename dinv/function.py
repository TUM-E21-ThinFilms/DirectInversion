import numpy
import scipy.interpolate

from scipy.integrate import trapz
from dinv.util import vslice

FUNCTION_INTERPOLATION_TYPE = 'linear'


class Function(object):
    """
    A mathematical function

    A function is in principle just a relation on a domain and the relation operation. Thus, every function
    here needs a domain (mesh/grid) together with a callable object (relation).

    Functions support the add, sub, mul, div and power operators:

    :Example:
    >>> f, g = Function(), Function
    >>> f + g, f + 3
    >>> f - g, f - 3
    >>> f * g, g * 3
    >>> f / g, f / 3
    >>> f ** g, f ** 3

    Composition is also possible:
    :Example:
    >>> f.apply(g) == g(f) # Use this if g is a build-in function, like abs
    >>> f.composeWith(g) == f(g)
    >>> g.composeWith(f) == g(f) # This is only possible if g is a Function

    Plotting is done
    :Example:
    >>> f.plot() # plots f on the whole domain (f.get_domain())
    >>> g.plot(domain, show=True) # plots g on domain
    """

    def __init__(self, domain, function_callable):
        """
        Creates a mathematical function based on the given domain and callable object.

        A function always needs a domain and a relation, i.e. f: X -> C
        with X being the domain, and C being the complex numbers.

        :Example:
        >>> f = Function(range(0, 10), lambda x: x**2)
        >>> g = Function(numpy.linspace(0, 10, 1000), lambda x: x**2)
        >>> h = Function(numpy.linspace(-10, 10, 200), abs)

        Function f and g have the same relation, however different domains. Function h is an example to use
        in-build function definitions.

        :param domain: list of points where the function is defined, equidistantly spaced!
        :param function_callable: callable function to evaluate this Function.
        """
        if not is_evenly_spaced_domain(self._dom):
            raise RuntimeWarning("Given domain is not equidistantly spaced")

        self._dom = numpy.array(domain)

        if not callable(function_callable):
            raise RuntimeError("function must be callable")

        if isinstance(function_callable, Function):
            function_callable = function_callable.get_function()

        self._f = function_callable

    def copy(self):
        """
        Copies and returns the copied function
        :return:
        """
        return Function(self._dom, self._f)

    def reinterpolate(self):
        """
        Uses the internal callable function, to interpolate it on the given domain.

        Useful after applying different functions to it, to increase the performance.

        :return:
        """
        self._f = to_function(self._dom, self._f)

    def shift(self, offset, domain=False):
        """
        Shifts the function to the right by offset.

        If domain is True, it additionally shifts the domain.

        :param offset:
        :param domain:
        :return:
        """
        dom = self._dom
        if domain is True:
            dom = self._dom + offset

        f = self._f
        return Function(dom, lambda x: f(x - offset))

    def apply(self, function):
        """
        Applies a function to Function. (Composition).

        In mathematical terms, let g be function, and f being the called Function. Then this method computes
        f.apply(g)(x) = g(f(x))

        :Example:
        >>> f = Function()
        >>> g = lambda x...
        >>> f.apply(g) # g(f(x))

        :param function: Callable function
        :return:
        """

        f = self._f
        return Function(self._dom, lambda x: function(f(x)))

    def composeWith(self, function):
        """
        Composition of two functions, similar to apply. However, the composition is the other way round.

        In mathematical terms, let g be function, and f being the called Function. Then this method computes
        f.composeWith(g) = f(g(x))

        :Example:
        >>> f = Function()
        >>> g = lambda x:
        >>> f.composeWith(g) # f(g(x))
        :param function:
        :return:
        """

        f = self._f
        return Function(self._dom, lambda x: f(function(x)))

    def conj(self):
        """
        Computes the complex conjugate and returns it.
        :return:
        """
        return self.apply(numpy.conj)

    def abs(self):
        """
        Computes the absolute value and returns it.
        :return:
        """
        return self.apply(abs)

    def log(self):
        """
        Computes the natural logarithm and returns it.
        :return:
        """
        return self.apply(numpy.log)

    def log10(self):
        """
        Computes the logarithm (base 10) and returns it.
        :return:
        """
        return self.apply(numpy.log10)

    def max(self):
        """
        Computes the maximum value and returns it.
        :return:
        """
        return numpy.max(self.eval())

    def min(self):
        """
        Computes the minimum value and returns it.
        :return:
        """
        return numpy.min(self.eval())

    def argmax(self):
        """
        Computes the argument which attains the maximum value
        :return:
        """
        return self.get_domain()[numpy.argmax(self.eval())]

    def argmin(self):
        """
        Computes the argument which attains the minimum value
        :return:
        """
        return self.get_domain()[numpy.argmin(self.eval())]

    def get_domain(self):
        return self._dom

    def eval(self):
        return self(self.get_domain())

    @classmethod
    def get_dx(cls, domain):
        if len(domain) < 2:
            return 0

        # Assuming equidistantly spaced domain
        return domain[1] - domain[0]

    def get_function(self):
        return self._f

    def __call__(self, x):
        return self._f(x)

    @classmethod
    def to_function(cls, domain, feval):
        return cls(domain, to_function(domain, feval))

    def remesh(self, new_mesh):
        """
        Remeshes the function using the new_mesh.

        :param new_mesh:
        :return:
        """
        return Function(new_mesh, to_function(new_mesh, self._f(new_mesh)))

    def vremesh(self, *selectors, dstart=0, dstop=0):
        """
        Remeshes the grid/domain using vslice.

        Particularly useful if you want to restrict you function

        :Example:
        >>> f.vremesh((None, None)) # does nothing in principle
        >>> f.vremesh((0, None)) # remeshes from 0 to the end of domain
        >>> f.vremesh((None, 0)) # remeshes from the start of the domain till 0

        >>> f = Function(np.linspace(-1, 1, 100), numpy.sin)
        >>> g = f.vremesh((-0.1, 0.1)) # == Function(np.linspace(-0.1, 0.1, 10), numpy.sin)

        >>> h = f.vremesh((-1.0, -0.1), (0.1, 1.0)) # remeshes the function on ([-1, -0.1] union [0.1, 1.0])
        >>> f == g + h

        :param selectors:
        :param dstart:
        :param dstop:
        :return:
        """

        return self.remesh(vslice(self.get_domain(), *selectors, dstart=dstart, dstop=dstop))

    @classmethod
    def from_function(cls, fun: 'Function'):
        return cls.to_function(fun.get_domain(), fun.get_function())

    def add(self, other):
        return self.__add__(other)

    @staticmethod
    def _is_number(other):
        return (isinstance(other, int) or
                isinstance(other, float) or
                isinstance(other, numpy.complex)
                or isinstance(other, numpy.float))

    def __add__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) + other.get_function()(x))
        if self._is_number(other):
            return Function(self._dom, lambda x: self._f(x) + other)

    def __sub__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) - other.get_function()(x))
        if self._is_number(other):
            return Function(self._dom, lambda x: self._f(x) - other)

    def __pow__(self, power):
        if isinstance(power, Function):
            return Function(self._dom, lambda x: self._f(x) ** power.get_function()(x))
        if self._is_number(power):
            return Function(self._dom, lambda x: self._f(x) ** power)

    def __mul__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) * other.get_function()(x))
        if self._is_number(other):
            return Function(self._dom, lambda x: self._f(x) * other)

    def __truediv__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) / other.get_function()(x))
        if self._is_number(other):
            return Function(self._dom, lambda x: self._f(x) / other)

    def __neg__(self):
        f = self._f
        return Function(self._dom, lambda x: -f(x))

    def plot(self, plot_space=None, show=False, real=True, **kwargs):
        import pylab
        if plot_space is None:
            plot_space = self.get_domain()

        feval = self._f(plot_space)

        lbl_re = {}
        lbl_im = {}

        try:
            lbl = kwargs.pop("label")
            if not lbl is None:
                lbl_re["label"] = lbl
                if not real:
                    lbl_re["label"] = lbl + ' (Re)'
                    lbl_im["label"] = lbl + ' (Im)'

        except KeyError:
            lbl = None

        pylab.plot(plot_space, feval.real, **kwargs, **lbl_re)

        if not real:
            pylab.plot(plot_space, feval.imag, **kwargs, **lbl_im)

        if not lbl is None:
            pylab.legend()

        if show:
            pylab.show()

    @property
    def real(self):
        return Function(self._dom, lambda x: self._f(x).real)

    @property
    def imag(self):
        return Function(self._dom, lambda x: self._f(x).imag)

    def find_zeros(self):
        f0 = self._f(self._dom[0])
        roots = []
        for el in self._dom:
            fn = self._f(el)
            if (f0.real * fn.real) < 0:
                # there was a change in sign.
                roots.append(el)
                f0 = fn
        return roots


class Antiderivative(Function):
    @classmethod
    def to_function(cls, domain, feval, C=0):
        r"""
        Returns the integral function starting from the first element of domain, i.e.
        ::math..
            F(x) = \int_{x0}^{x} f(z) dz

        where x0 = domain[0] and f is the given function (feval).
        :param domain:
        :param feval:
        :return:
        """
        # TODO: test
        dx = cls.get_dx(domain)
        Feval = scipy.integrate.cumtrapz(y=evaluate(domain, feval), dx=dx, initial=0) + C

        return Function.to_function(domain, Feval)

    @classmethod
    def from_function(cls, fun: Function, x0=None, C=0):
        if x0 is None:
            return cls.to_function(fun.get_domain(), fun, C=C)
        else:
            F = cls.from_function(fun)
            return F - F(x0)

    @classmethod
    def integral(cls, fun: Function, x0=None, x1=None):
        r"""
        Calculates the definite integral of the Function fun.

        If x0 or x1 are given, the function is re-meshed at these points, and thus this function returns
        ::math..
            \int_{x_0}^{x_1} f(x) dx

        If x0 and x1 are both None, the integral is evaluated over the whole domain
        x0, x1 = domain[0], domain[-1], i.e.
        ::math..
            \int_{x_0}^{x_1} f(x) dx = \int f(x) dx

        :param fun:
        :param x0: lower bound of the integral limit or None
        :param x1: upper bound of the integral limit or None
        :return: definite integral value (Not a function!)
        """
        if not any(numpy.array([x0, x1]) is None):
            fun = fun.vremesh((x0, x1))

        dx = cls.get_dx(fun.get_domain())
        return scipy.integrate.trapz(fun.eval(), dx=dx)


class Derivative(Function):
    @classmethod
    def to_function(cls, domain, feval):
        # TODO: test
        feval = evaluate(domain, feval)
        fprime = numpy.gradient(feval, cls.get_dx(domain), edge_order=2)
        return Function.to_function(domain, fprime)

    @classmethod
    def from_function(cls, fun: Function):
        return cls.to_function(fun.get_domain(), fun)

class FourierTransform(Function):
    @classmethod
    def to_function(cls, domain, feval, frequency_domain):
        w = numpy.array(frequency_domain).reshape((len(frequency_domain), 1))
        domain = numpy.array(domain).reshape((1, len(domain)))
        feval = evaluate(domain, feval)
        F = trapz(feval * numpy.exp(- 1j * numpy.dot(w, domain)), x=domain)
        return Function.to_function(frequency_domain, F)

        """
            dx = cls.get_dx(domain)
            w = numpy.array(frequency_domain).reshape((len(frequency_domain), 1))
            domain = numpy.array(domain).reshape((1, len(domain)))
            feval = evaluate(domain, feval)
            F = trapz(feval * numpy.exp(- 1j * numpy.dot(w, domain)), dx=dx)
            return Function.to_function(frequency_domain, F)
        """

    @classmethod
    def from_function(cls, frequency_domain, fun: Function):
        return cls.to_function(fun.get_domain(), fun.get_function(), frequency_domain)


class InverseFourierTransform(Function):
    @classmethod
    def to_function(cls, frequency_domain, feval, x_domain):
        dx = cls.get_dx(frequency_domain)
        w = numpy.array(x_domain).reshape((len(x_domain), 1))
        domain = numpy.array(frequency_domain).reshape((1, len(frequency_domain)))
        feval = evaluate(domain, feval)
        F = 1 / (2 * numpy.pi) * trapz(feval * numpy.exp(1j * numpy.dot(w, domain)), dx=dx)
        return Function.to_function(x_domain, F)

    @classmethod
    def from_function(cls, x_domain, fun: Function):
        return cls.to_function(fun.get_domain(), fun.get_function(), x_domain)


class InverseCosineTransform(InverseFourierTransform):
    @classmethod
    def to_function(cls, frequency_domain, feval, x_domain):
        dx = cls.get_dx(frequency_domain)
        w = numpy.array(x_domain).reshape((len(x_domain), 1))
        domain = numpy.array(frequency_domain).reshape((1, len(frequency_domain)))
        feval = evaluate(domain, feval)

        F = 1 / (numpy.pi) * trapz(feval * numpy.cos(numpy.dot(w, domain)), dx=dx)
        return Function.to_function(x_domain, F)


class CosineTransform(FourierTransform):
    @classmethod
    def to_function(cls, frequency_domain, feval, x_domain):
        dx = cls.get_dx(frequency_domain)
        w = numpy.array(x_domain).reshape((len(x_domain), 1))
        domain = numpy.array(frequency_domain).reshape((1, len(frequency_domain)))
        feval = evaluate(domain, feval)

        F = trapz(feval * numpy.cos(numpy.dot(w, domain)), dx=dx)
        return Function.to_function(x_domain, F)


class AbstractConvolution(Function):
    @classmethod
    def get_kernel(cls, fun: Function, **kwargs):
        raise RuntimeError("Not implemented")

    @classmethod
    def from_function(cls, fun: Function, **kwargs):
        domain = fun.get_domain()
        conv = numpy.convolve(fun(domain), cls.get_kernel(fun, **kwargs), mode='same')
        return Function.to_function(domain, conv)


class GaussianSmoothing(AbstractConvolution):
    @classmethod
    def get_kernel(cls, fun: Function, **kwargs):
        """
        Returns the gaussian kernel, centered at \mu = 0 and variance = sigma

        :param fun: Function to apply the gaussian kernel (needed for spacing purposes)
        :param kwargs: possible variable: sigma
        :return:
        """
        sigma = kwargs.get('sigma', 1.0)
        spacing = fun.get_dx(fun.get_domain())
        width = numpy.arange(-5 * sigma, 5 * sigma, spacing)
        kernel = 1.0 / numpy.sqrt(2 * numpy.pi * sigma ** 2) * numpy.exp(
            -numpy.square(width / sigma) / 2.0) * spacing
        return kernel


def evaluate(domain, function):
    """
    Evaluates a function on its domain.
    If function is callable, it's simply evaluated using the callable
    If function is a numpy array, then its simply returned (assuming it was already evaluated elsewhere)
    Otherwise a RuntimeError is raised
    :param domain: numpy.array
    :param function: callable/np.array
    :raise RuntimeError: Unknown type of function given
    :return:
    """
    if callable(function):
        return numpy.array([function(x) for x in domain])
    elif isinstance(function, numpy.ndarray) and len(domain) == len(function):
        return function
    elif isinstance(function, list) and len(domain) == len(function):
        return numpy.array(function)
    else:
        raise RuntimeError("Cannot evaluate, unknown type")


def set_interpolation_type(interpolation_type):
    """
    Sets the interpolation type used for all Functions

    :param interpolation_type: "linear", "cubic", "quadratic", etc.. see scipy.interpolation.interp1d
    :return: previous interpolation type
    """
    global FUNCTION_INTERPOLATION_TYPE
    previous_type = FUNCTION_INTERPOLATION_TYPE
    FUNCTION_INTERPOLATION_TYPE = interpolation_type
    return previous_type


def is_evenly_spaced_domain(domain):
    """
    Checks whether the given domain (list) is evenly (equdistantly) spcaed

    :param domain: numpy.array. domain to check
    :return: boolean
    """

    diff = numpy.diff(domain)

    if numpy.all(numpy.isclose(diff - diff[0], numpy.zeros(len(diff)))):
        return True

    return False


def to_function(x_space, feval, interpolation=None, to_zero=True):
    if interpolation is None:
        global FUNCTION_INTERPOLATION_TYPE
        interpolation = FUNCTION_INTERPOLATION_TYPE

    if callable(feval):
        feval = numpy.array([feval(x) for x in x_space])

    if len(x_space) == 0:
        return lambda x: 0

    feval = numpy.array(feval)

    if to_zero:
        fill = (0, 0)
    else:
        fill = numpy.nan

    real = scipy.interpolate.interp1d(x_space, feval.real, fill_value=fill, bounds_error=False,
                                      kind=interpolation)
    imag = scipy.interpolate.interp1d(x_space, feval.imag, fill_value=fill, bounds_error=False,
                                      kind=interpolation)

    return lambda x: real(x) + 1j * imag(x)


def fourier_matrix(t_space, f_space):
    # Important, otherwise t_space changes outside the function
    t_space = numpy.array(t_space)

    dt = t_space[1] - t_space[0]

    if dt == 0:
        raise RuntimeError("Given t_space has an incorrect format")

    f = numpy.array(f_space).reshape((len(f_space), 1))
    t = numpy.array(t_space).reshape((1, len(t_space)))

    f_t_matrix = numpy.dot(f, t)
    e_matrix = numpy.exp(-1j * f_t_matrix)

    # this is kinda the weighting of the trapezoidal integration rule
    e_matrix[:, 0] *= 0.5
    e_matrix[:, -1] *= 0.5

    return e_matrix * dt


def invfourier_matrix(f_space, t_space):
    r"""
    Returns a matrix representing a inverse fourier transform.

    Let :math:`R \colon f_space \to RR` (RR being the real numbers) be a function. The inverse fourier
    transform is then defined as
    ..math::
        F^{-1}[R](t) = \int_{RR}{e^{itf} R(f) df}

    This can be represented by a matrix multiplication. For this, assume you want to evaluate the inverse
    fourier transform :math:`F^-1[R] at t \in t_space` and you know the function R at :math:`f \in f_space`.
    Then this function returns a matrix A with the following properties
    ..math::
        F^{-1}[R] = A * R

    with * denoting the usual matrix-vector-product (i.e. numpy.dot) and the resulting vector will be
    evaluated exactly at :math:`t \in t_space`, i.e. :math:`(A*R)[i] = F^{-1}[R](t_space[i])`

    ..note::
        More details in the case of reflectometry: The resulting matrix A looks like the following
        with k (wavevector transfer) being f and x (depth) being t:

                k
            ----------▶
            |    ikx
          x |   e
            |
            ▼

        Thus :math:`A * R(k) = F[R](x) = V(x)`

    :Example:
    >>> x_space = numpy.linspace(0, 200, 200)
    >>> k_space = numpy.linspace(-0.1, 0.1, 400)
    >>> # R ... being the reflection evaluated at k_space, i.e. R = R_function(k_space)
    >>> A = invfourier_matrix(k_space, x_space)
    >>> V = numpy.dot(A, R) # V being evaluated at x_space, i.e. V = V_function(x_space)

    with R, V being the reflection and potential function, respectively.

    :param f_space: frequency space. The space where the function to do the inverse fourier transform is
    known. Has to be equidistantly spaced.
    :param t_space: time space. The space where the inverse fourier transform shall be evaluated.
    :return: A matrix representing the inverse fourier transform
    :raises:
        RuntimeError: If the spacing of f_space is zero, i.e. delta f_space = 0, df = 0.
    """

    # Important, otherwise f_space changes outside the function
    f_space = numpy.array(f_space)

    df = f_space[1] - f_space[0]
    if df == 0:
        raise RuntimeError("Given f_space has an incorrect format")

    f = numpy.array(f_space).reshape((1, len(f_space)))
    t = numpy.array(t_space).reshape((len(t_space), 1))

    t_f_matrix = numpy.dot(t, f)
    e_matrix = numpy.exp(1j * t_f_matrix)

    # this is kinda the weighting of the trapezoidal integration rule
    e_matrix[:, 0] *= 0.5
    e_matrix[:, -1] *= 0.5

    return 1 / (2 * numpy.pi) * e_matrix * df


def Lp(f1: Function, f2: Function, p=2, domain=None):
    if domain is None:
        domain = f1.get_domain()

    lp = numpy.power(numpy.abs(f1(domain) - f2(domain)), p)

    return numpy.power(scipy.integrate.trapz(lp, dx=f1.get_dx(domain)), 1 / p)
