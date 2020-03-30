import numpy
import scipy.interpolate
from scipy.integrate import trapz


class Function(object):
    def __init__(self, domain, function_callable):
        self._dom = numpy.array(domain)

        if not callable(function_callable):
            raise RuntimeError("function must be callable")

        if isinstance(function_callable, Function):
            function_callable = function_callable.get_function()

        self._f = function_callable

    def shift(self, offset, domain=False):
        f = self._f
        self._f = lambda x: f(x - offset)

        if domain is True:
            self._dom = self._dom + offset

    def get_domain(self):
        return self._dom

    def extend_domain(self, dx_steps=1):
        domain = self.get_domain()
        dx = self.get_dx(domain)

        pre = -dx * numpy.array(range(dx_steps, 0, -1)) + domain[0]
        post = dx * numpy.array(range(1, dx_steps + 1, 1)) + domain[-1]

        new_domain = numpy.append(pre, domain)
        new_domain = numpy.append(new_domain, post)

        self._dom = new_domain

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

    def remesh(self, new_mesh, return_function=False):
        f = to_function(new_mesh, self._f(new_mesh))
        dom = new_mesh

        if not return_function:
            self._f = f
            self._dom = dom
        else:
            return Function(dom, f)

    @classmethod
    def from_function(cls, fun: 'Function'):
        return cls.to_function(fun.get_domain(), fun.get_function())

    def __add__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) + other.get_function()(x))
        if isinstance(other, int) or isinstance(other, float):
            return Function(self._dom, lambda x: self._f(x) + other)

    def __sub__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) - other.get_function()(x))
        if isinstance(other, int) or isinstance(other, float):
            return Function(self._dom, lambda x: self._f(x) - other)

    def __mul__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) * other.get_function()(x))
        if isinstance(other, int) or isinstance(other, float):
            return Function(self._dom, lambda x: self._f(x) * other)

    def __truediv__(self, other):
        if isinstance(other, Function):
            return Function(self._dom, lambda x: self._f(x) / other.get_function()(x))
        if isinstance(other, int) or isinstance(other, float):
            return Function(self._dom, lambda x: self._f(x) / other)

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
    def to_function(cls, domain, feval):
        # TODO: test
        dx = cls.get_dx(domain)
        feval = evaluate(domain, feval)
        # not the most efficient way, but it's ok ...
        Feval = numpy.array([trapz(feval[0:idx], dx=dx) for idx in range(0, len(feval))])
        return Function.to_function(domain, Feval)


class Derivative(Function):
    @classmethod
    def to_function(cls, domain, feval):
        # TODO: test
        feval = evaluate(domain, feval)
        dx = cls.get_dx(domain)
        fprime = numpy.gradient(feval, dx, edge_order=2)
        return Function.to_function(domain, fprime)


class FourierTransform(Function):
    @classmethod
    def to_function(cls, domain, feval, frequency_domain):
        dx = cls.get_dx(domain)
        w = numpy.array(frequency_domain).reshape((len(frequency_domain), 1))
        domain = numpy.array(domain).reshape((1, len(domain)))
        feval = evaluate(domain, feval)
        F = trapz(feval * numpy.exp(- 1j * numpy.dot(w, domain)), dx=dx)
        return Function.to_function(frequency_domain, F)

    @classmethod
    def from_function(cls, frequency_domain, fun: Function):
        # fun.extend_domain()
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
        # fun.extend_domain()
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
        sigma = kwargs.get('sigma', 1.0)
        spacing = fun.get_dx(fun.get_domain())
        width = numpy.arange(-5 * sigma, 5 * sigma, spacing)
        kernel = 1.0 / numpy.sqrt(2 * numpy.pi * sigma ** 2) * numpy.exp(
            -numpy.square(width / sigma) / 2.0) * spacing
        return kernel


def evaluate(x_space, function):
    if callable(function):
        return numpy.array([function(x) for x in x_space])
    elif isinstance(function, numpy.ndarray) and len(x_space) == len(function):
        return function
    elif isinstance(function, list) and len(x_space) == len(function):
        return numpy.array(function)
    else:
        raise RuntimeError("Cannot evaluate, unknown type")


def to_function(x_space, feval, interpolation='linear', to_zero=True):
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

    :param f_space: frequency space. The space where the function to do the inverse fourier transform is known
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
