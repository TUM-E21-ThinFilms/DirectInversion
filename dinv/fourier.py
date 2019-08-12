import numpy
import pylab

import scipy.integrate
import scipy.interpolate

from numpy import array, pi


class GeneralFourierTransform(object):
    def __init__(self, function_support_range, function):
        self._range = function_support_range
        self._f = function

        self._spacing = self._range[1] - self._range[0]

        self._feval = [self._f(x) for x in self._range]
        self._cache = {}
        self.method = self.fourier

    def reset(self):
        self._cache = {}

    def __call__(self, *args, **kwargs):
        w = args[0]
        if w not in self._cache:
            self._cache[w] = self.method(w)

        return self._cache[w]

    def update(self, k_range, values):
        old = list(self._feval[0:len(values)])

        if len(values) == len(self._feval):
            self._feval = values
        else:
            self._feval[0:len(values)] = values

        self._cache = {}

        return numpy.max(abs(old - values))

    def fourier(self, w):
        return scipy.integrate.trapz(self._feval * numpy.exp(-1j * w * self._range), dx=self._spacing)

    def fourier_inverse(self, w):
        return 1 / (2 * pi) * scipy.integrate.trapz(self._feval * numpy.exp(1j * w * self._range), dx=self._spacing)


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

        if args[0] <= 0:
            return 0.0

        if not w in self._cache:
            # print("calculating at {}".format(w))
            self._cache[w] = self.method(w)

        return self._cache[w]

    def update(self, k_range, values):
        r = values.real
        i = values.imag

        old_r = list(self._r[0:len(values)])
        if len(values) == len(self._r):
            self._r = r
            self._i = i
        else:
            self._r[0:len(values)] = r
            self._i[0:len(values)] = i

        self._cache = {}

        # Todo: maybe include the imaginary part in diff, too?
        return numpy.max(abs(old_r - r))

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


class UpdateableFourierTransform(FourierTransform):
    def __init__(self, f1, f2):
        self._f1 = f1
        self._f2 = f2

    def update(self, k_range, values):
        # just update f1, which contains R(k) for small k
        return self._f1.update(k_range, values)

    def fourier_inverse(self, w):
        return self._f1.fourier_inverse(w) + self._f2.fourier_inverse(w)

    def __call__(self, *args, **kwargs):
        w = args[0]

        return self._f1(w) + self._f2(w)
