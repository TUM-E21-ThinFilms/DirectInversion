import numpy
import pylab

from dinv.glm import FourierTransform

from numpy import sqrt

exact_phase = numpy.loadtxt("data/amplitude.real").T
q = exact_phase[0]
real = exact_phase[1]
imag = exact_phase[2]


# Now sample the reflection coefficients
transform = FourierTransform(q / 2, real, imag, offset=0)


# frequency space
#w_space = numpy.linspace(70*2, 80*2, 10000)
w_space = numpy.linspace(-50, 500, 10000)


# Comparison to the numerical approximations
pylab.plot(w_space, [transform(w) for w in w_space], '-.')
pylab.plot(w_space, [transform.cosine_transform(w) for w in w_space])
pylab.plot(w_space, [transform.sine_transform(w) for w in w_space], '--')

pylab.axvline(color='black')
pylab.axvline(2*75, color='black')
pylab.axhline(color='black')

pylab.legend(['fourier', 'cosine', 'sine'])



pylab.show()
exit(1)


R = lambda k: (1 / (k ** 2 + 1j * k * sqrt(2) - 1))

RealR = lambda k: R(k).real
ImagR = lambda k: R(k).imag

q = numpy.linspace(0, 1000, 10000)

# analytical fourier transform
# F = lambda w: - 2 * numpy.sin(w)*numpy.exp(-w) * numpy.heaviside(w, 1)
F = lambda w: numpy.heaviside(w, 1) * (
            -1j / sqrt(2) * (numpy.exp(-sqrt(2) / 2 * (1 + 1j) * w) - (numpy.exp(-sqrt(2) / 2 * (1 - 1j) * w))))

transform = FourierTransform(q/2, RealR(q/2), ImagR(q/2))
w_space = numpy.linspace(-1, 100, 5000)
pylab.plot(w_space, [transform(w).real for w in w_space], '-.')
pylab.plot(w_space, [F(w) for w in w_space])
pylab.show()