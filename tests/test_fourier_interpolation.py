import numpy
import pylab
from math import sqrt
from dinv.glm import FourierTransform
from scipy import interpolate

# We're using the relfection
# R(k) = 2/(k**2 + 2ik - 2)
# which has the analytical inverse fourier transformation
# - 2 * sin(w) e^(-w) * heaviside(t)
# where heaviside(t) is the heaviside function with h(0) = 1

# which has the analytical scattering potential
# V(x) = 8/((1+2x)**2)

# R = lambda k: (2 / (k ** 2 + 2 * 1j * k - 2))
R = lambda k: (1 / (k ** 2 + 1j * k * sqrt(2) - 1))

RealR = lambda k: R(k).real
ImagR = lambda k: R(k).imag

# analytical fourier transform
# F = lambda w: - 2 * numpy.sin(w)*numpy.exp(-w) * numpy.heaviside(w, 1)
F = lambda w: numpy.heaviside(w, 1) * (
            -1j / sqrt(2) * (numpy.exp(-sqrt(2) / 2 * (1 + 1j) * w) - (numpy.exp(-sqrt(2) / 2 * (1 - 1j) * w))))




exact_phase = numpy.loadtxt("../../AMOR_Fe/sim/amplitude.real").T
k = (exact_phase[0] / 2)
real = exact_phase[1]
imag = -exact_phase[2]





reduce_size = 10
k_reduced = k[0:-1:reduce_size]
real_cont = interpolate.interp1d(k_reduced, real[0:-1:reduce_size], kind='cubic')
imag_cont = interpolate.interp1d(k_reduced, imag[0:-1:reduce_size], kind='cubic')


print("k_range min: full {}, reduced: {}".format(min(k), min(k_reduced)))
print("k_range max: full {}, reduced: {}".format(max(k), max(k_reduced)))


k_interp = numpy.linspace(min(k_reduced), max(k_reduced), reduce_size * len(k_reduced))
real_interp = real_cont(k_interp)
imag_interp = imag_cont(k_interp)




transform = FourierTransform(k, real, imag)
transform_interp = FourierTransform(k_interp, real_interp, imag_interp)
transform.method=transform.sine_transform
transform_interp.method=transform_interp.sine_transform


pylab.title("Real space")
transform_interp.plot_data(False)
transform.plot_data()
pylab.show()

pylab.title("Fourier space")
w_space = numpy.linspace(-1, 2000, 2000)
transform_interp.plot(w_space, show=False)
transform.plot(w_space)
pylab.show()
