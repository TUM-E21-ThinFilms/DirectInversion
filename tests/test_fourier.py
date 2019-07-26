import numpy
import pylab

from dinv.glm import FourierTransform



exact_phase = numpy.loadtxt("../../AMOR_Fe/sim/amplitude.real").T
q = exact_phase[0]
real = exact_phase[1]
imag = -exact_phase[2]

# Now sample the reflection coefficients
transform = FourierTransform(q / 2, real, imag, offset=-280)


# frequency space
w_space = numpy.linspace(-500, 500, 2000)

# Comparison to the numerical approximations
pylab.plot(w_space, [transform(w) for w in w_space], '-.')
transform.method = transform.cosine_transform
pylab.plot(w_space, [transform(w) for w in w_space])
transform.method = transform.sine_transform
pylab.plot(w_space, [transform(w) for w in w_space], '--')

pylab.legend(['fourier', 'cosine', 'sine'])



pylab.show()