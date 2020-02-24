import numpy as np
import pylab

from dinv.function import fourier_matrix, Function, FourierTransform, InverseFourierTransform, invfourier_matrix, Lp, Derivative
from dinv.fourier import smooth

from dinv.helper import load_potential

def rect(x):

    if x <= -1:
        return 0
    if x <= -0.5:
        return 2.56
    if x <= 0:
        return -5*x
    if x <= 0.5:
        return 3*x**2 + 1
    if x <= 1:
        return 2
    return 0


    if x <= -0.5:
        return 0
    elif x < -0.25:
        return 1.2
    elif x < 0.0:
        #return 5*x+1.5
        return 2.7
    elif x < 0.5:
        return -3*x**2 * 5 + 2.7
    #elif x < 0.75:
    #    return 2.466
    else:
        return 0


x_space = np.linspace(-800, 800, 4000)

#potential = lambda x: rect(x / 400)
#potential = smooth(potential, (x_space[0], x_space[-1]), 0.1, 5)

potential = load_potential("../extrapolation/initial.dat")


f = Function.to_function(x_space, potential)

w_space = np.linspace(-2, 2, 6000)
barrier = 0.1
idx_lower = np.argmax(w_space >= -barrier)
idx_upper = np.argmin(w_space <= barrier)

w_space_lower = w_space[idx_lower: idx_upper + 1]
w_space_upper1 = w_space[0: idx_lower + 1]
w_space_upper2 = w_space[idx_upper:]
w_space_upper = np.append(w_space_upper1, w_space_upper2)


F = FourierTransform.from_function(w_space, f)
#error = Function(w_space, lambda x: np.random.normal(1, 0.1))
#F = F * error

Flower = Function.to_function(w_space_lower, F)

# Flowerm = np.dot(fourier_matrix(x_space, w_space_lower), f(x_space))


Fupper1 = Function.to_function(w_space_upper1, F)
Fupper2 = Function.to_function(w_space_upper2, F)
Fupper = Fupper1 + Fupper2

flower = InverseFourierTransform.from_function(x_space, Flower)
fupper1 = InverseFourierTransform.from_function(x_space, Fupper1)
fupper2 = InverseFourierTransform.from_function(x_space, Fupper2)

fupper = fupper1 + fupper2
ffull = InverseFourierTransform.from_function(x_space, F)

w_plot_space = np.linspace(-0.3, 0.3, 1000)
x_plot_space = np.linspace(300, 3000, 1000)

"""
mupper1 = invfourier_matrix(w_space_upper1, x_plot_space)
mupper2 = invfourier_matrix(w_space_upper2, x_plot_space)
mlower = invfourier_matrix(w_space_lower, x_plot_space)
"""

pylab.subplot(2, 2, 1)
pylab.ylabel("SLD V")
f.real.plot(x_space)

pylab.subplot(2, 2, 2)
pylab.ylabel("F[V] (Fourier transform truncated)")
Flower.real.plot(w_plot_space)
Fupper.real.plot(w_plot_space)
Flowerm = np.dot(fourier_matrix(x_space, w_space_lower), f(x_space))
pylab.plot(w_space_lower, Flowerm.real  )

# Fupper.real.plot(w_plot_space)


pylab.subplot(2, 2, 4)
pylab.ylabel("F[V] (Fourier transform) reconstructed")

#(ffull - f).real.plot(x_plot_space)
#(ffull - f).imag.plot(x_plot_space)
#(flower + fupper - ffull).real.plot(x_plot_space)
#(fupper).real.plot(x_plot_space)
#pylab.plot(x_plot_space, np.dot(invfourier_matrix(w_space_lower, x_plot_space), Flower(w_space_lower)) + np.dot(invfourier_matrix(w_space_upper1, x_plot_space), Fupper1(w_space_upper1)) + np.dot(invfourier_matrix(w_space_upper2, x_plot_space), Fupper2(w_space_upper2)))

x_known = np.linspace(190, 220, 30)
y_known = np.array(len(x_known) * [8.0239e-6])

x_constrain_space = np.append(np.linspace(-1000, 50, 1000), np.linspace(330, 1000, 1000))
#x_constrain_space = np.append(x_constrain_space, x_known)

#x_constrain_space = np.linspace(0, 500, 5000)

m1 = invfourier_matrix(w_space_upper1, x_constrain_space)
m2 = invfourier_matrix(w_space_upper2, x_constrain_space)
m = np.append(m1, m2, axis=1)
#print(np.linalg.cond(m))

ml = invfourier_matrix(w_space_lower, x_constrain_space)
Fl = Flower(w_space_lower)

b = f(x_constrain_space) - np.dot(ml, Fl)
#b[-1-len(x_known):-1] += y_known

#b[]

fupperm, residuals, rank, s = np.linalg.lstsq(m, b, rcond=1e-10)
#print(residuals)
#print(np.linalg.cond(m1))
#print(np.linalg.cond(m2))





F = np.append(Fupper1(w_space_upper1), Fupper2(w_space_upper2))

#pylab.plot(x_plot_space, np.dot(invfourier_matrix(w_space_upper1, x_plot_space), Fupper1(w_space_upper1)) + np.dot(invfourier_matrix(w_space_upper2, x_plot_space), Fupper2(w_space_upper2)))
#pylab.plot(x_plot_space, flower(x_plot_space))
#pylab.plot(x_plot_space, np.dot(m, F) + b)

nbarrier = 0.2
idx = abs(w_space_upper) < nbarrier

idx1 = w_space_upper1 > -nbarrier
idx2 = w_space_upper2 < nbarrier

idx = np.append(idx1, idx2)

fmiddle = fupperm[idx]
pylab.plot(w_space_upper[idx], fmiddle.real)
#pylab.plot(w_space_upper[idx], fmiddle.imag)
#pylab.plot(w_space_upper, fupperm)
pylab.plot(w_space_upper[idx], F[idx].real)


m1 = invfourier_matrix(w_space_upper1[idx1], x_space)
m2 = invfourier_matrix(w_space_upper2[idx2], x_space)

mp = np.append(m1, m2, axis=1)

fimproved = Function.to_function(x_space, flower(x_space) + np.dot(mp, fmiddle))






#pylab.plot(w_space_upper[idx], F[idx].imag)



#pylab.plot(x_plot_space, np.dot(m1, Fupper1(w_space_upper1)) + np.dot(m2, Fupper2(w_space_upper2)))
#pylab.plot(x_plot_space, np.dot(m, fupper(w_space_upper)))
#pylab.plot(w_space_upper, fupperm)

#pylab.plot(x_plot_space, np.dot(invfourier_matrix(w_space_upper1, x_plot_space), Fupper1(w_space_upper1)) + np.dot(invfourier_matrix(w_space_upper2, x_plot_space), Fupper2(w_space_upper2)))
#pylab.plot(x_plot_space, )

# pylab.plot(x_plot_space, 2 * np.dot(mupper1, Fupper1(w_space_upper1)))
# pylab.plot(x_plot_space, np.dot(mlower, Flower(w_space_lower)))

pylab.subplot(2, 2, 3)
pylab.ylabel("SLD (Potential) reconstructed")
#f.real.plot(x_space)
flower.real.plot(x_space)
#fupper.real.plot(x_space)

fp = Derivative.to_function(np.linspace(70, 320, 1000), fimproved)
fpp = Derivative.from_function(fp)


#fp = Derivative.to_function(flower.get_domain(), flower.get_function())
#fpp = Derivative.to_function(fp.get_domain(), fp.get_function()) * 100
(fpp).real.plot(x_space)
zeros = fpp.find_zeros()
for z in zeros:
    pylab.axvline(z)


#(flower + fupper).real.plot(x_space)
ffull.real.plot(x_space)


"""
m1 = invfourier_matrix(w_space_upper1, x_space)
m2 = invfourier_matrix(w_space_upper2, x_space)
mp = np.append(m1, m2, axis=1)

pylab.plot(x_plot_space, np.dot(ml, Fl) + np.dot(m, fupperm)  , color='green')
pylab.plot(x_space, flower(x_space) + np.dot(mp, fupperm), color='green')
"""


fimproved.real.plot()
#pylab.plot(x_space, (flower(x_space) + np.dot(mp, fmiddle)).real, color='green')


#print(Lp(f, ffull))
print(Lp(ffull, fimproved))
#print(Lp(ffull, flower))




pylab.show()
