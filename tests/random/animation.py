import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure(figsize=(15, 6), dpi=200)
camera = Camera(fig)


def load_pot(ext):
    z, V = np.loadtxt('store/iteration/pot_it_'+ext+'.dat', dtype=float).T
    return z, V

def load_phase(ext):
    k, re, im = np.loadtxt("store/iteration/phase_it_"+ext+'.dat', dtype=float, usecols=(0, 1, 2)).T
    return k, re, im

handles = []
legends = []


def create_plots(exts):

    zeq, Veq = np.loadtxt('store/iteration/pot_exact.dat', dtype=float).T
    keq, Req, Ieq = np.loadtxt("store/iteration/phase_exact.dat", dtype=float, usecols=(0, 1, 2)).T

    pidx = keq < 0.02


    for ext in exts:
        z, V = load_pot(str(ext))

        k, re, im = load_phase(str(ext))

        plt.subplot(121)
        plt.plot(zeq, Veq, color='red')
        plt.plot(z, V, color='blue')
        plt.legend(['Exact', 'Iteration {}'.format(ext)])
        plt.xlabel("z [A]")
        plt.ylabel("SLD [1/A^2]")

        plt.subplot(222)
        plt.plot(keq[pidx], Req[pidx], color='red')
        plt.plot(k, re, color='blue')
        plt.legend(['Re R exact', 'Re R approx'])

        plt.xlabel("k [A^-1]")
        plt.ylabel("Re R [1]")

        plt.subplot(224)
        plt.plot(keq[pidx], Ieq[pidx], color='red')
        plt.plot(k, im, color='blue')
        plt.legend(['Im R exact', 'Im R approx'], loc = 'lower right')
        plt.xlabel("k [A^-1]")
        plt.ylabel("Im R [1]")



        print("{:4.2f}".format(ext))

        camera.snap()

        """
        z, V = load_pot(to_str(ext), file)
        q, refl = load_refl(to_str(ext), file)

        plt.subplot(121)
        plt.plot(q, refl, color='blue')
        plt.xlabel("q [A]")
        plt.ylabel("log |R(q)|^2")
        plt.yscale('log')
        plt.subplot(122)
        t = plt.plot(z, V, color='blue')
        plt.xlabel("z [A]")
        plt.ylabel("SLD [1/A^2]")
        print("{:4.2f}".format(ext))
        plt.legend(t, ['{:4.2f}'.format(ext)])
        camera.snap()
        """

#create_plots(DEFAULT_FILE, np.arange(0, 30, 0.2))
#create_plots(CONVEX_INTERPOLATION_FILE, np.arange(0, 30, 0.2))
create_plots(np.arange(0, 67, 1))

animation = camera.animate(interval=100, blit=True)
animation.save('test_new.mp4', dpi=200)
