import numpy as np
from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure(figsize=(15, 6), dpi=200)
camera = Camera(fig)


DEFAULT_FILE = ''
CONVEX_INTERPOLATION_FILE = 'conv_'
NEW_FILE = 'new_'

def to_str(fl):
    fl = round(fl, 4)
    return str(fl).replace('.', 'd')

def load_refl(ext, file):
    q, refl = np.loadtxt('calc/'+file+'refl_'+ext+'.dat').T
    return q, refl

def load_pot(ext, file):
    z, V = np.loadtxt('calc/'+file+'pot_'+ext+'.dat', dtype=float).T
    return z, V



handles = []
legends = []


def create_plots(file, exts):
    for ext in exts:
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

#create_plots(DEFAULT_FILE, np.arange(0, 30, 0.2))
#create_plots(CONVEX_INTERPOLATION_FILE, np.arange(0, 30, 0.2))
create_plots(NEW_FILE, np.arange(0, 90, 0.5))


#plt.xlegend("z [A]")
#plt.ylegend("SLD")

animation = camera.animate(interval=50, blit=True)
#animation.save('refl.gif', writer = 'imagemagick', dpi=100)
animation.save('test_new.mp4', dpi=200)
