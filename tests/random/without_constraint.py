import numpy
import scipy.interpolate
import os

from dinv.helper import TestRun

numpy.random.seed(1)
numpy.set_printoptions(precision=2, linewidth=210)


def constrain(potential, x_space):
    data = potential(x_space)

    data[(x_space >= 400)] = 0e-6

    interpolation = scipy.interpolate.interp1d(x_space, data, fill_value=(0, 0), bounds_error=False)
    return interpolation

q_test_space = list(map(lambda x: round(x, 8), numpy.linspace(0.0001, 0.012, 21)))
q_test_space = 0.0005 * numpy.array(range(23, 25))
print(q_test_space)

for q in q_test_space:
    print(q)
    print("\n\n\n")

    test = TestRun("simulation-1-profile.dat")
    q_as_string = str(q).replace(".", 'd')

    test.cutoff = q
    test.noise = 0
    test.iterations = 5000
    test.tolerance = 1e-8
    test.offset = 20
    test.thickness = 350
    test.precision = 1
    test.pot_cutoff = 2
    test.use_only_real_part = False
    test.q_max = 0.5
    test.plot_every_nth = 100
    test.store_path = 'data/test/kc/' + q_as_string + "/"

    # iteration 5054
    test.start = 0
    #[(-1-0j), (-0.987891610083803-0.1539156965177988j), (-0.9517258427457719-0.3044192834952758j), (-0.8919880484776215-0.4480803133684385j), (-0.8095118126936875-0.5814323174166595j), (-0.7055190259187244-0.7009570937023191j), (-0.5816823356805126-0.8030738362058469j), (-0.4402167017001642-0.8841389392087976j), (-0.28400874294310985-0.9404675875962406j), (-0.11679333065595936-0.9683972150692793j), (0.056616156088289696-0.9644273481235854j), (0.23004997059509139-0.9254916152351943j), (0.39569521680216435-0.8494439866391528j), (0.5439084337896348-0.7358605648579846j), (0.6633592903877372-0.5872341768416709j), (0.7419030215104829-0.4104944309970644j), (0.7686778473562942-0.2184152700180505j), (0.7376613094684382-0.02986821141968694j), (0.6519137835813343+0.13255890166786335j), (0.5261438208483313+0.2483474713688639j), (0.38459859048898204+0.30648511446897303j), (0.25365484305644787+0.3101337367754309j), (0.15286014190572325+0.274899944237922j), (0.08981748118121585+0.22176843832117002j), (0.061007605713638134+0.16946007287038006j), (0.05632541421772976+0.13005411025177321j)]

    test.plot_potential = True
    test.plot_phase = False
    test.plot_reflectivity = False
    test.show_plot = False


    try:
        os.mkdir(os.getcwd() + "/" + test.store_path)
    except:
        pass


    test.run(constrain)
