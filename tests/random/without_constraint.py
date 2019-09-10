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
q_test_space = [0.013]#, 0.014, 0.0145, 0.015]
for q in q_test_space:
    print(q)
    print("\n\n\n")

    test = TestRun("simulation-1-profile.dat")
    q_as_string = str(q).replace(".", 'd')

    test.cutoff = q
    test.noise = 0
    test.iterations = 10000
    test.tolerance = 1e-8
    test.offset = 20
    test.thickness = 340
    test.precision = 1
    test.pot_cutoff = 2
    test.use_only_real_part = True
    test.q_max = 0.5
    test.plot_every_nth = 100
    #test.store_path = 'data/test/kc/' + q_as_string + "/"
    test.q_precision = 2

    # iteration 6128
    test.start = 'exact'
    #test.start = [(-1-0j), (-0.9969192379860731-0.0778245779293796j), (-0.9876872514796058-0.15521203543693282j), (-0.9723350526861599-0.23172467592602183j), (-0.9509147106588488-0.3069236548694596j), (-0.9234999362597571-0.38036841808887895j), (-0.8901869251607519-0.4516161581862981j), (-0.8510954835308336-0.5202213012654389j), (-0.8063704688842149-0.5857350419684108j), (-0.7561835868581847-0.6477049530986034j), (-0.7007355933819727-0.7056747073728344j), (-0.6402589605644029-0.7591839640211422j), (-0.5750210731964294-0.8077684931512148j), (-0.5053280301873025-0.8509606374113192j), (-0.4315291301211496-0.8882902452012932j), (-0.35402212019847845-0.9192862544392807j), (-0.2732592796931028-0.9434791627866953j), (-0.18975438762592567-0.960404691269262j), (-0.10409058229208457-0.9696090348254547j), (-0.016929047290273225-0.9706561954215824j), (0.07098165912048958-0.9631380080115202j), (0.1587959951444067-0.9466875885203564j), (0.24556217606939157-0.9209970391406901j), (0.3302142669142325-0.885840308810713j), (0.41156713241654996-0.8411020752001539j), (0.48831639030296575-0.7868133117991352j), (0.5590462714680229-0.7231937216502965j), (0.6222490888598381-0.6507003211638903j), (0.6763606874399608-0.5700799964141988j), (0.7198164923657578-0.4824217203089617j), (0.7511321402264435-0.3892013280196081j), (0.7690105932533412-0.2923085777247161j), (0.772473533634519-0.19404337218497425j), (0.7610084502514226-0.09706670716421076j), (0.7347146446693017-0.0042937990401927415j), (0.694423083608633+0.08127645957780266j), (0.6417596475026373+0.15678940862483912j), (0.5791226382617743+0.2197974156908233j), (0.5095562980886744+0.2685191237597597j), (0.43652237069318206+0.3020472665875287j), (0.3635967104087875+0.3204600486817518j), (0.294139244577416+0.3248082294935709j), (0.23099469815777532+0.3169773033405085j), (0.17627421940481997+0.299452389614634j), (0.1312470700757582+0.27503278091258954j), (0.09634500360888823+0.246547801705901j), (0.07125914085031146+0.21661638836678523j), (0.05509618703898359+0.18747532052524327j), (0.04655885104125813+0.1608824624949658j), (0.04412149969392424+0.13808704922304207j), (0.046182034077162405+0.1198511404377536j), (0.051180918502778124+0.10650433721917446j)]
    #test.start = [(-1-0j), (-0.9873716238341044-0.1569816863693837j), (-0.9496609476754702-0.3103168080291433j), (-0.8874000171004759-0.45633707151479486j), (-0.8015055762507602-0.5913318657178316j), (-0.693327770514606-0.711530868855267j), (-0.5647259007661407-0.813094044049147j), (-0.4181789611706198-0.8921172416549689j), (-0.2569403513212944-0.9446686620455736j), (-0.0852454732537842-0.9668830679110235j), (0.09142603546850436-0.9551585430487207j), (0.2660528415868237-0.9065249177618688j), (0.42980764299510527-0.8192777453167812j), (0.5719890419267292-0.6939746952947405j), (0.6804593733085524-0.5348180920139823j), (0.7430470525935872-0.35120487942477857j), (0.7503331564209758-0.1587385657880014j), (0.6996620369665865+0.021581142236055277j), (0.5988922466380926+0.1672740313461656j), (0.46708891125944063+0.2611465056447111j), (0.3299960662024506+0.29785336805300466j), (0.21162763849994168+0.28588477524232286j), (0.12676037768229775+0.24333701274782552j), (0.07842846969701107+0.19040658628896354j), (0.060456177541694106+0.14314430546417528j), (0.06212964609423854+0.11076817379381274j), (0.07229129442227315+0.09601916454080324j), (0.08178993253141291+0.09696247842105464j)]


    #[(-1-0j), (-0.987891610083803-0.1539156965177988j), (-0.9517258427457719-0.3044192834952758j), (-0.8919880484776215-0.4480803133684385j), (-0.8095118126936875-0.5814323174166595j), (-0.7055190259187244-0.7009570937023191j), (-0.5816823356805126-0.8030738362058469j), (-0.4402167017001642-0.8841389392087976j), (-0.28400874294310985-0.9404675875962406j), (-0.11679333065595936-0.9683972150692793j), (0.056616156088289696-0.9644273481235854j), (0.23004997059509139-0.9254916152351943j), (0.39569521680216435-0.8494439866391528j), (0.5439084337896348-0.7358605648579846j), (0.6633592903877372-0.5872341768416709j), (0.7419030215104829-0.4104944309970644j), (0.7686778473562942-0.2184152700180505j), (0.7376613094684382-0.02986821141968694j), (0.6519137835813343+0.13255890166786335j), (0.5261438208483313+0.2483474713688639j), (0.38459859048898204+0.30648511446897303j), (0.25365484305644787+0.3101337367754309j), (0.15286014190572325+0.274899944237922j), (0.08981748118121585+0.22176843832117002j), (0.061007605713638134+0.16946007287038006j), (0.05632541421772976+0.13005411025177321j)]

    test.plot_potential = True
    test.plot_phase = False
    test.plot_reflectivity = False
    test.show_plot = True


    try:
        os.mkdir(os.getcwd() + "/" + test.store_path)
    except:
        pass


    test.run(constrain)
