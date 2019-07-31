from glob import glob
from refl1d.names import *
from refl1d.model import Repeat
from itertools import chain
import bumps.bounds
import copy
import pylab


# ======= User options =======

num_iron_layers = 5

# ======= User options =======
import numpy as np

data = {
    filename[-9:-4]: load4(filename, name=filename[-9:-4], columns="Q R dQ dR",
                           radiation="neutron", L=0.1, dL=0.00001 / 4, simulation_range=np.linspace(0, 10, 10000))
    # filename[-9:-4]: load4(filename, name=filename[-9:-4], columns="Q R dQ dR", radiation="neutron", L=4, dL=0.0004 / 4, data_range=(6, None),  back_reflectivity=True, simulation_range=np.linspace(0.0001, 0.5, 2000))
    for filename in glob("data/fecu_d*.Rqz")
}

# Contains 00, 01, 02, ... 18
measurement_names = np.sort(np.unique([name[0:2] for name in data.keys()]))

silicon = si = Material("Si", name="Si (substrate)")
copper = cu = Material("Cu")
anything = SLD(rho=1)
silicon_copper = SLD(rho=6, name='Si/Cu')
iron_fcc = fe_fcc = Material("Fe")

air_slab = Slab(air)
si_slab = Slab(silicon, 0, 0)  # Substrate

si_slab2 = Slab(silicon, 10, 0)
cu_slab = Slab(copper, 450, 0)  # Copper layer on substrate

iron_fcc_slab = num_iron_layers * [None]

for i in range(0, num_iron_layers):
    iron_fcc_slab[i] = Slab(Material("Fe", name="Fe fcc {}".format(i + 1)), 15 * (i + 1), 0,
                            name="Fe fcc {}".format(i + 1))
    # iron_fcc_slab[i].magnetism = Magnetism(rhoM=0, interfaceM=None)
    # iron_fcc_slab[i].thickness.value = 40*(i+1)




sur2 = Slab(anything)
sur3 = Slab(gold)
au_slab = Slab(gold, 15, 0)
iron_slab = Slab(iron_fcc, 20, 0)

test_material = SLD(rho=7)
test_material2 = SLD(rho=6)
test_material3 = SLD(rho=4.662)


"""
sur1 = Slab(air, 0, 0)
t1 = Slab(test_material, 10, 0)
t2 = Slab(test_material2, 20, 0)
t3 = Slab(test_material3, 15, 0)
"""


sur1 = Slab(air, 0, 2)
t1 = Slab(test_material, 20, 3)
t2 = Slab(test_material2, 10, 1)
t3 = Slab(test_material3, 50, 3)

samples = {
    # '01': Stack([si_slab, cu_slab, au_slab, qsur1]),
    # '02': Stack([si_slab, cu_merge_slab, au_slab, sur2]),
    # '03': Stack([si_slab, cu_merge_slab, au_slab, sur3]),
    # 'free': Stack([sur1, cu_merge_slab, au_slab, sur1]),
    # 'free_rev': Stack([sur1, au_slab, cu_merge_slab, sur1]),
    #'free': Stack([sur1, test_slab, test_slab2, sur1]),
    'free2': Stack([si_slab, cu_slab, iron_fcc_slab[0], air]),
    'free': Stack([sur1, 2*Stack([t1, t2]), t3, sur1]),
    '01': Stack([si_slab, cu_slab, iron_fcc_slab[1], air_slab]),
    '02': Stack([si_slab, cu_slab, iron_fcc_slab[2], air_slab]),
    '03': Stack([si_slab, cu_slab, iron_fcc_slab[3], air_slab]),
    '04': Stack([si_slab, cu_slab, iron_fcc_slab[4], air_slab]),
}

# iron_fcc_slab[0].thickness.range(0, 20)
t2.material.rho.pmp(1)
au_slab.material.density.pmp(20)
cu_slab.material.density.pmp(20)

data['01_dn'] = copy.copy(data['04_dn'])
data['02_dn'] = copy.copy(data['04_dn'])
data['03_dn'] = copy.copy(data['04_dn'])

probes = {
    name: PolarizedNeutronProbe([data[name + '_dn'], None, None, None]) for name in measurement_names
}

#probes['free'] = copy.copy(probes['04'])
probes['free2'] = copy.copy(probes['04'])
probes['free'] = copy.copy(probes['04'])
probes['free_rev'] = copy.copy(probes['04'])
# probes['free'].mm.resolution.value=1e-6


for name, probe in probes.items():
    # Set the wavelength to 4. Actually it should be between 4 and 14 but since the SLD doesnt change
    # with the wavelength in this regime, we're ok ...
    probe.unique_L = np.array([4])

models = {
    name: Experiment(probe=probes[name], sample=samples[name], name='Si/Cu/Fe_' + name)
    for name in ['01', '02', '03', '04', 'free']
}

fit_iron_layer = 4

#problem = MultiFitProblem([models[key] for key in measurement_names[1:fit_iron_layer + 1]] + [models['free']])
#problem = MultiFitProblem([models[key] for key in measurement_names[1:fit_iron_layer + 1]] + [models['free']])
problem = MultiFitProblem([models['free']])

ampl = models['free'].amplitude()


numpy.savetxt("sim/amplitude.real", zip(ampl[0], ampl[1].real, -ampl[1].imag))

#pylab.plot(ampl[0], ampl[1].real * ampl[0]**2, '-', color='#990000')
#pylab.show()

#exit(1)