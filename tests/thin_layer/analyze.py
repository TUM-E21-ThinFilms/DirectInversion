import os
import re
import numpy

result = {}

for subdir, dirs, files in os.walk('store/test/K/'):
    for file in files:
        qmax = os.path.basename(subdir)
        qmax = qmax.replace('d', '.')
        qmax = int(round(float(qmax) * 10000, 3))
        #qmax = int(float(qmax)) + 20
        #print os.path.join(subdir, file)

        if file.endswith('.dat'):
            iteration = re.findall(r'\d+', file)

            if len(iteration) == 0:
                continue



            iteration = int(iteration[0])
            if qmax in result:
                if iteration > result[qmax]:
                    result[qmax] = iteration
            else:
                result[qmax] = iteration

#print(result)

q, it = [], []
for i in sorted (result.keys()):
    q.append(i)
    it.append(result[i])

print(zip(q, it))

numpy.savetxt('K_simulation.dat', numpy.array(zip(q, it), dtype=int), fmt='%i', header="q_max * 100 [1/Ang]\tmax_iteration [1]", delimiter="\t")
