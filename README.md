# DirectInversion
This python library computes from the reflection coefficient the 
corresponding scattering potential, assuming no absorption and no 
bound states. The algorithm is a straight-forward discretization of the
Gelfand-Levitan-Marchenko integral equation.
 ![GLM integral equation](https://quicklatex.com/cache3/00/ql_0153fa716f34233471dfab251d0b0400_l3.png)

# Usage
First load the reflection data. Either the real part or the imaginary
part of the reflection coefficient is required. Of course, also both can
be used.

```python
q, RReal, RImag = numpy.loadtxt("reflection.dat").T
``` 

From the data, calculate the Fourier transform, and reconstruct the data
using the PotentialReconstruction class. The parameter precision 
determines the discretization step, the higher the smaller the step. 
The shift parameter shifts the potential to the right, usefull for 
potentials with rough surfaces at the air interface. 

```python

from dinv.fourier import FourierTransform
from dinv.glm import PotentialReconstruction

fourier = FourierTransform(q/2, RReal, RImag)

precision = 1
film_thickness = 350
reconstruction = PotentialReconstruction(film_thickness, precision, shift=20)
 
potential = reconstruction.reconstruct(fourier)
``` 

The reconstructed potential is a callable function (scipy interpolation 
object). For a simple plott, simply use

```python
import pylab
import numpy

x_space = numpy.linspace(0, 360, 1)
pylab.plot(x_space, potential(x_space))
pylab.show()
```

## Using only the real/imag part of the reflection coefficient
It is possible to use only the real or only the imaginary part for the 
Fourier transform. The fourier transform degenerates to the cosine 
transform if only the real part is used. Analogously, the sine transform
uses only the imaginary part. This can be achieved simply by
 
```python
# Use only real part
fourier.method = fourier.cosine_transform
# Use only imaginary part
fourier.method = fourier.cosine_transform
# Use both
fourier.method = fourier.fourier_transform
```
