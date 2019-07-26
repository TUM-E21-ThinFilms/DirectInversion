"""
Copyright (C) 2006-2010, University of Maryland

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/ or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from numpy import pi, sqrt, exp, array, asarray, isscalar, empty, ones

"""
This code was taken from

https://github.com/reflectometry/direfl 
"""



# This program is public domain.
# Author: Paul Kienzle
"""
Optical matrix form of the reflectivity calculation.

O.S. Heavens, Optical Properties of Thin Solid Films
"""
def refl(Qz, depth, rho, mu=0, wavelength=1, sigma=0):
    """
    Reflectometry as a function of Qz and wavelength.

    **Parameters:**
        *Qz:* float|A
            Scattering vector 4*pi*sin(theta)/wavelength. This is an array.
        *depth:* float|A
            Thickness of each layer.  The thickness of the incident medium
            and substrate are ignored.
        *rho, mu (uNb):* (float, float)|
            Scattering length density and absorption of each layer.
        *wavelength:* float|A
            Incident wavelength (angstrom).
        *sigma:* float|A
            Interfacial roughness. This is the roughness between a layer
            and the subsequent layer. There is no interface associated
            with the substrate. The sigma array should have at least n-1
            entries, though it may have n with the last entry ignored.

    :Returns:
        *r* array of float
    """

    if isscalar(Qz): Qz = array([Qz], 'd')
    n = len(rho)
    nQ = len(Qz)

    # Make everything into arrays
    kz = asarray(Qz, 'd') / 2
    depth = asarray(depth, 'd')
    rho = asarray(rho, 'd')
    mu = mu * ones(n, 'd') if isscalar(mu) else asarray(mu, 'd')
    wavelength = wavelength * ones(nQ, 'd') \
        if isscalar(wavelength) else asarray(wavelength, 'd')
    sigma = sigma * ones(n - 1, 'd') if isscalar(sigma) else asarray(sigma, 'd')

    # Scale units
    rho = rho * 1e-6
    mu = mu * 1e-6

    ## For kz < 0 we need to reverse the order of the layers
    ## Note that the interface array sigma is conceptually one
    ## shorter than rho,mu so when reversing it, start at n-1.
    ## This allows the caller to provide an array of length n
    ## corresponding to rho,mu or of length n-1.
    idx = (kz >= 0)
    r = empty(len(kz), 'D')
    r[idx] = _refl_calc(kz[idx], wavelength[idx], depth, rho, mu, sigma)
    r[~idx] = _refl_calc(abs(kz[~idx]), wavelength[~idx],
                         depth[-1::-1], rho[-1::-1], mu[-1::-1],
                         sigma[n - 2::-1])
    r[abs(kz) < 1.e-6] = -1  # reflectivity at kz=0 is -1
    return r


def _refl_calc(kz, wavelength, depth, rho, mu, sigma):
    """Abeles matrix calculation."""
    if len(kz) == 0: return kz

    ## Complex index of refraction is relative to the incident medium.
    ## We can get the same effect using kz_rel^2 = kz^2 + 4*pi*rho_o
    ## in place of kz^2, and ignoring rho_o.
    kz_sq = kz ** 2 + 4 * pi * rho[0]
    k = kz

    # According to Heavens, the initial matrix should be [ 1 F; F 1],
    # which we do by setting B=I and M0 to [1 F; F 1].  An extra matrix
    # multiply versus some coding convenience.
    B11 = 1
    B22 = 1
    B21 = 0
    B12 = 0
    for i in range(0, len(rho) - 1):
        k_next = sqrt(kz_sq - (4 * pi * rho[i + 1] + 2j * pi * mu[i + 1] / wavelength))
        F = (k - k_next) / (k + k_next)
        F *= exp(-2 * k * k_next * sigma[i] ** 2)
        M11 = exp(1j * k * depth[i]) if i > 0 else 1
        M22 = exp(-1j * k * depth[i]) if i > 0 else 1
        M21 = F * M11
        M12 = F * M22
        C1 = B11 * M11 + B21 * M12
        C2 = B11 * M21 + B21 * M22
        B11 = C1
        B21 = C2
        C1 = B12 * M11 + B22 * M12
        C2 = B12 * M21 + B22 * M22
        B12 = C1
        B22 = C2
        k = k_next

    r = B12 / B11
    return r
