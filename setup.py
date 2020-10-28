from setuptools import setup, find_packages

requires = ['numpy', 'scipy', 'matplotlib', 'refl1d', 'skipi']

setup(
    name='dinv',
    version=__import__('dinv').__version__,
    description='Library for calculating from a reflection phase the corresponding scattering length density potential using the GLM algorithm',
    author='Alexander Book',
    author_email='alexander.book@frm2.tum.de',
    license = 'GNU General Public License (GPL), Version 3',
    url='https://github.com/TUM-E21-ThinFilms/DirectInversion',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
)
