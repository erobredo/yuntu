"""Instalation script."""
import os
import sys
import version


from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

pyversion = sys.version[:3]

def read_requirements():
    requirements = []
    with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as req:
        for line in req:
            nline = line.replace("\n", "")
            if nline != "":
            	requirements.append(line.replace("\n", ""))
    return requirements


if sys.version[:3] == '3.6':
    install_requires = read_requirements()
else:
    install_requires = ['numpy<1.21.0',
                        'numba>=0.50.1',
                        'psycopg2',
                        'pony',
                        'dill',
                        'pickle-mixin',
                        'matplotlib',
                        'librosa',
                        'scikit-image',
                        'shapely',
                        'requests',
                        'tqdm',
                        'dask[complete]',
                        'fastparquet',
                        'pyarrow',
                        'pygraphviz',
                        'pymongo']
# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='yuntu',
    version=version.__version__,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    license='BSD License',
    description='Acoustic Analysis tools for Conabio',
    long_description=README,
    url='https://github.com/CONABIO-audio/yuntu',
    author=(
        'CONABIO, '
        'Santiago Martínez Balvanera, '
        'Everardo Gustavo Robredo Esquivelzeta'
    ),
    author_email=(
        'smartinez@conabio.gob.mx, '
        'erobredo@conabio.gob.mx'
    ),
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
