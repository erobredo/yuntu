"""Instalation script."""
import os
from setuptools import find_packages, setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='yuntu-conabio',
    version='0.0.1',
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    license='BSD License',
    description='Acoustic Analysis tools for Conabio',
    long_description=README,
    url='https://github.com/CONABIO-audio/axon',
    author=(
        'CONABIO, '
        'Dalia Camacho García Formentí, '
        'Santiago Martínez Balvanera, '
        'Gustavo Everardo Robredo Esquivelzeta'

    ),
    author_email=(
        'dcamacho@conabio.gob.mx, '
        'smartinez@conabio.gob.mx, '
        'erobredo@conabio.gob.mx'

    ),
    install_requires=[
        'dask',
        'librosa',
        'sqlite3',
        'psycopg2',
        'pymongo'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ],
)
