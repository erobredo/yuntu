# Yúntú

Audio analysis toolset for ecoacoustics.

## Description

The package *yuntu* is intended to be a self-contained toolbox for:
* Acoustic data representation and transformation.
* Integration of multi-source data, from plain directories to a combination of databases and remote storages (working as an ETL for acoustic data).
* Scalable ecoacoustic analysis

Thanks to *dask*, yuntu can scale up from single machine processing to many parallel processing configurations using different clients.

## Install
To install yuntu, *libsndfile1* and *graphviz* (with dev packages) have to be present in the system.

Currently, pip installation is out of date. To install from source run:

```
python setup.py install

```
