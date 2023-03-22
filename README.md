# UNRAVEL - UtiliziNg tRActography to uncoVEr muLti-fixel microstructure

<p align="center">
  <img src="https://user-images.githubusercontent.com/70629561/224088594-7bcbded3-9f68-4389-955b-6141569b3c06.png" width="600" />
</p>

Welcome to the UNRAVEL's Github repository!

[![Documentation Status](https://readthedocs.org/projects/unravel/badge/?version=latest)](https://unravel.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/unravel-python?label=pypi%20package)](https://pypi.org/project/unravel-python/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/TIME-python)](https://pypi.org/project/unravel-python/)
![GitHub repo size](https://img.shields.io/github/repo-size/DelinteNicolas/unravel)
[![DOI](https://zenodo.org/badge/455556787.svg)](https://zenodo.org/badge/latestdoi/455556787)

The documentation of the code is available on [readthedocs](https://time.readthedocs.io/en/latest/)

## Description

To *unravel* has two meanings :

* to disentangle the fibers of
* to resolve the intricacy, complexity, or obscurity of

With the UNRAVEL framework, we utilize tractography to unravel the microstructure of multi-fixel models. 

This repository contains the code used to combine macroscopic tractography information with microscopic multi-fixel model estimates in order to improve the accuracy in the estimation of the microstructural properties of neural fibers in a specified tract.

## Installing & importing

### Online install

The UNRAVEL package is available through ```pip install``` under the name ```unravel-python```. Note that the online version might not always be up to date with the latest changes.

```
pip install unravel-python
```
To upgrade the current version : ```pip install unravel-python --upgrade```.

To install a specific version of the package use
```
pip install unravel-python==1.0.0
```
All available versions are listed in [PyPI](https://pypi.org/project/unravel-python/). The package names follow the rules of [semantic versioning](https://semver.org/).

### Local install

If you want to download the latest version directly from GitHub, you can clone this repository
```
git clone https://github.com/DelinteNicolas/unravel.git
```
For a more frequent use of the library, you may wish to permanently add the package to your current Python environment. Navigate to the folder where this repository was cloned or downloaded (the folder containing the ```setup.py``` file) and install the package as follows
```
cd UNRAVEL
pip install .
```

If you have an existing install, and want to ensure package and dependencies are updated use --upgrade
```
pip install --upgrade .
```
### Importing
At the top of your Python scripts, import the library as
```
import unravel
```

### Checking current version installed

The version of the UNRAVEL package installed can be displayed by typing the following command in your python environment
```
unravel.__version__
``` 

### Uninstalling
```
pip uninstall unravel-python
```

## Example data and code

An example use of the main methods and outputs of UNRAVEL is written in the `example.py` file. A tractogram of the middle anterior section of the corpus callosum is used as tractography input.

<p align="center">
  <img src="https://user-images.githubusercontent.com/70629561/169159877-ffbb9b99-ab99-451a-b6a1-24c0b1b5d124.gif" />

