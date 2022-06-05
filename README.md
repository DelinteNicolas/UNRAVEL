# TIME - Tractography Informed Multi-fascicle microstructure Estimation

<p align="center">
  <img src="https://user-images.githubusercontent.com/70629561/171500194-2f3afd6a-e0fe-4925-9aba-ba462091e5ff.png" />
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/70629561/172073125-b9535681-c5ae-4e05-a403-908e6f9f02ef.png" />
</p>

Welcome to the TIME's Github repository!

[![Documentation Status](https://readthedocs.org/projects/time/badge/?version=latest)](https://time.readthedocs.io/en/latest/?badge=latest)

The documentation of the code is available on [readthedocs](https://time.readthedocs.io/en/latest/)

## Description

This repository contains the code used to combine macroscopic tractography information with microscopic multi-fixel model estimates in order to improve the accuracy in the estimation of the microstructural properties of neural fibers in a specified tract.

## Installing & importing
Download a copy of this repository
```
git clone https://github.com/DelinteNicolas/TIME.git
```

### Installing into your python environment
For a more frequent use of the library, you may wish to permanently add the package to your current Python environment.

Navigate to the folder where this repository was cloned or downloaded (the folder containing the ```setup.py``` file) and install the package as follows
```
cd TIME
pip install .
```

If you have an existing install, and want to ensure package and dependencies are updated use --upgrade
```
pip install --upgrade .
```
(To uninstall :)
```
pip uninstall TIME
```
At the top of your Python scripts, import the library as
```
import TIME
```

## Example data and code

An example use of the main methods and outputs of TIME is written in the `example.py` file. A tractogram of the middle anterior section of the corpus callosum is used as tractography input.

<p align="center">
  <img src="https://user-images.githubusercontent.com/70629561/169159877-ffbb9b99-ab99-451a-b6a1-24c0b1b5d124.gif" />

