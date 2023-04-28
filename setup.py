# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:19:22 2022

@author: DELINTE Nicolas
"""

from setuptools import setup

import unravel

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='unravel-python',
    version=unravel.__version__,
    description='Implementation of UNRAVEL',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DelinteNicolas/UNRAVEL',
    author='Nicolas Delinte',
    author_email='nicolas.delinte@uclouvain.be',
    license='GNU General Public License v3.0',
    packages=['unravel'],
    install_requires=['dipy',
                      'nibabel',
                      'numpy',
                      'scipy',
                      'tqdm',
                      ],

    classifiers=['Development Status :: 4 - Beta',
                 'Intended Audience :: Science/Research',
                 'Natural Language :: English',
                 'Programming Language :: Python'],
)
