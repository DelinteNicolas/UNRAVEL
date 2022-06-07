# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 21:19:22 2022

@author: DELINTE Nicolas
"""

from setuptools import setup

import TIME

setup(
    name='TIME-python',
    version=TIME.__version__,
    description='Implementation of TIME',
    url='https://github.com/shuds13/pyexample',
    author='Nicolas Delinte',
    author_email='nicolas.delinte@uclouvain.be',
    license='GNU General Public License v3.0',
    packages=['TIME'],
    install_requires=['dipy',
                      'nibabel',
                      'numpy',
                      'scipy',
                      ],

    classifiers=['Development Status :: 3 - Alpha',
                 'Natural Language :: English',
                 'Programming Language :: Python'],
)
