#!/usr/bin/env python

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

# This fetch values from package.xml
d = generate_distutils_setup(
    packages=['uncom'],
    package_dir={'': 'src'}
)

setup(**d)