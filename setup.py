#!/usr/bin/env python
from __future__ import print_function

from distutils.core import setup

from setuptools import find_packages

CLASSIFIERS = [
    "Development Status :: 1 - Beta",
    "Intended Audience :: Industry",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: LGPL License",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: MacOS",
    "Natural Language :: English",
]

# with open("README.md") as f:
#     LONG_DESCRIPTION = "".join(f.readlines())

setup(
    name="geoh5py",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "h5py"],
    author="Mira Geoscience",
    author_email="domfournier@mirageoscience.com",
    description="Geoscience Analyst API",
    keywords="geophysics, geologists",
    url="https://geoh5py.readthedocs.io/en/latest/",
    download_url="https://github.com/MiraGeoscience/GeoH5io.git",
    classifiers=CLASSIFIERS,
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    license="LGPL License",
    use_2to3=False,
)
