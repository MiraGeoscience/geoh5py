|coverage| |maintainability| |precommit_ci| |docs| |style| |version| |status| |pyversions|


.. |docs| image:: https://readthedocs.org/projects/geoh5py/badge/
    :alt: Documentation Status
    :target: https://geoh5py.readthedocs.io/en/latest/?badge=latest

.. |coverage| image:: https://codecov.io/gh/MiraGeoscience/geoh5py/branch/develop/graph/badge.svg
    :alt: Code coverage
    :target: https://codecov.io/gh/MiraGeoscience/geoh5py

.. |style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :alt: Coding style
    :target: https://github.com/pf/black

.. |version| image:: https://img.shields.io/pypi/v/geoh5py.svg
    :alt: version on PyPI
    :target: https://pypi.python.org/pypi/geoh5py/

.. |status| image:: https://img.shields.io/pypi/status/geoh5py.svg
    :alt: version status on PyPI
    :target: https://pypi.python.org/pypi/geoh5py/

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/geoh5py.svg
    :alt: Python versions
    :target: https://pypi.python.org/pypi/geoh5py/

.. |precommit_ci| image:: https://results.pre-commit.ci/badge/github/MiraGeoscience/geoh5py/develop.svg
    :alt: pre-commit.ci status
    :target: https://results.pre-commit.ci/latest/github/MiraGeoscience/geoh5py/develop

.. |maintainability| image:: https://api.codeclimate.com/v1/badges/68beb6badd223d4c4809/maintainability
   :target: https://codeclimate.com/github/MiraGeoscience/geoh5py/maintainability
   :alt: Maintainability


geoh5py: Python API for geoh5, an open file format for geoscientific data
=========================================================================
The **geoh5py** library has been created for the manipulation and storage of a wide range of
geoscientific data (points, curve, surface, 2D and 3D grids) in
``*.geoh5`` file format. Users will be able to directly leverage the powerful visualization
capabilities of `Geoscience ANALYST <https://mirageoscience.com/mining-industry-software/geoscience-analyst/>`_.

.. contents:: Table of Contents
   :local:
   :depth: 3

Documentation
^^^^^^^^^^^^^

`Online documentation <https://geoh5py.readthedocs.io/en/latest/>`_

See also documentation for the `geoh5 file format`_.

.. _geoh5 file format: docs/content/geoh5_file_format.textile


Installation
^^^^^^^^^^^^
**geoh5py** is currently written for Python 3.7 or higher, and depends on `NumPy <https://numpy.org/>`_ and
`h5py <https://www.h5py.org/>`_. Users will likely want to also make use of advanced processing
techniques made available under the python ecosystem. We therefore recommend installing
`Miniforge <https://github.com/conda-forge/miniforge#download>`_ `(Windows x86_64) <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe>`_ to handle the various packages.

Install **geoh5py** from PyPI::

    $ pip install geoh5py


Setup for development
^^^^^^^^^^^^^^^^^^^^^

To configure the development environment and tools, please see `README-dev.rst`_.

.. _README-dev.rst: README-dev.rst


Third Party Software
^^^^^^^^^^^^^^^^^^^^
The geoh5 Software may provide links to third party libraries or code (collectively "Third Party Software")
to implement various functions. Third Party Software does not comprise part of the Software.
The use of Third Party Software is governed by the terms of such software license(s).
Third Party Software notices and/or additional terms and conditions are located in the
`THIRD_PARTY_SOFTWARE.rst`_ file.

.. _THIRD_PARTY_SOFTWARE.rst: THIRD_PARTY_SOFTWARE.rst


Citing geoh5py
^^^^^^^^^^^^^^

If you use **geoh5py** in your research, please cite it as follows:

.. image:: https://zenodo.org/badge/207860560.svg
   :target: https://zenodo.org/badge/latestdoi/207860560
