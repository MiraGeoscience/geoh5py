.. image:: https://codecov.io/gh/MiraGeoscience/GeoH5io/branch/development/graph/badge.svg?token=cBBxmt1WiA
  :target: https://codecov.io/gh/MiraGeoscience/GeoH5io

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black

geoh5py: Python API for Geoscience Analyst
==========================================
The **geoh5py** library has been created for the manipulation and storage of a wide range of
geoscientific data (points, curve, surface, 2D and 3D grids) in
``*.geoh5`` file format. Users will be able to directly leverage the powerful visualization
capabilities of `Geoscience Analyst <https://mirageoscience.com/mining-industry-software/geoscience-analyst/>`_.

Installation
^^^^^^^^^^^^
**geoh5py** is currently written for Python 3.6 or higher, and depends on `NumPy <https://numpy.org/>`_ and
`h5py <https://www.h5py.org/>`_. Users will likely want to also make use of advanced processing
techniques made available under the python ecosystem. We therefore recommend installing
Anaconda to handle the various packages.


Install **geoh5py** from PyPI::

    $ pip install geoh5py


Setup for development
^^^^^^^^^^^^^^^^^^^^^
Source code is available at https://github.com/MiraGeoscience/GeoH5io.git

After having cloned the Git repository you will need to setup `Poetry`_.
`Poetry`_ makes it easy to install the dependencies and start a virtual environment.

To install poetry:

.. code-block:: bash

  curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

Then, create and activate the virtual environment simply with:

.. code-block:: bash

  poetry shell

.. _Poetry: https://poetry.eustace.io/docs/


Feedback
^^^^^^^^

Have comments or suggestions? Submit feedback.
All the content can be found on our github_ repository.

.. _github: https://github.com/MiraGeoscience/GeoH5io


License
^^^^^^^

geoh5py is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

geoh5py is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.


Copyright
^^^^^^^^^
Copyright (c) 2020 Mira Geoscience Ltd.
