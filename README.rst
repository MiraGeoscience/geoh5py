|coverage| |docs| |style| |version| |status| |pyversions|


geoh5py: Python API for geoh5, an open file format for geoscientific data
=========================================================================
The **geoh5py** library has been created for the manipulation and storage of a wide range of
geoscientific data (points, curve, surface, 2D and 3D grids) in
``*.geoh5`` file format. Users will be able to directly leverage the powerful visualization
capabilities of `Geoscience ANALYST <https://mirageoscience.com/mining-industry-software/geoscience-analyst/>`_.


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
Anaconda to handle the various packages.

Install **geoh5py** from PyPI::

    $ pip install geoh5py


Setup for development
^^^^^^^^^^^^^^^^^^^^^
After having cloned the Git repository you will need to set up `Poetry`_.
`Poetry`_ makes it easy to install the dependencies and start a virtual environment.

To install poetry:

.. code-block:: bash

  curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

Then, create and activate the virtual environment simply with:

.. code-block:: bash

  poetry shell

.. _Poetry: https://poetry.eustace.io/docs/

Configure the pre-commit hooks
------------------------------
To have the `pre-commit`_ hooks installed and running, first make sure you have pip installed
on your system. Then execute:

.. code-block:: bash

  pip install --user pre-commit
  pre-commit install

All the pre-commit checks run automatically for you, and reformat files when required. Enjoy...

.. _pre-commit: https://pre-commit.com/

IDE
---
`PyCharm`_, by JetBrains, is a very good IDE for developing with Python.
Some suggested PyCharm plugins for working in this project:

- `Toml`_

For Vim lovers, also check out `IdeaVim`_.

.. _PyCharm: https://www.jetbrains.com/pycharm/

.. _Toml: https://plugins.jetbrains.com/plugin/8195-toml/
.. _IdeaVim: https://plugins.jetbrains.com/plugin/164-ideavim/

Build the docs
--------------

To build the api docs using autodocs

.. code-block:: bash

  sphinx-apidoc -o source/ ../geoh5py

.. |docs| image:: https://readthedocs.org/projects/geoh5py/badge/
    :alt: Documentation Status
    :target: https://geoh5py.readthedocs.io/en/latest/?badge=latest

.. |coverage| image:: https://codecov.io/gh/MiraGeoscience/geoh5py/branch/development/graph/badge.svg
    :alt: Test coverage
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


Third Party Software
^^^^^^^^^^^^^^^^^^^^
The geoh5 Software may provide links to third party libraries or code (collectively “Third Party Software”)
to implement various functions. Third Party Software does not comprise part of the Software.
The use of Third Party Software is governed by the terms of such software license(s).
Third Party Software notices and/or additional terms and conditions are located in the
`THIRD_PARTY_SOFTWARE.rst`_ file.

.. _THIRD_PARTY_SOFTWARE.rst: THIRD_PARTY_SOFTWARE.rst


Copyright
^^^^^^^^^
Copyright (c) 2021 Mira Geoscience Ltd.
