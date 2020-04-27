.. image:: https://codecov.io/gh/MiraGeoscience/GeoH5io/branch/development/graph/badge.svg?token=cBBxmt1WiA
  :target: https://codecov.io/gh/MiraGeoscience/GeoH5io
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
  :target: https://github.com/psf/black

geoh5py: Python API for geoh5, an open file format for geoscientific data
=========================================================================
The **geoh5py** library has been created for the manipulation and storage of a wide range of
geoscientific data (points, curve, surface, 2D and 3D grids) in
``*.geoh5`` file format. Users will be able to directly leverage the powerful visualization
capabilities of `Geoscience Analyst <https://mirageoscience.com/mining-industry-software/geoscience-analyst/>`_.

See also the description of `the geoh5 file format`_.

.. _the geoh5 file format: doc/source/GeoH5.textile


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
After having cloned the Git repository you will need to setup `Poetry`_.
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
- `Thrift Support`_

For Vim lovers, also check out `IdeaVim`_.

.. _PyCharm: https://www.jetbrains.com/pycharm/

.. _Toml: https://plugins.jetbrains.com/plugin/8195-toml/
.. _Thrift Support: https://plugins.jetbrains.com/plugin/7331-thrift-support/
.. _IdeaVim: https://plugins.jetbrains.com/plugin/164-ideavim/

Build the docs
--------------

To build the api docs using autodocs

.. code-block:: bash

  sphinx-apidoc -o source/ ../geoh5py


Language agnostic interface and services with Thrift [WIP]
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The API is described in terms data types and service interfaces in  `Thrift`_ definition files.
You are free to generate a client from these `definition files`_  in one of the many
languages supported by Thrift. You can then use the client in your custom code
to communicate with the GeoH5 server.

.. _Thrift: https://thrift.apache.org/
.. _definition files: interfaces/


Configuration
-------------
Both the server and the example client are reading the configuration from config.toml file
in the current directory. If this file does not exists, defaults are applied.

config-example.toml at the root of the project is provided as an example.

To start the server
-------------------
To start the server, execute:

.. code-block:: bash

  python server/Server.py

Run the simple client example
-----------------------------
Once the server is started:

.. code-block:: bash

  python example/Client.py

Run the stand-alone application example
---------------------------------------
Without any server running:

.. code-block:: bash

  python example/Client.py

For contributors: create the .pyi from thrift files
---------------------------------------------------
To regenerate the Python interface .pyi files from the thrift files, execute:

.. code-block:: bash

  thriftpyi interfaces --output geoh5io/interfaces

On Windows, it most likely terminates with an exception running ``autoflake``,
but files are created nonetheless.
And file will be re-formatted by the pre-commit hook anyway.


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
