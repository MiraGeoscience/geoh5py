.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

GeoH5io
=======
API to read and write geoh5 files from Python 3.

The API is described in terms data types and service interfaces in  `Thrift`_ definition files.
You are free to generate a client from these `definition files`_  in one of the many
languages supported by Thrift. You can then use the client in your custom code
to communicate with the GeoH5 server.

See also the description of `the geoh5 file format`_.

.. _Thrift: https://thrift.apache.org/
.. _the geoh5 file format: doc/source/GeoH5.textile
.. _definition files: interfaces/



Setup for development
=====================
`Poetry`_ makes it easy to install the dependencies and
start a virtual environment.

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

Create the .pyi from thrift files
---------------------------------
To regenerate the Python interface .pyi files from the thrift files, execute:

.. code-block:: bash

  thriftpyi interfaces --output geoh5io/interfaces

On Windows, it most likely terminate with an exception running ``autoflake``,
but files are created nonetheless.
And file while be re-formatted by the pre-commit hook anyway.


Configuration
=============
Both the server and the example client are reading the configuration from config.toml file
in the current directory. If this file does not exists, defaults are applied.

config-example.toml at the root of the project is provided as an example.

To start the server
===================
To start the server, execute:

.. code-block:: bash

  python server/Server.py

To run the examples
====================
Run the simple client
---------------------
Once the server is started:

.. code-block:: bash

  python example/Client.py

Run the stand-alone application
-------------------------------
Without any server running:

.. code-block:: bash

  python example/Client.py
