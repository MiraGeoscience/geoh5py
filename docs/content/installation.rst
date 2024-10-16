Installation
============

**geoh5py** is currently written for Python 3.10 or higher, and depends on `NumPy <https://numpy.org/>`_ and
`h5py <https://www.h5py.org/>`_.



.. note:: Users will likely want to take advantage of other packages available in the Python ecosystem. We therefore recommend using `Miniforge <https://github.com/conda-forge/miniforge#download>`_ `(Windows x86_64) <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe>`_ to manage the installation.

	.. figure:: ../images/installation/MinicondaInstaller.png
	    :align: center
	    :width: 200


Install **geoh5py** from PyPI::

    $ pip install geoh5py

To install the latest development version of **geoh5py**, you can use ``pip`` with the
latest GitHub ``development`` branch::

    $ pip install git+https://github.com/MiraGeoscience/geoh5py.git

To work with **geoh5py** source code in development, install from GitHub::

    $ git clone --recursive https://github.com/MiraGeoscience/geoh5py.git
    $ cd geoh5py
    $ python setup.py install
