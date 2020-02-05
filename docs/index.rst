gh5py: Python API for Geoscience Analyst
========================================


Welcome to the documentation page for **gh5py**!

In short
^^^^^^^^

The **gh5py** library is an extension of the `h5py <https://www.h5py.org/>`_ package.
It has been created for the manipulation and storage of a wide range of
geoscientific data (points, curve, surface, 2D and 3D grids) in
``*.geoh5`` file format. Users will be able to directly leverage the powerful visualization
capabilities of `Geoscience Analyst <https://mirageoscience.com/mining-industry-software/geoscience-analyst/>`_.

.. figure:: ./images/GA_demo.png
	    :align: center
	    :width: 400


Installation
^^^^^^^^^^^^

**gh5py** is currently written for Python 3.6 or higher, and depends on `NumPy <https://numpy.org/>`_ and
`h5py <https://www.h5py.org/>`_. Users will likely want to also make use of advanced processing
techniques made available under the python ecosystem. We therefore recommend installing
Anaconda to handle the various packages.


Step 1- Setup Anaconda
^^^^^^^^^^^^^^^^^^^^^^
`Download Anaconda <https://www.anaconda.com/download/>`_

- Launch the installation

	.. figure:: ./images/installation/MinicondaInstaller.png
	    :align: center
	    :width: 400

- We recommended letting Anaconda set the Environment Path:

	.. figure:: ./images/installation/AnacondaPath.png
	    :align: center
	    :width: 400


Step 2: Install **gh5py**
^^^^^^^^^^^^^^^^^^^^^^^^^

Install **gh5py** from PyPI::

    $ pip install gh5py

To install the latest development version of **gh5py**, you can use pip with the
latest GitHub master::

    $ pip install git+https://github.com/domfournier/GeoH5io.git

To work with **gh5py** source code in development, install from GitHub::

    $ git clone --recursive https://github.com/domfournier/GeoH5io.git
    $ cd gh5py
    $ python setup.py install


Contents:
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   content/tutorials/Tutorial_Basics.ipynb
   content/api/geoh5io.rst


Feedback:
^^^^^^^^^

Have comments or suggestions? Submit feedback.
All the content can be found on our github_ repository.

.. _github: https://github.com/domfournier/GeoH5io


Contributors:
^^^^^^^^^^^^^

.. include:: AUTHORS.rst
