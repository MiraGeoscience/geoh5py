GEOH5 File Format
=================

.. figure:: ./images/logo.png
       :align: left
       :width: 100%

About
^^^^^

The GEOH5 format aims to provide a robust means of handling large quantities
of diverse data required in geoscience. The file structure builds on the
generic qualities of the `Geoscience ANALYST
<http://www.mirageoscience.com/our-products/software-product/geoscience-analyst>`_
data model, and attempts to maintain a certain level of simplicity and
consistency throughout. It is based entirely on free and open `HDF5 technology
<https://www.hdfgroup.org/about/hdf_technologies.html>`__. Given that this specification is public, the
file format could, with further investment and involvement, become a useful
exchange format for the broader geoscientific community.


Why GEOH5?
----------

- Leverages properties of HDF5.
   Fast I/O, compression, cross-platform

- Content readable and writeable by third party software.
   We recommend using `HDFView <https://support.hdfgroup.org/products/java/hdfview/>`__, along with Geoscience ANALYST, when
   learning the format.

- Easily extensible to new data types.
   It is intended for Geoscience ANALYST to preserve data it does not
   understand (and generally be very tolerant with regards to missing
   information) when loading and saving geoh5 files. This will allow third
   parties to write to this format fairly easily, as well as include
   additional information not included in this spec for their own purposes.


Definition
^^^^^^^^^^

The following sections define the structure and components making up the GEOH5 file format.

.. toctree::
   :maxdepth: 1

   hierarchy.rst
   groups.rst
   objects.rst
   data.rst
   types.rst


External Links
^^^^^^^^^^^^^^

-  The contents of an HDF5 file can be viewed using
   `HDFView <https://support.hdfgroup.org/products/java/hdfview/>`__ .
-  Precompiled binaries for multiple platforms can be found
   `here <https://www.hdfgroup.org/products/java/release/download.html>`__
- Libraries for accessing HDF5 data
   -  `C, C, .NET <https://www.hdfgroup.org/downloads/>`__
   -  `Python <http://www.h5py.org/>`__
   -  `Matlab <http://www.mathworks.com/help/matlab/hdf5-files.html>`__
   -  etc

-  `Matlab <http://www.mathworks.com/help/matlab/hdf5-files.html>`__

Future development
^^^^^^^^^^^^^^^^^^

-  Evaluate the `blosc <http://www.blosc.org>`__ compression filter for
   HDF5 for smaller file sizes and sometimes even improved performance.
-  Evaluate holding large grid data in 2D or 3D chunked datasets for
   better I/O performance.
-  Investigate use of h5repack for delivering smaller files.
-  Investigate use of h5copy to merge data between files.
