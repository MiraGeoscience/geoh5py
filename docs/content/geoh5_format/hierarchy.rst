Workspace
=========

.. figure:: ./images/hierarchy.png
       :align: right
       :width: 50%

The bulk of the data is accessible both directly by UUID through the
“flat” HDF5 groups or through a **hierarchy of hard links**
While all groups/objects/data are written into their respective base
folder, each group/object also has links to its children, allowing
traversal. (There is no data duplication, merely multiple references to
the same data.)

Types are shared (and thus generally written to file first), and all
groups/objects/data must include a hard link to their type. Details follow.


-  **GEOSCIENCE**

   -  Types
   -  Groups (flat container for all workspace groups)
   -  Objects (flat container for all workspace objects)
   -  Data (flat container for all workspace data)
   -  Root (mandatory hard link to “workspace” group, top of group
      hierarchy)

Attributes


-  Version : (double) Version of specification used by this file
-  Distance unit : (string) Distance unit of all data enclosed
   (“metres”/“feet”)
-  Contributors : (optional, 1D array strings) List of users who
   contributed to this workspace

.. note::

   -  All text data and attributes are variable-length and use UTF-8
      encoding
   -  All numeric data uses intel pc native types
   -  Boolean values are stored using char (0:false, 1:true)
   -  Anything found in a geoh5 v1.0 file which is not mentioned in this
      document is optional information


Main Types
----------

-  ``Groups`` (simple containers for objects)
-  ``Objects`` (containers of data with spatial information)
-  ``Data`` of various types (integer, floating point, text, binary
   data, etc)
-  ``Attributes`` of various types on groups or datasets
-  Links to groups, datasets or subsets of datasets (“hard” with
   reference counting and “soft” without)

While they are structured similarly, **each group, object or set of data
has a type that defines how its HDF5 datasets should be interpreted**.
This type is shared among any number of entities.


Root
----

The ``Root`` group defines the tree structure used in Geoscience ANALYST
describing the parent-child relationships of entities.
