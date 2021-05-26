Workspace
*********

The bulk of the data is accessible both directly by ``UUID`` through the
``flat`` HDF5 groups or through a **hierarchy of hard links** structured as follow

.. figure:: ./images/hierarchy.png
    :align: right
    :height: 200

    As seen in `HDFView <https://support.hdfgroup.org/products/java/hdfview/>`_


-  **GEOSCIENCE (Workspace)**
    -  **Data**: Flat container for all data entities
    -  **Groups**: Flat container for all group entities
    -  **Objects**: Flat container for all object entities
    -  **Root**: Mandatory hard link to ``workspace`` group, top of group hierarchy.
    -  **Types**
        - :ref:`Data Types <data_types>`: Flat container for all data types
        - :ref:`Group Types <group_types>`: Flat container for all group types
        - :ref:`Object Types <object_types>`: Flat container for all object types

**Attributes**

-  Version: (double) Version of specification used by this file
-  Distance unit: (``string``) Distance unit of all data enclosed
   (``metres``/``feet``)
-  Contributors: (optional, 1D array ``string``) List of users who
    contributed to this workspace

While all groups, objects and data entities are written into their respective base
folder, they also hold links to their children entities to allow for
traversals. There is no data duplication, merely multiple references (pointers) to
the data storage on file.

.. figure:: ./images/entity_links.png
    :align: center
    :height: 400

Types are shared (and thus generally written to file first). All
groups, objects and data must include a hard link to their type.



Groups
******

.. figure:: ./images/groups.png
    :align: right
    :width: 300

Groups are used as container for Objects to organize the Workspace.
See the :ref:`Group Types <group_types>` section for the list of supported groups.

**Attributes**

-  Name : (string)
-  ID : (string, UUID of this entity)
-  Visible : (optional, char, 0 or (default) 1) will be visible in the
   3D camera (checked in the object tree)
-  Public : (optional, char, 0 or (default) 1) accessible in the object
   tree and other parts of the the user interface
-  Clipping IDs : (optional, 1D array UUID strings of clipping plane
   objects)
-  Allow delete : (optional, char, 0 or (default) 1) user interface will
   allow deletion
-  Allow move : (optional, char, 0 or (default) 1) user interface will
   allow moving to another parent group
-  Allow rename : (optional, char, 0 or (default) 1) user interface will
   allow renaming

.. note:: Though this file format technically allows objects/groups to appear
   within multiple groups simultaneously (overlapping lists), this is not
   currently supported by Geoscience ANALYST.


Objects
*******

.. figure:: ./images/objects.png
    :align: right
    :width: 300

Objects are containers for ``Data`` with spatial information.
Most (not all) object geometry is described in terms of vertices (3D
locations) and cells (groupings of vertices such as triangles or
segments). The exact requirements and interpretation depends on the
type. Additional information may also be stored for some specific types.
See the :ref:`Object Types <object_types>` section for the list of supported objects.

**Attributes**

-  Name : (string)
-  ID : (string, UUID of this entity)
-  Visible : (optional, char, 0 or (default) 1) will be visible in the
   3D camera (checked in the object tree)
-  Clipping IDs : (optional, 1D array UUID strings of clipping plane
   objects)
-  Allow delete : (optional, char, 0 or (default) 1) user interface will
   allow deletion
-  Allow move : (optional, char, 0 or (default) 1) user interface will
   allow moving to another parent group
-  Allow rename : (optional, char, 0 or (default) 1) user interface will
   allow renaming
-  Public : (optional, char, 0 or (default) 1) accessible in the object
   tree and other parts of the the user interface


Data
****

.. figure:: ./images/data.png
    :align: right
    :width: 300

Container for data values of various types.
Data are currently **always stored as a 1D array**, even in the case of
single-value data with the ``Object`` association (in which case it is a
1D array of length 1).
See the :ref:`Data Types <data_types>` section for the list of supported data types.

**Attributes**

-  Association : (string) “Object”, “Cell” or “Vertex” - describes
   whether the property is tied to cells, vertices, or the object/group
   itself.
-  Name : (string)
-  ID : (string, UUID of this entity)
-  Visible : (optional, char, 0 or (default) 1) will be visible in the
   3D camera (checked in the object tree)
-  Allow delete : (optional, char, 0 or (default) 1) user interface will
   allow deletion
-  Allow rename : (optional, char, 0 or (default) 1) user interface will
   allow renaming
-  Public : (optional, char, 0 or (default) 1) accessible in the object
   tree and other parts of the the user interface


Types
*****

.. figure:: ./images/types.png
    :align: right
    :width: 300

While they are structured similarly, **each group, object or set of data
has a type that defines how its HDF5 datasets should be interpreted**.
This type is shared among any number of entities (groups/objects/data sets).

.. _group_types:

Group Types
===========

While groups can simply be an arbitrary container of random objects, it
is often useful to assign special meanings (and specialized software
functionality).

**Attributes**

-  Name : (``string``)
-  ID : (``string``, ``UUID`` of this Group type, referring to the Group
   implementation)
-  Description : (``string``, optional)
-  Allow move contents : (char, optional, 0(false) or 1(true), default
   1)
-  Allow delete contents : (char, optional, 0(false) or 1(true), default
   1)

The following section describes the supported group types.

.. toctree::
   :maxdepth: 1

   analyst/groups
   giftools/groups
   integrator/groups
   integrator/themes


.. _object_types:

Object Types
============

Containers of data sets with spatial information

**Attributes**

-  Name : (``string``)
-  ID : (``string``, ``UUID`` of this Object type, referring to the Object
   implementation)
-  Description : (``string``, optional)

The following section describes the supported object types.

.. toctree::
   :maxdepth: 1

   analyst/objects
   integrator/objects


.. _data_types:

Data Types
==========

New data types can be created at will by software or users to describe
object or group properties. Data of the same type can exist on any
number of objects or groups of any type, and each instance can be
associated with vertices, cells or the object/group itself. Some data
type identifiers can also be reserved as a means of identifying a
specific kind of data. Each of them must be of one of the following
**primitive types**, which dictate the contents of the “Data” HDF5
dataset for each instance :


**Attributes**

-  Name: (``string``)
-  ID: (``string``, ``UUID`` of this Data type)
-  Primitive type: (``string``) : describing the kind of data contained in
   the associated (see :ref:`Data <core_data>` section)
-  Description: (``string``, optional)
-  Units: (``string``, optional)
-  Color map: (1D compound array dataset - Value(double),
    Red(unsigned char), Green(unsigned char), Blue(unsigned
    char), Alpha(unsigned char) : Optional, records colors
    assigned to value ranges (where Value is the start of the
    range)
-  Value map: (1D compound array dataset - Key(unsigned int),
    Value(``string``)) : Required only for reference data types (aka
    classifications)

The following section describes the supported data types.

.. toctree::
   :maxdepth: 1

   analyst/data
   integrator/data
