Types
=====

.. figure:: ./images/types.png
    :align: right
    :width: 150

While they are structured similarly, **each group, object or set of data
has a type that defines how its HDF5 datasets should be interpreted**.
This type is shared among any number of entities (groups/objects/data sets).

.. _group_types:

Group Types
-----------

**Attributes**

:Name: ``str``
    Name of the group displayed in the project tree.
:ID: ``str``
    Unique identifier (*UUID*) of the group type.
:Description: ``str``
    (Optional) Description of the type.
:Allow move contents: ``int``, 0 or (default) 1
    (Optional) User interface allows deletion of the content.
:Allow delete contents: ``int``, 0 or (default) 1
    (Optional) User interface allows deletion of the content.



.. _object_types:

Object Types
------------

Objects are containers for data values with spatial information.

**Attributes**

:Name: ``str``
    Name of the object displayed in the project tree.
:ID: ``str``
    Unique identifier (*UUID*) of the group type.
:Description: ``str``
    (Optional) Description of the type.


.. _data_types:

Data Types
----------

New data types can be created at will by software or users to describe
object or group properties. Data of the same type can exist on any
number of objects or groups of any type, and each instance can be
associated with vertices, cells or the object/group itself. Some data
type identifiers can also be reserved as a means of identifying a
specific kind of data.


**Attributes**

:Name: ``str``
    Name of the object displayed in the project tree.
:ID: ``str``
    Unique identifier (*UUID*) of the data type.

    Unlike ``Groups`` and ``Objects``, ``Data`` entities do not generally have fixed identifier ``Type``.
    Multiple data entities linked by a type will share common properties (color map, units, etc.).

:Primitive type: ``str``
    Specifies the kind of data values stored as ``HDF5 dataset``.

:Description: ``str``
    (Optional) Description of the type.
:Units: ``str``
    (Optional) Data units
:Color map: 1D compound array

    [*Value* ``double``, *Red* ``uint``, *Green* ``uint``, *Blue* ``uint``, *Alpha* ``uint``]

    (Optional) Records colors assigned to value ranges. The *Value* mark the start of the range)
:Value map: (1D compound array dataset)

    [*Key* ``uint``, *Value* ``str``]

    Required only for reference data types (classifications)
:Transparent no data: ``int``, 0 or (default) 1
    (Optional) Whether or not absence of data/filtered data should be hidden in the viewport.
:Hidden: ``int``, 0 or (default) 1
    (Optional) Whether or not the data type should appear in the data type list.
:Scientific notation: ``int``, 0 or (default) 1
    (Optional) Whether or not the data values of this type should be displayed in scientific notation.
:Precision: ``int``
    (Optional) The number of decimals (or significant digits in case of scientific notation) used when displayed data values of this type.
:Number of bins: ``int``, default=50
    (Optional) Number of bins used when displaying histogram
:Duplicate type on copy: ``int``, 0 or (default) 1
    (Optional) When enabled, a separate copy of this data type will be created and used when data of this type is copied.
