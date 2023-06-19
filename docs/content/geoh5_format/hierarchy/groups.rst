Groups
======

.. figure:: ../images/groups.png
    :align: right
    :width: 300

Groups are simple container for other groups and objects. They are often used to assign
special meanings to a collection of entities or to create specialized software
functionality.

Attributes
----------
:Name: ``str``
    Name of the object displayed in the project tree.
:ID: ``str``, *UUID*
    Unique identifier of the group.
:Visible: ``int``, 0 or (default) 1
    Set visible in the 3D camera (checked in the object tree).
:Public: ``int``, 0 or (default) 1
    Set accessible in the object tree and other parts of the the user interface.
:Clipping IDs: 1D array of *UUID*
    (Optional) List of unique identifiers of clipping plane objects.
:Allow delete: ``int``, 0 or (default) 1
    (Optional) User interface allows deletion.
:Allow move: ``int``, 0 or (default) 1
    (Optional) User interface allows moving to another parent group.
:Allow rename: ``int``, 0 or (default) 1
    (Optional) User interface allows renaming.
:Metadata: (``int``, optional)
    (Optional) Any additional text attached to the group.

The following section describes the supported group types.

.. toctree::
   :maxdepth: 1

   ../analyst/groups
   ../giftools/groups
   ../integrator/groups
   ../integrator/themes

.. note:: Though this file format technically allows objects/groups to appear
   within multiple groups simultaneously (overlapping lists), this is not
   currently supported by ``Geoscience ANALYST``.
