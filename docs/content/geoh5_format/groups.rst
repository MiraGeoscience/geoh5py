Groups
======

.. figure:: ./images/groups.png
    :align: right
    :width: 300

Groups are simple container for other groups and objects. They are often used to assign
special meanings to a collection of entities or to create specialized software
functionality.

**Attributes**

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

.. note:: Though this file format technically allows objects/groups to appear
   within multiple groups simultaneously (overlapping lists), this is not
   currently supported by ``Geoscience ANALYST``.


The following are the default group types recognized by Geoscience ANALYST.

Container
---------

**UUID : {61FBB4E8-A480-11E3-8D5A-2776BDF4F982}**

Simple container with no special meaning. Default in Geoscience ANALYST.


Drillholes
----------

**UUID : {825424FB-C2C6-4FEA-9F2B-6CD00023D393}**

Container restricted to containing drillhole objects, and which may
provide convenience functionality for the drillholes within.


No Type (Root)
--------------

**UUID : {dd99b610-be92-48c0-873c-5b5946ea2840}**

The ``Root`` group defines the tree structure used in Geoscience ANALYST
describing the parent-child relationships of entities. If absent, any Groups/Objects/Data
will be brought into Geoscience ANALYST under the workspace group, still respecting any defined hierarchy links.

SimPEG
------

**UUID : {55ed3daf-c192-4d4b-a439-60fa987fe2b8}**

Container group for SimPEG inversions. Contains

Datasets
^^^^^^^^

:Metadata: json formatted ``string``

    Dictionary of inversion options.

:options: ui.json formatted ``string``

    Dictionary holding the corresponding ui.json.


Tools
-----

**UUID : {a2befc38-3207-46aa-95a2-16b40117a5d8}**

Group for slicer and label objects.

*Not yet geoh5py implemented*

*To be further documented*

Maxwell
-------

**UUID : {1c4122b2-8e7a-4ec3-8d6e-c818495adac7}**

Group for Maxwell plate modeling application.

*Not yet geoh5py implemented*

*To be further documented*
