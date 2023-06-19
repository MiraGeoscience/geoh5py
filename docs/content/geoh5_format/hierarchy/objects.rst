Objects
=======

.. figure:: ../images/objects.png
    :align: right
    :width: 300

Objects are containers for ``Data`` with spatial information.
Most (not all) object geometry is described in terms of vertices (3D
locations) and cells (groupings of vertices such as triangles or
segments). The exact requirements and interpretation depends on the
type.


Attributes
----------

:Name: ``str``
    Name of the object displayed in the project tree.
:ID: ``str``
    Unique identifier (*UUID*) of the group.
:Visible: ``int``, 0 or (default) 1
    Set visible in the 3D camera (checked in the object tree).
:Public: ``int``, 0 or (default) 1
    Set accessible in the object tree and other parts of the the user interface.
:Clipping IDs: 1D array of ``UUID``
    (Optional) List of unique identifiers of clipping plane objects.
:Allow delete: ``int``, 0 or (default) 1
    (Optional) User interface allows deletion.
:Allow move: ``int``, 0 or (default) 1
    (Optional) User interface allows moving to another parent group.
:Allow rename: ``int``, 0 or (default) 1
    (Optional) User interface allows renaming.
:Metadata: (``int``, optional)
    (Optional) Any additional text attached to the group.

The following section describes the supported object types and their specific attributes.

.. toctree::
   :maxdepth: 1

   ../analyst/objects
   ../integrator/objects
