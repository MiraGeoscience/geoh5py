Data
====

.. figure:: ../images/data.png
    :align: right
    :width: 300

Containers for data values of various types.
Data are currently **always stored as a 1D array**, even in the case of
single-value data with the ``Object`` association (in which case it is a
1D array of length 1).
See the :ref:`Data Types <data_types>` section for the list of supported data types.

**Attributes**

:Association: ``str``
    Describes which part the property is tied to. Must be one of:
    *Unknown*, *Object*, *Cell*, *Vertex*, *Face* or *Group*

:Name: ``str``
    Name of the data displayed in the project tree.
:ID: ``str``
    Unique identifier (*UUID*) of the group.
:Visible: ``int``, 0 or 1
    (Optional) Set visible in the 3D camera (checked in the object tree).
:Clipping IDs: 1D array of ``UUID``
    (Optional) List of unique identifiers of clipping plane objects.
:Allow delete: ``int``, 0 or (default) 1
    (Optional) User interface allows deletion.
:Allow rename: ``int``, 0 or (default) 1
    (Optional) User interface allows renaming.
:Public: ``int``, 0 or (default) 1
    (Optional) Set accessible in the object tree and other parts of the the user interface.
