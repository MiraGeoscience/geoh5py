.. _analyst_objects:

ANALYST Objects
===============

Entities with spatial information used to store data.

.. _geoh5_points:

Points
------

**UUID : {202C5DB1-A56D-4004-9CAD-BAAFD8899406}**

3-D scatter points object defined by vertices with fixed coordinates in Cartesian system (x, y and z).

**Datasets**

:Vertices: 1D composite array

    [*x* ``double``, *y* ``double``, *z* ``double``]


.. _geoh5_curve:

Curve
-----

**UUID : {6A057FDC-B355-11E3-95BE-FD84A7FFCB88}**

Polyline object defined by a series of line segments (cells) connecting vertices.
Data can be associated to either the vertices or cells.

Attributes
^^^^^^^^^^

:Current line property ID: ``str``, *UUID*

    Unique identifier of a reference data for naming of curve parts.


Datasets
^^^^^^^^

:Cells: Array of ``int32``, shape(N, 2)

    Array defining the connection (line segment) between pairs of vertices.

Surface
-------

**UUID : {F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}**

Triangulated mesh object defined by cells (triangles) and vertices.

Datasets
^^^^^^^^

:Cells: Array of ``int32``, shape(N, 3)

    Array defining the connection between triplets of vertices, representing triangles.


Block model
-----------

**UUID : {B020A277-90E2-4CD7-84D6-612EE3F25051}**

Rectilinear grid of cells defined along three orthogonal axes (U,V and Z)
of length nU, nV and nZ respectively. The conversion between the array coordinates of a cell
to a 1-D vector index can be calculated from

::

   cell index = k + i*nZ + j*nU*nZ

Without rotation angles, U points eastwards, V points northwards, and Z points upwards.
Since their geometry is defined entirely by the additional data described below, block models do not require a ``Vertices`` or ``Cells`` dataset.

Datasets
^^^^^^^^
:U cell delimiters: array of ``double``, shape(nU,)

    Distances of cell edges from origin along the U axis (first value should be 0)
:V cell delimiters: array of ``double``, shape(nV,)

    Distances of cell edges from origin along the V axis (first value should be 0)
:Z cell delimiters: array of ``double``, shape(nZ,)

    Distances of cell edges from origin upwards along the vertical axis (first value should be 0)

Attributes
^^^^^^^^^^

:Origin: composite type

    [*X* ``double``, *Y* ``double``, *Z* ``double``]

    Origin point of grid
:Rotation: ``double`` (default) 0
    Counterclockwise angle (degrees) of rotation around the vertical axis in degrees.

2D Grid
-------

**UUID : {48f5054a-1c5c-4ca4-9048-80f36dc60a06}**

Rectilinear grid of cells defined along two orthogonal axes (U and V) of length nU and nV.
The conversion between the grid coordinates of a cell to its 1-D vector index can be calculated from

::

   cell index = i + j*nU

Without rotation angles, U points eastwards and V points northwards. Since their geometry is defined entirely by the additional data
described below, 2D grids do not require a Vertices or Cells dataset.

Attributes
^^^^^^^^^^

:Origin: composite type

    [*X* ``double``, *Y* ``double``, *Z* ``double``]

    Origin point of the grid.
:U Size: ``double``
    Length of U axis
:U Count: ``double``
    Number of cells along U axis
:V Size: ``double``
    Length of V axis
:V Count: ``double``
    Number of cells along V axis
:Rotation: ``double``
    (Optional) Counterclockwise angle (degrees) of rotation around the vertical axis at the Origin.
:Vertical: ``char``, 0(false, default) or 1(true))
    (Optiona) If true, V axis is vertical (and rotation defined around the V axis)

Drillhole
---------

**UUID : {7CAEBF0E-D16E-11E3-BC69-E4632694AA37}**

Object representing boreholes defined by a collar location and survey parameters.
Vertices represent points along the drillhole path (support for data rather than the drillhole geometry itself) and must have a ``Depth`` property value.
Cells contain two vertices and represent intervals along the drillhole path (and are a support for interval data as well).
Cells may overlap with each other to accommodate the different sampling intervals of various data.

Attributes
^^^^^^^^^^

:Collar: composite type, shape(3,)

    [*X* ``double``, *Y* ``double``, *Z* ``double``]

    Collar location

Datasets
^^^^^^^^
:Surveys: composite array, shape(3,)

    [*Depth* ``double``, *Dip* ``double``, *Azimuth* ``double``]

    Survey locations
:Trace: 1D composite array

    [*X* ``double``, *Y* ``double``, *Z* ``double``]

    Points forming the drillhole path from collar to end of hole. Must contain at least two points.

Geoimage
--------

**UUID : {77AC043C-FE8D-4D14-8167-75E300FB835A}**

*Not yet geoh5py implemented*

*To be further documented*

Vertices represent the four corners of the geolocated image. No cell data. An object-associated file-type data containing the image to display
is expected to exist under this object.

.. note:: Should be arranged as a rectangle currently, since Geoscience ANALYST
   does not currently support skewed images.

Label
-----

**UUID : {E79F449D-74E3-4598-9C9C-351A28B8B69E}**

*Not yet geoh5py implemented*

*To be further documented*

Has no vertices nor cell data

Attributes
^^^^^^^^^^

:Target position: composite type, shape(3,)

    [*X* ``double``, *Y* ``double``, *Z* ``double``]

    The target location of the label

:Label position: composite type, shape(3,)

    [*X* ``double``, *Y* ``double``, *Z* ``double``]
    (Optional - Defaults to same as target position ) The location where the text of the label is displayed


Slicer
------

**UUID : {238f961d-ae63-43de-ab64-e1a079271cf5}**

*Not yet geoh5py implemented*

*To be further documented*


Target
------

**UUID : {46991a5c-0d3f-4c71-8661-354558349282}**

*Not yet geoh5py implemented*

*To be further documented*


ioGAS Points
------------

**UUID : {d133341e-a274-40e7-a8c1-8d32fb7f7eaf}**

*Not yet geoh5py implemented*

*To be further documented*


Maxwell Plate
-------------

**UUID : {878684e5-01bc-47f1-8c67-943b57d2e694}**

*Not yet geoh5py implemented*

*To be further documented*


Octree
------

**UUID : {4ea87376-3ece-438b-bf12-3479733ded46}**

Semi-structured grid that stores cells in a tree structure with eight octants.

Datasets
^^^^^^^^

:Octree Cells: composite type, shape(N, 4)

    [*I* ``integer``, *J* ``integer``, *K* ``integer``, *NCells* ``integer``]

    Array defining the position (I, J, K) and size (NCells) of every cell within
    the base octree grid.

Attributes
^^^^^^^^^^

:NU: ``integer``
    Number of base cells along the U-axis.

:NV: ``integer``
    Number of base cells along the V-axis.

:NW: ``integer``
    Number of base cells along the W-axis.

:Origin: composite type, shape(3,)

    [*X* ``double``, *Y* ``double``, *Z* ``double``]

    Origin point of the grid.

:Rotation: ``double`` (default) 0
    Counterclockwise angle (degrees) of rotation around the vertical axis in degrees.

:U Cell Size: ``double``
    Base cell dimension along the U-axis.

:V Cell Size: ``double``
    Base cell dimension along the V-axis.

:W Cell Size: ``double``
    Base cell dimension along the W-axis.


Text Object
-----------

**UUID : {c00905d1-bc3b-4d12-9f93-07fcf1450270}**

*Not yet geoh5py implemented*

*To be further documented*


.. _geoh5_potential_electrode:

Potential Electrode
-------------------

**UUID : {275ecee9-9c24-4378-bf94-65f3c5fbe163}**

:ref:`Curve <geoh5_curve>` object representing the receiver electrodes of a direct-current resistivity survey.

Datasets
^^^^^^^^

:Metadata: json formatted ``string``

    Dictionary defining the link between the source and receiver objects.

    - "Current Electrodes" ``uuid``: Identifier for the linked :ref:`Current Electrode <geoh5_current_electrode>`

    - "Potential Electrodes" ``uuid``: Identifier for the linked :ref:`Potential Electrode <geoh5_potential_electrode>`


Requirements
^^^^^^^^^^^^

:A-B Cell ID: Data entity

    Reference data named "A-B Cell ID" with ``association=CELL`` expected.
    The values define the source dipole (cell) from the :ref:`Current Electrode <geoh5_current_electrode>`
    to every potential measurement.


.. _geoh5_current_electrode:

Current Electrode
-----------------

**UUID : {9b08bb5a-300c-48fe-9007-d206f971ea92}**

:ref:`Curve <geoh5_curve>` object representing the transmitter electrodes of a direct-current resistivity survey.

Datasets
^^^^^^^^

:Metadata: json formatted ``string``

    Dictionary defining the link between the source and receiver objects. Same definition as
    the :ref:`Potential Electrode <geoh5_potential_electrode>` object.


Requirements
^^^^^^^^^^^^

:A-B Cell ID: Data entity

    Reference data named "A-B Cell ID" with ``association=CELL`` defining
    a unique identifier to every unique dipole sources. For "pole" sources, the ``cell``
    attribute references twice to the same vertex.


VP Model
--------

**UUID : {7d37f28f-f379-4006-984e-043db439ee95}**

*Not yet geoh5py implemented*

*To be further documented*



Airborne EM
-----------
**UUID : {fdf7d01e-97ab-43f7-8f2c-b99cc10d8411}**

*Not yet geoh5py implemented*

*To be further documented*

.. _geoh5_atem_rx:

Airborne TEM Rx
---------------

**UUID : {19730589-fd28-4649-9de0-ad47249d9aba}**

:ref:`Curve <geoh5_curve>` object representing an array of time-domain electromagnetic receiver dipoles.

Attributes
^^^^^^^^^^

:Target position: composite type

Datasets
^^^^^^^^

:Metadata: json formatted ``string``

    Dictionary of survey parameters shared with the :ref:`Transmitters <geoh5_atem_tx>`. The following items are core
    parameters stored under the "EM Dataset" key.

    - "Channels": ``list`` of ``double``
        Time channels at which data are recorder.
    - "Input type": ``string``
        Type of survey from "Rx", "Tx" or "Tx and Rx"
    - "Loop radius": ``double``
        Transmitter loop radius.
    - "Property groups": ``list`` of ``uuid``
        Reference to property groups containing data at every channel.
    - "Receivers": ``uuid``
        Unique identifier referencing to itself.
    - "Survey type": ``string``
        Defaults to "Airborne TEM".
    - "Transmitters": ``uuid``
        Unique identifier referencing to the linked transmitters entity.
    - "Unit": ``string``
        Sampling units, must be one of "Seconds (s)", "Milliseconds (ms)",
        "Microseconds (us)" or "Nanoseconds (ns)".
    - "Crossline offset property" ``uuid`` OR  "Crossline offset value" ``double``:
        Offline offset between the receivers and transmitters,
        either defined locally on vertices as a ``property`` OR globally as a constant ``value``.
    - "Inline offset property" ``uuid`` OR  "Crossline offset value" ``double``:
        Inline offset between the receivers and transmitters,
        either defined locally on vertices as a ``property`` OR globally as a constant ``value``.
    - "Inline offset property" ``uuid`` OR  "Crossline offset value" ``double``:
        Vertical offset between the receivers and transmitters,
        either defined locally on vertices as a ``property`` OR globally as a constant ``value``.
    - "Yaw property" ``uuid`` OR  "Yaw value" ``double``:
        Rotation (angle) of the transmitter loop as measured on the UV-plane (+ clockwise),
        either defined locally on vertices as a ``property`` OR globally as a constant ``value``.
    - "Pitch property" ``uuid`` OR  "Pitch value" ``double``:
        Tilt angle of the transmitter loop as measured on the VW-plane (+ nose up),
        either defined locally on vertices as a ``property`` OR globally as a constant ``value``.
    - "Roll property" ``uuid`` OR  "Roll value" ``double``:
        Banking angle of the transmitter loop as measured on the UW-plane (+ right-wing down),
        either defined locally on vertices as a ``property`` OR globally as a constant ``value``.
    - "Waveform" ``dict``:
        - "Discretization" array of ``double``, shape(N, 2):
            Array of times and normalized currents (Amp) describing the source impulse
            over a discrete interval (e.g. [[t_1, c_1], [t_2, c_2], ..., [t_N, c_N]])
        - "Timing mark" ``double``:
            Reference timing mark measured from the beginning of the "Discretization".
            Generally used as the reference (t_i=0.0) for the provided data channels:
            (-) on-time an (+) off-time.

.. _geoh5_atem_tx:

Airborne TEM Tx
---------------

**UUID : {58c4849f-41e2-4e09-b69b-01cf4286cded}**

:ref:`Curve <geoh5_curve>` object representing an array of time-domain electromagnetic transmitter loops.

Datasets
^^^^^^^^

:Metadata: json formatted ``string``

    See definition from the :ref:`Airborne TEM Rx <geoh5_atem_rx>` object. The "Transmitters" ``uuid`` value
    should point to itself, while the "Receivers" ``uuid`` refers the linked
    :ref:`Airborne TEM Rx <geoh5_atem_rx>` object.


Airborne FEM Rx
---------------

**UUID : {b3a47539-0301-4b27-922e-1dde9d882c60}**

*Not yet geoh5py implemented*

*To be further documented*


Airborne FEM Tx
---------------

**UUID : {a006cf3e-e24a-4c02-b904-2e57b9b5916d}**

*Not yet geoh5py implemented*

*To be further documented*


Airborne Gravity
----------------

**UUID : {b54f6be6-0eb5-4a4e-887a-ba9d276f9a83}**

*Not yet geoh5py implemented*

*To be further documented*


Airborne Magnetics
------------------

**UUID : {4b99204c-d133-4579-a916-a9c8b98cfccb}**

*Not yet geoh5py implemented*

*To be further documented*


Ground Gravity
--------------

**UUID : {5ffa3816-358d-4cdd-9b7d-e1f7f5543e05}**

*Not yet geoh5py implemented*

*To be further documented*


Ground Magnetics
----------------

**UUID : {028e4905-cc97-4dab-b1bf-d76f58b501b5}**

*Not yet geoh5py implemented*

*To be further documented*


Ground Gradient IP
------------------

**UUID : {68b16515-f424-47cd-bb1a-a277bf7a0a4d}**

*Not yet geoh5py implemented*

*To be further documented*


Ground EM
---------

**UUID : {09f1212f-2bdd-4dea-8bbd-f66b1030dfcd}**

*Not yet geoh5py implemented*

*To be further documented*


Ground TEM Rx
-------------

**UUID : {41018a45-01a0-4c61-a7cb-9f32d8159df4}**

*Not yet geoh5py implemented*

*To be further documented*


Ground TEM Tx
-------------

**UUID : {98a96d44-6144-4adb-afbe-0d5e757c9dfc}**

*Not yet geoh5py implemented*

*To be further documented*


Ground TEM Rx (large-loop)
--------------------------

**UUID : {deebe11a-b57b-4a03-99d6-8f27b25eb2a8}**

*Not yet geoh5py implemented*

*To be further documented*


Ground TEM Tx (large-loop)
--------------------------

**UUID : {17dbbfbb-3ee4-461c-9f1d-1755144aac90}**

*Not yet geoh5py implemented*

*To be further documented*


Ground FEM Rx
-------------

**UUID : {a81c6b0a-f290-4bc8-b72d-60e59964bfe8}**

*Not yet geoh5py implemented*

*To be further documented*


Ground FEM Tx
-------------

**UUID : {f59d5a1c-5e63-4297-b5bc-43898cb4f5f8}**

*Not yet geoh5py implemented*

*To be further documented*


Magnetotellurics
----------------

**UUID : {b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}**

:ref:`Points <geoh5_points>` object representing a magnetotelluric survey.

:Metadata: json formatted ``string``

    Dictionary of survey parameters. The following items are core parameters stored under the
    "EM Dataset" key.

    - "Channels": ``list`` of ``double``
        Frequency channels at which data are recorder.
    - "Input type": ``string``
        Static field set to "Rx only"
    - "Property groups": ``list`` of ``uuid``
        Reference to property groups containing data at every channel.
    - "Receivers": ``uuid``
        Reference to itself.
    - "Survey type": ``string``
        Static field set to "Magnetotellurics"
    - "Unit": ``string``
        Sampling units, must be one of "Hertz (Hz)", "KiloHertz (kHz)", "MegaHertz (MHz)" or
        "Gigahertz (GHz)".

.. _geoh5_tipper_receivers:

Tipper Rx
---------

**UUID : {0b639533-f35b-44d8-92a8-f70ecff3fd26}**

:ref:`Curve <geoh5_curve>` object representing a tipper survey.

:Metadata: json formatted ``string``

    Dictionary of survey parameters. The following items are core parameters stored under the
    "EM Dataset" key.

    - "Channels": ``list`` of ``double``
        Frequency channels at which data are recorder.
    - "Input type": ``string``
        Static field set to "Rx and base stations"
    - "Property groups": ``list`` of ``uuid``
        Reference to property groups containing data at every channel.
    - "Receivers": ``uuid``
        Reference to itself.
    - "Base stations: ``uuid``
        Reference to :ref:`Tipper Base stations <geoh5_tipper_base_stations>`
    - "Survey type": ``string``
        Static field set to "Magnetotellurics"
    - "Unit": ``string``
        Sampling units, must be one of "Hertz (Hz)", "KiloHertz (kHz)", "MegaHertz (MHz)" or
        "Gigahertz (GHz)".

.. _geoh5_tipper_base_stations:

Tipper Base stations
--------------------

**UUID : {f495cd13-f09b-4a97-9212-2ea392aeb375}**

:ref:`Points <geoh5_points>` object representing a tipper survey.

:Metadata: json formatted ``string``

    See definition from the :ref:`Tipper Rx <geoh5_tipper_receivers>` object. The "Base stations" ``uuid`` value
    should point to itself, while the "Receivers" ``uuid`` refers the linked
    :ref:`Tipper Rx <geoh5_tipper_receivers>` object.
