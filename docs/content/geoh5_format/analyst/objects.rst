ANALYST Objects
===============

Points
------

``UUID : {202C5DB1-A56D-4004-9CAD-BAAFD8899406}``

Object defined by vertices only - no cell data.

Curve
-----

``UUID : {6A057FDC-B355-11E3-95BE-FD84A7FFCB88}``

Each cell contains two vertex indices, representing a segment.

Surface
-------

``UUID : {F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}``

Each cell contains three vertex indices, representing a triangle.

Block model
-----------

``UUID : {B020A277-90E2-4CD7-84D6-612EE3F25051}``

Each cell represents a point of a 3D rectilinear grid. For a 3D cell
index (i,j,k) along axes U,V and Z of length nU, nV and nZ respectively,

::

   cell index = k + i*nZ + j*nU*nZ

-  Without rotation angles, U points eastwards, V points northwards, and
   Z points upwards.
-  Since their geometry is defined entirely by the additional data
   described below, block models do not require a Vertices or Cells
   dataset.
-  Additional datasets :

   -  U cell delimiters : (1D array doubles) distances of cell edges
      from origin along the U axis (first value should be 0)
   -  V cell delimiters : (1D array doubles) distances of cell edges
      from origin along the V axis (first value should be 0)
   -  Z cell delimiters : (1D array doubles) distances of cell edges
      from origin upwards along the vertical axis (first value should be
      0)

-  Additional attributes :

   -  Origin : (composite type - X (double), Y (double), Z (double) )
      origin point of grid
   -  Rotation : (double, default 0) counterclockwise angle of rotation
      around the vertical axis in degrees.

2D Grid
-------

``UUID : {48f5054a-1c5c-4ca4-9048-80f36dc60a06}``

Each cell represents a point in a regular 2D grid. For a 2D cell
index (i,j) within axes U and V containing nU and nV cells respectively,

::

   cell index = i + j*nU

-  Without rotation angles, U points eastwards and V points northwards
-  Since their geometry is defined entirely by the additional data
   described below, 2D grids do not require a Vertices or Cells dataset.
-  Additional attributes :

   -  Origin : (composite type - X (double), Y (double), Z (double) )
      origin point of grid
   -  U Size : (double) length of U axis
   -  U Count : (double) number of cells along U axis
   -  V Size : (double) length of V axis
   -  V Count : (double) number of cells along V axis
   -  Rotation : (optional double) counterclockwise angle of rotation
      around the vertical axis at the Origin in degrees
   -  Vertical : (optional char, 0(false, default) or 1(true)) when
      true, V axis is vertical (and rotation defined around the V axis)

Drillhole
---------

``UUID : {7CAEBF0E-D16E-11E3-BC69-E4632694AA37}``

-  Vertices represent points along the drillhole path (support for data rather than the drillhole geometry itself) and must have a “Depth”
    property value.
-  Cells contain two vertices and represent intervals along the
    drillhole path (and are a support for interval data as well)
-  Cells may overlap with each other to accommodate the different
    sampling intervals of various data.
-  Additional attribute :

   -  Collar : (composite - X, Y, Z) - collar location

-  Additional datasets :

   -  Surveys : (1D composite array) - Depth(float), Dip(float),
      Azimuth(float) - survey locations
   -  Trace : (1D composite array - X, Y, Z, containing at least two
      points) the actual drillhole geometry - points forming the
      drillhole path, from collar to end of hole (optional if surveys
      and collar are present)

Geoimage
--------

``UUID : {77AC043C-FE8D-4D14-8167-75E300FB835A}``

-  Vertices represent the four corners of the geolocated image. Note :
   Should be arranged as a rectangle currently, since Geoscience ANALYST
   does not currently support skewed images.
-  No cell data.
-  An object-associated file-type data containing the image to display
   is expected to exist under this object.

Label
-----

``UUID : {E79F449D-74E3-4598-9C9C-351A28B8B69E}``

-  Has no vertices nor cell data
-  Additional attributes :

   -  Target position : (composite type, X (double), Y (double), Z
      (double) ) The target location of the label
   -  Label position : (optional composite type, X (double), Y (double),
      Z (double), defaults to same as target position ) The location
      where the text of the label is displayed


Slicer
------

``UUID : {238f961d-ae63-43de-ab64-e1a079271cf5}``
-  ...

Target
------

``UUID : {46991a5c-0d3f-4c71-8661-354558349282}``
-  ...

ioGAS Points
------------

``UUID : {d133341e-a274-40e7-a8c1-8d32fb7f7eaf}``
-  ...

Maxwell Plate
-------------

``UUID : {878684e5-01bc-47f1-8c67-943b57d2e694}``
-  ...

Octree
------

``UUID : {4ea87376-3ece-438b-bf12-3479733ded46}``
-  ...

Text Object
-----------

``UUID : {c00905d1-bc3b-4d12-9f93-07fcf1450270}``
-  ...

Potential Electrode
-------------------

``UUID : {275ecee9-9c24-4378-bf94-65f3c5fbe163}``
-  ...

Current Electrode
-----------------

``UUID : {9b08bb5a-300c-48fe-9007-d206f971ea92}``
-  ...

VP Model
--------

``UUID : {7d37f28f-f379-4006-984e-043db439ee95}``
-  ...


Airborne EM
-----------
``UUID : {fdf7d01e-97ab-43f7-8f2c-b99cc10d8411}``
-  ...

Airborne TEM Rx
---------------

``UUID : {19730589-fd28-4649-9de0-ad47249d9aba}``
-  ...

Airborne TEM Tx
---------------

``UUID : {58c4849f-41e2-4e09-b69b-01cf4286cded}``
-  ...

Airborne FEM Rx
---------------

``UUID : {b3a47539-0301-4b27-922e-1dde9d882c60}``
-  ...

Airborne FEM Tx
---------------

``UUID : {a006cf3e-e24a-4c02-b904-2e57b9b5916d}``
-  ...

Airborne Gravity
----------------

``UUID : {b54f6be6-0eb5-4a4e-887a-ba9d276f9a83}``
-  ...

Airborne Magnetics
------------------

``UUID : {4b99204c-d133-4579-a916-a9c8b98cfccb}``
-  ...

Ground Gravity
--------------

``UUID : {5ffa3816-358d-4cdd-9b7d-e1f7f5543e05}``
-  ...

Ground Magnetics
----------------

``UUID : {028e4905-cc97-4dab-b1bf-d76f58b501b5}``
-  ...

Ground Gradient IP
------------------

``UUID : {68b16515-f424-47cd-bb1a-a277bf7a0a4d}``
-  ...

Ground EM
---------

``UUID : {09f1212f-2bdd-4dea-8bbd-f66b1030dfcd}``
-  ...

Ground TEM Rx
-------------

``UUID : {41018a45-01a0-4c61-a7cb-9f32d8159df4}``
-  ...

Ground TEM Tx
-------------

``UUID : {98a96d44-6144-4adb-afbe-0d5e757c9dfc}``
-  ...

Ground TEM Rx (large-loop)
--------------------------

``UUID : {deebe11a-b57b-4a03-99d6-8f27b25eb2a8}``
-  ...

Ground TEM Tx (large-loop)
--------------------------

``UUID : {17dbbfbb-3ee4-461c-9f1d-1755144aac90}``
-  ...

Ground FEM Rx
-------------

``UUID : {a81c6b0a-f290-4bc8-b72d-60e59964bfe8}``
-  ...

Ground FEM Tx
-------------

``UUID : {f59d5a1c-5e63-4297-b5bc-43898cb4f5f8}``
-  ...

Magnetotellurics
----------------

``UUID : {b99bd6e5-4fe1-45a5-bd2f-75fc31f91b38}``
-  ...

ZTEM Rx
-------

``UUID : {0b639533-f35b-44d8-92a8-f70ecff3fd26}``
-  ...

ZTEM Base Stations
------------------

``UUID : {f495cd13-f09b-4a97-9212-2ea392aeb375}``
-  ...
