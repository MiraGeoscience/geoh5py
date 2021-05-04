Objects
=======

Most (not all) object geometry is described in terms of vertices (3D
locations) and cells (groupings of vertices such as triangles or
segments). The exact requirements and interpretation depends on the
type. Additional information may also be stored for some specific types.
(see type descriptions section)

-  Objects

   -

      -  Type
      -  Vertices : (1D array compound dataset - double x, double y,
         double z) : 3D points
      -  Cells : (2D array unsigned int dataset) : sets of indices from
         “Vertices” dataset (number of indices per cell depends on type)
      -  Data

         -  {0782cc42-74f9-4ebf-b9b7-372939999204}
         -  …

   -
   -  …


Attributes
^^^^^^^^^^


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


Types
-----

Some object types are straightforward enough that vertices and cells are
enough to define their geometry. In other cases it is insufficient or
impractical to do so, and these types have additional datasets or
attributes defining their geometry.

Points type
^^^^^^^^^^^

-  UUID : {202C5DB1-A56D-4004-9CAD-BAAFD8899406}
-  No cell data.

Curve type
^^^^^^^^^^

-  UUID : {6A057FDC-B355-11E3-95BE-FD84A7FFCB88}
-  Each cell contains two vertex indices, representing a segment.

Surface type
^^^^^^^^^^^^

-  UUID : {F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}
-  Each cell contains three vertex indices, representing a triangle.

Block model type
^^^^^^^^^^^^^^^^

\* UUID : {B020A277-90E2-4CD7-84D6-612EE3F25051}

\* Each cell represents a point of a 3D rectilinear grid. For a 3D cell
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

2D grid type
^^^^^^^^^^^^

\* UUID : {48f5054a-1c5c-4ca4-9048-80f36dc60a06}

\* Each cell represents a point in a regular 2D grid. For a 2D cell
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

Drillhole type
^^^^^^^^^^^^^^

-  UUID : {7CAEBF0E-D16E-11E3-BC69-E4632694AA37}
-  Vertices represent points along the drillhole path (support for data
   rather than the drillhole geometry itself) and must have a “Depth”
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

Geoimage type
^^^^^^^^^^^^^

-  UUID : {77AC043C-FE8D-4D14-8167-75E300FB835A}
-  Vertices represent the four corners of the geolocated image. Note :
   Should be arranged as a rectangle currently, since Geoscience ANALYST
   does not currently support skewed images.
-  No cell data.
-  An object-associated file-type data containing the image to display
   is expected to exist under this object.

Label type
^^^^^^^^^^

-  UUID : {E79F449D-74E3-4598-9C9C-351A28B8B69E}
-  Has no vertices nor cell data
-  Additional attributes :

   -  Target position : (composite type, X (double), Y (double), Z
      (double) ) The target location of the label
   -  Label position : (optional composite type, X (double), Y (double),
      Z (double), defaults to same as target position ) The location
      where the text of the label is displayed
