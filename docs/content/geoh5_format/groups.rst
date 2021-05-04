Groups
======

-  Groups

   -

      -  Type
      -  Groups

         -  {2d0eaf4c-bbc7-4c7f-8496-e36e4727fbda}
         -  …

      -  Objects

         -  {0b561ab2-87a4-4b82-a5fe-a7c1f6c3c8f8}
         -  …

      -  Data

         -  {0803f944-1392-4a8d-87f1-c7e4c4f3f499}
         -  …

   -
   -  …


Attributes
----------

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


Group types
-----------

While groups can simply be an arbitrary container of random objects, it
is often useful to assign special meanings (and specialized software
functionality).

Container
^^^^^^^^^

Simple container with no special meaning. Default in Geoscience ANALYST.

-  UUID : {61FBB4E8-A480-11E3-8D5A-2776BDF4F982}

Drillholes group
^^^^^^^^^^^^^^^^

Container restricted to containing drillhole objects, and which may
provide convenience functionality for the drillholes within.

-  UUID : {825424FB-C2C6-4FEA-9F2B-6CD00023D393}
