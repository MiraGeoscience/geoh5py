Types
=====

Each type can be shared by any number of groups/objects/data sets.

-  Types

   -  Group types

      -
      -  …

   -  Object types

      -
      -  …

   -  Data types

      -

         -  Color map : (1D compound array dataset - Value(double),
            Red(unsigned char), Green(unsigned char), Blue(unsigned
            char), Alpha(unsigned char) : Optional, records colors
            assigned to value ranges (where Value is the start of the
            range)
         -  Value map : (1D compound array dataset - Key(unsigned int),
            Value(string)) : Required only for reference data types (aka
            classifications)

      -  …

Group type attributes


-  Name : (string)
-  ID : (string, UUID of this Group type, referring to the Group
   implementation)
-  Description : (string, optional)
-  Allow move contents : (char, optional, 0(false) or 1(true), default
   1)
-  Allow delete contents : (char, optional, 0(false) or 1(true), default
   1)

Object type attributes


-  Name : (string)
-  ID : (string, UUID of this Object type, referring to the Object
   implementation)
-  Description : (string, optional)

Data type attributes


-  Name : (string)
-  ID : (string, UUID of this Data type)
-  Primitive type : (string) : describing the kind of data contained in
   the associated “Data” tables - “Integer”, “Float”, “Referenced”,
   “Text”, “Filename”, “DateTime” or “Blob” (see “Data types” section)
-  Description : (string, optional)
-  Units : (string, optional)
