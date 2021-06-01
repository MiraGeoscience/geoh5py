
.. _primitive_type:

Float
^^^^^

-  Stored as a 1D array of 32-bit float type
-  No data value: 1.175494351e-38 (FLT_MIN, considering use of NaN)

Integer
^^^^^^^

-  Stored as a 1D array of 32-bit integer type
-  No data value: –2147483648 (INT_MIN, considering use of NaN)

Text
^^^^

-  Stored as a 1D array of UTF-8 encoded, variable-length string type
-  No data value : empty string

Referenced
^^^^^^^^^^

-  Stored as a 1D array of 32-bit unsigned integer type (native)
-  Value map : (1D composite type array dataset - Key (unsigned int),
   Value (variable-length utf8 string) ) must exist under type
-  No data value : 0 (key is tied to value “Unknown”)

DateTime
^^^^^^^^

-  Stored as a 1D array of variable-length strings formatted according
   to the `ISO 8601 <https://en.wikipedia.org/wiki/ISO_8601>`__ extended
   specification for representations of UTC dates and times (Qt
   implementation), taking the form YYYY-MM-DDTHH:mm:ss[Z|[+|-]HH:mm]
-  No data value : empty string

Filename
^^^^^^^^

-  Stored as a 1D array of UTF-8 encoded, variable-length string type
   designating a file name
-  For each file name within “Data”, an opaque dataset named after the
   filename must be added under the Data instance, containing a complete
   binary dump of the file
-  Different files (under the same object/group) must be saved under
   different names
-  No data value : empty string

Blob
^^^^

-  Stored as a 1D array of 8-bit char type (native) (value ‘0’ or ‘1’)
-  For each index set to 1, an opaque dataset named after the index
   (e.g. “1”, “2”, etc) must be added under the Data instance,
   containing the binary data tied to that index
-  No data value : 0
