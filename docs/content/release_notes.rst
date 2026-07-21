Release Notes
=============

Release 0.13.0 (2026-07-04)
---------------------------

- GEOPY-2495: Better string conversion of UIJSon class used by print
- GEOPY-2499: BaseUIJson crashes in to_params validation when skipping over disabled fields.
- GEOPY-2488: UIJson should have a flatten method to retrieve value/property of all forms
- GEOPY-2507: Support all data types in DataForm
- GEOPY-2506: Support all data associations in DataForm
- GEOPY-2581: Deprecation error accessing model_fields from pydantic class
- GEOPY-2582: UIJson flatten method does not promote geoh5 to Workspace
- GEOPY-2540: Copy from extent GeoImage takes forever.
- GEOPY-2508: Add multi-select data form type
- GEOPY-2525: Support placeholderText in uijson forms
- GEOPY-2623: colour for Curve is not working the same way as other objects
- GEOPY-2635: Remove code associated with the aborted Form/UIJson classes
- GEOPY-2492: Support FloatRangeSlider as geoh5py form
- GEOPY-2591: geoimage/grid2d conversion of non rectangular geoimage
- GEOPY-2449: Create UI and Driver for the classification of EM anomalies
- GEOPY-2363: geoh5py.data.visual_parameters.colour returns BGR not RGB
- GEOPY-2654: DOCS: Add example on assigning parent to entities
- GEOPY-2632: Simplify the BaseForm.infer method
- GEOPY-2639: Documentation improvement
- GEOPY-2662: Accept meshType and groupType as string name instead of UUID
- GEOPY-2676: FileForm is out of date with current GA behaviour
- GEOPY-2450: Implement Maxwell plate model
- GEOPY-2697: Implement Airborne Apparent Conductivity survey type
- GEOPY-2693: Group types in mapper do not match GA definitions
- GEOPY-2700: Random failure of name incrementor test
- GEOPY-2733: clean GeoApps-Error for concatenate groups if not the same association
- GEOPY-2746: implement "Group Value" in BaseUIJSON
- GEOPY-2702: Validate "geoh5" for existing path
- GEOPY-2744: Migrate core utilities to geoapps-utils and geoh5py
- GEOPY-2786: add a method to access the reshaped values in geoh5py for 2d grid and 3dgrid
- GEOPY-2657: Allow UIJson forms to accept geoh5py.Entity
- GEOPY-2762: Add mechanisms to update the UIJson class with values
- GEOPY-2714: Refactor mask_by_extent utilities for surfaces
- GEOPY-2739: Accept BaseUIJson in the start of driver
- GEOPY-2798: merge main to develop branch (conflicts)
- GEOPY-2793: Add conversion between Plate model from and to Maxwell plates
- GEOPY-2816: data_map.entity_type.value_map can have map is None leading to error
- GEOPY-2833: Regularization parameters of sub-drivers always over-written by joint parameters
- GEOPY-2602: Validate grid size for clipping 2D grids for a given amount of RAM
- GEOPY-2576: class attribute are forced to be unique if strings: subclass ReferenceValueMap
- GEOPY-2854: restore catch str if Enum
- GEOPY-2860: Accept form dependency type based on group_optional
- GEOPY-1306: Add new attributes for value statistics
- GEOPY-2815: address report of updated code linters
- GEOPY-2873: Implement Texture 2D data type
- GEOPY-2756: H5 flags to optimize performances
- GEOPY-2456: Solve warning issues raised by RTD
- GEOPY-2893: h5 compression level seems ignore (from mira-omf)


Release 0.12.0 (2025-12-17)
---------------------------

- GEOPY-2232: Refactor components of inversion options
- GEOPY-2256: bug for get_parent_reference function ion data_type if a property_group is in children
- GEOPY-2258: running several times add_data_map + monitoring directory copy lead to error for inexisting data.
- GEOPY-2188: single entry point to run any application
- GEOPY-2292: Use scientific format with fixed number of decimals to store means
- GEOPY-2091: Profile drillhole group read time and streamline reading process
- GEOPY-2259: Implement VP mesh object
- GEOPY-2272: geoh5py: when naming the files, the extra value (n) should come before the extension
- DEVOPS-690: migrate to poetry 2
- GEOPY-2121: add the name of the application when saving ui.json
- GEOPY-2317: Expose filter basement visual parameters to the VPmesh object.
- GEOPY-2301: The fxmean attached to a referenced data is getting always the same name
- GEOPY-2375: Duplicated data map on re-run of domain mapper
- GEOPY-2413: Error copying Geometric data on cell_objects
- GEOPY-2418: Change the copy method of UIJsonGroup to also copy objects in the uijson
- GEOPY-2429: Support copy_from_extent for DrapeModel objects
- GEOPY-2459: Support directoryOnly field in FileForm.
- GEOPY-2465: make uijson add_ui_json() return the file data
- GEOPY-2440: add a copy group+object base in inputfile as it exists uin uijson group
- GEOPY-2474: UIJson to_params method skips over disabled parameters
- GEOPY-1875: UIJson class: Infer Form type on read of ui.json file
- GEOPY-425: Crash on Zarr file shape for tiled inversions with disk storage
- GEOPY-1667: Use raise instead of asserts inside all validations
- GEOPY-2453: object stock on display after use of geoh5py
- GEOPY-2246: Document uijson dependencyType: show or hide
- GEOPY-2409: Support radial button option
- GEOPY-2409: support infer
- GEOPY-2527: copy function for InputFile
- GEOPY-1261: Return None if path2workspace does not find a file
- GEOPY-2548: Random failure of unit test for surface manipulation
- GEOPY-2564: Basement added to the referenced value map not visible in GA
- GEOPY-2575: Add data_map crashes on float values trying to rename
- GEOPY-2629: warning with Pydantic >=2.12



Release 0.11.0 (2025-06-18)
---------------------------

- GEOPY-1822: update The names of PropertyGroup if it already exists
- GEOPY-1701: Add validation with mesh_type on ObjectForm
- GEOPY-1836: return referenced values in table
- GEOPY-1846: pass drillholes group in InputFile data
- GEOPY-1820: Add a depth of investigation application
- GEOPY-1776: accept "object" association for data in ui.json and "Filename" data
- GEOPY-1878: Error creating value map from values
- GEOPY-1813: Change typing for input PropertyGroup.properties to list[Data] | list[UUID] | list[str] instead list[Data | UUID | str]
- GEOPY-1910: Throw warning if any of the common auto-saving drive is found in mode="r+"
- GEOPY-1946: Octree mesh upside down
- DEVOPS-635: build conda package faster with rattler-build
- GEOPY-1932: Desurveying can produce a divide by zero warning with some paths
- GEOPY-2033: Error copying reference data with GeometryDataValueMap
- GEOPY-2003: Fixes to make simpeg-drivers uijson round-trip test pass
- GEOPY-2025: Duplicated survey object when monitoring_directory is used
- GEOPY-2025: Duplicated survey object when monitoring_directory is used
- GEOPY-2040: geoh5py threshold slider widget is not returning all its values in uijson.data
- GEOPY-1987: Error from cmd.exe printed if h5repack is not found
- GEOPY-1950: Add docs describing the sorting/reshaping of BlockModel objects.
- GEOPY-2065: Crash on monitoring directory copy with reference data
- GEOPY-2039: Deal with data values vector of wrong length
- GEOPY-2035: Preserve unknown (extra) fields in UIJson from ui.json file
- GEOPY-2056: Handle extra fields and deprecations in UIJson version validation
- GEOPY-2068: Clean up pydantic warnings
- GEOPY-2079: petro-lingo composite bug when one of the association is selected
- GEOPY-2096: ChoiceList parameters are converted to list when value is a string
- GEOPY-2053: Support colour data type
- GEOPY-1900: Implement Text Data object
- GEOPY-2153: Increase length of possible string in value map


Release 0.10.0 (2024-10-31)
---------------------------

- Drop support for Python 3.9
- GEOPY-1602: Major refactor of class instantiation and inheritance
    - Entity type passed as a keyword argument to the constructor.
    - Deprecate property 'default_type_uid' in favour of class attribute '_TYPE_UID'.
    - Explicit definition of objects attributes in the signature of the class
    - Enforce assignment of geometric attributes such as vertices, cells, and centroids.
    - Improve object classes docstrings.
    - Add private default name for objects. Assign name to entity type.
    - Change base class of DrapeModel to ObjectBase.
    - Store dtypes of arrays (prisms, layers, etc.) as class attributes
    - Standardize attributes of sub-classes of GridObject.
    - Add a method to get the number of vertices and cells of an object.



Release 0.9.1 (2024-07-02)
--------------------------

- Adjust dependencies to allow for Python 3.9 to 3.12

Release 0.9.0 (2024-06-17)
--------------------------

- GEOPY-1062: Add merging utility for Points
- GEOPY-1092: Add collect_values method to collect values from property_groups.
- GEOPY-1045: Add merging of DrapeModel objects
- GEOPY-1102: Add merging of curve objects
- GEOPY-1144: 1147: Increment property_group name if already exists.
- GEOPY-1287: Fix handling of byte strings.
- GEOPY-1297: Expose data tables from DrillholeGroup.
- GEOPY-1296: Allow access of Concatenated data in array form.
- GEOPY-1335: Improve drillholes tables memory and access.
- GEOPY-1332: Adapt geoh5py for drillhole group in ui.json form.
- GEOPY-1328: Issue loading Bool data in mode="r".
- GEOPY-1364: Facilitate metadata assignation to object.
- GEOPY-1351: Implement referenced data in depth_tables add_data.
- GEOPY-1349: Fix get_data() for ConcatenatedPropertyGroup.
- GEOPY-1441: Suppress unnecessary warnings triggered in set_enabled.
- GEOPY-1434: Fix DrillholeGroup exported for monitoring directory.
- GEOPY-532: Re-order curve parts for ANALYST efficiency.
- GEOPY-1440: Better handling of integer data.
- GEOPY-1418: Clean out empty property groups on drillhole objects after data removal.
- GEOPY-1439: Allow FileData associated to DrillholeGroup.
- GEOPY-1418: Avoid removing empty arrays.
- GEOPY-1460: Fix crash on remove_vertices of points object.
- GEOPY-1450: Handle Geoimage from tiff with float layers.
- GEOPY-1418: Clean out empty property groups on drillhole objects after data removal.
- GEOPY-1503: Octree mesh cell definition not update if record array.
- GEOPY-1539: Add a locations property to expose vertices or centroids on geoh5py.objects.ObjectBase.
- GEOPY-1032, 1111, 1217, 1229, 1230, 1311, 1321, 1346, 1349, 1562: Better handling of Drillhole objects.
- GEOPY-1375, 1456, 1472, 1481, 1541, 1548, 1564: General maintenance.


Release 0.8.0 (2023-10-31)
--------------------------

- GEOPY-241, 550, 993: Explicit creation of a Workspace object.
- GEOPY-262, 479: Improved documentation.
- GEOPY-438: Add access to property_groups as child of objects.
- GEOPY-516, 652: New functionality for UI.json forms.
- GEOPY-652: Expose coordinate system information.
- GEOPY-666: Add clipping by extent for mesh type objects.
- GEOPY-742, 747: Implement all remaining EM survey types for TEM, FEM, airbone and ground.
- GEOPY-776: Support mask data type (BoolData).
- GEOPY-846: Bgin support of object VisualParameters (Color only).
- GEOPY-915, 919, 991, 1002, 1013, 1014: Improve clipping by extent for Grid2D and GeoImage.
- GEOPY-923, 1025, 1050: Add documentation and docstrings.
- GEOPY-870, 897, 918, 976, 979, 987, 992, 1000, 1004, 1030, 1042, 1055: Bug fixes
- GEOPY-1092: Add "collect_values" method to collect values from property_groups.
- GEOPY-1102: Add functionality to merge curve objects


Release 0.7.0 (2023-03-26)
--------------------------

- GEOPY-857, 877: Add and improve function to copy entities from extent.
- GEOPY-537: Throw user warning if change mode to "r+" to "r"
- GEOPY-667, 668, 723, 848: Fixes on drillhole copy and data selection
- GEOPY-851: Fix NDV not recognized on concatenated data
- GEOPY-862: Add measure of maintainability with code climate
- GEOPY-876: Handle geoh5 conversion from 4.2 (geoh5 v2.1) saved as 4.1 (geoh5 v2.0) format.


Release 0.6.1 (2023-02-09)
--------------------------

- GEOPY-848: Fix the issue of copying drill holes with DateTime.
- GEOPY-847: Fix the issue with clipping by extent with 2d coordinates
- GEOPY-537: Add a convenience method to get an active workspace in a different mode "fetch_active_workspace".


Release 0.6.0 (2023/01/26)
--------------------------

- GEOPY-700, 701, 721, 726: Add functionality to convert between Grid2D and GeoImages.
- GEOPY-843: Update drillhole group compatibility with ANALYST v4.2
- GEOPY-746: Implement ground TEM (large-loop) survey type.


Release 0.5.0 (2022/10/26)
--------------------------

- GEOPY-624: Add functionality to remove vertices and cells
- GEOPY-644: Functionality to copy object within box extent. Only implemented for vertex-based object.
- Bug fixes:
    - GEOPY-650: Deal with INTEGRATOR text data in byte format.
    - GEOPY-615: Fix de-survey method for older geoh5 v1 format.


Release 0.4.0 (2022/08/26)
--------------------------

Major release adding new classes and fixing issues with the DrillholeGroup class.

- Fixes for concatenated DrillHoleGroup
    - GEOPY-598: Implement IntegratorDrillholeGroup class
    - GEOPY-583: Better handling of adding and removing concatenaned drillholes and data intervals.
- GEOPY-584: Preserve integer values on IntegerData class.
- GEOPY-548: Allow TextData values on vertices and cells.
- GEOPY-329: API implementation of DrapeModel object class.
- GEOPY-462: Documentation fixes



Release 0.3.1 (2022/08/26)
--------------------------

This release addresses issues encountered after the 0.3.0 release.

- GEOPY-608: Check for 'allow_delete' status before removing.
- GEOPY-600: Fix crash on missing 'Group types' group from project written by ANALYST.
- GEOPY-587: Increase PEP8 compliance after pylint update.
- GEOPY-575: Improve ui.json documentation.


Release 0.3.0 (2022/06/30)
--------------------------

This release addresses changes introduced by the geoh5 v2.0 standard.

- Drillhole objects and associated data are stored as Concatenated entities under the DrillholeGroup.
- Use of context manager for the Workspace with options for read/write mode specifications added.
- Implementation of a SimPEGGroup entity.


Release 0.2.0 (2022/04/18)
--------------------------

- Add MT, tipper and airborne time-domain survey objects.
- Add ui.json read/write with validations
- Bug fixes and documentation.


Release 0.1.6 (2021/12/09)
--------------------------

- Fix StatsCache on value changes.
- Fix crash if data values are None.
- Clean up for linters


Release 0.1.5 (2021/11/05)
--------------------------

- Fix for copying of direct-current survey.
- Fix documentation.


Release 0.1.4 (2021/08/31)
--------------------------

- Add direct_current survey type and related documentation.
- Fix for drillholes with single survey location anywhere along the borehole.
- Fix for entity.parent setter. Changes are applied directly to the target workspace.
- Improve Typing.
