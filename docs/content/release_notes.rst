Release Notes
=============

Release 0.13.0 (2026-07-04)
---------------------------

- Better string conversion of UIJson class used by print
- BaseUIJson crashes in to_params validation when skipping over disabled fields
- UIJson should have a flatten method to retrieve value/property of all forms
- Support all data types in DataForm
- Support all data associations in DataForm
- Deprecation error accessing model_fields from pydantic class
- UIJson flatten method does not promote geoh5 to Workspace
- Copy from extent GeoImage takes forever
- Add multi-select data form type
- Support placeholderText in uijson forms
- Colour for Curve is not working the same way as other objects
- Remove code associated with the aborted Form/UIJson classes
- Support FloatRangeSlider as geoh5py form
- Geoimage/grid2d conversion of non rectangular geoimage
- Create UI and Driver for the classification of EM anomalies
- Geoh5py.data.visual_parameters.colour returns BGR not RGB
- Add example on assigning parent to entities
- Simplify the BaseForm.infer method
- Documentation improvement
- Accept meshType and groupType as string name instead of UUID
- FileForm is out of date with current GA behaviour
- Implement Maxwell plate model
- Implement Airborne Apparent Conductivity survey type
- Group types in mapper do not match GA definitions
- Random failure of name incrementor test
- Clean GeoApps-Error for concatenate groups if not the same association
- Implement "Group Value" in BaseUIJSON
- Validate "geoh5" for existing path
- Migrate core utilities to geoapps-utils and geoh5py
- Add a method to access the reshaped values in geoh5py for 2d grid and 3dgrid
- Allow UIJson forms to accept geoh5py.Entity
- Add mechanisms to update the UIJson class with values
- Refactor mask_by_extent utilities for surfaces
- Accept BaseUIJson in the start of driver
- Merge main to develop branch (conflicts)
- Add conversion between Plate model from and to Maxwell plates
- Data_map.entity_type.value_map can have map is None leading to error
- Regularization parameters of sub-drivers always overwritten by joint parameters
- Validate grid size for clipping 2D grids for a given amount of RAM
- Class attribute are forced to be unique if strings subclass ReferenceValueMap
- Restore catch str if Enum
- Accept form dependency type based on group_optional
- Add new attributes for value statistics
- Address report of updated code linters
- Implement Texture 2D data type
- H5 flags to optimize performances
- Solve warning issues raised by RTD
- H5 compression level seems ignore (from mira-omf)


Release 0.12.0 (2025-12-17)
---------------------------

- Refactor components of inversion options
- Bug for get_parent_reference function ion data_type if a property_group is in children
- running several times add_data_map + monitoring directory copy lead to error for inexisting data
- single entry point to run any application
- Use scientific format with fixed number of decimals to store means
- Profile drillhole group read time and streamline reading process
- Implement VP mesh object
- When naming the files, the extra value (n) should come before the extension
- Migrate to poetry 2
- Add the name of the application when saving ui.json
- Expose filter basement visual parameters to the VPmesh object
- The fxmean attached to a referenced data is getting always the same name
- Duplicated data map on re-run of domain mapper
- Error copying Geometric data on cell_objects
- Change the copy method of UIJsonGroup to also copy objects in the uijson
- Support copy_from_extent for DrapeModel objects
- Support directoryOnly field in FileForm
- Make uijson add_ui_json() return the file data
- Add a copy group+object base in inputfile as it exists uin uijson group
- UIJson to_params method skips over disabled parameters
- Infer Form type on read of ui.json file
- Crash on Zarr file shape for tiled inversions with disk storage
- Use raise instead of asserts inside all validations
- object stock on display after use of geoh5py
- Document uijson dependencyType show or hide
- Support radial button option
- support infer
- Copy function for InputFile
- Return None if path2workspace does not find a file
- Random failure of unit test for surface manipulation
- Basement added to the referenced value map not visible in GA
- Add data_map crashes on float values trying to rename
- Warning with Pydantic >=2.12



Release 0.11.0 (2025-06-18)
---------------------------

- update The names of PropertyGroup if it already exists
- Add validation with mesh_type on ObjectForm
- return referenced values in table
- Pass drillholes group in InputFile data
- Add a depth of investigation application
- Accept "object" association for data in ui.json and "Filename" data
- Error creating value map from values
- Change typing for input PropertyGroup.properties to list[Data] | list[UUID] | list[str] instead list[Data | UUID | str]
- Throw warning if any of the common auto-saving drive is found in mode="r+"
- Octree mesh upside down
- Build conda package faster with rattler-build
- Desurveying can produce a divide by zero warning with some paths
- Error copying reference data with GeometryDataValueMap
- Fixes to make simpeg-drivers uijson round-trip test pass
- Duplicated survey object when monitoring_directory is used
- Duplicated survey object when monitoring_directory is used
- Geoh5py threshold slider widget is not returning all its values in uijson.data
- Error from cmd.exe printed if h5repack is not found
- Add docs describing the sorting/reshaping of BlockModel objects
- Crash on monitoring directory copy with reference data
- Deal with data values vector of wrong length
- Preserve unknown (extra) fields in UIJson from ui.json file
- Handle extra fields and deprecations in UIJson version validation
- Clean up pydantic warnings
- Petro-lingo composite bug when one of the association is selected
- ChoiceList parameters are converted to list when value is a string
- Support colour data type
- Implement Text Data object
- Increase length of possible string in value map


Release 0.10.0 (2024-10-31)
---------------------------

- Drop support for Python 3.9
- Major refactor of class instantiation and inheritance
- Entity type passed as a keyword argument to the constructor
- Deprecate property 'default_type_uid' in favour of class attribute '_TYPE_UID'
- Explicit definition of objects attributes in the signature of the class
- Enforce assignment of geometric attributes such as vertices, cells, and centroids
- Improve object classes docstrings
- Add private default name for objects. Assign name to entity type
- Change base class of DrapeModel to ObjectBase
- Store dtypes of arrays (prisms, layers, etc.) as class attributes
- Standardize attributes of sub-classes of GridObject
- Add a method to get the number of vertices and cells of an object



Release 0.9.1 (2024-07-02)
--------------------------

- Adjust dependencies to allow for Python 3.9 to 3.12

Release 0.9.0 (2024-06-17)
--------------------------

- Add merging utility for Points
- Add collect_values method to collect values from property_groups
- Add merging of DrapeModel objects
- Add merging of curve objects
- Increment property_group name if already exists
- Fix handling of byte strings
- Expose data tables from DrillholeGroup
- Allow access of Concatenated data in array form
- Improve drillholes tables memory and access
- Adapt geoh5py for drillhole group in ui.json form
- Issue loading Bool data in mode="r"
- Facilitate metadata assignation to object
- Implement referenced data in depth_tables add_data
- Fix get_data() for ConcatenatedPropertyGroup
- Suppress unnecessary warnings triggered in set_enabled
- Fix DrillholeGroup exported for monitoring directory
- Re-order curve parts for ANALYST efficiency
- Better handling of integer data
- Clean out empty property groups on drillhole objects after data removal
- Allow FileData associated to DrillholeGroup
- Avoid removing empty arrays
- Fix crash on remove_vertices of points object
- Handle Geoimage from tiff with float layers
- Clean out empty property groups on drillhole objects after data removal
- Octree mesh cell definition not update if record array
- Add a locations property to expose vertices or centroids on geoh5py.objects.ObjectBase
- Better handling of Drillhole objects
- General maintenance


Release 0.8.0 (2023-10-31)
--------------------------

- Explicit creation of a Workspace object
- Improved documentation
- Add access to property_groups as child of objects
- New functionality for UI.json forms
- Expose coordinate system information
- Add clipping by extent for mesh type objects
- Implement all remaining EM survey types for TEM, FEM, airbone and ground
- Support mask data type (BoolData)
- Bgin support of object VisualParameters (Color only)
- Improve clipping by extent for Grid2D and GeoImage
- Add documentation and docstrings
- Bug fixes
- Add "collect_values" method to collect values from property_groups
- Add functionality to merge curve objects


Release 0.7.0 (2023-03-26)
--------------------------

- Add and improve function to copy entities from extent
- Throw user warning if change mode to "r+" to "r"
- Fixes on drillhole copy and data selection
- Fix NDV not recognized on concatenated data
- Add measure of maintainability with code climate
- Handle geoh5 conversion from 4.2 (geoh5 v2.1) saved as 4.1 (geoh5 v2.0) format


Release 0.6.1 (2023-02-09)
--------------------------

- Fix the issue of copying drill holes with DateTime
- Fix the issue with clipping by extent with 2d coordinates
- Add a convenience method to get an active workspace in a different mode "fetch_active_workspace"


Release 0.6.0 (2023/01/26)
--------------------------

- Add functionality to convert between Grid2D and GeoImages
- Update drillhole group compatibility with ANALYST v4.2
- Implement ground TEM (large-loop) survey type


Release 0.5.0 (2022/10/26)
--------------------------

- Add functionality to remove vertices and cells
- Functionality to copy object within box extent. Only implemented for vertex-based object

Bug fixes and maintenance
^^^^^^^^^^^^^^^^^^^^^^^^^
- Deal with INTEGRATOR text data in byte format
- Fix de-survey method for older geoh5 v1 format


Release 0.4.0 (2022/08/26)
--------------------------

Major release adding new classes and fixing issues with the DrillholeGroup class

- Fixes for concatenated DrillHoleGroup
- Implement IntegratorDrillholeGroup class
- Better handling of adding and removing concatenaned drillholes and data intervals
- Preserve integer values on IntegerData class
- Allow TextData values on vertices and cells
- API implementation of DrapeModel object class
- Documentation fixes



Release 0.3.1 (2022/08/26)
--------------------------

This release addresses issues encountered after the 0.3.0 release

- Check for 'allow_delete' status before removing
- Fix crash on missing 'Group types' group from project written by ANALYST
- Increase PEP8 compliance after pylint update
- Improve ui.json documentation


Release 0.3.0 (2022/06/30)
--------------------------

This release addresses changes introduced by the geoh5 v2.0 standard

- Drillhole objects and associated data are stored as Concatenated entities under the DrillholeGroup
- Use of context manager for the Workspace with options for read/write mode specifications added
- Implementation of a SimPEGGroup entity


Release 0.2.0 (2022/04/18)
--------------------------

- Add MT, tipper and airborne time-domain survey objects
- Add ui.json read/write with validations
- Bug fixes and documentation


Release 0.1.6 (2021/12/09)
--------------------------

- Fix StatsCache on value changes
- Fix crash if data values are None
- Clean up for linters


Release 0.1.5 (2021/11/05)
--------------------------

- Fix for copying of direct-current survey
- Fix documentation


Release 0.1.4 (2021/08/31)
--------------------------

- Add direct_current survey type and related documentation
- Fix for drillholes with single survey location anywhere along the borehole
- Fix for entity.parent setter. Changes are applied directly to the target workspace
- Improve Typing
