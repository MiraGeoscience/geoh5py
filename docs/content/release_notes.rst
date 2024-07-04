Release Notes
=============

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
