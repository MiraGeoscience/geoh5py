Release Notes
=============

Release 0.3.0 - 2022/06/30
--------------------------

This release addresses changes introduced by the geoh5 v2.0 standard.

- Drillhole objects and associated data are stored as Concatenated entities under the DrillholeGroup.
- Use of context manager for the Workspace with options for read/write mode specifications added.
- Implementation of a SimPEGGroup entity.


Release 0.2.0 - 2022/04/18
--------------------------

- Add MT, tipper and airborne time-domain survey objects.
- Add ui.json read/write with validations
- Bug fixes and documentation.


Release 0.1.6 - 2021/12/09
--------------------------

- Fix StatsCache on value changes.
- Fix crash if data values are None.
- Clean up for linters


Release 0.1.5 - 2021/11/05
--------------------------

- Fix for copying of direct-current survey.
- Fix documentation.


Release 0.1.4 - 2021/08/31
--------------------------

- Add direct_current survey type and related documentation.
- Fix for drillholes with single survey location anywhere along the borehole.
- Fix for entity.parent setter. Changes are applied directly to the target workspace.
- Improve Typing.
