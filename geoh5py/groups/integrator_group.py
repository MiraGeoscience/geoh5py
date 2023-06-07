#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import uuid

from .group import Group, GroupType


class AirborneTheme(Group):
    """The type for a INTEGRATOR Airborne Theme."""

    __TYPE_UID = uuid.UUID("{3d0e8578-7764-48cf-8db8-6c83d6411762}")

    _name = "Airborne Theme"
    _description = "Airborne Theme"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class EarthModelsTheme(Group):
    """The type for a INTEGRATOR Earth Models Theme."""

    __TYPE_UID = uuid.UUID("{adee3b2a-3829-11e4-a70e-fcddabfddab1}")

    _name = "Earth Models Theme"
    _description = "Earth Models Theme"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class GeochemistryMineralogyTheme(Group):
    """The type for a INTEGRATOR Geochemistry & Mineralogy Theme."""

    __TYPE_UID = uuid.UUID("{ed00094f-3da1-485f-8c4e-b52f6f171ea4}")

    _name = "Geochemistry & Mineralogy Theme"
    _description = "Geochemistry & Mineralogy Theme"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class GeochemistryMineralogyDataSet(Group):
    """The type for a INTEGRATOR Geochemistry & Mineralogy DataSet."""

    __TYPE_UID = uuid.UUID("{72f29283-a4f6-4fc0-a1a8-1417ce5fcbec}")

    _name = "Geochemistry & Mineralogy DataSet"
    _description = "Geochemistry & Mineralogy DataSet"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class GeophysicsTheme(Group):
    """The type for a INTEGRATOR Geophysics Theme."""

    __TYPE_UID = uuid.UUID("{151778d9-6cc0-4e72-ba08-2a80a4fb967f}")

    _name = "Geophysics Theme"
    _description = "Geophysics Theme"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class GroundTheme(Group):
    """The type for a INTEGRATOR Ground Theme."""

    __TYPE_UID = uuid.UUID("{47d6f059-b56a-46c7-8fc7-a0ded87360c3}")

    _name = "Ground Theme"
    _description = "Ground Theme"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class IntegratorProject(Group):
    """The type for a INTEGRATOR group."""

    __TYPE_UID = uuid.UUID("{56f6f03e-3833-11e4-a7fb-fcddabfddab1}")

    _name = "Geoscience INTEGRATOR Project"
    _description = "Geoscience INTEGRATOR Project"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class IntegratorGroup(Group):
    """The type for a INTEGRATOR group."""

    __TYPE_UID = uuid.UUID("{61449477-3833-11e4-a7fb-fcddabfddab1}")

    _name = "Geoscience INTEGRATOR"
    _description = "Geoscience INTEGRATOR"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class QueryGroup(Group):
    """The type for a INTEGRATOR Query Group."""

    __TYPE_UID = uuid.UUID("{85756113-592a-4088-b374-f32c8fac37a2}")

    _name = "Query Group"
    _description = "Query Group"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class ObservationPointsTheme(Group):
    """The type for a INTEGRATOR Observation Points Theme."""

    __TYPE_UID = uuid.UUID("{f65e521c-a763-427b-97bf-d0b4e5689e0d}")

    _name = "Observation Points Theme"
    _description = "Observation Points Theme"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class RockPropertiesTheme(Group):
    """The type for a INTEGRATOR Rock Properties Theme."""

    __TYPE_UID = uuid.UUID("{cbeb3920-a1a9-46f8-ab2b-7dfdf79c8a00}")

    _name = "Rock Properties Theme"
    _description = "Rock Properties Theme"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID


class SamplesTheme(Group):
    """The type for a INTEGRATOR Samples Theme."""

    __TYPE_UID = uuid.UUID("{1cde9996-cda7-40f0-8c20-faeb4e926748}")

    _name = "Samples Theme"
    _description = "Samples Theme"

    def __init__(self, group_type: GroupType, **kwargs):
        assert group_type is not None
        super().__init__(group_type, **kwargs)

        group_type.workspace._register_group(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
