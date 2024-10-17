#  Copyright (c) 2024 Mira Geoscience Ltd.
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

from .base import Group


class AirborneTheme(Group):
    """The type for a INTEGRATOR Airborne Theme."""

    _TYPE_UID = uuid.UUID("{3d0e8578-7764-48cf-8db8-6c83d6411762}")
    _default_name = "Airborne Theme"


class EarthModelsTheme(Group):
    """The type for a INTEGRATOR Earth Models Theme."""

    _TYPE_UID = uuid.UUID("{adee3b2a-3829-11e4-a70e-fcddabfddab1}")
    _default_name = "Earth Models Theme"


class GeochemistryMineralogyTheme(Group):
    """The type for a INTEGRATOR Geochemistry & Mineralogy Theme."""

    _TYPE_UID = uuid.UUID("{ed00094f-3da1-485f-8c4e-b52f6f171ea4}")
    _default_name = "Geochemistry & Mineralogy Theme"


class GeochemistryMineralogyDataSet(Group):
    """The type for a INTEGRATOR Geochemistry & Mineralogy DataSet."""

    _TYPE_UID = uuid.UUID("{72f29283-a4f6-4fc0-a1a8-1417ce5fcbec}")
    _default_name = "Geochemistry & Mineralogy DataSet"


class GeophysicsTheme(Group):
    """The type for a INTEGRATOR Geophysics Theme."""

    _TYPE_UID = uuid.UUID("{151778d9-6cc0-4e72-ba08-2a80a4fb967f}")
    _default_name = "Geophysics Theme"


class GroundTheme(Group):
    """The type for a INTEGRATOR Ground Theme."""

    _TYPE_UID = uuid.UUID("{47d6f059-b56a-46c7-8fc7-a0ded87360c3}")
    _default_name = "Ground Theme"


class IntegratorProject(Group):
    """The type for a INTEGRATOR group."""

    _TYPE_UID = uuid.UUID("{56f6f03e-3833-11e4-a7fb-fcddabfddab1}")
    _default_name = "Geoscience INTEGRATOR Project"


class IntegratorGroup(Group):
    """The type for a INTEGRATOR group."""

    _TYPE_UID = uuid.UUID("{61449477-3833-11e4-a7fb-fcddabfddab1}")
    _default_name = "Geoscience INTEGRATOR"


class QueryGroup(Group):
    """The type for a INTEGRATOR Query Group."""

    _TYPE_UID = uuid.UUID("{85756113-592a-4088-b374-f32c8fac37a2}")
    _default_name = "Query Group"


class ObservationPointsTheme(Group):
    """The type for a INTEGRATOR Observation Points Theme."""

    _TYPE_UID = uuid.UUID("{f65e521c-a763-427b-97bf-d0b4e5689e0d}")
    _default_name = "Observation Points Theme"


class RockPropertiesTheme(Group):
    """The type for a INTEGRATOR Rock Properties Theme."""

    _TYPE_UID = uuid.UUID("{cbeb3920-a1a9-46f8-ab2b-7dfdf79c8a00}")
    _default_name = "Rock Properties Theme"


class SamplesTheme(Group):
    """The type for a INTEGRATOR Samples Theme."""

    _TYPE_UID = uuid.UUID("{1cde9996-cda7-40f0-8c20-faeb4e926748}")
    _default_name = "Samples Theme"
