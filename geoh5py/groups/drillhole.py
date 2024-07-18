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


class DrillholeGroup(Group):
    """The type for the group containing drillholes."""

    _TYPE_UID = uuid.UUID(
        fields=(0x825424FB, 0xC2C6, 0x4FEA, 0x9F, 0x2B, 0x6CD00023D393)
    )
    _default_name = "Drillhole Group"


class IntegratorDrillholeGroup(DrillholeGroup):
    """The type for the group containing drillholes."""

    _TYPE_UID = uuid.UUID("{952829b6-76a2-4d0b-b908-7f8d2482dc0d}")
    _default_name = "Integrator Drillhole Group"
