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

from .object_base import ObjectBase


class Slicer(ObjectBase):
    """
    Slicer object.

    It's an empty object used by
    :class:`geoh5py.groups.interpretation_section.InterpretationSection`
    to store the slicing VisualParameter information.
    """

    _attribute_map: dict = ObjectBase._attribute_map.copy()
    _attribute_map.pop("Clipping IDs")
    _attribute_map.pop("PropertyGroups")
    _default_name = "Slicer"
    _TYPE_UID = uuid.UUID("{238F961D-AE63-43DE-AB64-E1A079271CF5}")
