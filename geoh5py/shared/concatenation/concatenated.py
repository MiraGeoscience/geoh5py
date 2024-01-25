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

from abc import ABC
from typing import TYPE_CHECKING

from geoh5py.shared.entity import Entity

if TYPE_CHECKING:
    from .concatenator import Concatenator


class Concatenated(Entity, ABC):
    """
    Base class modifier for concatenated objects and data.
    """

    _parent: Concatenated | Concatenator
    _concat_attr_str: str = "Attributes"

    def __init__(self, entity_type, **kwargs):
        attribute_map = getattr(self, "_attribute_map", {})
        attr = {"name": "Entity", "parent": None}
        for key, value in kwargs.items():
            attr[attribute_map.get(key, key)] = value

        super().__init__(entity_type, **attr)

    @property
    def concat_attr_str(self) -> str:
        """String identifier for the concatenated attributes."""
        return self._concat_attr_str

    @property
    def concatenator(self) -> Concatenator:
        """
        Parental Concatenator entity.
        """
        if isinstance(self._parent, Concatenated):
            return self._parent.concatenator

        return self._parent
