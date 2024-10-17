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

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .concatenator import Concatenator


class Concatenated:
    """
    Base class modifier for concatenated objects and data.
    """

    _concat_attr_str: str = "Attributes"

    def __init__(self, *args, **kwargs):
        attribute_map = getattr(self, "_attribute_map", {})
        attr = {"name": "Entity", "parent": None}

        for key, value in kwargs.items():
            attr[attribute_map.get(key, key)] = value

        super().__init__(*args, **attr)

    @property
    def concat_attr_str(self) -> str:
        """String identifier for the concatenated attributes."""
        return self._concat_attr_str

    @property
    def concatenator(self) -> Concatenator:
        """
        Parental Concatenator entity.

        Warning: Parent is not an attribute of Concatenated, but of the derived class.
        """
        parent: Concatenated | Concatenator = getattr(self, "parent", None)  # type: ignore
        if parent is None:
            raise UserWarning("Parent of concatenated object is not defined.")

        if isinstance(parent, Concatenated):
            return parent.concatenator

        return parent
