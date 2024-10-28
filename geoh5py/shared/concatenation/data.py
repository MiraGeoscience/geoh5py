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

from ...data import Data
from ..utils import as_str_if_uuid
from .concatenated import Concatenated


if TYPE_CHECKING:
    from .object import ConcatenatedObject
    from .property_group import ConcatenatedPropertyGroup


class ConcatenatedData(Concatenated, Data):
    def __init__(self, **kwargs):
        if kwargs.get("parent") is None:
            raise UserWarning(
                "Creating a concatenated data must have a parent "
                "of type Concatenated."
            )

        super().__init__(**kwargs)

    @property
    def property_group(self) -> ConcatenatedPropertyGroup | None:
        """Get the property group containing the data interval."""
        if self.parent.property_groups is None:
            return None

        for prop_group in self.parent.property_groups:
            if prop_group.properties is None:
                continue

            if self.uid in prop_group.properties:
                return prop_group

        return None

    @property
    def parent(self) -> ConcatenatedObject:
        return self._parent

    @parent.setter
    def parent(self, parent: ConcatenatedObject):
        if not hasattr(parent, "add_children"):
            raise ValueError(
                "The 'parent' of a concatenated data must have an 'add_children' method."
            )
        parent.add_children([self])
        self._parent: ConcatenatedObject = parent

        parental_attr = self.concatenator.get_concatenated_attributes(self.parent.uid)

        if f"Property:{self.name}" not in parental_attr:
            parental_attr[f"Property:{self.name}"] = as_str_if_uuid(self.uid)

    @property
    def n_values(self) -> int | None:
        """Number of values in the data."""

        n_values = None
        depths = getattr(self.property_group, "depth_", None)
        if depths and depths is not self:
            n_values = len(depths.values)
        intervals = getattr(self.property_group, "from_", None)
        if intervals and intervals is not self:
            n_values = len(intervals.values)

        return n_values
