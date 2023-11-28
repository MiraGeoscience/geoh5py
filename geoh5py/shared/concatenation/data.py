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

import numpy as np

from geoh5py.shared.utils import as_str_if_uuid

from .entity import Concatenated, ConcatenatedObject


class ConcatenatedData(Concatenated):
    _parent: ConcatenatedObject

    def __init__(self, entity_type, **kwargs):
        if kwargs.get("parent") is None or not isinstance(
            kwargs.get("parent"), ConcatenatedObject
        ):
            raise UserWarning(
                "Creating a concatenated data must have a parent "
                "of type Concatenated."
            )

        super().__init__(entity_type, **kwargs)

    @property
    def property_group(self):
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
    def parent(self, parent):
        if not isinstance(parent, ConcatenatedObject):
            raise AttributeError(
                "The 'parent' of a concatenated Data must be of type 'Concatenated'."
            )
        self._parent = parent
        self._parent.add_children([self])  # type: ignore

        parental_attr = self.concatenator.get_concatenated_attributes(self.parent.uid)

        if f"Property:{self.name}" not in parental_attr:
            parental_attr[f"Property:{self.name}"] = as_str_if_uuid(self.uid)

    @property
    def n_values(self) -> np.ndarray:
        """Number of values in the data."""

        n_values = None
        depths = getattr(self.property_group, "depth_", None)
        if depths and depths is not self:
            n_values = len(depths.values)
        intervals = getattr(self.property_group, "from_", None)
        if intervals and intervals is not self:
            n_values = len(intervals.values)

        return n_values
