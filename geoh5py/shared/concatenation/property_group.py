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
from typing import TYPE_CHECKING

import numpy as np

from geoh5py.data import Data
from geoh5py.groups import PropertyGroup

if TYPE_CHECKING:
    from .concatenator import Concatenator
    from .object import ConcatenatedObject


class ConcatenatedPropertyGroup(PropertyGroup):
    _parent: ConcatenatedObject

    def __init__(self, parent: ConcatenatedObject, **kwargs):
        super().__init__(parent, **kwargs)

    @property
    def concatenator(self) -> Concatenator:
        """
        Parental Concatenator entity.
        """
        return self._parent.concatenator

    def is_collocated(
        self,
        locations: np.ndarray,
        collocation_distance: float,
    ) -> bool:
        """
        True if locations are collocated with property group.

        :param locations: Locations to check.
        :param collocation_distance: tolerance for similarity check.
        """

        if (
            self.locations is None
            or locations.ndim != self.locations.ndim
            or len(locations) != len(self.locations)
            or not np.allclose(locations, self.locations, atol=collocation_distance)
        ):
            return False

        return True

    @property
    def locations(self) -> np.ndarray:
        """Return depths or intervals array if either exists else None."""

        if self.depth_ is not None:
            return self.depth_.values

        if self.from_ is not None and self.to_ is not None:
            return np.c_[self.from_.values, self.to_.values]

        return None

    @property
    def depth_(self):
        if self.properties is None or len(self.properties) < 1:
            return None

        data = self.parent.get_data(  # pylint: disable=no-value-for-parameter
            self.properties[0]
        )

        if any(data) and isinstance(data[0], Data) and "depth" in data[0].name.lower():
            return data[0]

        return None

    @property
    def from_(self):
        """Return the data entities defined the 'from' depth intervals."""
        if self.properties is None or len(self.properties) < 1:
            return None

        data = self.parent.get_data(  # pylint: disable=no-value-for-parameter
            self.properties[0]
        )

        if any(data) and isinstance(data[0], Data) and "from" in data[0].name.lower():
            return data[0]

        return None

    @property
    def to_(self):
        """Return the data entities defined the 'to' depth intervals."""
        if self.properties is None or len(self.properties) < 2:
            return None

        data = self.parent.get_data(  # pylint: disable=no-value-for-parameter
            self.properties[1]
        )

        if any(data) and isinstance(data[0], Data) and "to" in data[0].name.lower():
            return data[0]

        return None

    @property
    def parent(self) -> ConcatenatedObject:
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self._parent is not None:
            raise AttributeError("Cannot change parent of a property group.")

        if not hasattr(parent, "add_children"):
            raise ValueError(
                "The 'parent' of a concatenated data must have an 'add_children' method."
            )

        parent.add_children([self])
        self._parent: ConcatenatedObject = parent

        parent.workspace.add_or_update_property_group(self)

    def remove_properties(self, data: Data | list[Data | uuid.UUID] | uuid.UUID):
        """
        Remove data from the properties.

        The property group is removed if only the depth or from/to data are left.
        """
        super().remove_properties(data)

        if (
            self._properties is not None
            and len(self._properties) == 1
            and self.depth_ is not None
        ):
            self.depth_.allow_delete = True
            self.parent.remove_children(self)

        elif (
            self._properties is not None
            and len(self._properties) == 2
            and self.from_ is not None
            and self.to_ is not None
        ):
            self.from_.allow_delete = True
            self.to_.allow_delete = True
            self.parent.remove_children(self)
