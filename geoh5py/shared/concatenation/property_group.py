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
from uuid import UUID

import numpy as np

from geoh5py.data import Data
from geoh5py.groups.property_group import GroupTypeEnum, PropertyGroup


if TYPE_CHECKING:
    from .concatenator import Concatenator
    from .object import ConcatenatedObject


class ConcatenatedPropertyGroup(PropertyGroup):
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
        if (
            self.properties is None
            or len(self.properties) < 1
            or self._property_group_type != GroupTypeEnum.DEPTH
        ):
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
        if (
            self.properties is None
            or len(self.properties) < 1
            or self._property_group_type != GroupTypeEnum.INTERVAL
        ):
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
        if (
            self.properties is None
            or len(self.properties) < 2
            or self._property_group_type != GroupTypeEnum.INTERVAL
        ):
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

    def _clear_data_list(
        self, data: str | Data | list[str | Data | UUID] | UUID
    ) -> list[str | Data | UUID]:
        """
        Clear the data list of any data that is a depth or from/to data.

        :param data: List of data to clear.

        :return: List of data with depth and from/to data removed.
        """
        if not isinstance(data, (list, tuple)):
            data = [data]

        # avoid suppressing depth and from-to directly
        avoid = []
        if self.depth_ is not None:
            avoid.extend([self.depth_, self.depth_.uid, self.depth_.name])
        if self.from_ is not None:
            avoid.extend([self.from_, self.from_.uid, self.from_.name])
        if self.to_ is not None:
            avoid.extend([self.to_, self.to_.uid, self.to_.name])

        return [d for d in data if d not in avoid]

    def _remove_depth_from_to(self):
        """
        Remove the depth or from/to data from the property group
            if they are the only properties left.
        """
        if (
            self._properties is not None
            and len(self._properties) == 1
            and self.depth_ is not None
        ):
            self.depth_.allow_delete = True
            self._properties.remove(self.depth_.uid)
            self.parent.remove_children([self.depth_, self])

        elif (
            self._properties is not None
            and len(self._properties) == 2
            and self.from_ is not None
            and self.to_ is not None
        ):
            self.to_.allow_delete = True
            self._properties.remove(self.to_.uid)
            self.parent.remove_children([self.to_])

            self.from_.allow_delete = True
            self._properties.remove(self.from_.uid)
            self.parent.remove_children([self.from_, self])

    def remove_properties(self, data: str | Data | list[str | Data | UUID] | UUID):
        """
        Remove data from the properties.

        The property group is removed if only the depth or from/to data are left.

        :param data: Data to remove.
        """
        data = self._clear_data_list(data)
        if data:
            super().remove_properties(data)

        self._remove_depth_from_to()
