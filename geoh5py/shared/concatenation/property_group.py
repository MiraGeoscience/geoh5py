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

from geoh5py.data import Data
from geoh5py.groups import PropertyGroup

from .utils import is_concatenated_object

if TYPE_CHECKING:
    from .object import ConcatenatedObject


class ConcatenatedPropertyGroup(PropertyGroup):
    _parent: ConcatenatedObject

    def __init__(self, parent: ConcatenatedObject, **kwargs):
        if not is_concatenated_object(parent):
            raise UserWarning(
                "Creating a concatenated data must have a parent "
                "of type Concatenated."
            )

        super().__init__(parent, **kwargs)

    @property
    def depth_(self):
        if self.properties is None or len(self.properties) < 1:
            return None

        data = self.parent.get_data(  # pylint: disable=no-value-for-parameter
            self.properties[0]
        )[0]

        if isinstance(data, Data) and "depth" in data.name.lower():
            return data

        return None

    @property
    def from_(self):
        """Return the data entities defined the 'from' depth intervals."""
        if self.properties is None or len(self.properties) < 1:
            return None

        data = self.parent.get_data(  # pylint: disable=no-value-for-parameter
            self.properties[0]
        )[0]

        if isinstance(data, Data) and "from" in data.name.lower():
            return data

        return None

    @property
    def to_(self):
        """Return the data entities defined the 'to' depth intervals."""
        if self.properties is None or len(self.properties) < 2:
            return None

        data = self.parent.get_data(  # pylint: disable=no-value-for-parameter
            self.properties[1]
        )[0]

        if isinstance(data, Data) and "to" in data.name.lower():
            return data

        return None

    @property
    def parent(self) -> ConcatenatedObject:
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self._parent is not None:
            raise AttributeError("Cannot change parent of a property group.")

        if not is_concatenated_object(parent):
            raise AttributeError(
                "The 'parent' of a concatenated Data must be of type 'Concatenated'."
            )
        parent.add_children([self])
        self._parent = parent
        parent.workspace.add_or_update_property_group(self)
