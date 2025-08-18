# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2025 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoh5py.                                               '
#                                                                              '
#  geoh5py is free software: you can redistribute it and/or modify             '
#  it under the terms of the GNU Lesser General Public License as published by '
#  the Free Software Foundation, either version 3 of the License, or           '
#  (at your option) any later version.                                         '
#                                                                              '
#  geoh5py is distributed in the hope that it will be useful,                  '
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              '
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               '
#  GNU Lesser General Public License for more details.                         '
#                                                                              '
#  You should have received a copy of the GNU Lesser General Public License    '
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.           '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


from __future__ import annotations

from typing import TYPE_CHECKING, cast
from uuid import UUID

from geoh5py.shared.utils import get_unique_name_from_entities

from .data import Data


if TYPE_CHECKING:
    from geoh5py.data import ReferencedData


class GeometricDataConstants(Data):
    """
    Base class for geometric data constants.

    :param allow_move: Defaults coordinate to remain on object.
    :param visible: Defaults to not visible.
    """

    _TYPE_UID: UUID

    def __init__(
        self,
        allow_move=False,
        visible=False,
        allow_delete=False,
        allow_rename=False,
        **kwargs,
    ):
        super().__init__(
            allow_move=allow_move,
            allow_delete=allow_delete,
            allow_rename=allow_rename,
            visible=visible,
            **kwargs,
        )

    def validate_values(self, values: None) -> None:
        """
        Validate values for GeometricDataConstants.
        """
        if values is not None:
            raise TypeError(
                f"GeometricDataConstants does not accept values. Got {values}."
            )

        return values

    def copy(
        self,
        parent: ReferencedData | None = None,
        *,
        clear_cache: bool = False,
        name: str | None = None,
        **kwargs,
    ) -> GeometricDataConstants:
        """
        Copy the GeometricDataConstants to a new parent with a unique name.

        Note the parent must be a ReferencedData instance that is associated with a
        GeometricDataValueMapType. If the parent is None, it will return None.

        :param parent: The ReferencedData parent to copy to.
        :param clear_cache: The flag to clear the cache.
        :param name: The name of the new GeometricDataConstants.

        :return: A new GeometricDataConstants instance or None.
        """
        if self.entity_type.value_map is None or parent is None:
            raise AttributeError("GeometricDataConstants must have a value_map")

        if parent is None or not hasattr(parent, "data_maps"):
            raise TypeError(
                "Parent must have a 'data_maps' attribute, typically a ReferencedData."
            )

        name = get_unique_name_from_entities(
            name or self.name,
            parent.parent.children,
            types=GeometricDataConstants,
        )

        geometric_data = cast(
            GeometricDataConstants,
            super().copy(parent.parent, name=name, clear_cache=clear_cache),
        )
        data_type = parent.parent.add_data_map_type(
            name, self.entity_type.value_map.map, parent.entity_type.name
        )

        geometric_data.entity_type = data_type

        return geometric_data
