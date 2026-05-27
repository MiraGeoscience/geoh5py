# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2020-2026 Mira Geoscience Ltd.                                '
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

from uuid import UUID

import numpy as np

from .data import Data


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
        parent=None,
        *,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        """
        Overload of the base Data.copy method to prevent direct copy of GeometricData.

        :return: A new GeometricDataConstants instance or None.
        """
        return None

    @property
    def name(self) -> str:
        """
        :obj:`str` Name of the entity
        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = self.fix_up_name(new_name)

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

            if hasattr(self._entity_type, "set_parent_reference"):
                self._entity_type.set_parent_reference(self, new_name)
