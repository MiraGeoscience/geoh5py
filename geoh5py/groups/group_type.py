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

from ..shared import EntityType


if TYPE_CHECKING:
    from ..workspace import Workspace
    from . import base


class GroupType(EntityType):
    _attribute_map = EntityType._attribute_map.copy()
    _attribute_map.update(
        {
            "Allow move contents": "allow_move_content",
            "Allow delete contents": "allow_delete_content",
        }
    )

    def __init__(
        self,
        workspace: Workspace,
        allow_move_content: bool = True,
        allow_delete_content: bool = True,
        **kwargs,
    ):
        super().__init__(workspace, **kwargs)

        self.allow_move_content = allow_move_content
        self.allow_delete_content = allow_delete_content

    @property
    def allow_move_content(self) -> bool:
        """
        Allow to move the group.
        """
        return self._allow_move_content

    @allow_move_content.setter
    def allow_move_content(self, allow: bool):
        if not isinstance(allow, bool) and allow != 1 and allow != 0:
            raise TypeError("'allow_move_content must be a boolean.")
        self._allow_move_content = bool(allow)

    @property
    def allow_delete_content(self) -> bool:
        """
        :obj:`bool`: [True] Allow to delete the group
        :obj:`~geoh5py.shared.entity.Entity.children`.
        """
        return self._allow_delete_content

    @allow_delete_content.setter
    def allow_delete_content(self, allow: bool):
        if not isinstance(allow, bool) and allow != 1 and allow != 0:
            raise TypeError("'allow_delete_content must be a boolean.")
        self._allow_delete_content = bool(allow)
