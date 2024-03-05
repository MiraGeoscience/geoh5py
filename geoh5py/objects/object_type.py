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


class ObjectType(EntityType):
    """
    Object type class
    """

    @staticmethod
    def create_custom(workspace: Workspace) -> ObjectType:
        """Creates a new instance of ObjectType for an unlisted custom Object type with a
        new auto-generated UUID.

        :param workspace: An active Workspace class
        """
        return ObjectType(workspace)
