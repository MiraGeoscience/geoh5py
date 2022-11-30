#  Copyright (c) 2022 Mira Geoscience Ltd.
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

from ...workspace import Workspace

if TYPE_CHECKING:
    from ..entity import Entity


class ConversionBase:
    def __init__(self, entity: Entity):
        """
        Converter class from an :obj:geoh5py.shared.entity.Entity to another.
        :param entity: the entity to convert.
        """
        self._entity = entity
        self._workspace = self.entity.workspace
        self._name = self.entity.name

    def change_workspace(self, grid2d_kwargs: dict):
        """
        Change the workspace of the object based on a dictionary.
        :param grid2d_kwargs: the dict of the kwargs verify if parent exists.
        """
        # verify parent
        parent = grid2d_kwargs.get("parent", None)

        if parent:
            if not hasattr(parent, "workspace"):
                raise AttributeError("The parent kwarg must has a 'workspace'")

            self._workspace = parent.workspace

    @property
    def entity(self) -> Entity:
        return self._entity

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name
