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

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...workspace.workspace import Workspace
    from ..entity import Entity


class ConversionBase:
    def __init__(self, entity: Entity):
        """
        Converter class from an :obj:geoh5py.shared.entity.Entity to another.
        :param entity: the entity to convert.
        """
        self._entity = entity
        self._workspace_output = self.entity.workspace
        self._name = self.entity.name
        self._output = None

    def change_workspace_parent(self, **grid2d_kwargs):
        """
        Change the workspace of the object based on a dictionary.
        :param grid2d_kwargs: the dict of the kwargs verify if parent exists.
        """
        # verify if workspace in grid2d_kwargs
        if grid2d_kwargs.get("workspace", None) is not None:
            raise ValueError("workspace cannot be set in the object")

        # verify parent
        parent = grid2d_kwargs.get("parent", None)

        if grid2d_kwargs.get("parent", None):
            if not hasattr(parent, "workspace"):
                raise AttributeError("The parent kwarg must has a 'workspace'")

            self._workspace_output = parent.workspace

    @abstractmethod
    def get_attributes(self):
        pass

    @abstractmethod
    def create_output(self):
        pass

    @abstractmethod
    def add_data_output(self):
        pass

    @abstractmethod
    def verify_kwargs(self):
        pass

    def __call__(self, **kwargs):
        self.verify_kwargs(**kwargs)
        self.get_attributes(**kwargs)
        self.create_output(**kwargs)
        self.add_data_output(**kwargs)

        return self._output

    @property
    def entity(self) -> Entity:
        return self._entity

    @property
    def workspace_output(self) -> Workspace:
        return self._workspace_output

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name
