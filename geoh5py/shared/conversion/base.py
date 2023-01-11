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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ...objects import Points

if TYPE_CHECKING:
    from ...workspace.workspace import Workspace
    from ..entity import Entity
    from ...objects import ObjectBase


class ConversionBase(ABC):
    _entity: ObjectBase
    
    def __init__(self, entity: ObjectBase):
        """
        Converter class from an :obj:geoh5py.shared.entity.Entity to another.
        :param entity: the entity to convert.
        """
        self._entity = entity

    @abstractmethod
    def get_attributes(self, **kwargs):
        """"""

    @abstractmethod
    def create_output(self):
        """"""

    @abstractmethod
    def add_data_output(self):
        """"""

    @abstractmethod
    def verify_kwargs(self):
        """"""

    def __call__(self, **kwargs):
        self.verify_kwargs(**kwargs)
        self.get_attributes(**kwargs)
        self.create_output(**kwargs)
        self.add_data_output(**kwargs)

        return self._output

    @property
    def entity(self) -> ObjectBase:
        """Input object"""
        return self._entity


def VertexObject(ConversionBase):
    def __init__(self, entity: ObjectBase):
        """
        Converter class from grid-based object to Points.
        :param entity: the entity to convert.
        """

        if getattr(entity, "vertices", None) is None:
            raise TypeError("Input entity for `VertexObject` conversion must have vertices.")

        super().__init__(entity)

    def to_points(self, workspace=None, **kwargs):
        """Cell-based object conversion to Points"""
        if workspace is None:
            workspace = self.entity.workspace

        points = Points.create(workspace, vertices=self.vertices, **kwargs)

        return points


class GridObject(ConversionBase):

    def __init__(self, entity: ObjectBase):
        """
        Converter class from grid-based object to Points.
        :param entity: the entity to convert.
        """

        if getattr(entity, "centroids"):
            raise TypeError("Input entity for `GridObject` conversion must have centroids.")

        super().__init__(entity)

    def to_points(self, workspace=None, **kwargs):
        """Cell-based object conversion to Points"""
        if workspace is None:
            workspace = self.entity.workspace

        points = Points.create(workspace, vertices=self.centroids, **kwargs)
        for child in entity.children:
            if child.association == "CELL":
                child.copy(parent=points, association="VERTEX")

        return points