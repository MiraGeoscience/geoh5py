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

from ... import objects

if TYPE_CHECKING:
    from ...objects import BlockModel, Grid2D, ObjectBase, Octree, Points
    from ...shared.entity import Entity
    from ...workspace import Workspace

entity_properties = ["name", "allow_rename", "allow_move", "allow_delete"]


class ConversionBase(ABC):
    _entity: ObjectBase

    def __init__(self, entity: ObjectBase):
        """
        Converter class from an :obj:geoh5py.shared.entity.Entity to another.
        :param entity: the entity to convert.
        """
        self._entity: Entity = entity
        self._output = None
        self._workspace_output: Workspace | None = None

    @abstractmethod
    def verify_kwargs(self):
        """
        Abstract method to verify the kwargs passed to the converter.
        """

    def change_workspace_parent(self, **kwargs):
        """
        Define the workspace of the converter class if the workspace is defined in the kwargs;
        else the workspace of the input entity is used.
        """
        if "workspace" in kwargs:
            self._workspace_output = kwargs["workspace"]
        else:
            self._workspace_output = self._entity.workspace

    def copy_properties(self):
        """
        Copy all the properties from the original entity to the new one.
        """
        if self._output is not None:
            for property_ in entity_properties:
                setattr(self._output, property_, getattr(self._entity, property_))
        else:
            raise ValueError("Output has not been created yet.")

    @property
    def workspace_output(self) -> Workspace:
        """Workspace of the output object"""
        if self._workspace_output is not None:
            return self._workspace_output
        raise ValueError(
            "Workspace has not been defined yet,\
        please run change_workspace_parent()."
        )

    @property
    def entity(self) -> Entity:
        """Input object"""
        return self._entity

    @property
    def output(self):
        """Output object"""
        return self._output

    @output.setter
    def output(self, value):
        """
        Set the output object.
        :param value: any values to be pass to output.
        """
        self._output = value


class CellObject(ConversionBase):
    def __init__(self, entity: ObjectBase):
        """
        Converter class from grid-based (association: cell) object to Points.
        :param entity: the entity to convert.
        """
        # verify if the entity contains centroids
        if not hasattr(entity, "centroids"):
            raise TypeError(
                "Input entity for `GridObject` conversion must have centroids."
            )

        super().__init__(entity)
        self.output: Points
        self.entity: Grid2D | BlockModel | Octree

    def copy_child_properties(self, association: str):
        """
        Copy child properties from the original entity to the new one.
        :param association: association of the children to copy.
        """
        if self.output is not None:
            for child in self.entity.children:
                child.copy(
                    parent=self.output,
                    association=association
                    if child.association == "CELL"
                    else child.association,
                )
        else:
            raise ValueError("Output has not been created yet.")

    def to_points(self, parent=None, **kwargs) -> Points:
        """Cell-based object conversion to Points"""
        if parent is None:
            workspace = self.entity.workspace
        else:
            workspace = parent.workspace

        # create the point object
        self.output = objects.Points.create(
            workspace, parent=parent, vertices=self.entity.centroids, **kwargs
        )

        # copy the properties of the original object
        self.copy_properties()

        # change the association of the children
        self.copy_child_properties(association="VERTEX")

        return self.output
