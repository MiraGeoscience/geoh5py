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

from abc import ABC
from typing import TYPE_CHECKING

from ... import objects

if TYPE_CHECKING:
    from ...objects import Points
    from ...shared.entity import Entity


entity_properties = ["name", "allow_rename", "allow_move", "allow_delete"]


class ConversionBase(ABC):
    def __init__(self):
        """
        Converter class from an :obj:geoh5py.shared.entity.Entity to another..
        """

    @classmethod
    def verify_kwargs(cls, input_entity, **kwargs) -> dict:
        """
        Verify if the kwargs are valid.
        :param input_entity: The input entity to convert.
        :return: A dictionary of the valid kwargs.
        """
        output_properties = {}
        output_properties["workspace"] = cls.change_workspace_parent(
            input_entity, **kwargs
        )

        return output_properties

    @classmethod
    def change_workspace_parent(cls, input_entity: Entity, **kwargs) -> tuple:
        """
        Define the parent of the converter class if the parent is defined in the kwargs;
        and the workspace to use.
        :param input_entity: The input entity to convert.
        :return: a tuple containing the (parent, workspace)
        """
        # pylint: disable=R1715
        if "parent" in kwargs and kwargs["parent"] is not None:
            workspace = kwargs["parent"].workspace
        elif "workspace" in kwargs:
            workspace = kwargs["workspace"]
        else:
            workspace = input_entity.workspace

        return workspace

    @classmethod
    def copy_properties(cls, input_entity: Entity, output: Entity):
        """
        Copy all the properties from the original entity to the new one.
        :param input_entity: The input entity to convert.
        :param output: the new entity.
        """
        for property_ in entity_properties:
            setattr(output, property_, getattr(input_entity, property_))


class CellObject(ConversionBase):
    @classmethod
    def copy_child_properties(
        cls, input_entity: Entity, output: Entity, association: str
    ):
        """
        Copy child properties from the original entity to the new one.
        :param input_entity: The input entity to convert.
        :param output: the new entity.
        :param association: association of the children to copy.
        """
        for child in input_entity.children:
            child.copy(
                parent=output,
                association=association
                if child.association == "CELL"
                else child.association,
            )

    @classmethod
    def to_points(cls, input_entity: Entity, **kwargs) -> Points:
        """
        Cell-based object conversion to Points
        :param input_entity: The input entity to convert.
        :return: a Points object.
        """
        # verify if the entity contains centroids
        if not hasattr(input_entity, "centroids"):
            raise TypeError(
                "Input entity for `GridObject` conversion must have centroids."
            )

        # get the properties
        properties = cls.verify_kwargs(input_entity, **kwargs)

        # create the point object
        output = objects.Points.create(
            properties["workspace"], vertices=input_entity.centroids, **kwargs
        )

        # copy the properties of the original object
        cls.copy_properties(input_entity, output)

        # change the association of the children
        cls.copy_child_properties(input_entity, output, association="VERTEX")

        return output
