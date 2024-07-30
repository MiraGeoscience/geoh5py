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

from abc import ABC
from typing import TYPE_CHECKING

from ... import objects


if TYPE_CHECKING:
    from ...objects import ObjectBase
    from ...workspace import Workspace

CORE_PROPERTIES = [
    "name",
    "allow_rename",
    "allow_move",
    "allow_delete",
]


class BaseConversion(ABC):
    def __init__(self):
        """
        Converter class from an :obj:geoh5py.shared.entity.Entity to another.
        """

    @classmethod
    def copy_child_properties(
        cls, input_entity: ObjectBase, output: ObjectBase, association: str, **kwargs
    ):
        """
        Copy child properties from the original entity to the new one.

        :param input_entity: The input entity to convert.
        :param output: The new entity.
        :param association: Association of the children to copy.
        """
        for child in input_entity.children:
            child.copy(parent=output, association=association, **kwargs)

    @classmethod
    def verify_kwargs(cls, input_entity, **kwargs) -> dict:
        """
        Verify if the kwargs are valid and update kwargs with core properties.

        :param input_entity: The input entity to convert.
        :param kwargs: Additional keyword arguments.

        :return: A dictionary of the valid kwargs.
        """
        output_properties = {key: getattr(input_entity, key) for key in CORE_PROPERTIES}

        for key, value in kwargs.items():
            if hasattr(input_entity, key):
                output_properties[key] = value

        return output_properties

    @classmethod
    def validate_workspace(
        cls, input_entity: ObjectBase, **kwargs
    ) -> tuple[Workspace, dict]:
        """
        Define the parent of the converter class if the parent is defined in the kwargs;
        and the workspace to use.
        :param input_entity: The input entity to convert.
        :return: a tuple containing the (parent, workspace)
        """
        # pylint: disable=R1715
        if "parent" in kwargs and kwargs["parent"] is not None:
            workspace = kwargs["parent"].workspace
            kwargs.pop("parent")
        elif "workspace" in kwargs:
            workspace = kwargs["workspace"]
            kwargs.pop("workspace")
        else:
            workspace = input_entity.workspace

        return workspace, kwargs


class CellObjectConversion(BaseConversion):
    """
    Converter class from a :obj:geoh5py.objects.CellObject to
    a :obj:geoh5py.objects.Points.
    """

    @classmethod
    def to_points(
        cls, input_entity: ObjectBase, copy_children=True, **kwargs
    ) -> objects.Points:
        """
        Cell-based object conversion to Points

        :param input_entity: The input entity to convert.

        :return: A Points object.
        """
        # verify if the entity contains centroids
        if getattr(input_entity, "centroids", None) is None:
            raise TypeError(
                "Input entity for `GridObject` conversion must have centroids."
            )

        workspace, kwargs = cls.validate_workspace(input_entity, **kwargs)

        # get the properties
        kwargs = cls.verify_kwargs(input_entity, **kwargs)
        kwargs["vertices"] = getattr(input_entity, "centroids", None)

        # create the point object
        output = objects.Points.create(workspace, **kwargs)

        # change the association of the children
        if copy_children:
            cls.copy_child_properties(input_entity, output, association="VERTEX")

        return output
