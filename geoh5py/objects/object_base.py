#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import uuid
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from ..data import CommentsData, Data
from ..data.data_association_enum import DataAssociationEnum
from ..data.primitive_type_enum import PrimitiveTypeEnum
from ..groups import PropertyGroup
from ..shared import Entity
from .object_type import ObjectType

if TYPE_CHECKING:
    from .. import workspace


class ObjectBase(Entity):
    """
    Object base class.
    """

    _attribute_map = Entity._attribute_map.copy()
    _attribute_map.update(
        {"Last focus": "last_focus", "PropertyGroups": "property_groups"}
    )

    def __init__(self, object_type: ObjectType, **kwargs):
        assert object_type is not None
        self._entity_type = object_type
        self._property_groups: List[PropertyGroup] = []
        self._last_focus = "None"
        self._comments = None
        # self._clipping_ids: List[uuid.UUID] = []

        if not any(key for key in kwargs if key in ["name", "Name"]):
            kwargs["name"] = type(self).__name__

        super().__init__(**kwargs)

        if self.entity_type.name == "Entity":
            self.entity_type.name = type(self).__name__

    def add_comment(self, comment: str, author: str = None):
        """
        Add text comment to an object.

        :param comment: Text to be added as comment.
        :param author: Name of author or defaults to
            :obj:`~geoh5py.workspace.workspace.Workspace.contributors`.
        """

        date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        if author is None:
            author = ",".join(self.workspace.contributors)

        comment_dict = {"Author": author, "Date": date, "Text": comment}

        if self.comments is None:
            self.add_data(
                {
                    "UserComments": {
                        "values": [comment_dict],
                        "association": "OBJECT",
                        "entity_type": {"primitive_type": "TEXT"},
                    }
                }
            )
        else:
            self.comments.values = self.comments.values + [comment_dict]

    def add_data(
        self, data: dict, property_group: str = None
    ) -> Union[Data, List[Data]]:
        """
        Create :obj:`~geoh5py.data.data.Data` from dictionary of name and arguments.
        The provided arguments can be any property of the target Data class.

        :param data: Dictionary of data to be added to the object, e.g.

        .. code-block:: python

            data = {
                "data_A": {
                    'values', [v_1, v_2, ...],
                    'association': 'VERTEX'
                    },
                "data_B": {
                    'values', [v_1, v_2, ...],
                    'association': 'CELLS'
                    },
            }

        :return: List of new Data objects.
        """
        data_objects = []
        for name, attr in data.items():
            assert isinstance(attr, dict), (
                f"Given value to data {name} should of type {dict}. "
                f"Type {type(attr)} given instead."
            )
            assert "values" in list(
                attr.keys()
            ), f"Given attr for data {name} should include 'values'"

            attr["name"] = name

            self.validate_data_association(attr)
            entity_type = self.validate_data_type(attr)

            # Re-order to set parent first
            kwargs = {"parent": self, "association": attr["association"]}
            for key, val in attr.items():
                if key in ["parent", "association", "entity_type", "type"]:
                    continue
                kwargs[key] = val

            data_object = self.workspace.create_entity(
                Data, entity=kwargs, entity_type=entity_type
            )

            if property_group is not None:
                self.add_data_to_group(data_object, property_group)

            data_objects.append(data_object)

        if len(data_objects) == 1:
            return data_object

        return data_objects

    def add_data_to_group(
        self, data: Union[List, Data, uuid.UUID, str], name: str
    ) -> PropertyGroup:
        """
        Append data children to a :obj:`~geoh5py.groups.property_group.PropertyGroup`
        All given data must be children of the parent object.

        :param data: :obj:`~geoh5py.data.data.Data` object,
            :obj:`~geoh5py.shared.entity.Entity.uid` or
            :obj:`~geoh5py.shared.entity.Entity.name` of data.
        :param name: Name of a :obj:`~geoh5py.groups.property_group.PropertyGroup`.
            A new group is created if none exist with the given name.

        :return: The target property group.
        """
        prop_group = self.find_or_create_property_group(name=name)
        if isinstance(data, list):
            uids = []
            for datum in data:
                uids += self.reference_to_uid(datum)
        else:
            uids = self.reference_to_uid(data)

        for uid in uids:
            assert uid in [
                child.uid for child in self.children
            ], f"Given data with uuid {uid} does not match any known children"
            if uid not in prop_group.properties:
                prop_group.properties.append(uid)
                self.modified_attributes = "property_groups"

        return prop_group

    def remove_data_from_group(
        self, data: Union[List, Data, uuid.UUID, str], name: str = None
    ):
        """
        Remove data children to a :obj:`~geoh5py.groups.property_group.PropertyGroup`
        All given data must be children of the parent object.

        :param data: :obj:`~geoh5py.data.data.Data` object,
            :obj:`~geoh5py.shared.entity.Entity.uid` or
            :obj:`~geoh5py.shared.entity.Entity.name` of data.
        :param name: Name of a :obj:`~geoh5py.groups.property_group.PropertyGroup`.
            A new group is created if none exist with the given name.
        """
        if getattr(self, "property_groups", None) is not None:

            if isinstance(data, list):
                uids = []
                for datum in data:
                    uids += self.reference_to_uid(datum)
            else:
                uids = self.reference_to_uid(data)

            if name is not None:
                prop_groups = [
                    prop_group
                    for prop_group in self.property_groups
                    if prop_group.name == name
                ]
            else:
                prop_groups = self.property_groups

            for prop_group in prop_groups:
                for uid in uids:
                    if uid in prop_group.properties:
                        prop_group.properties.remove(uid)
                        self.modified_attributes = "property_groups"

    @property
    def cells(self):
        """
        :obj:`numpy.array` of :obj:`int`: Array of indices
        defining the connection between
        :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`.
        """
        ...

    @property
    def comments(self):
        """
        Fetch a :obj:`~geoh5py.data.text_data.CommentsData` entity from children.
        """
        for child in self.children:
            if isinstance(child, CommentsData):
                return child

        return None

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        ...

    @property
    def entity_type(self) -> ObjectType:
        """
        :obj:`~geoh5py.shared.entity_type.EntityType`: Object type.
        """
        return self._entity_type

    @property
    def faces(self):
        ...

    @classmethod
    def find_or_create_type(
        cls, workspace: "workspace.Workspace", **kwargs
    ) -> ObjectType:
        """
        Find or create a type instance for a given object class.

        :param workspace: Target :obj:`~geoh5py.workspace.workspace.Workspace`.

        :return: The ObjectType instance for the given object class.
        """
        return ObjectType.find_or_create(workspace, cls, **kwargs)

    def find_or_create_property_group(self, **kwargs) -> PropertyGroup:
        """
        Find or create :obj:`~geoh5py.groups.property_group.PropertyGroup`
        from given name and properties.

        :param kwargs: Any arguments taken by the
            :obj:`~geoh5py.groups.property_group.PropertyGroup` class.

        :return: A new or existing :obj:`~geoh5py.groups.property_group.PropertyGroup`
        """
        if "name" in list(kwargs.keys()) and any(
            pg.name == kwargs["name"] for pg in self.property_groups
        ):
            prop_group = [
                pg for pg in self.property_groups if pg.name == kwargs["name"]
            ][0]
        else:
            prop_group = PropertyGroup(**kwargs)
            self.property_groups = [prop_group]

        return prop_group

    def get_data(self, name: str) -> List[Data]:
        """
        Get a child :obj:`~geoh5py.data.data.Data` by name.

        :param name: Name of the target child data

        :return: A list of children Data objects
        """
        entity_list = []

        for child in self.children:
            if isinstance(child, Data) and child.name == name:
                entity_list.append(child)

        return entity_list

    def get_data_list(self) -> List[str]:
        """
        Get a list of names of all children :obj:`~geoh5py.data.data.Data`.

        :return: List of names of data associated with the object.
        """
        name_list = []
        for child in self.children:
            if isinstance(child, Data):
                name_list.append(child.name)
        return sorted(name_list)

    @property
    def last_focus(self) -> str:
        """
        :obj:`bool`: Object visible in camera on start.
        """
        return self._last_focus

    @last_focus.setter
    def last_focus(self, value: str):
        self._last_focus = value

    @property
    def n_cells(self) -> Optional[int]:
        """
        :obj:`int`: Number of cells.
        """
        if self.cells is not None:
            return self.cells.shape[0]
        return None

    @property
    def n_vertices(self) -> Optional[int]:
        """
        :obj:`int`: Number of vertices.
        """
        if self.vertices is not None:
            return self.vertices.shape[0]
        return None

    @property
    def property_groups(self) -> List[PropertyGroup]:
        """
        :obj:`list` of :obj:`~geoh5py.groups.property_group.PropertyGroup`.
        """
        return self._property_groups

    @property_groups.setter
    def property_groups(self, prop_groups: List[PropertyGroup]):
        # Check for existing property_group
        for prop_group in prop_groups:
            if not any(
                pg.uid == prop_group.uid for pg in self.property_groups
            ) and not any(pg.name == prop_group.name for pg in self.property_groups):
                prop_group.parent = self

                self.modified_attributes = "property_groups"
                self._property_groups = self.property_groups + [prop_group]

    @property
    def vertices(self):
        r"""
        :obj:`numpy.array` of :obj:`float`, shape (\*, 3): Array of x, y, z coordinates
        defining the position of points in 3D space.
        """
        ...

    def validate_data_association(self, attribute_dict):
        """
        Get a dictionary of attributes and validate the data 'association' keyword.
        """

        if "association" in list(attribute_dict.keys()):
            assert attribute_dict["association"] in [
                enum.name for enum in DataAssociationEnum
            ], (
                "Data 'association' must be one of "
                + f"{[enum.name for enum in DataAssociationEnum]}. "
                + f"{attribute_dict['association']} provided."
            )
        else:
            attribute_dict["association"] = "OBJECT"
            if (
                getattr(self, "n_cells", None) is not None
                and attribute_dict["values"].ravel().shape[0] == self.n_cells
            ):
                attribute_dict["association"] = "CELL"
            elif (
                getattr(self, "n_vertices", None) is not None
                and attribute_dict["values"].ravel().shape[0] == self.n_vertices
            ):
                attribute_dict["association"] = "VERTEX"

    @staticmethod
    def validate_data_type(attribute_dict):
        """
        Get a dictionary of attributes and validate the type of data.
        """

        if "entity_type" in list(attribute_dict.keys()):
            entity_type = attribute_dict["entity_type"]
        elif "type" in list(attribute_dict.keys()):
            assert attribute_dict["type"].upper() in list(
                PrimitiveTypeEnum.__members__.keys()
            ), f"Data 'type' should be one of {list(PrimitiveTypeEnum.__members__.keys())}"
            entity_type = {"primitive_type": attribute_dict["type"].upper()}
        else:
            if isinstance(attribute_dict["values"], np.ndarray) and (
                attribute_dict["values"].dtype in [np.float32, np.float64]
            ):
                entity_type = {"primitive_type": "FLOAT"}
            elif isinstance(attribute_dict["values"], np.ndarray) and (
                attribute_dict["values"].dtype in [np.uint32, np.int32]
            ):
                entity_type = {"primitive_type": "INTEGER"}
            elif isinstance(attribute_dict["values"], str):
                entity_type = {"primitive_type": "TEXT"}
            else:
                raise NotImplementedError(
                    "Only add_data values of type FLOAT, INTEGER and TEXT have been implemented"
                )

        return entity_type
