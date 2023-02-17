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

# pylint: disable=R0904

from __future__ import annotations

import uuid
import warnings
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ..data import CommentsData, Data
from ..data.data_association_enum import DataAssociationEnum
from ..data.primitive_type_enum import PrimitiveTypeEnum
from ..groups import PropertyGroup
from ..shared import Entity
from ..shared.conversion import BaseConversion
from ..shared.utils import clear_array_attributes, mask_by_extent
from .object_type import ObjectType

if TYPE_CHECKING:
    from .. import workspace


class ObjectBase(Entity):
    """
    Object base class.
    """

    _attribute_map: dict = Entity._attribute_map.copy()
    _attribute_map.update(
        {"Last focus": "last_focus", "PropertyGroups": "property_groups"}
    )
    _converter: type[BaseConversion] | None = None

    def __init__(self, object_type: ObjectType, **kwargs):
        assert object_type is not None
        self._comments = None
        self._entity_type = object_type
        self._extent = None
        self._last_focus = "None"
        self._property_groups: list[PropertyGroup] | None = None
        # self._clipping_ids: list[uuid.UUID] = []

        if not any(key for key in kwargs if key in ["name", "Name"]):
            kwargs["name"] = type(self).__name__

        super().__init__(**kwargs)

        if self.entity_type.name == "Entity":
            self.entity_type.name = type(self).__name__

    def add_comment(self, comment: str, author: str | None = None):
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
        self, data: dict, property_group: str | PropertyGroup | None = None
    ) -> Data | list[Data]:
        """
        Create :obj:`~geoh5py.data.data.Data` from dictionary of name and arguments.
        The provided arguments can be any property of the target Data class.

        :param data: Dictionary of data to be added to the object, e.g.

        .. code-block:: python

            data = {
                "data_A": {
                    'values': [v_1, v_2, ...],
                    'association': 'VERTEX'
                    },
                "data_B": {
                    'values': [v_1, v_2, ...],
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
            attr["name"] = name
            self.validate_data_association(attr)
            entity_type = self.validate_data_type(attr)
            kwargs = {"parent": self, "association": attr["association"]}
            for key, val in attr.items():
                if key in ["parent", "association", "entity_type", "type"]:
                    continue
                kwargs[key] = val

            data_object = self.workspace.create_entity(
                Data, entity=kwargs, entity_type=entity_type
            )

            if not isinstance(data_object, Data):
                continue

            if property_group is not None:
                self.add_data_to_group(data_object, property_group)

            data_objects.append(data_object)

        if len(data_objects) == 1:
            return data_objects[0]

        return data_objects

    def add_data_to_group(
        self, data: list | Data | uuid.UUID, property_group: str | PropertyGroup
    ) -> PropertyGroup:
        """
        Append data children to a :obj:`~geoh5py.groups.property_group.PropertyGroup`
        All given data must be children of the parent object.

        :param data: :obj:`~geoh5py.data.data.Data` object,
            :obj:`~geoh5py.shared.entity.Entity.uid` or
            :obj:`~geoh5py.shared.entity.Entity.name` of data.
        :param property_group: Name or :obj:`~geoh5py.groups.property_group.PropertyGroup`.
            A new group is created if none exist with the given name.

        :return: The target property group.
        """
        if isinstance(data, list):
            uids = []
            for datum in data:
                uids += self.reference_to_uid(datum)
        else:
            uids = self.reference_to_uid(data)

        association = None
        template = self.workspace.get_entity(uids[0])[0]
        if isinstance(template, Data):
            association = template.association

        if isinstance(property_group, str):
            property_group = self.find_or_create_property_group(
                name=property_group,
                association=association,
            )
        for uid in uids:
            assert uid in [
                child.uid for child in self.children
            ], f"Given data with uuid {uid} does not match any known children"
            if uid not in property_group.properties:
                property_group.properties.append(uid)

        self.workspace.update_attribute(self, "property_groups")

        return property_group

    @property
    def cells(self):
        """
        :obj:`numpy.array` of :obj:`int`: Array of indices
        defining the connection between
        :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`.
        """

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

    def copy_from_extent(
        self,
        bounds: np.ndarray,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
    ) -> ObjectBase | None:
        """
        Find indices of vertices within a rectangular bounds.

        :param bounds: shape(2, 2) Bounding box defined by the South-West and
            North-East coordinates. Extents can also be provided as 3D coordinates
            with shape(2, 3) defining the top and bottom limits.
        :param attributes: Dictionary of attributes to clip by extent.
        """
        if not any(mask_by_extent(bounds, self.extent)) and not any(
            mask_by_extent(self.extent, bounds)
        ):
            return None

        new_entity = self.copy(
            parent=parent, copy_children=copy_children, clear_cache=clear_cache
        )
        new_entity.clip_by_extent(bounds)

        if clear_cache:
            clear_array_attributes(new_entity, recursive=copy_children)

        return new_entity

    def clip_by_extent(self, bounds: np.ndarray) -> ObjectBase | None:
        """
        Find indices of cells within a rectangular bounds.

        :param bounds: shape(2, 2) Bounding box defined by the South-West and
            North-East coordinates. Extents can also be provided as 3D coordinates
            with shape(2, 3) defining the top and bottom limits.
        :param attributes: Dictionary of attributes to clip by extent.
        """

        # TODO Clip entity within bounds.
        warnings.warn(
            f"Method 'clip_by_extent' for entity {type(self)} not fully implemented. "
            f"Bounds {bounds} ignored."
        )
        return self

    @property
    def entity_type(self) -> ObjectType:
        """
        :obj:`~geoh5py.shared.entity_type.EntityType`: Object type.
        """
        return self._entity_type

    @property
    def extent(self):
        if self._extent is None:
            locations = getattr(self, "vertices", None)
            if locations is None:
                locations = getattr(self, "centroids", None)

            if locations is not None:
                self._extent = np.c_[locations.min(axis=0), locations.max(axis=0)].T

        return self._extent

    @property
    def faces(self):
        ...

    @classmethod
    def find_or_create_type(
        cls, workspace: workspace.Workspace, **kwargs
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
        property_groups = []
        if self._property_groups is not None:
            property_groups = self._property_groups

        if "name" in kwargs and any(
            pg.name == kwargs["name"] for pg in property_groups
        ):
            prop_group = [pg for pg in property_groups if pg.name == kwargs["name"]][0]
        else:
            if (
                "property_group_type" not in kwargs
                and "Property Group Type" not in kwargs
            ):
                kwargs["property_group_type"] = "Multi-element"

            prop_group = PropertyGroup(self, **kwargs)

            property_groups += [prop_group]

        self._property_groups = property_groups
        return prop_group

    def get_data(self, name: str | uuid.UUID) -> list[Data]:
        """
        Get a child :obj:`~geoh5py.data.data.Data` by name.

        :param name: Name of the target child data

        :return: A list of children Data objects
        """
        entity_list = []

        for child in self.children:
            if isinstance(child, Data):
                if (
                    isinstance(name, uuid.UUID) and child.uid == name
                ) or child.name == name:
                    entity_list.append(child)

        return entity_list

    def get_data_list(self, attribute="name") -> list[str]:
        """
        Get a list of names of all children :obj:`~geoh5py.data.data.Data`.

        :return: List of names of data associated with the object.
        """
        name_list = []
        for child in self.children:
            if isinstance(child, Data):
                name_list.append(getattr(child, attribute))
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
    def n_cells(self) -> int | None:
        """
        :obj:`int`: Number of cells.
        """
        if self.cells is not None:
            return self.cells.shape[0]
        return None

    @property
    def n_vertices(self) -> int | None:
        """
        :obj:`int`: Number of vertices.
        """
        if self.vertices is not None:
            return self.vertices.shape[0]
        return None

    @property
    def property_groups(self) -> list[PropertyGroup] | None:
        """
        List of :obj:`~geoh5py.groups.property_group.PropertyGroup`.
        """
        return self._property_groups

    def remove_property_groups(
        self, property_groups: PropertyGroup | list[PropertyGroup]
    ):
        if self.property_groups is None:
            return

        if isinstance(property_groups, PropertyGroup):
            property_groups = [property_groups]

        keepers = []
        for property_group in self.property_groups:
            if property_group not in property_groups:
                keepers += [property_group]

        if not keepers:
            self._property_groups = None
        else:
            self._property_groups = keepers

        self.workspace.update_attribute(self, "property_groups")

    def remove_children_values(self, indices: list[int], association: str):
        for child in self.children:
            if (
                getattr(child, "values", None) is not None
                and isinstance(child.association, DataAssociationEnum)
                and child.association.name == association
            ):
                child.values = np.delete(child.values, indices, axis=0)

    @property
    def vertices(self):
        r"""
        :obj:`numpy.array` of :obj:`float`, shape (\*, 3): Array of x, y, z coordinates
        defining the position of points in 3D space.
        """

    @property
    def converter(self):
        """
        :return: The converter for the object.
        """
        return self._converter

    def validate_data_association(self, attribute_dict):
        """
        Get a dictionary of attributes and validate the data 'association' keyword.
        """
        if attribute_dict.get("association") is not None:
            return

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
        else:
            attribute_dict["association"] = "OBJECT"

    @staticmethod
    def validate_data_type(attribute_dict):
        """
        Get a dictionary of attributes and validate the type of data.
        """
        entity_type = attribute_dict.get("entity_type")
        if entity_type is None:
            primitive_type = attribute_dict.get("type")
            if primitive_type is not None:
                assert (
                    primitive_type.upper() in PrimitiveTypeEnum.__members__
                ), f"Data 'type' should be one of {PrimitiveTypeEnum.__members__}"
                entity_type = {"primitive_type": primitive_type.upper()}
            else:
                values = attribute_dict.get("values")
                if values is None or (
                    isinstance(values, np.ndarray)
                    and (values.dtype in [np.float32, np.float64])
                ):
                    entity_type = {"primitive_type": "FLOAT"}
                elif isinstance(values, np.ndarray) and (
                    values.dtype in [np.uint32, np.int32]
                ):
                    entity_type = {"primitive_type": "INTEGER"}
                elif isinstance(values, str):
                    entity_type = {"primitive_type": "TEXT"}
                else:
                    raise NotImplementedError(
                        "Only add_data values of type FLOAT, INTEGER and TEXT have been implemented"
                    )

        return entity_type
