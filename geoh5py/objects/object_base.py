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
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from ..data import CommentsData, Data, VisualParameters
from ..data.data_association_enum import DataAssociationEnum
from ..data.primitive_type_enum import PrimitiveTypeEnum
from ..groups import PropertyGroup
from ..shared import Entity
from ..shared.conversion import BaseConversion
from ..shared.utils import clear_array_attributes
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
        self._last_focus = "None"
        self._property_groups: list[PropertyGroup] | None = None
        # self._clipping_ids: list[uuid.UUID] = []
        self._visual_parameters: VisualParameters | None = None

        if not any(key for key in kwargs if key in ["name", "Name"]):
            kwargs["name"] = type(self).__name__

        super().__init__(**kwargs)

        if self.entity_type.name == "Entity":
            self.entity_type.name = type(self).__name__

    def add_children(self, children: list[Entity] | list[PropertyGroup]):
        """
        :param children: Add a list of entities as
            :obj:`~geoh5py.shared.entity.Entity.children`
        """
        property_groups = self._property_groups or []

        for child in children:
            if child not in self._children:
                self._children.append(child)

            if isinstance(child, PropertyGroup) and child not in property_groups:
                property_groups.append(child)

            if property_groups:
                self._property_groups = property_groups

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

        :param property_group: Name or :obj:`~geoh5py.groups.property_group.PropertyGroup`.
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
        if not isinstance(data, list):
            data = [data]

        children = []
        associations = []
        for entity in data:
            if isinstance(entity, (uuid.UUID, str)):
                entity = self.get_entity(entity)[0]
            children.append(entity)
            associations.append(entity.association)

        associations = list(set(associations))
        if len(associations) > 1:
            raise ValueError("All data must have the same association.")

        if isinstance(property_group, str):
            property_group = self.find_or_create_property_group(
                name=property_group,
                association=associations[0],
            )

        property_group.add_properties(children)

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

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Function to copy an entity to a different parent entity.

        :param parent: New parent for the copied object.
        :param copy_children: Copy children entities.
        :param clear_cache: Clear cache of data values.
        :param mask: Array of indices to sub-sample the input entity.
        :param kwargs: Additional keyword arguments.

        :return: New copy of the input entity.
        """

        if parent is None:
            parent = self.parent

        new_object = self.workspace.copy_to_parent(
            self,
            parent,
            clear_cache=clear_cache,
            **kwargs,
        )

        if copy_children:
            children_map = {}
            for child in self.children:
                if isinstance(child, PropertyGroup):
                    continue
                if isinstance(child, Data) and child.association in (
                    DataAssociationEnum.VERTEX,
                    DataAssociationEnum.CELL,
                ):
                    child_copy = child.copy(
                        parent=new_object,
                        clear_cache=clear_cache,
                        mask=mask,
                    )
                else:
                    child_copy = self.workspace.copy_to_parent(
                        child, new_object, clear_cache=clear_cache
                    )
                children_map[child.uid] = child_copy.uid

            if self.property_groups:
                self.workspace.copy_property_groups(
                    new_object, self.property_groups, children_map
                )
                new_object.workspace.update_attribute(new_object, "property_groups")

        return new_object

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        Default entity type unique identifier
        """

    @abstractmethod
    def mask_by_extent(
        self, extent: np.ndarray, inverse: bool = False
    ) -> np.ndarray | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        """

    @property
    def entity_type(self) -> ObjectType:
        """
        :obj:`~geoh5py.shared.entity_type.EntityType`: Object type.
        """
        return self._entity_type

    @property
    @abstractmethod
    def extent(self):
        """
        Geography bounding box of the object.

        :return: shape(2, 3) Bounding box defined by the bottom South-West and
            top North-East coordinates.
        """

    @property
    def faces(self):
        """Object faces."""

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

    def get_property_group(self, name: uuid.UUID | str) -> list:
        """
        Get a child :obj:`~geoh5py.groups.property_group.PropertyGroup` by name.
        :param name: the reference of the property group to get.
        :return: A list of children Data objects
        """
        if self.property_groups is None:
            return [None]

        entities = []

        for child in self.property_groups:
            if (
                isinstance(name, uuid.UUID) and child.uid == name
            ) or child.name == name:
                entities.append(child)

        if len(entities) == 0:
            return [None]

        return entities

    def create_property_group(self, on_file=False, **kwargs) -> PropertyGroup:
        """
        Create a new :obj:`~geoh5py.groups.property_group.PropertyGroup`.
        :param kwargs: Any arguments taken by the
            :obj:`~geoh5py.groups.property_group.PropertyGroup` class.
        :return: A new :obj:`~geoh5py.groups.property_group.PropertyGroup`
        """
        if (
            "name" in kwargs
            and self.property_groups is not None
            and any(pg.name == kwargs["name"] for pg in self.property_groups)
        ):
            raise KeyError(
                f"A Property Group with name {kwargs['name']} already exists."
            )

        if "property_group_type" not in kwargs and "Property Group Type" not in kwargs:
            kwargs["property_group_type"] = "Multi-element"

        prop_group = PropertyGroup(self, on_file=on_file, **kwargs)

        return prop_group

    def find_or_create_property_group(self, **kwargs) -> PropertyGroup:
        """
        Find or create :obj:`~geoh5py.groups.property_group.PropertyGroup`
        from given name and properties.

        :param kwargs: Any arguments taken by the
            :obj:`~geoh5py.groups.property_group.PropertyGroup` class.

        :return: A new or existing :obj:`~geoh5py.groups.property_group.PropertyGroup`
        """

        prop_group = None
        if "name" in kwargs:
            prop_group = self.get_property_group(kwargs["name"])[0]

        if prop_group is None:
            prop_group = self.create_property_group(**kwargs)

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

    def remove_children(self, children: list[Entity] | list[PropertyGroup]):
        """
        Remove children from the list of children entities.

        :param children: List of entities

        .. warning::
            Removing a child entity without re-assigning it to a different
            parent may cause it to become inactive. Inactive entities are removed
            from the workspace by
            :func:`~geoh5py.shared.weakref_utils.remove_none_referents`.
        """
        if not isinstance(children, list):
            children = [children]

        for child in children:
            if child not in self._children:
                continue

            self._children.remove(child)

            if not self._property_groups:
                continue

            if isinstance(child, PropertyGroup):
                self._property_groups.remove(child)
            elif isinstance(child, Data):
                self.remove_data_from_groups(child)

        self.workspace.remove_children(self, children)

    def remove_children_values(
        self,
        indices: list[int] | np.ndarray,
        association: str,
        clear_cache: bool = False,
    ):
        if isinstance(indices, list):
            indices = np.array(indices)

        if not isinstance(indices, np.ndarray):
            raise TypeError("Indices must be a list or numpy array.")

        for child in self.children:
            if (
                getattr(child, "values", None) is not None
                and isinstance(child.association, DataAssociationEnum)
                and child.association.name == association
            ):
                child.values = np.delete(child.values, indices, axis=0)
                if clear_cache:
                    clear_array_attributes(child)

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
                elif isinstance(values, np.ndarray) and (values.dtype == bool):
                    entity_type = {"primitive_type": "BOOLEAN"}
                else:
                    raise NotImplementedError(
                        "Only add_data values of type FLOAT, INTEGER,"
                        "BOOLEAN and TEXT have been implemented"
                    )

        return entity_type

    def add_default_visual_parameters(self):
        """
        Add default visual parameters to the object.
        """
        if self.visual_parameters is not None:
            raise UserWarning("Visual parameters already exist.")

        self.workspace.create_entity(  # type: ignore
            Data,
            save_on_creation=True,
            **{
                "entity": {
                    "name": "Visual Parameters",
                    "parent": self,
                    "association": "OBJECT",
                },
                "entity_type": {"name": "XmlData", "primitive_type": "TEXT"},
            },
        )

        return self._visual_parameters

    def remove_data_from_groups(self, data: list[Data] | Data) -> None:
        """
        Remove data children to all
        :obj:`~geoh5py.groups.property_group.PropertyGroup` of the object.

        :param data: :obj:`~geoh5py.data.data.Data` object,
            :obj:`~geoh5py.shared.entity.Entity.uid` or
            :obj:`~geoh5py.shared.entity.Entity.name` of data.
        """
        if not isinstance(data, list):
            data = [data]

        if not self.property_groups:
            return

        for property_group in self.property_groups:
            property_group.remove_properties(data)

    @property
    def visual_parameters(self) -> VisualParameters | None:
        """
        Access the visual parameters of the object.
        """
        if self._visual_parameters is None:
            for child in self.children:
                if isinstance(child, VisualParameters):
                    self._visual_parameters = child
                    break

        return self._visual_parameters

    @visual_parameters.setter
    def visual_parameters(self, value: VisualParameters):
        if not isinstance(value, VisualParameters):
            raise TypeError("visual_parameters must be a VisualParameters object.")

        self._visual_parameters = value
