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

import uuid
from typing import TYPE_CHECKING

from ...data import Data
from ...groups.property_group import GroupTypeEnum
from ...objects import ObjectBase
from ..utils import INV_KEY_MAP
from .concatenated import Concatenated
from .data import ConcatenatedData
from .property_group import ConcatenatedPropertyGroup


if TYPE_CHECKING:
    from ..entity import Entity
    from .concatenator import Concatenator


class ConcatenatedObject(Concatenated, ObjectBase):
    def __init__(self, **kwargs):
        if kwargs.get("parent") is None:
            raise UserWarning(
                "Creating a concatenated object must have a parent "
                "of type Concatenator."
            )

        self._parent: Concatenator
        self._property_groups: list | None = None

        super().__init__(**kwargs)

    def create_property_group(
        self,
        name=None,
        property_group_type: GroupTypeEnum | str = GroupTypeEnum.INTERVAL,
        **kwargs,
    ) -> ConcatenatedPropertyGroup:
        """
        Create a new :obj:`~geoh5py.groups.property_group.PropertyGroup`.

        :param name: Name of the property group.
        :param property_group_type: Type of property group.
        :param kwargs: Any arguments taken by the
            :obj:`~geoh5py.groups.property_group.PropertyGroup` class.

        :return: A new :obj:`~geoh5py.groups.property_group.PropertyGroup`
        """
        if self._property_groups is not None and name in [
            pg.name for pg in self._property_groups
        ]:
            raise KeyError(f"A Property Group with name '{name}' already exists.")

        prop_group = ConcatenatedPropertyGroup(
            self, name=name, property_group_type=property_group_type, **kwargs
        )

        return prop_group

    def _fetch_concatenated_children(self):
        """
        Method to generate concatenated children.
        """
        attr = self.concatenator.get_concatenated_attributes(self.uid).copy()

        for key, value in attr.items():
            if "Property:" in key:
                child_data = self.workspace.get_entity(uuid.UUID(value))[0]
                if child_data is None:
                    attributes: dict = self.concatenator.get_concatenated_attributes(
                        value
                    ).copy()
                    attributes["parent"] = self
                    self.workspace.create_from_concatenation(attributes)
                elif not isinstance(child_data, ConcatenatedPropertyGroup):
                    self.add_children([child_data])

    def get_entity(self, name: str | uuid.UUID) -> list[Entity | None]:
        """
        Get a child :obj:`~geoh5py.data.data.Data` by name.

        :param name: Name of the target child data.

        :return: A list of children Data objects
        """
        if not any(child for child in self.children if isinstance(child, Data)):
            self._fetch_concatenated_children()

        if isinstance(name, uuid.UUID):
            entity_list = [child for child in self.children if child.uid == name]
        else:
            entity_list = [child for child in self.children if child.name == name]

        if not entity_list:
            return [None]

        return entity_list

    def get_data_list(self, attribute="name"):
        """
        Lazy loading of data names from concatenated attributes.
        """
        data_list = [
            attr.replace("Property:", "").replace("\u2044", "/")
            for attr in self.concatenator.get_concatenated_attributes(self.uid)
            if "Property:" in attr
        ]

        return data_list

    @property
    def parent(self) -> Concatenator:
        return self._parent

    @parent.setter
    def parent(self, parent: Concatenator):
        if not hasattr(parent, "add_children"):
            raise ValueError(
                "The 'parent' of a concatenated Object must have an "
                "'add_children' method."
            )
        parent.add_children([self])
        self._parent = parent

    @property
    def property_groups(self) -> list | None:
        if self._property_groups is None:
            property_groups = self.concatenator.fetch_values(self, "property_group_ids")

            if property_groups is None or isinstance(self, ConcatenatedData):
                property_groups = []

            for key in property_groups:
                attributes = {
                    INV_KEY_MAP.get(key, key): val
                    for key, val in self.concatenator.get_concatenated_attributes(
                        key
                    ).items()
                }
                self.fetch_property_group(**attributes, on_file=True)

            property_groups = [
                child
                for child in self.children
                if isinstance(child, ConcatenatedPropertyGroup)
            ]

            if len(property_groups) > 0:
                self._property_groups = property_groups

        return self._property_groups

    def remove_children(
        self, children: list | Concatenated | ConcatenatedPropertyGroup
    ):
        """
        Remove children from object.

        This method calls the ObjectBase parent class to remove children from the
        object children, but also deletes the children from the workspace.

        :param children: List of children to remove.
        """
        if not isinstance(children, list):
            children = [children]

        for child in children:
            if child not in self._children:
                continue

            self.concatenator.remove_entity(child)
            self._children.remove(child)
