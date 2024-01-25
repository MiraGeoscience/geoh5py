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
import warnings
from abc import ABC
from typing import TYPE_CHECKING

from ...objects import ObjectBase
from .concatenated import Concatenated
from .data import ConcatenatedData
from .property_group import ConcatenatedPropertyGroup

if TYPE_CHECKING:
    from .concatenator import Concatenator


class ConcatenatedObject(Concatenated, ObjectBase, ABC):
    _parent: Concatenator
    _property_groups: list | None = None

    def __init__(self, entity_type, **kwargs):
        if kwargs.get("parent") is None:
            raise UserWarning(
                "Creating a concatenated object must have a parent "
                "of type Concatenator."
            )

        super().__init__(entity_type, **kwargs)

    def add_children(
        self,
        children,
    ):
        """
        :param children: Add a list of entities as
            :obj:`~geoh5py.shared.entity.Entity.children`
        """
        property_groups = self._property_groups or []

        for child in children:
            if child not in self._children and isinstance(
                child, (ConcatenatedData, ConcatenatedPropertyGroup)
            ):
                self._children.append(child)

            if (
                isinstance(child, ConcatenatedPropertyGroup)
                and child not in property_groups
            ):
                property_groups.append(child)

            if property_groups:
                self._property_groups = property_groups

    def create_property_group(
        self, name=None, on_file=False, uid=None, **kwargs
    ) -> ConcatenatedPropertyGroup:
        """
        Create a new :obj:`~geoh5py.groups.property_group.PropertyGroup`.

        :param name: Name of the property group.
        :param on_file: If True, the property group is saved to file.
        :param uid: Unique ID of the property group.
        :param kwargs: Any arguments taken by the
            :obj:`~geoh5py.groups.property_group.PropertyGroup` class.

        :return: A new :obj:`~geoh5py.groups.property_group.PropertyGroup`
        """
        if self._property_groups is not None and name in [
            pg.name for pg in self._property_groups
        ]:
            raise KeyError(f"A Property Group with name '{name}' already exists.")

        if "property_group_type" not in kwargs and "Property Group Type" not in kwargs:
            kwargs["property_group_type"] = "Interval table"

        prop_group = ConcatenatedPropertyGroup(
            self, name=name, on_file=on_file, **kwargs
        )

        return prop_group

    def get_data(self, name: str | uuid.UUID):
        """
        Generic function to get data values from object.
        """
        entity_list = []
        attr = self.concatenator.get_concatenated_attributes(
            getattr(self, "uid")
        ).copy()

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
                else:
                    warnings.warn(f"Failed: '{name}' is a property group, not a Data.")

        for child in getattr(self, "children"):
            if (
                isinstance(name, str) and hasattr(child, "name") and child.name == name
            ) or (
                isinstance(name, uuid.UUID)
                and hasattr(child, "uid")
                and child.uid == name
            ):
                entity_list.append(child)

        return entity_list

    def get_data_list(self, attribute="name"):
        """
        Get list of data names.
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
                self.find_or_create_property_group(
                    **self.concatenator.get_concatenated_attributes(key), on_file=True
                )

            property_groups = [
                child
                for child in self.children
                if isinstance(child, ConcatenatedPropertyGroup)
            ]

            if len(property_groups) > 0:
                self._property_groups = property_groups

        return self._property_groups
