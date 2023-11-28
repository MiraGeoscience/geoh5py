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

from geoh5py.data import Data
from geoh5py.groups import PropertyGroup
from geoh5py.objects import ObjectBase
from geoh5py.shared.entity import Entity

from .group import Concatenator
from .property_group import ConcatenatedPropertyGroup


class Concatenated(Entity):
    """
    Base class modifier for concatenated objects and data.
    """

    _parent: Concatenated | Concatenator
    _concat_attr_str: str = "Attributes"

    def __init__(self, entity_type, **kwargs):
        attribute_map = getattr(self, "_attribute_map", {})
        attr = {"name": "Entity", "parent": None}
        for key, value in kwargs.items():
            attr[attribute_map.get(key, key)] = value

        super().__init__(entity_type, **attr)

    @property
    def concat_attr_str(self) -> str:
        """String identifier for the concatenated attributes."""
        return self._concat_attr_str

    @property
    def concatenator(self) -> Concatenator:
        """
        Parental Concatenator entity.
        """
        if isinstance(self._parent, Concatenated):
            return self._parent.concatenator

        return self._parent


class ConcatenatedObject(Concatenated, ObjectBase):
    _parent: Concatenator
    _property_groups: list | None = None

    def __init__(self, entity_type, **kwargs):
        if kwargs.get("parent") is None or not isinstance(
            kwargs.get("parent"), Concatenator
        ):
            raise UserWarning(
                "Creating a concatenated object must have a parent "
                "of type Concatenator."
            )

        super().__init__(entity_type, **kwargs)

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

    def get_data(self, name: str | uuid.UUID) -> list[Data]:
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
                elif not isinstance(child_data, PropertyGroup):
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
    def parent(self, parent):
        if not isinstance(parent, Concatenator):
            raise AttributeError(
                "The 'parent' of a concatenated Object must be of type "
                "'Concatenator'."
            )
        self._parent = parent
        self._parent.add_children([self])

    @property
    def property_groups(self) -> list | None:
        if self._property_groups is None:
            property_groups = self.concatenator.fetch_values(self, "property_group_ids")

            if property_groups is None or isinstance(self, Data):
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
