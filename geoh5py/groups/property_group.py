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

import uuid
from abc import ABC
from typing import TYPE_CHECKING

from geoh5py.data import Data, DataAssociationEnum

if TYPE_CHECKING:
    from geoh5py.objects import ObjectBase
    from geoh5py.shared import Entity


class PropertyGroup(ABC):
    """
    Property group listing data children of an object.
    This group is not registered to the workspace and only visible to the parent object.
    """

    _attribute_map = {
        "Association": "association",
        "Group Name": "name",
        "ID": "uid",
        "Properties": "properties",
        "Property Group Type": "property_group_type",
    }

    def __init__(self, parent: ObjectBase, on_file=False, **kwargs):
        self._parent: Entity
        self._name = "prop_group"
        self._uid = uuid.uuid4()
        self._allow_delete = True
        self.on_file = on_file
        self._association: DataAssociationEnum = DataAssociationEnum.VERTEX
        self._properties: list[uuid.UUID] = []
        self._property_group_type = "Multi-element"
        self.parent: ObjectBase = parent

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map:
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue

        self.parent.workspace.register_property_group(self)

    @property
    def allow_delete(self) -> bool:
        """
        :obj:`bool` Allow deleting the group
        """
        return self._allow_delete

    @allow_delete.setter
    def allow_delete(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("allow_delete must be a boolean")
        self._allow_delete = value

    @property
    def association(self) -> DataAssociationEnum:
        """
        :obj:`~geoh5py.data.data_association_enum.DataAssociationEnum` Data association
        """
        return self._association

    @association.setter
    def association(self, value: str | DataAssociationEnum):
        if isinstance(value, str):
            value = getattr(DataAssociationEnum, value.upper())

        if not isinstance(value, DataAssociationEnum):
            raise TypeError(
                f"Association must be 'VERTEX', 'CELL' or class of type {DataAssociationEnum}"
            )

        self._association = value

    @property
    def attribute_map(self) -> dict:
        """
        :obj:`dict` Attribute names mapping between geoh5 and geoh5py
        """
        return self._attribute_map

    @property
    def name(self) -> str:
        """
        :obj:`str` Name of the group
        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = new_name

    @property
    def on_file(self):
        """
        Property group is on geoh5 file.
        """
        return self._on_file

    @on_file.setter
    def on_file(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("Attribute 'on_file' must be a boolean.")

        self._on_file = value

    @property
    def parent(self) -> Entity:
        """
        The parent :obj:`~geoh5py.objects.object_base.ObjectBase`
        """
        return self._parent

    @parent.setter
    def parent(self, parent: Entity):
        if self._parent is not None:
            raise AttributeError("Cannot change parent of a property group.")

        if not hasattr(parent, "_property_groups"):
            raise AttributeError(
                f"Parent {parent} must have a 'property_groups' attribute"
            )

        parent.add_children([self])
        self._parent = parent

    @property
    def properties(self) -> list[uuid.UUID]:
        """
        List of unique identifiers for the :obj:`~geoh5py.data.data.Data`
        contained in the property group.
        """
        return self._properties

    @properties.setter
    def properties(self, uids: list[str | uuid.UUID]):
        properties = []
        for uid in uids:
            if isinstance(uid, str):
                uid = uuid.UUID(uid)
            properties.append(uid)
        self._properties = properties
        self.parent.workspace.add_or_update_property_group(self)

    @property
    def property_group_type(self) -> str:
        return self._property_group_type

    @property_group_type.setter
    def property_group_type(self, group_type: str):
        self._property_group_type = group_type

    def remove_data(self, data: Data | list[Data]):
        """
        Remove data from the properties.
        """
        if isinstance(data, Data):
            data = [data]

        if self._properties is None:
            return

        for entity in data:
            if entity.uid in self._properties:
                self._properties.remove(entity.uid)

        self.parent.workspace.add_or_update_property_group(self)

    @property
    def uid(self) -> uuid.UUID:
        """
        :obj:`uuid.UUID` Unique identifier
        """
        return self._uid

    @uid.setter
    def uid(self, uid: str | uuid.UUID):
        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        assert isinstance(
            uid, uuid.UUID
        ), f"Could not convert input uid {uid} to type uuid.UUID"
        self._uid = uid
