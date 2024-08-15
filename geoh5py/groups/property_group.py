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
from collections.abc import Iterable
from enum import Enum
from typing import TYPE_CHECKING, Literal

from ..data import Data, DataAssociationEnum
from ..shared.utils import map_attributes

if TYPE_CHECKING:
    from ..objects import ObjectBase


class GroupTypeEnum(str, Enum):
    """
    Supported property group types.
    """

    DEPTH = "Depth table"
    DIPDIR = "Dip direction & dip"
    INTERVAL = "Interval table"
    MULTI = "Multi-element"
    STRIKEDIP = "Strike & dip"
    VECTOR = "3D vector"


class PropertyGroup:
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
    _name: str
    _uid: uuid.UUID

    def __init__(  # pylint: disable=too-many-arguments
        self,
        parent: ObjectBase,
        association: DataAssociationEnum = DataAssociationEnum.VERTEX,
        name=None,
        on_file=False,
        uid=None,
        property_group_type: Literal[GroupTypeEnum.MULTI] = GroupTypeEnum.MULTI,
        **kwargs,
    ):
        self.name = name or "property_group"
        self.uid = uid or uuid.uuid4()
        self._allow_delete = True
        self.on_file = on_file
        self.association = association

        if not hasattr(parent, "_property_groups"):
            raise AttributeError(
                f"Parent {parent} must have a 'property_groups' attribute"
            )

        self._parent: ObjectBase = parent
        self._properties: list[uuid.UUID] | None = None
        self.property_group_type = property_group_type

        parent.add_children([self])

        map_attributes(self, **kwargs)

        self.parent.workspace.register(self)

    def add_properties(self, data: Data | list[Data | uuid.UUID] | uuid.UUID):
        """
        Remove data from the properties.
        """
        if isinstance(data, (Data, uuid.UUID)):
            data = [data]

        properties = self._properties or []
        for elem in data:
            if isinstance(elem, uuid.UUID):
                entity = self.parent.get_entity(elem)[0]
            elif isinstance(elem, Data) and elem in self.parent.children:
                entity = elem
            else:
                continue

            if isinstance(entity, Data) and entity.uid not in properties:
                properties.append(entity.uid)

        if properties:
            self._properties = properties
            self.parent.workspace.add_or_update_property_group(self)

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
    def collect_values(self) -> list | None:
        """
        The values of the properties in the group.
        """

        if self._properties is None:
            return None

        return [self._parent.get_data(data)[0].values for data in self._properties]

    @property
    def name(self) -> str:
        """
        :obj:`str` Name of the group
        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        if not isinstance(new_name, str):
            raise TypeError("Name must be a string")

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
    def parent(self) -> ObjectBase:
        """
        The parent :obj:`~geoh5py.objects.object_base.ObjectBase`
        """
        return self._parent

    @property
    def properties(self) -> list[uuid.UUID] | None:
        """
        List of unique identifiers for the :obj:`~geoh5py.data.data.Data`
        contained in the property group.
        """
        return self._properties

    @properties.setter
    def properties(self, uids: list[str | uuid.UUID]):
        if self._properties is not None:
            raise UserWarning(
                "Cannot modify properties of an existing property group. "
                "Consider using 'add_properties'."
            )

        if not isinstance(uids, Iterable):
            return

        properties = []
        for uid in uids:
            if isinstance(uid, str):
                uid = uuid.UUID(uid)
            properties.append(uid)

        if not all(isinstance(uid, uuid.UUID) for uid in properties):
            raise TypeError("All uids must be of type uuid.UUID")

        self._properties = properties

    @property
    def property_group_type(self) -> str:
        return self._property_group_type

    @property_group_type.setter
    def property_group_type(self, value: str | GroupTypeEnum):
        if isinstance(value, str):
            try:
                value = GroupTypeEnum(value)
            except ValueError as error:
                raise ValueError(
                    f"Property group type must be one of "
                    f"{', '.join(GroupTypeEnum.__members__)}. Provided {value}"
                ) from error

        if not isinstance(value, GroupTypeEnum):
            raise TypeError(f"Association must be of type {GroupTypeEnum}")

        self._property_group_type = value

    def remove_properties(self, data: Data | list[Data | uuid.UUID] | uuid.UUID):
        """
        Remove data from the properties.
        """
        if isinstance(data, (Data, uuid.UUID)):
            data = [data]

        if self._properties is None:
            return

        for elem in data:

            if isinstance(elem, Data):
                elem = elem.uid

            if elem in self._properties:
                self._properties.remove(elem)

        if len(self._properties) == 0:
            self.parent.workspace.remove_entity(self)
            return

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

        if not isinstance(uid, uuid.UUID):
            raise TypeError(f"Could not convert input uid {uid} to type uuid.UUID")

        self._uid = uid
