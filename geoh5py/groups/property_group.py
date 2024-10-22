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

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING
from uuid import UUID, uuid4
from warnings import warn

from ..data import Data, DataAssociationEnum
from ..shared.utils import (
    find_unique_name,
    remove_duplicates_in_list,
)
from .property_group_table import PropertyGroupTable
from .property_group_type import GroupTypeEnum


if TYPE_CHECKING:  # pragma: no cover
    from ..objects import ObjectBase


class PropertyGroup:
    """
    Property group listing data children of an object.
    This group is not registered to the workspace and only visible to the parent object.

    :param parent: Parent object.
    :param association: Association of the data.
    :param allow_delete: Allow deleting the group.
    :param name: Name of the group.
    :param on_file: Property group is on file.
    :param uid: Unique identifier.
    :param property_group_type: Type of property group.
    :param properties: List of data or unique identifiers for the data.
    """

    _attribute_map = {
        "Association": "association",
        "Group Name": "name",
        "ID": "uid",
        "Properties": "properties",
        "Property Group Type": "property_group_type",
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        parent: ObjectBase,
        *,
        association: str | DataAssociationEnum | None = None,
        allow_delete: bool = True,
        name: str = "Property Group",
        on_file: bool = False,
        uid: UUID | None = None,
        property_group_type: GroupTypeEnum | str = GroupTypeEnum.SIMPLE,
        properties: list[UUID | Data | str] | None = None,
        **_,
    ):
        self._parent: ObjectBase = self._validate_parent(parent)
        self._property_group_type = self._validate_group_type(property_group_type)

        self.allow_delete = allow_delete
        self.name = name
        self.on_file = on_file
        self.uid = uid or uuid4()

        properties_list = self._initialize_properties(properties)

        self._association = self._validate_association(association, properties_list)
        self._properties = self._validate_properties(properties_list)

        self.parent.add_children([self])
        self.parent.workspace.register(self)

    def _initialize_properties(
        self, properties: str | UUID | Data | list[UUID | str | Data] | None
    ) -> list[Data] | None:
        """
        Initialize the properties list.

        :param properties: List of Data entities to validate.

        :return: List of unique identifiers for the Data entities.
        """
        if properties is None:
            return None

        if not isinstance(properties, Iterable):
            properties = [properties]

        return [self.parent.reference_to_data(prop) for prop in properties]

    @staticmethod
    def _validate_association(
        value: str | DataAssociationEnum | None, properties: list[Data] | None
    ) -> DataAssociationEnum:
        """
        Verify that the association is valid, or infer it from the properties.

        :param value: The association to validate.
        :param properties: A list of properties to infer the association from.
        """
        if properties is None and value is None:
            raise ValueError(
                "At least one of 'properties' or 'association' must be provided."
            )

        if value is None and properties is not None:
            value = properties[0].association

        if isinstance(value, str):
            value = getattr(DataAssociationEnum, value.upper())

        if not isinstance(value, DataAssociationEnum):
            raise TypeError(f"Association must be one of type {DataAssociationEnum}")

        return value

    def _validate_data(self, data: Data | UUID | str) -> Data:
        """
        Verify that the data is in the parent and has the same association as the group.

        :param data: The data to verify.
            It can be the name, the uuid or the data itself.

        :return: The uuid of the data.
        """
        data = self.parent.reference_to_data(data)

        if self.association != data.association:
            raise ValueError(
                f"Data '{data.name}' association ({data.association}) "
                f"does not match group association ({self.association})."
            )

        return data

    @staticmethod
    def _validate_group_type(value: str | GroupTypeEnum) -> GroupTypeEnum:
        """
        Verify that the group type is a valid GroupTypeEnum.

        :param value: The group type to validate.

        :return: The validated group type.
        """
        if isinstance(value, str):
            try:
                value = GroupTypeEnum(value)
            except ValueError as error:
                raise ValueError(
                    f"'Property group type' must be one of "
                    f"{', '.join(GroupTypeEnum.__members__)}. Provided {value}"
                ) from error

        if not isinstance(value, GroupTypeEnum):
            raise TypeError(
                f"'Property group type' must be of type {GroupTypeEnum}, "
                f"provided {type(value)}"
            )

        return value

    @staticmethod
    def _validate_parent(parent: ObjectBase) -> ObjectBase:
        """
        Verify that the parent is valid.

        :param parent: The parent Object to validate.

        :return: The parent Object.
        """
        # define the parent
        if not hasattr(parent, "_property_groups"):
            raise TypeError(f"Parent {parent} must have a 'property_groups' attribute")
        return parent

    def _validate_properties(
        self, data_list: Sequence[str | UUID | Data] | None
    ) -> list[UUID] | None:
        """
        Validate the properties list.

        :param data_list: List of Data entities to validate.

        :return: List of unique identifiers for the Data entities.
        """
        if not data_list:
            return None

        data_list_ = remove_duplicates_in_list(
            [self._validate_data(uid) for uid in data_list]
        )

        self._property_group_type.verify(data_list_)

        return [data.uid for data in data_list_]

    def add_properties(self, data: str | Data | list[str | Data | UUID] | UUID):
        """
        Add data to properties.

        :param data: Data to add to the group.
            It can be the name, the uuid or the data itself in a list or alone.
        """
        if self._property_group_type.no_modify:
            raise ValueError(
                f"Cannot add properties to '{self._property_group_type}' property group type."
            )

        if not isinstance(data, list):
            data = [data]

        properties = self._validate_properties(
            data if self.properties is None else self.properties + data
        )

        if properties:
            self._properties = remove_duplicates_in_list(properties)
            self.parent.workspace.add_or_update_property_group(self)

    @property
    def allow_delete(self) -> bool:
        """
        Allow deleting the group
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
        The association of the data.
        """
        return self._association

    @property
    def attribute_map(self) -> dict:
        """
        Attribute names mapping between geoh5 and geoh5py
        """
        return self._attribute_map

    @property
    def collect_values(self) -> list | None:
        """
        The values of the properties in the group.
        """
        warn(
            "PropertyGroup.collect_values is deprecated, use PropertyGroup.table instead.",
            DeprecationWarning,
        )

        if self._properties is None:
            return None

        return [self._parent.get_data(data)[0].values for data in self._properties]

    @property
    def name(self) -> str:
        """
        Name of the group
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
        The parent of the PropertyGroup.
        """
        return self._parent

    @property
    def properties(self) -> list[UUID] | None:
        """
        List of unique identifiers for the :obj:`~geoh5py.data.data.Data`
        contained in the property group.
        """
        return self._properties

    @property
    def properties_name(self) -> list[str] | None:
        """
        List of names of the properties`
        """
        if self._properties is None:
            return None

        names: list[str] = []
        for uid in self._properties:
            data = self.parent.get_data(uid)[0]
            name = str(data.uid) if data.name is None else data.name
            names.append(find_unique_name(name, names))

        return names

    @property
    def property_group_type(self) -> GroupTypeEnum:
        """
        Type of property group.
        """
        return self._property_group_type

    def remove_properties(self, data: str | Data | list[str | Data | UUID] | UUID):
        """
        Remove data from the properties.

        :param data: Data to remove from the group.
            It can be the name, the uuid or the data itself in a list or alone.
        """
        if self._property_group_type.no_modify:
            raise ValueError(
                f"Cannot remove properties from '{self._property_group_type}' property group type."
            )

        if self.properties is None:
            return

        if not isinstance(data, list):
            data = [data]

        properties = self.properties
        for elem in data:
            elem = self.parent.reference_to_data(elem).uid
            if elem in properties:
                properties.remove(elem)

        self._properties = self._validate_properties(properties)

        if not self._properties:
            self.parent.workspace.remove_entity(self)
            return

        self.parent.workspace.add_or_update_property_group(self)

    @property
    def table(self) -> PropertyGroupTable:
        """
        Create an object to access the data of the property group.
        """
        return PropertyGroupTable(self)

    @property
    def uid(self) -> UUID:
        """
        Unique identifier
        """
        return self._uid

    @uid.setter
    def uid(self, uid: str | UUID):
        if isinstance(uid, str):
            uid = UUID(uid)

        if not isinstance(uid, UUID):
            raise TypeError(f"Could not convert input uid {uid} to type UUID")

        self._uid = uid
