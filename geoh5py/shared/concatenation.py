#  Copyright (c) 2022 Mira Geoscience Ltd.
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

import numpy

from ..shared.utils import as_str_if_utf8_bytes, as_str_if_uuid


class Concatenator:
    """
    Class modifier for concatenation of objects and data.
    """

    _attribute_map = {
        "Attributes": "attributes",
        "Property Groups IDs": "property_group_ids",
        "Concatenated object IDs": "concatenated_object_ids",
        "Concatenated Data": "concatenated_data",
    }
    _attributes = None
    _attributes_keys = None
    _concatenated_data = None
    _concatenated_object_ids = None
    _data: dict = {}
    _indices: dict = {}
    _property_group_ids = None
    _property_groups = None

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    @property
    def attributes(self):
        """Dictionary of concatenated objects and data attributes."""
        if self._attributes is None:
            self._attributes = getattr(self, "workspace").fetch_concatenated_values(
                self, "attributes"
            )

            if self._attributes is not None:
                self._attributes_keys = [
                    elem["ID"] for elem in self._attributes["Attributes"]
                ]

        return self._attributes["Attributes"]

    @attributes.setter
    def attributes(self, attr: dict):
        if not isinstance(attr, dict):
            raise AttributeError("Input value for 'attributes' must be of type dict.")

        if "Attributes" not in attr:
            raise AttributeError("The first key of 'attributes' must be 'Attributes'.")

        self._attributes = attr

        getattr(self, "workspace").update_attribute(self, "attributes")

    def attr_index(self, uid: bytes | str | uuid.UUID):
        """
        Fast reference index to attribute keys.
        """
        uid = as_str_if_utf8_bytes(uid)

        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        uid = as_str_if_uuid(uid)

        try:
            return getattr(self, "_attributes_keys").index(
                as_str_if_uuid(as_str_if_utf8_bytes(uid))
            )
        except KeyError as error:
            raise KeyError(
                f"Identifier {uid} not present in Concatenator 'Attributes'."
            ) from error

    @property
    def concatenated_object_ids(self):
        """Dictionary of concatenated objects and data concatenated_object_ids."""
        if getattr(self, "_concatenated_object_ids", None) is None:
            self._concatenated_object_ids = getattr(
                self, "workspace"
            ).fetch_array_attribute(self, "concatenated_object_ids")

        return self._concatenated_object_ids

    @concatenated_object_ids.setter
    def concatenated_object_ids(self, object_ids: list[uuid.UUID]):
        if not isinstance(object_ids, list):
            raise AttributeError(
                "Input value for 'concatenated_object_ids' must be of type list."
            )

        self._concatenated_object_ids = object_ids

    @property
    def concatenator(self):
        return self

    @property
    def data(self) -> dict:
        """
        Concatenated data values stored as a dictionary.
        """
        return self._data

    @property
    def indices(self) -> dict:
        """
        Concatenated indices stored as a dictionary.
        """
        return self._indices

    def fetch_concatenated_objects(self):
        """
        Load all concatenated children.
        :param group: Concatenator group
        :param attributes: Entities stored as list of dictionaries.
        """
        attr_dict = {}
        for key in self.concatenated_object_ids:
            attrs = {
                attr: val
                for attr, val in self.attributes[self.attr_index(key)].items()
                if "Property" not in attr
            }
            attrs["parent"] = self
            attr_dict[key] = getattr(self, "workspace").create_from_concatenation(attrs)

        return attr_dict

    def fetch_values(self, entity, field: str):
        """
        Get values from a concatenated array.
        """
        if field not in self.data:
            data, indices = getattr(self, "workspace").fetch_concatenated_values(
                self, field
            )
            self.data[field] = data
            self.indices[field] = indices

        try:
            start, size = self.indices[field][as_str_if_uuid(entity.uid).encode()][:2]
        except KeyError:
            start, size = self.indices[field][
                as_str_if_uuid(entity.parent.uid).encode()
            ][:2]

        return self.data[field][start : start + size]

    @property
    def property_group_ids(self):
        """Dictionary of concatenated objects and data property_group_ids."""
        if getattr(self, "_property_group_ids", None) is None:
            self._property_group_ids = getattr(
                self, "workspace"
            ).fetch_concatenated_values(self, "property_group_ids")
        return self._property_group_ids

    @property_group_ids.setter
    def property_group_ids(self, object_ids: list[uuid.UUID]):
        if not isinstance(object_ids, list):
            raise AttributeError(
                "Input value for 'property_group_ids' must be of type list."
            )

        self._property_group_ids = object_ids

    @property
    def property_groups(self):
        """
        Property groups for the concatenated data.
        """
        if getattr(self, "_property_groups", None) is None:
            self._property_groups = getattr(self, "workspace").fetch_property_groups(
                self
            )

        return self._property_groups

    def update_attributes(self, entity, field):
        """Update the attributes of a concatenated entity."""
        if field == "attributes":
            ref = self.attr_index(entity.uid)
            for key, attr in entity.attribute_map.items():
                val = getattr(entity, attr)

                if val is None:
                    continue

                if isinstance(val, numpy.ndarray):
                    val = "{" + ", ".join(str(e) for e in val.tolist()) + "}"
                elif isinstance(val, uuid.UUID):
                    val = as_str_if_uuid(val)

                self.attributes[ref][key] = val

            getattr(self, "workspace").update_attribute(self, "attributes")

    # def add_children(self, children):
    #     """
    #     :param children: Add a list of entities as
    #         :obj:`~geoh5py.shared.entity.Entity.children`
    #     """
    #     for child in children:
    #         if child not in self._children:
    #             self._children.append(child)


class Concatenated:
    """
    Class modifier for concatenated objects and data.
    """

    def __init__(self, **kwargs):
        self._parent: Concatenated | Concatenator | None = None

        super().__init__(**kwargs)

    def add_data(self, data: dict, property_group: str = None):
        """
        Overloaded :obj:`~geoh5py.objects.ObjectBase.add_data` method.
        """
        raise NotImplementedError(
            "Concatenated entity `add_data` method not yet implemented."
        )

    def add_data_to_group(self, data: dict, property_group: str = None):
        """
        Overloaded :obj:`~geoh5py.objects.ObjectBase.add_data_to_group` method.
        """
        raise NotImplementedError(
            "Concatenated entity `add_data_to_group` method not yet implemented."
        )

    @property
    def concatenator(self):
        """
        Parental Concatenator entity.
        """
        return self.parent.concatenator if self.parent is not None else None

    def find_or_create_property_group(self, **kwargs):
        """
        Overloaded :obj:`~geoh5py.objects.ObjectBase.find_or_create_property_group` method.
        """
        raise NotImplementedError(
            "Concatenated entity `find_or_create_property_group` method not yet implemented."
        )

    def fetch_values(self, entity, field: str):
        """
        Get values from the parent entity.
        """
        return self.concatenator.fetch_values(entity, field)

    def get_data(self, name: str) -> list:
        """
        Generic function to get data values from object.
        """
        entity_list = []
        uid = self.parent.attr_index(getattr(self, "uid"))
        attr = self.parent.attributes[uid]
        if f"Property:{name}" in attr:
            uid = self.parent.attr_index(attr.get(f"Property:{name}"))
            attributes = self.parent.attributes[uid]
            attributes["parent"] = self
            getattr(self, "workspace").create_from_concatenation(attributes)

        for child in getattr(self, "children"):
            if hasattr(child, "name") and child.name == name:
                entity_list.append(child)

        return entity_list

    def get_data_list(self):
        """
        Get list of data names.
        """
        uid = self.parent.attr_index(getattr(self, "uid"))
        data_list = [
            attr.replace("Property:", "")
            for attr in self.parent.attributes[uid]
            if "Property:" in attr
        ]

        return data_list

    @staticmethod
    def update_attributes(entity, field: str):
        """
        Update the attributes on the concatenated entity.
        """
        return getattr(entity, "parent").update_attributes(entity, field)

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if (parent.concatenation is not Concatenated) and (
            parent.concatenation is not Concatenator
        ):
            raise AttributeError(
                "The 'parent' of a concatenated Entity must be of type 'Concatenator'."
            )
        self._parent = parent

        current_parent = self._parent

        if parent is not None:
            self._parent = parent
            self._parent.add_children([self])

            if current_parent is not None and current_parent != self._parent:
                current_parent.remove_children([self])
