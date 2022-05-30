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

from geoh5py.data import Data


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
    _concatenated_data = None
    _concatenated_object_ids = None
    _property_group_ids = None
    _property_groups = None
    _attributes = None
    _data: dict = {}
    _indices: dict = {}

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    @property
    def attributes(self):
        """Dictionary of concatenated objects and data attributes."""
        return self._attributes

    @attributes.setter
    def attributes(self, attr: dict):
        if not isinstance(attr, dict):
            raise AttributeError("Input value for 'attributes' must be of type dict.")

        self._attributes = attr

    @property
    def concatenated_object_ids(self):
        """Dictionary of concatenated objects and data concatenated_object_ids."""
        if getattr(self, "_concatenated_object_ids", None) is None:
            self._concatenated_object_ids = getattr(
                self, "workspace"
            ).fetch_concatenated_data(self, "Concatenated object IDs")
        return self._concatenated_object_ids

    @concatenated_object_ids.setter
    def concatenated_object_ids(self, object_ids: list[uuid.UUID]):
        if not isinstance(object_ids, list):
            raise AttributeError(
                "Input value for 'concatenated_object_ids' must be of type list."
            )

        self._concatenated_object_ids = object_ids

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

    def get_concatenated_data(self, data_id: str | uuid.UUID):
        """
        Get values from a concatenated array.
        """
        if data_id not in self.data:
            data, indices = getattr(self, "workspace").fetch_concatenated_data(
                self, data_id
            )
            self.data[data_id] = data
            self.indices[data_id] = indices

        return self.data[data_id][self.indices[data_id]]

    @property
    def property_group_ids(self):
        """Dictionary of concatenated objects and data property_group_ids."""
        if getattr(self, "_property_group_ids", None) is None:
            self._property_group_ids = getattr(
                self, "workspace"
            ).fetch_concatenated_data(self, "Property Group IDs")
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


class Concatenated:
    """
    Class modifier for concatenated objects and data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_data(self, name: str) -> list:
        """
        Generic function to get data values from object.
        """
        entity_list = []

        if f"Property:{name}" in self.__dict__:
            if isinstance(getattr(self, f"Property:{name}"), str):
                uid = getattr(self, f"Property:{name}")
                attributes = getattr(self, "parent").attributes[uid]
                attributes["parent"] = self

                getattr(self, "workspace").create_concatenated_entity(attributes)
                #
                # getattr(self, "workspace").get_concatenated_data(
                #     self, getattr(self, f"Property:{name}")
                # )

        for child in getattr(self, "children"):
            if isinstance(child, Data) and child.name == name:
                entity_list.append(child)

        return entity_list
