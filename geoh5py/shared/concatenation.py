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

import numpy as np
from h5py import special_dtype

from geoh5py.shared.utils import KEY_MAP, as_str_if_utf8_bytes, as_str_if_uuid


class Concatenator:
    """
    Class modifier for concatenation of objects and data.
    """

    _attribute_map = {
        "Attributes": "concatenated_attributes",
        "Property Groups IDs": "property_group_ids",
        "Concatenated object IDs": "concatenated_object_ids",
        "Concatenated Data": "concatenated_data",
    }
    _concatenated_attributes = None
    _attributes_keys = None
    _concatenated_data = None
    _concatenated_object_ids = None
    _data: dict | None = None
    _index: dict | None = None
    _property_group_ids = None
    _property_groups = None

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    @property
    def concatenated_attributes(self) -> list[dict]:
        """Dictionary of concatenated objects and data attributes."""
        if self._concatenated_attributes is None:
            self._concatenated_attributes = getattr(
                self, "workspace"
            ).fetch_concatenated_values(self, "concatenated_attributes")

            if self._concatenated_attributes is not None:
                self._attributes_keys = [
                    elem["ID"] for elem in self._concatenated_attributes["Attributes"]
                ]
            else:
                self._concatenated_attributes = {"Attributes": []}
                self._attributes_keys = []

        return self._concatenated_attributes["Attributes"]

    @concatenated_attributes.setter
    def concatenated_attributes(self, attr: dict):
        if not isinstance(attr, dict):
            raise AttributeError(
                "Input value for 'concatenated_attributes' must be of type dict."
            )

        if "Attributes" not in attr:
            raise AttributeError(
                "The first key of 'concatenated_attributes' must be 'Attributes'."
            )

        self._concatenated_attributes = attr

        getattr(self, "workspace").update_attribute(self, "concatenated_attributes")

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

        if self._concatenated_object_ids is None:
            self.concatenated_object_ids = []

        return self._concatenated_object_ids

    @concatenated_object_ids.setter
    def concatenated_object_ids(self, object_ids: list[uuid.UUID]):
        if not isinstance(object_ids, list):
            raise AttributeError(
                "Input value for 'concatenated_object_ids' must be of type list."
            )

        self._concatenated_object_ids = object_ids
        getattr(self, "workspace").update_attribute(self, "concatenated_object_ids")

    @property
    def concatenator(self):
        return self

    @property
    def data(self):
        """
        Concatenated data values stored as a dictionary.
        """
        if getattr(self, "_data", None) is None:
            getattr(self, "index")

        return self._data

    @property
    def index(self):
        """
        Concatenated index stored as a dictionary.
        """
        if getattr(self, "_index", None) is None:
            data_list = getattr(self, "workspace").fetch_concatenated_values(
                self, "Index"
            )

            if data_list is None:
                self._data, self._index = {}, {}
                return self._index

            data = {}
            index = {}
            for key in data_list:
                arrays = getattr(self, "workspace").fetch_concatenated_values(self, key)
                if arrays is not None:
                    data[key], index[key] = arrays

            self._data, self._index = data, index

        return self._index

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
                for attr, val in self.concatenated_attributes[
                    self.attr_index(key)
                ].items()
                if "Property" not in attr
            }
            attrs["parent"] = self
            attr_dict[key] = getattr(self, "workspace").create_from_concatenation(attrs)

        return attr_dict

    def fetch_index(self, entity, field: str):
        """
        Fetch the array index for specific entity and data field.
        """
        field = KEY_MAP.get(field, field)

        if field not in self.index:
            return None

        uid = as_str_if_uuid(entity.uid).encode()
        if uid in self.index[field]["Object ID"].tolist():
            return self.index[field]["Object ID"].tolist().index(uid)

        if uid in self.index[field]["Data ID"].tolist():
            return self.index[field]["Data ID"].tolist().index(uid)

        return None

    def fetch_values(self, entity, field: str):
        """
        Get values from a concatenated array.
        """
        field = KEY_MAP.get(field, field)

        index = self.fetch_index(entity, field)

        if index is None:
            return None

        start, size = self.index[field][index][0], self.index[field][index][1]

        return self.data[field][start : start + size]

    @property
    def n_property_groups(self):
        """
        Return the number of property groups.
        """
        if self.property_group_ids is None:
            return 0

        return len(self.property_group_ids)

    @property
    def property_group_ids(self):
        """Dictionary of concatenated objects and data property_group_ids."""
        if getattr(self, "_property_group_ids", None) is None:
            property_groups_ids = getattr(self, "workspace").fetch_concatenated_values(
                self, "property_group_ids"
            )

            if property_groups_ids is not None:
                self._property_group_ids = property_groups_ids[0]

        return self._property_group_ids

    @property
    def property_groups(self):
        """
        Property groups for the concatenated data.
        """
        if getattr(self, "_property_groups", None) is None:
            self._property_groups = getattr(self, "workspace").fetch_property_groups(
                self
            )
            # property_groups = []
            # for key in self.property_group_ids:
            #     attrs = {
            #         attr: val
            #         for attr, val in self.concatenated_attributes[
            #             self.attr_index(key)
            #         ].items()
            #         if "Property" not in attr
            #     }
            #     attrs["parent"] = self
            #     attrs["concatenation"] = Concatenated
            #     prop_group = PropertyGroup(**attrs)
            #     property_groups += [prop_group]
            #
            # if property_groups:
            #     self._property_groups = property_groups

        return self._property_groups

    def update_attributes(self, entity, label):
        """
        Update a concatenated entity.
        """
        if label == "attributes":
            self.update_concatenated_attributes(entity)
        elif label ==  "property_groups":
            if getattr(entity, "property_groups", None) is not None:

                for pg in getattr(entity, "property_groups"):
                    self.add_save_concatenated(pg)
        else:
            if hasattr(entity, "values"):
                label = entity.name

            self.update_array_attribute(entity, label)

    def update_concatenated_attributes(self, entity):
        """Update the concatenated attributes."""
        index = self.attr_index(entity.uid)
        for key, attr in entity.attribute_map.items():
            val = getattr(entity, attr)

            if val is None:
                continue

            if isinstance(val, np.ndarray):
                val = "{" + ", ".join(str(e) for e in val.tolist()) + "}"
            elif isinstance(val, uuid.UUID):
                val = as_str_if_uuid(val)
            elif isinstance(val, list):
                val = [as_str_if_uuid(uid) for uid in val]
            elif attr == "association":
                val = val.name

            self.concatenated_attributes[index][key] = val

        if hasattr(entity, "values"):
            self.concatenated_attributes[index]["Type ID"] = as_str_if_uuid(
                entity.entity_type.uid
            )
        else:
            self.concatenated_attributes[index]["Object Type ID"] = as_str_if_uuid(
                entity.entity_type.uid
            )
        getattr(self, "workspace").repack = True

    def update_array_attribute(self, entity, field):
        """
        Update values stored as data.
        """
        if hasattr(entity, f"_{field}"):
            values = getattr(entity, f"_{field}", None)
            obj_id = as_str_if_uuid(entity.uid).encode()
            data_id = as_str_if_uuid(uuid.UUID(int=0)).encode()
        elif getattr(entity, "name") == field:
            values = getattr(entity, "values", None)
            obj_id = as_str_if_uuid(entity.parent.uid).encode()
            data_id = as_str_if_uuid(entity.uid).encode()
        else:
            raise UserWarning(
                f"Input entity {entity} does not have a property or values "
                f"for the requested field {field}"
            )

        if field == "property_groups":
            alias = "Property Group IDs"
            values = [as_str_if_uuid(val.uid).encode() for val in values]
        else:
            alias = KEY_MAP.get(field, field)
        index = self.fetch_index(entity, field)
        if index is not None:  # First remove the old data
            start, size = self.index[alias][index][0], self.index[alias][index][1]
            self.data[alias] = np.delete(
                self.data[alias], np.arange(start, start + size), axis=0
            )
            # Shift indices
            self.index[alias]["Start index"][
                self.index[alias]["Start index"] > start
            ] -= size
            start = self.data[alias].shape[0] - 1
            self.index[alias] = np.delete(self.index[alias], index, axis=0)
        else:
            start = 0

        if values is not None:
            indices = np.hstack(
                [
                    np.core.records.fromarrays(
                        (start, len(values), obj_id, data_id),
                        dtype=[
                            ("Start index", "<u4"),
                            ("Size", "<u4"),
                            ("Object ID", special_dtype(vlen=str)),
                            ("Data ID", special_dtype(vlen=str)),
                        ],
                    )
                ]
            )
            if alias in self.index:
                indices = np.hstack([self.index[alias], indices])

            self.index[alias] = indices

            if alias in self.data:
                values = np.hstack([self.data[alias], values])

            self.data[alias] = values

        getattr(self, "workspace").update_attribute(self, "index", alias)

        if hasattr(entity, f"_{field}"):  # For group property
            if field == "property_groups":
                field = "property_group_ids"
            setattr(self, f"_{field}", self.data.get(alias))
            getattr(self, "workspace").update_attribute(self, field)
        else:  # For data values
            getattr(self, "workspace").update_attribute(self, "data", field)

    def add_save_concatenated(self, child):
        """
        Add or save a concatenated entity.

        :param child: Concatenated entity
        """
        if as_str_if_uuid(child.uid) not in self._attributes_keys:
            self._attributes_keys.append(as_str_if_uuid(child.uid))
            self.concatenated_attributes.append({})

        self.update_concatenated_attributes(child)

        if hasattr(child, "values"):
            self.update_array_attribute(child, child.name)
        else:
            if child.uid not in self.concatenated_object_ids:
                self.concatenated_object_ids = self.concatenated_object_ids + [
                    as_str_if_uuid(child.uid)
                ]
            if hasattr(child, "surveys"):  # Specific to drillholes
                self.update_array_attribute(child, "surveys")
                self.update_array_attribute(child, "trace")

        child.on_file = True


class Concatenated:
    """
    Class modifier for concatenated objects and data.
    """

    def __init__(self, parent: Concatenated | Concatenator, **kwargs):
        self.parent = parent

        super().__init__(**kwargs)

    @property
    def concatenator(self) -> Concatenator:
        """
        Parental Concatenator entity.
        """
        return self._parent.concatenator

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
        uid = self.concatenator.attr_index(getattr(self, "uid"))
        attr = self.concatenator.concatenated_attributes[uid]
        if f"Property:{name}" in attr:
            uid = self.concatenator.attr_index(attr.get(f"Property:{name}"))
            attributes: dict = self.concatenator.concatenated_attributes[uid].copy()
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
        uid = self.concatenator.attr_index(getattr(self, "uid"))
        data_list = [
            attr.replace("Property:", "")
            for attr in self.concatenator.concatenated_attributes[uid]
            if "Property:" in attr
        ]

        return data_list

    def update_attributes(self, entity, field: str):
        """
        Update the attributes on the concatenated entity.
        """
        return self.concatenator.update_attributes(entity, field)

    @property
    def parent(self) -> Concatenated | Concatenator:
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
        self._parent.add_children([self])

    @property
    def property_groups(self):
        if getattr(self, "_property_groups", None) is None:
            prop_groups = self.concatenator.fetch_values(self, "property_group_ids")

            if prop_groups is None:
                return

            for key in prop_groups:
                getattr(self, "find_or_create_property_group")(
                    **self.concatenator.concatenated_attributes[
                        self.concatenator.attr_index(key)
                    ]
                )

        return self._property_groups

    def save(self):
        """
        Save the concatenated object or data to concatenator.
        """
        self.concatenator.add_save_concatenated(self)
