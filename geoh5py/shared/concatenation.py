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


from __future__ import annotations

import uuid

import numpy as np
from h5py import special_dtype

from geoh5py.shared.utils import KEY_MAP, as_str_if_utf8_bytes, as_str_if_uuid


class Concatenator:
    """
    Class modifier for concatenation of objects and data.
    """

    _concatenated_attributes = None
    _attributes_keys = None
    _concatenated_data = None
    _concatenated_object_ids = None
    _data: dict | None = None
    _index: dict | None = None
    _property_group_ids = None
    _property_groups = None

    def __init__(self, group_type, **kwargs):

        super().__init__(group_type, **kwargs)

        getattr(self, "_attribute_map").update(
            {
                "Attributes": "concatenated_attributes",
                "Property Groups IDs": "property_group_ids",
                "Concatenated object IDs": "concatenated_object_ids",
                "Concatenated Data": "concatenated_data",
            }
        )

    def add_attribute(self, uid):
        """Add new element to the concatenated attributes."""
        if self.attributes_keys is None:
            self._attributes_keys = []

        self.attributes_keys.append(uid)

        if self.concatenated_attributes is None:
            self._concatenated_attributes = {"Attributes": []}

        self.concatenated_attributes["Attributes"].append({})

    @property
    def attributes_keys(self):
        """List of uuids present in the concatenated attributes."""
        if (
            getattr(self, "_attributes_keys", None) is None
            and self.concatenated_attributes is not None
        ):
            self._attributes_keys = [
                elem["ID"] for elem in self.concatenated_attributes["Attributes"]
            ]

        return self._attributes_keys

    @property
    def concatenated_attributes(self) -> dict | None:
        """Dictionary of concatenated objects and data attributes."""
        if self._concatenated_attributes is None:
            self.concatenated_attributes = getattr(
                self, "workspace"
            ).fetch_concatenated_values(self, "concatenated_attributes")

        return self._concatenated_attributes

    @concatenated_attributes.setter
    def concatenated_attributes(self, attr: dict):
        if attr is None:
            self._concatenated_attributes = None
            return

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
                for attr, val in self.get_attributes(key).items()
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

        return self._property_groups

    def update_attributes(self, entity, label):
        """
        Update a concatenated entity.
        """
        if label == "attributes":
            self.update_concatenated_attributes(entity)
        elif label == "property_groups":
            if getattr(entity, "property_groups", None) is not None:

                for prop_group in getattr(entity, "property_groups"):
                    self.add_save_concatenated(prop_group)
                self._property_group_ids = None

            self.update_array_attribute(entity, label)

        else:
            if hasattr(entity, "values"):
                label = entity.name

            self.update_array_attribute(entity, label)

    def update_concatenated_attributes(self, entity):
        """Update the concatenated attributes."""
        target_attributes = self.get_attributes(entity.uid)
        for key, attr in entity.attribute_map.items():
            val = getattr(entity, attr)

            if val is None or attr == "property_groups":
                continue

            if isinstance(val, np.ndarray):
                val = "{" + ", ".join(str(e) for e in val.tolist()) + "}"
            elif isinstance(val, uuid.UUID):
                val = as_str_if_uuid(val)
            elif isinstance(val, list):
                val = [as_str_if_uuid(uid) for uid in val]
            elif attr == "association":
                val = val.name.lower().capitalize()

            target_attributes[key] = val

        if hasattr(entity, "values"):
            target_attributes["Type ID"] = as_str_if_uuid(entity.entity_type.uid)
        elif hasattr(entity, "properties"):
            pass
        else:
            target_attributes["Object Type ID"] = as_str_if_uuid(entity.entity_type.uid)
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

        start = self.fetch_start_index(entity, alias)

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

        property_kwarg = {
            "property_group_ids": {
                "dtype": special_dtype(vlen=str),
                "maxshape": (None,),
            },
            "surveys": {"maxshape": (None,)},
        }

        if hasattr(entity, f"_{field}"):  # For group property
            if field == "property_groups":
                field = "property_group_ids"

            getattr(self, "workspace").update_attribute(
                self,
                field,
                values=self.data.get(alias),
                **property_kwarg.get(field, {}),
            )
        else:  # For data values
            getattr(self, "workspace").update_attribute(self, "data", field)

    def add_save_concatenated(self, child):
        """
        Add or save a concatenated entity.

        :param child: Concatenated entity
        """
        self.update_concatenated_attributes(child)

        if hasattr(child, "values"):
            self.update_array_attribute(child, child.name)
        elif hasattr(child, "surveys"):  # Specific to drillholes
            if as_str_if_uuid(child.uid) not in self.concatenated_object_ids:
                self.concatenated_object_ids = self.concatenated_object_ids + [
                    as_str_if_uuid(child.uid)
                ]

            self.update_array_attribute(child, "surveys")
            self.update_array_attribute(child, "trace")

        child.on_file = True

    def get_attributes(self, uid: bytes | str | uuid.UUID):
        """
        Fast reference index to concatenated attribute keys.
        """

        uid = as_str_if_utf8_bytes(uid)

        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        uid = as_str_if_utf8_bytes(as_str_if_uuid(uid))

        if (
            self.attributes_keys is not None
            and as_str_if_uuid(uid) in self.attributes_keys
        ):
            index = self.attributes_keys.index(as_str_if_uuid(uid))
        else:
            self.add_attribute(uid)
            index = -1

        return self.concatenator.concatenated_attributes["Attributes"][index]

    def fetch_start_index(self, entity, label: str):
        """
        Fetch starting index for a given entity and label.
        Existing date is removed such that new entries can be appended.
        """
        index = self.fetch_index(entity, label)
        if index is not None:  # First remove the old data
            self.delete_index_data(label, index)
            start = self.data[label].shape[0]

        elif label in self.index:
            start = np.sum(self.index[label]["Size"])
        else:
            start = 0

        return start

    def delete_index_data(self, label: str, index: int):
        start, size = self.index[label][index][0], self.index[label][index][1]
        self.data[label] = np.delete(
            self.data[label], np.arange(start, start + size), axis=0
        )
        # Shift indices
        self.index[label]["Start index"][
            self.index[label]["Start index"] > start
        ] -= size
        self.index[label] = np.delete(self.index[label], index, axis=0)


class Concatenated:
    """
    Class modifier for concatenated objects and data.
    """

    _parent: Concatenated | Concatenator
    _property_groups = None

    def __init__(self, entity_type, parent=None, **kwargs):

        if not isinstance(parent, (Concatenated, Concatenator)):
            raise UserWarning(
                "Creating a concatenated entity must have a parent "
                "of type Concatenator for 'objects', or Concatenated for 'data'."
            )

        super().__init__(entity_type, parent=parent, **kwargs)

    @property
    def concatenator(self) -> Concatenator:
        """
        Parental Concatenator entity.
        """
        if isinstance(self._parent, Concatenated):
            return self._parent.concatenator

        return self._parent

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
        attr = self.concatenator.get_attributes(getattr(self, "uid"))

        if f"Property:{name}" in attr and name not in self.get_data_list():
            attributes: dict = self.concatenator.get_attributes(
                attr.get(f"Property:{name}")
            ).copy()
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
        data_list = [
            attr.replace("Property:", "")
            for attr in self.concatenator.get_attributes(getattr(self, "uid"))
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
        if not isinstance(parent, (Concatenated, Concatenator)):
            raise AttributeError(
                "The 'parent' of a concatenated Entity must be of type "
                "'Concatenator' or 'Concatenated'."
            )
        self._parent = parent
        self._parent.add_children([self])

        if hasattr(self, "values"):
            parental_attr = self.concatenator.get_attributes(self.parent.uid)
            if f"Property:{self.name}" not in parental_attr:
                parental_attr[f"Property:{self.name}"] = as_str_if_uuid(self.uid)

    @property
    def property_groups(self):
        if getattr(self, "_property_groups", None) is None:
            prop_groups = self.concatenator.fetch_values(self, "property_group_ids")

            if prop_groups is None:
                return None

            for key in prop_groups:
                getattr(self, "find_or_create_property_group")(
                    **self.concatenator.get_attributes(key)
                )

        return self._property_groups

    def save(self):
        """
        Save the concatenated object or data to concatenator.
        """
        self.concatenator.add_save_concatenated(self)
