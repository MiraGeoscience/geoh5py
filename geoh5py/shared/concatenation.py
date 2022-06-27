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
from typing import TYPE_CHECKING

import numpy as np
from h5py import special_dtype

from geoh5py.data import Data
from geoh5py.groups import Group
from geoh5py.shared.entity import Entity
from geoh5py.shared.utils import KEY_MAP, as_str_if_utf8_bytes, as_str_if_uuid

if TYPE_CHECKING:
    from ..groups import GroupType


class Concatenator(Group):
    """
    Class modifier for concatenation of objects and data.
    """

    _concatenated_attributes: dict | None = None
    _attributes_keys: list[uuid.UUID] | None = None
    _concatenated_object_ids: list[bytes] | None = None
    _data: dict
    _index: dict
    _property_group_ids: np.ndarray | None = None
    _property_groups: list | None = None

    def __init__(self, group_type: GroupType, **kwargs):

        super().__init__(group_type, **kwargs)

        getattr(self, "_attribute_map").update(
            {
                "Attributes": "concatenated_attributes",
                "Property Groups IDs": "property_group_ids",
                "Concatenated object IDs": "concatenated_object_ids",
            }
        )

    def add_attribute(self, uid: str) -> None:
        """
        Add new element to the concatenated attributes.

        :param uid: Unique identifier of the new concatenated entity in str format.
        """
        if self.attributes_keys is not None:
            self.attributes_keys.append(uid)

        if self.concatenated_attributes is not None:
            self.concatenated_attributes["Attributes"].append({})

    @property
    def attributes_keys(self) -> list | None:
        """List of uuids present in the concatenated attributes."""
        if getattr(self, "_attributes_keys", None) is None:
            attributes_keys = []
            if self.concatenated_attributes is not None:
                attributes_keys = [
                    elem["ID"] for elem in self.concatenated_attributes["Attributes"]
                ]

            self._attributes_keys = attributes_keys

        return self._attributes_keys

    @property
    def concatenated_attributes(self) -> dict | None:
        """Dictionary of concatenated objects and data attributes."""
        if self._concatenated_attributes is None:
            concatenated_attributes = self.workspace.fetch_concatenated_attributes(
                self, "Attributes"
            )

            if concatenated_attributes is None:
                concatenated_attributes = {"Attributes": []}

            self._concatenated_attributes = concatenated_attributes

        return self._concatenated_attributes

    @property
    def concatenated_object_ids(self) -> list[bytes] | None:
        """Dictionary of concatenated objects and data concatenated_object_ids."""
        if getattr(self, "_concatenated_object_ids", None) is None:
            concatenated_object_ids = self.workspace.fetch_array_attribute(
                self, "concatenated_object_ids"
            )
            if isinstance(concatenated_object_ids, np.ndarray):
                concatenated_object_ids = concatenated_object_ids.tolist()

            self._concatenated_object_ids = concatenated_object_ids

        return self._concatenated_object_ids

    @concatenated_object_ids.setter
    def concatenated_object_ids(self, object_ids: list[uuid.UUID] | np.ndarray | None):
        if isinstance(object_ids, np.ndarray):
            object_ids = object_ids.tolist()
        elif not isinstance(object_ids, (list, type(None))):
            raise AttributeError(
                "Input value for 'concatenated_object_ids' must be of type list."
            )

        self._concatenated_object_ids = object_ids
        self.workspace.update_attribute(self, "concatenated_object_ids")

    @property
    def data(self) -> dict:
        """
        Concatenated data values stored as a dictionary.
        """
        if getattr(self, "_data", None) is None:
            data_list = self.workspace.fetch_concatenated_list(self, "Data")
            if data_list is not None:
                self._data = {name: None for name in data_list}
            else:
                self._data = {}

        return self._data

    @property
    def index(self) -> dict:
        """
        Concatenated index stored as a dictionary.
        """
        if getattr(self, "_index", None) is None:
            data_list = self.workspace.fetch_concatenated_list(self, "Index")
            if data_list is not None:
                self._index = {name: None for name in data_list}

        return self._index

    def fetch_concatenated_objects(self) -> dict:
        """
        Load all concatenated children.
        """
        attr_dict = {}
        if self.concatenated_object_ids is None:
            return {}

        for key in self.concatenated_object_ids:
            attrs = {
                attr: val
                for attr, val in self.get_attributes(key).items()
                if "Property" not in attr
            }
            attrs["parent"] = self
            attr_dict[key] = self.workspace.create_from_concatenation(attrs)

        return attr_dict

    def fetch_index(self, entity: Concatenated, field: str) -> int | None:
        """
        Fetch the array index for specific concatenated object and data field.

        :param entity: Parent entity with data
        :param field: Name of the target data.
        """
        field = KEY_MAP.get(field, field)

        if field not in self.index:
            return None

        if self.index[field] is None:
            values = self.workspace.fetch_concatenated_values(self, field)
            if isinstance(values, tuple):
                self.data[field], self.index[field] = values

        uid = as_str_if_uuid(entity.uid).encode()
        ind = np.where(self.index[field]["Object ID"] == uid)[0]
        if len(ind) == 1:
            return ind[0]

        ind = np.where(self.index[field]["Data ID"] == uid)[0]
        if len(ind) == 1:
            return ind[0]

        return None

    def fetch_values(self, entity: Concatenated, field: str) -> np.ndarray | None:
        """
        Get an array of values from concatenated data.

        :param entity: Parent entity with data
        :param field: Name of the target data.
        """
        field = KEY_MAP.get(field, field)

        index = self.fetch_index(entity, field)

        if index is None:
            return None

        start, size = self.index[field][index][0], self.index[field][index][1]

        return self.data[field][start : start + size]

    @property
    def property_group_ids(self) -> list | None:
        """Dictionary of concatenated objects and data property_group_ids."""
        if self._property_group_ids is None:
            property_groups_ids = self.workspace.fetch_concatenated_values(
                self, "property_group_ids"
            )

            if property_groups_ids is not None:
                self._property_group_ids = property_groups_ids[0].tolist()

        return self._property_group_ids

    def update_attributes(self, entity: Concatenated, label: str) -> None:
        """
        Update a concatenated entity.
        """
        if label == "attributes":
            self.update_concatenated_attributes(entity)
        elif label == "property_groups":
            if getattr(entity, "property_groups", None) is not None:
                for prop_group in getattr(entity, "property_groups"):
                    self.add_save_concatenated(prop_group)
                    if (
                        self.property_group_ids is not None
                        and as_str_if_uuid(prop_group.uid).encode()
                        not in self.property_group_ids
                    ):
                        self.property_group_ids.append(
                            as_str_if_uuid(prop_group.uid).encode()
                        )

            self.update_array_attribute(entity, label)

        else:
            if isinstance(entity, Data):
                label = entity.name

            self.update_array_attribute(entity, label)

    def update_concatenated_attributes(self, entity: Concatenated) -> None:
        """
        Update the concatenated attributes.
        :param entity: Concatenated entity with attributes.
        """
        target_attributes = self.get_attributes(entity.uid)
        for key, attr in entity.attribute_map.items():
            val = getattr(entity, attr, None)

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

        if isinstance(entity, Data):
            target_attributes["Type ID"] = as_str_if_uuid(entity.entity_type.uid)
        elif hasattr(entity, "properties"):
            pass
        else:
            target_attributes["Object Type ID"] = as_str_if_uuid(entity.entity_type.uid)
        self.workspace.repack = True

    def update_array_attribute(self, entity: Concatenated, field: str) -> None:
        """
        Update values stored as data.
        Row data and indices are first remove then appended.

        :param entity: Concatenated entity with array values.
        :param field: Name of the valued field.
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

        if field == "property_groups" and isinstance(values, list):
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

        self.workspace.update_attribute(self, "index", alias)

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

            self.workspace.update_attribute(
                self,
                field,
                values=self.data.get(alias),
                **property_kwarg.get(field, {}),
            )
        else:  # For data values
            self.workspace.update_attribute(self, "data", field)

    def add_save_concatenated(self, child) -> None:
        """
        Add or save a concatenated entity.

        :param child: Concatenated entity
        """
        self.update_concatenated_attributes(child)

        if hasattr(child, "values"):
            self.update_array_attribute(child, child.name)
        elif hasattr(child, "surveys"):  # Specific to drillholes
            uid = as_str_if_uuid(child.uid).encode()
            concat_object_ids = [uid]
            if self._concatenated_object_ids is not None:
                if uid not in self._concatenated_object_ids:
                    concat_object_ids = (
                        self._concatenated_object_ids + concat_object_ids
                    )
                else:
                    concat_object_ids = self._concatenated_object_ids

            self.concatenated_object_ids = concat_object_ids
            self.update_array_attribute(child, "surveys")
            self.update_array_attribute(child, "trace")

        child.on_file = True

    def get_attributes(self, uid: bytes | str | uuid.UUID) -> dict:
        """
        Fast reference index to concatenated attribute keys.
        """
        if self.concatenated_attributes is None:
            return {}

        uid = as_str_if_utf8_bytes(uid)

        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        uid = as_str_if_utf8_bytes(as_str_if_uuid(uid))

        if self.attributes_keys is not None and uid in self.attributes_keys:
            index = self.attributes_keys.index(uid)
        else:
            self.add_attribute(uid)
            index = -1

        return self.concatenated_attributes["Attributes"][index]

    def fetch_start_index(self, entity: Concatenated, label: str) -> int:
        """
        Fetch starting index for a given entity and label.
        Existing date is removed such that new entries can be appended.

        :param entity: Concatenated entity to be added.
        :param label: Name of the attribute requiring an update.
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

    def delete_index_data(self, label: str, index: int) -> None:
        start, size = self.index[label][index][0], self.index[label][index][1]
        self.data[label] = np.delete(
            self.data[label], np.arange(start, start + size), axis=0
        )
        # Shift indices
        self.index[label]["Start index"][
            self.index[label]["Start index"] > start
        ] -= size
        self.index[label] = np.delete(self.index[label], index, axis=0)


class Concatenated(Entity):
    """
    Class modifier for concatenated objects and data.
    """

    _parent: Concatenated | Concatenator
    _property_groups = None

    def __init__(self, entity_type, **kwargs):
        attribute_map = getattr(self, "_attribute_map", {})
        attr = {"name": "Entity", "parent": None}
        for key, value in kwargs.items():
            attr[attribute_map.get(key, key)] = value

        if not isinstance(attr.get("parent"), (Concatenated, Concatenator)):
            raise UserWarning(
                "Creating a concatenated entity must have a parent "
                "of type Concatenator for 'objects', or Concatenated for 'data'."
            )

        super().__init__(entity_type, **attr)

    @property
    def concatenator(self) -> Concatenator:
        """
        Parental Concatenator entity.
        """
        if isinstance(self._parent, Concatenated):
            return self._parent.concatenator

        return self._parent

    def get_data(self, name: str) -> list[Data]:
        """
        Generic function to get data values from object.
        """
        entity_list = []
        attr = self.concatenator.get_attributes(getattr(self, "uid")).copy()

        for key, value in attr.items():
            if (
                "Property:" in key
                and self.workspace.get_entity(uuid.UUID(value))[0] is None
            ):
                attributes: dict = self.concatenator.get_attributes(value).copy()
                attributes["parent"] = self
                self.workspace.create_from_concatenation(attributes)

        for child in getattr(self, "children"):
            if hasattr(child, "name") and child.name == name.replace("/", "\u2044"):
                entity_list.append(child)

        return entity_list

    def get_data_list(self):
        """
        Get list of data names.
        """
        data_list = [
            attr.replace("Property:", "")
            for attr in self.concatenator.get_attributes(self.uid)
            if "Property:" in attr
        ]

        return data_list

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

        if isinstance(self, Data) and isinstance(self, Concatenated):
            parental_attr = self.concatenator.get_attributes(self.parent.uid)
            if f"Property:{self.name}" not in parental_attr:
                parental_attr[f"Property:{self.name}"] = as_str_if_uuid(self.uid)

    @property
    def property_groups(self) -> list | None:
        if self._property_groups is None:
            prop_groups = self.concatenator.fetch_values(self, "property_group_ids")

            if prop_groups is None or isinstance(self, Data):
                return None

            for key in prop_groups:
                getattr(self, "find_or_create_property_group")(
                    **self.concatenator.get_attributes(key)
                )

        return self._property_groups
