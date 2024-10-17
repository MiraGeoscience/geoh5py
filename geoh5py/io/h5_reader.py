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

import json
import uuid
from typing import Any

import h5py
import numpy as np

from ..shared import FLOAT_NDV, fetch_h5_handle
from ..shared.utils import (
    INV_KEY_MAP,
    KEY_MAP,
    as_str_if_utf8_bytes,
    as_str_if_uuid,
    str2uuid,
    str_json_to_dict,
)


class H5Reader:
    """
    Class to read information from a geoh5 file.
    """

    @classmethod
    def fetch_attributes(
        cls,
        file: str | h5py.File,
        uid: uuid.UUID,
        entity_type: str,
    ) -> tuple[dict, dict, dict] | tuple[None, None, None]:
        """
        Get attributes of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier
        :param entity_type: Type of entity from
            'group', 'data', 'object', 'group_type', 'data_type', 'object_type'

        Returns
        -------
        attributes: :obj:`dict` of attributes for the :obj:`~geoh5py.shared.entity.Entity`
        type_attributes: :obj:`dict` of attributes for the :obj:`~geoh5py.shared.entity.EntityType`
        property_groups: :obj:`dict` of data :obj:`uuid.UUID`
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]

            entity_type = cls.format_type_string(entity_type)

            if entity_type == "Root":
                entity = h5file[name].get(entity_type)
            else:
                entity = h5file[name][entity_type].get(as_str_if_uuid(uid))

            if entity is None:
                return None, None, None

            attributes: dict = {}
            type_attributes: dict = {}
            property_groups: dict = {}

            for key, value in entity.attrs.items():
                attributes[INV_KEY_MAP.get(key, key)] = value

            # TODO Use lazy pointer to data
            if entity_type != "Data":
                for key, value in entity.items():
                    if key.lower() in ["metadata", "options"]:
                        attributes[INV_KEY_MAP[key]] = cls.fetch_metadata(
                            file, uid, entity_type, key
                        )
                    elif (
                        key in INV_KEY_MAP
                        and isinstance(value, h5py.Dataset)
                        and value.ndim > 0
                    ):
                        attributes[INV_KEY_MAP[key]] = value[:]

            if "Type" in entity:
                type_attributes = cls.fetch_type_attributes(entity["Type"])
            # Check if the entity has property_group
            if "PropertyGroups" in entity:
                property_groups = cls.fetch_property_groups(file, uid)

        return attributes, type_attributes, property_groups

    @classmethod
    def fetch_array_attribute(
        cls, file: str | h5py.File, uid: uuid.UUID, entity_type: str, key: str
    ) -> np.ndarray | None:
        """
        Get an entity attribute stores as array such as
        :obj:`~geoh5py.objects.object_base.ObjectBase.cells`.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target object.
        :param entity_type: Group type to fetch entity from.
        :param key: Field attribute name.

        :return cells: :obj:`numpy.ndarray` of :obj:`int`.
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]
            label = KEY_MAP.get(key, key)

            if "types" in entity_type:
                root_handle = h5file[name]["Types"][entity_type]
            else:
                root_handle = h5file[name][entity_type]

            try:
                values = root_handle[as_str_if_uuid(uid)][label][:]
                return values
            except KeyError:
                return None

    @classmethod
    def fetch_children(
        cls, file: str | h5py.File, uid: uuid.UUID, entity_type: str
    ) -> dict:
        """
        Get :obj:`~geoh5py.shared.entity.Entity.children` of an
        :obj:`~geoh5py.shared.entity.Entity`.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier
        :param entity_type: Type of entity from
            'group', 'data', 'object', 'group_type', 'data_type', 'object_type'


        :return children: [{uuid: type}, ... ]
            List of dictionaries for the children uid and type
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]
            children: dict = {}
            entity_type = cls.format_type_string(entity_type)

            if (
                entity_type not in h5file[name]
                or as_str_if_uuid(uid) not in h5file[name][entity_type]
            ):
                return children

            entity = h5file[name][entity_type][as_str_if_uuid(uid)]

            for child_type, child_list in entity.items():
                if child_type in ["Type", "PropertyGroups", "Concatenated Data"]:
                    continue

                if isinstance(child_list, h5py.Group):
                    for uid_str in child_list:
                        children[str2uuid(uid_str)] = child_type.replace(
                            "s", ""
                        ).lower()

        return children

    @classmethod
    def fetch_concatenated_values(
        cls,
        file: str | h5py.File,
        uid: uuid.UUID,
        entity_type: str,
        label: str,
    ) -> tuple | None:
        """
        Get :obj:`~geoh5py.shared.entity.Entity.children` values of concatenated group.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier
        :param entity_type: Type of entity from
            'group', 'data', 'object', 'group_type', 'data_type', 'object_type'
        :param label: Group identifier for the attribute requested.

        :return children: [{uuid: type}, ... ]
            List of dictionaries for the children uid and type
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]
            entity_type = cls.format_type_string(entity_type)
            label = KEY_MAP.get(label, label)

            try:
                group = h5file[name][entity_type][as_str_if_uuid(uid)][
                    "Concatenated Data"
                ]
                indices = group["Index"].get(label.replace("/", "\u2044"))
                if indices is None:
                    return None

                array = group["Data"].get(label.replace("/", "\u2044"))
                if array is None:
                    array = group.get(label.replace("/", "\u2044"))

                attribute = array[:]

                if attribute.dtype in [float, "float64", "float32"]:
                    attribute[attribute == FLOAT_NDV] = np.nan

                if (
                    attribute.dtype == object
                    and len(attribute) > 0
                    and isinstance(attribute[0], bytes)
                ):
                    attribute = np.char.decode(attribute.astype(np.bytes_), "UTF-8")

                return attribute, indices[:]

            except KeyError:
                return None

    @classmethod
    def fetch_concatenated_attributes(
        cls,
        file: str | h5py.File,
        uid: uuid.UUID,
        entity_type: str,
        label: str,
    ) -> list | dict | None:
        """
        Get 'Attributes', 'Data' or 'Index' from Concatenator group.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier
        :param entity_type: Type of entity from
            'group', 'data', 'object', 'group_type', 'data_type', 'object_type'
        :param label: Group identifier for the attribute requested.

        :return children: [{uuid: type}, ... ]
            List of dictionaries for the children uid and type
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]
            entity_type = cls.format_type_string(entity_type)
            label = KEY_MAP.get(label, label)

            try:
                group = h5file[name][entity_type][as_str_if_uuid(uid)][
                    "Concatenated Data"
                ]
                if label == "Attributes":
                    attribute = group[label][()]
                    if isinstance(attribute, np.ndarray):
                        attribute = attribute[0]

                    return json.loads(as_str_if_utf8_bytes(attribute))

                if label == "Attributes Jsons":
                    attribute = group[label][()]
                    return {
                        "Attributes": [
                            json.loads(as_str_if_utf8_bytes(val)) for val in attribute
                        ]
                    }

                return list(group[label])

            except KeyError:
                return None

    @classmethod
    def fetch_metadata(
        cls,
        file: str | h5py.File,
        uid: uuid.UUID,
        entity_type: str = "Objects",
        argument: str = "Metadata",
    ) -> str | dict | None:
        """
        Fetch text of dictionary type attributes of an entity.

        :param file: Target h5 file.
        :param uid: Unique identifier of the target Entity.
        :param entity_type: Base type of the target Entity.
        :param argument: Label name of the dictionary.
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]

            try:
                value = np.r_[h5file[name][entity_type][as_str_if_uuid(uid)][argument]][
                    0
                ]

            except KeyError:
                return None

        return str_json_to_dict(value)

    @classmethod
    def fetch_project_attributes(cls, file: str | h5py.File) -> dict[Any, Any]:
        """
        Get attributes of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: :obj:`h5py.File` or name of the target geoh5 file

        :return attributes: :obj:`dict` of attributes.
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)
            if len(name) != 1:
                raise FileNotFoundError

            attributes = {}

            for key, value in h5file[name[0]].attrs.items():
                attributes[key] = value

        return attributes

    @classmethod
    def fetch_property_groups(
        cls, file: str | h5py.File, uid: uuid.UUID
    ) -> dict[str, dict[str, str]]:
        r"""
        Get the property groups.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target entity

        :return property_group_attributes: :obj:`dict` of property groups
            and respective attributes.

        .. code-block:: python

            property_group = {
                "group_1": {"attribute": value, ...},
                ...,
                "group_N": {"attribute": value, ...},
            }
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]
            property_groups: dict[str, dict[str, str]] = {}
            try:
                pg_handle = h5file[name]["Objects"][as_str_if_uuid(uid)][
                    "PropertyGroups"
                ]
                for pg_uid in pg_handle:
                    property_groups[pg_uid] = {}
                    for attr, value in pg_handle[pg_uid].attrs.items():
                        property_groups[pg_uid][INV_KEY_MAP.get(attr, attr)] = value
            except KeyError:
                pass
        return property_groups

    @classmethod
    def fetch_type(
        cls, file: str | h5py.File, uid: uuid.UUID, entity_type: str
    ) -> dict:
        """
        Fetch a type from the target geoh5.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target entity
        :param entity_type: One of 'Data', 'Object' or 'Group'
        :return property_group_attributes: :obj:`dict` of property groups
            and respective attributes.

        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]
            entity_type = entity_type + " types"
            type_handle = h5file[name]["Types"][entity_type][as_str_if_uuid(uid)]
            return cls.fetch_type_attributes(type_handle)

    @classmethod
    def fetch_type_attributes(cls, type_handle: h5py.Group) -> dict:
        """
        Fetch type attributes from a given h5 handle.
        """
        type_attributes = {}
        for key, value in type_handle.attrs.items():
            type_attributes[INV_KEY_MAP.get(key, key)] = value

        if "Color map" in type_handle:
            type_attributes["color_map"] = {}
            for key, value in type_handle["Color map"].attrs.items():
                type_attributes["color_map"][key] = value
            type_attributes["color_map"]["values"] = type_handle["Color map"][:]

        if "Value map" in type_handle:
            type_attributes["value_map"] = type_handle["Value map"][:].astype(
                [("Key", "<u4"), ("Value", h5py.special_dtype(vlen=str))]
            )
        return type_attributes

    @classmethod
    def fetch_uuids(cls, file: str | h5py.File, entity_type: str) -> list:
        """
        Fetch all uuids of a given type from geoh5

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param entity_type: Type of entity from
            'group', 'data', 'object', 'group_type', 'data_type', 'object_type'

        :return uuids: [uuid1, uuid2, ...]
            List of uuids
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]
            entity_type = cls.format_type_string(entity_type)
            try:
                uuids = [str2uuid(uid) for uid in h5file[name][entity_type]]
            except KeyError:
                uuids = []

        return uuids

    @classmethod
    def fetch_file_object(
        cls, file: str | h5py.File, uid: uuid.UUID, file_name: str
    ) -> bytes | None:
        """
        Load data associated with an image file

        :param file: Name of the target geoh5 file
        :param uid: Unique identifier of the target entity
        :param file_name: Name of the file stored as bytes data.

        :return values: Data file stored as bytes
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]

            try:
                bytes_value = h5file[name]["Data"][as_str_if_uuid(uid)][file_name][
                    ()
                ].tobytes()

            except KeyError:
                bytes_value = None

        return bytes_value

    @classmethod
    def fetch_values(
        cls, file: str | h5py.File, uid: uuid.UUID
    ) -> np.ndarray | str | float | None:
        """
        Get data :obj:`~geoh5py.data.data.Data.values`

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target entity

        :return values: :obj:`numpy.array` of :obj:`float`
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file)[0]

            try:
                values = np.r_[h5file[name]["Data"][as_str_if_uuid(uid)]["Data"]]
                if isinstance(values[0], (str, bytes)):
                    values = np.asarray([as_str_if_utf8_bytes(val) for val in values])
                    if len(values) == 1:
                        values = values[0]
                else:
                    if values.dtype in [float, "float64", "float32"]:
                        ind = values == FLOAT_NDV
                        values[ind] = np.nan

            except KeyError:
                values = None

        return values

    @staticmethod
    def format_type_string(string: str) -> str:
        """Format names used for types."""
        string = string.capitalize()
        if string in ["Group", "Object"]:
            string += "s"

        return string
