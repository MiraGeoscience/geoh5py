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

import json
import uuid
from typing import Any

import h5py
import numpy as np

from ..data.float_data import FloatData
from ..data.integer_data import IntegerData
from ..shared import fetch_h5_handle
from .utils import as_str_if_uuid, str2uuid, str_from_utf8_bytes


class H5Reader:
    """
    Class to read information from a geoh5 file.
    """

    key_map = {
        "values": "Data",
        "cells": "Cells",
        "surveys": "Surveys",
        "trace": "Trace",
        "trace_depth": "TraceDepth",
        "vertices": "Vertices",
        "octree_cells": "Octree Cells",
        "property_groups": "PropertyGroups",
        "color_map": "Color map",
    }

    @classmethod
    def fetch_attributes(
        cls,
        file: str | h5py.File,
        uid: uuid.UUID,
        entity_type: str,
    ) -> tuple[dict, dict, dict]:
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
            name = list(h5file.keys())[0]
            attributes: dict = {"entity": {}}
            type_attributes: dict = {"entity_type": {}}
            property_groups: dict = {}

            entity_type = cls.format_type_string(entity_type)
            if "type" in entity_type:
                entity_type = entity_type.replace("_", " ") + "s"
                entity = h5file[name]["Types"][entity_type][as_str_if_uuid(uid)]
            elif entity_type == "Root":
                entity = h5file[name][entity_type]
            else:
                entity = h5file[name][entity_type][as_str_if_uuid(uid)]

            for key, value in entity.attrs.items():
                attributes["entity"][key] = value

            if "Type" in entity:
                for key, value in entity["Type"].attrs.items():
                    type_attributes["entity_type"][key] = value

                if "Color map" in entity["Type"].keys():
                    type_attributes["entity_type"]["color_map"] = {}
                    for key, value in entity["Type"]["Color map"].attrs.items():
                        type_attributes["entity_type"]["color_map"][key] = value
                    type_attributes["entity_type"]["color_map"]["values"] = entity[
                        "Type"
                    ]["Color map"][:]

                if "Value map" in entity["Type"].keys():
                    mapping = cls.fetch_value_map(file, uid)
                    type_attributes["entity_type"]["value_map"] = mapping

            # Check if the entity has property_group
            if "PropertyGroups" in entity.keys():
                property_groups = cls.fetch_property_groups(file, uid)

            attributes["entity"]["existing_h5_entity"] = True

        return attributes, type_attributes, property_groups

    @classmethod
    def fetch_cells(cls, file: str | h5py.File, uid: uuid.UUID) -> np.ndarray | None:
        """
        Get an object's :obj:`~geoh5py.objects.object_base.ObjectBase.cells`.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target object.

        :return cells: :obj:`numpy.ndarray` of :obj:`int`.
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file.keys())[0]
            indices = None

            try:
                indices = h5file[name]["Objects"][as_str_if_uuid(uid)]["Cells"][:]
            except KeyError:
                pass

        return indices

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
            name = list(h5file.keys())[0]
            children = {}
            entity_type = cls.format_type_string(entity_type)
            entity = h5file[name][entity_type][as_str_if_uuid(uid)]

            for child_type, child_list in entity.items():
                if child_type in ["Type", "PropertyGroups"]:
                    continue

                if isinstance(child_list, h5py.Group):
                    for uid_str in child_list.keys():
                        children[str2uuid(uid_str)] = child_type.replace(
                            "s", ""
                        ).lower()

        return children

    @classmethod
    def fetch_delimiters(
        cls,
        file: str | h5py.File,
        uid: uuid.UUID,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the delimiters of a :obj:`~geoh5py.objects.block_model.BlockModel`.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target entity.


        Returns
        -------
        u_delimiters: :obj:`numpy.ndarray` of u_delimiters
        v_delimiters: :obj:`numpy.ndarray` of v_delimiters
        z_delimiters: :obj:`numpy.ndarray` of z_delimiters
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file.keys())[0]

            try:
                u_delimiters = np.r_[
                    h5file[name]["Objects"][as_str_if_uuid(uid)]["U cell delimiters"]
                ]
            except KeyError:
                u_delimiters = None

            try:
                v_delimiters = np.r_[
                    h5file[name]["Objects"][as_str_if_uuid(uid)]["V cell delimiters"]
                ]
            except KeyError:
                v_delimiters = None

            try:
                z_delimiters = np.r_[
                    h5file[name]["Objects"][as_str_if_uuid(uid)]["Z cell delimiters"]
                ]
            except KeyError:
                z_delimiters = None

        return u_delimiters, v_delimiters, z_delimiters

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
        """

        with fetch_h5_handle(file) as h5file:
            name = list(h5file.keys())[0]

            try:
                metadata = np.r_[
                    h5file[name][entity_type][as_str_if_uuid(uid)][argument]
                ]
                metadata = str_from_utf8_bytes(metadata[0])

            except KeyError:
                return None

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except ValueError:
                pass

        if isinstance(metadata, dict):
            for key, val in metadata.items():
                if isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        metadata[key][sub_key] = str2uuid(sub_val)
                else:
                    metadata[key] = str2uuid(val)

        return metadata

    @classmethod
    def fetch_octree_cells(cls, file: str | h5py.File, uid: uuid.UUID) -> np.ndarray:
        """
        Get :obj:`~geoh5py.objects.octree.Octree`
        :obj:`~geoh5py.objects.object_base.ObjectBase.cells`.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target entity.

        :return octree_cells: :obj:`numpy.ndarray` of :obj:`int`.
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file.keys())[0]

            try:
                octree_cells = np.r_[
                    h5file[name]["Objects"][as_str_if_uuid(uid)]["Octree Cells"]
                ]
            except KeyError:
                octree_cells = None

        return octree_cells

    @classmethod
    def fetch_project_attributes(cls, file: str | h5py.File) -> dict[Any, Any]:
        """
        Get attributes of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: :obj:`h5py.File` or name of the target geoh5 file

        :return attributes: :obj:`dict` of attributes.
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file.keys())[0]
            attributes = {}

            for key, value in h5file[name].attrs.items():
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
            name = list(h5file.keys())[0]
            property_groups: dict[str, dict[str, str]] = {}
            try:
                pg_handle = h5file[name]["Objects"][as_str_if_uuid(uid)][
                    "PropertyGroups"
                ]
                for pg_uid in pg_handle.keys():
                    property_groups[pg_uid] = {}
                    for attr, value in pg_handle[pg_uid].attrs.items():
                        property_groups[pg_uid][attr] = value
            except KeyError:
                pass
        return property_groups

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
            name = list(h5file.keys())[0]
            entity_type = cls.format_type_string(entity_type)
            try:
                uuids = [str2uuid(uid) for uid in h5file[name][entity_type].keys()]
            except KeyError:
                uuids = []

        return uuids

    @classmethod
    def fetch_value_map(cls, file: str | h5py.File, uid: uuid.UUID) -> dict:
        """
        Get data :obj:`~geoh5py.data.data.Data.value_map`

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target entity

        :return value_map: :obj:`dict` of {:obj:`int`: :obj:`str`}
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file.keys())[0]
            try:
                entity = h5file[name]["Data"][as_str_if_uuid(uid)]
                value_map = entity["Type"]["Value map"][:]
                mapping = {}
                for key, value in value_map.tolist():
                    value = str_from_utf8_bytes(value)
                    mapping[key] = value

            except KeyError:
                mapping = {}

        return mapping

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
            name = list(h5file.keys())[0]

            try:
                bytes_value = h5file[name]["Data"][as_str_if_uuid(uid)][file_name][
                    ()
                ].tobytes()

            except KeyError:
                bytes_value = None

        return bytes_value

    @classmethod
    def fetch_values(cls, file: str | h5py.File, uid: uuid.UUID) -> float | None:
        """
        Get data :obj:`~geoh5py.data.data.Data.values`

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target entity

        :return values: :obj:`numpy.array` of :obj:`float`
        """
        with fetch_h5_handle(file) as h5file:
            name = list(h5file.keys())[0]

            try:
                values = np.r_[h5file[name]["Data"][as_str_if_uuid(uid)]["Data"]]
                if isinstance(values[0], (str, bytes)):
                    values = str_from_utf8_bytes(values[0])
                else:
                    if values.dtype in [float, "float64", "float32"]:
                        ind = values == FloatData.ndv()
                    else:
                        ind = values == IntegerData.ndv()
                        values = values.astype("float64")
                    values[ind] = np.nan

            except KeyError:
                values = None

        return values

    @classmethod
    def fetch_coordinates(
        cls, file: str | h5py.File, uid: uuid.UUID, name: str
    ) -> np.ndarray | None:
        """
        Get an object coordinates data.

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target object
        :param name: Type of coordinates 'vertices', 'trace' or 'surveys'

        :return surveys: :obj:`numpy.ndarray` of [x, y, z] coordinates
        """
        with fetch_h5_handle(file) as h5file:
            root = list(h5file.keys())[0]

            coordinates = None
            try:
                coordinates = np.asarray(
                    h5file[root]["Objects"][as_str_if_uuid(uid)][cls.key_map[name]]
                )
            except KeyError:
                pass

        return coordinates

    @classmethod
    def fetch_trace_depth(cls, file: str | h5py.File, uid: uuid.UUID) -> np.ndarray:
        """
        Get an object :obj:`~geoh5py.objects.drillhole.Drillhole.trace_depth` data

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param uid: Unique identifier of the target object

        :return surveys: :obj:`numpy.ndarray` of [x, y, z] coordinates
        """
        with fetch_h5_handle(file) as h5file:

            try:
                root = list(h5file.keys())[0]
                trace_depth = h5file[root]["Objects"][as_str_if_uuid(uid)]["TraceDepth"]
            except KeyError:
                trace_depth = None

        return trace_depth

    @staticmethod
    def format_type_string(string):
        string = string.capitalize()
        if string in ["Group", "Object"]:
            string += "s"

        return string
