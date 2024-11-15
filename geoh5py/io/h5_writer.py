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

# pylint: disable=too-many-public-methods, too-many-lines

from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING
from warnings import warn

import h5py
import numpy as np

from ..data import (
    CommentsData,
    Data,
    DataType,
    FilenameData,
    GeometricDataValueMapType,
    ReferenceDataType,
    ReferencedData,
    ReferenceValueMap,
)
from ..groups import Group, GroupType, PropertyGroup, RootGroup
from ..objects import ObjectBase, ObjectType
from ..shared import FLOAT_NDV, Entity, EntityType, fetch_h5_handle
from ..shared.concatenation import Concatenator
from ..shared.utils import KEY_MAP, as_str_if_uuid, dict_mapper
from .utils import str_from_subtype, str_from_type


if TYPE_CHECKING:
    from .. import shared, workspace


class H5Writer:
    """
    Writing class to a geoh5 file.
    """

    str_type = h5py.special_dtype(vlen=str)

    @staticmethod
    def init_geoh5(
        file: str | h5py.File,
        workspace: workspace.Workspace,
    ):
        """
        Add the geoh5 core structure.

        :param file: Name or handle to a geoh5 file.
        :param workspace: :obj:`~geoh5py.workspace.workspace.Workspace` object
            defining the project structure.

        :return h5file: Pointer to a geoh5 file.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            project = h5file.create_group(workspace.name)
            H5Writer.write_attributes(h5file, workspace)
            project.create_group("Data")
            project.create_group("Groups")
            project.create_group("Objects")
            types = project.create_group("Types")
            types.create_group("Data types")
            types.create_group("Group types")
            types.create_group("Object types")

    @staticmethod
    def create_dataset(entity_handle, dataset: np.ndarray, label: str) -> None:
        """
        Create a dataset on geoh5.

        :param entity_handle: Pointer to a hdf5 group
        :param dataset: Array of values to be written
        :param label: Name of the dataset on file
        """
        entity_handle.create_dataset(
            label,
            data=dataset,
            dtype=dataset.dtype,
            compression="gzip",
            compression_opts=9,
        )

    @staticmethod
    def remove_child(
        file: str | h5py.File,
        uid: uuid.UUID,
        ref_type: str,
        parent: Entity,
    ) -> None:
        """
        Remove a child from a parent.

        :param file: Name or handle to a geoh5 file
        :param uid: uuid of the target :obj:`~geoh5py.shared.entity.Entity`
        :param ref_type: Input type from: 'Types', 'Groups', 'Objects' or 'Data
        :param parent: Remove entity from parent.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            uid_str = as_str_if_uuid(uid)
            parent_handle = H5Writer.fetch_handle(h5file, parent)

            if parent_handle is None or parent_handle.get(ref_type) is None:
                return

            if uid_str in parent_handle[ref_type]:
                del parent_handle[ref_type][uid_str]
                parent.workspace.repack = True

    @staticmethod
    def remove_entity(
        file: str | h5py.File,
        uid: uuid.UUID,
        ref_type: str,
        parent: Entity | None = None,
    ) -> None:
        """
        Remove an entity and its type from the target geoh5 file.

        :param file: Name or handle to a geoh5 file
        :param uid: uuid of the target :obj:`~geoh5py.shared.entity.Entity`
        :param ref_type: Input type from: 'Types', 'Groups', 'Objects' or 'Data
        :param parent: Remove entity from parent.

        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            base = list(h5file)[0]
            base_type_handle = h5file[base][ref_type]
            uid_str = as_str_if_uuid(uid)

            if ref_type == "Types":
                for e_type in ["Data types", "Group types", "Object types"]:
                    if uid_str in base_type_handle[e_type]:
                        del base_type_handle[e_type][uid_str]
            else:
                if uid_str in base_type_handle:
                    del base_type_handle[uid_str]

                if parent is not None:
                    H5Writer.remove_child(h5file, uid, ref_type, parent)

    @staticmethod
    def fetch_handle(
        file: str | h5py.File,
        entity,
        return_parent: bool = False,
    ) -> None | h5py.Group:
        """
        Get a pointer to an :obj:`~geoh5py.shared.entity.Entity` in geoh5.

        :param file: Name or handle to a geoh5 file
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`
        :param return_parent: Option to return the handle to the parent entity.

        :return entity_handle: HDF5 pointer to an existing entity, parent or None if not found.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            base = list(h5file)[0]
            base_handle = h5file[base]

            if entity.name == base:
                return base_handle

            uid = entity.uid
            hierarchy = {
                Data: "Data",
                ObjectBase: "Objects",
                Group: "Groups",
                DataType: "Data types",
                ObjectType: "Object types",
                GroupType: "Group types",
            }

            if isinstance(entity, EntityType):
                if "Types" in base_handle:
                    base_handle = base_handle["Types"]
                else:
                    base_handle = base_handle.create_group("Types")

            for key, value in hierarchy.items():
                if isinstance(entity, key):
                    if value in base_handle:
                        base_handle = base_handle[value]
                    else:
                        base_handle = base_handle.create_group(value)

            # Check if already in the project
            if as_str_if_uuid(uid) in base_handle:
                if return_parent:
                    return base_handle

                return base_handle[as_str_if_uuid(uid)]

        return None

    @staticmethod
    def save_entity(
        file: str | h5py.File,
        entity,
        compression: int = 5,
        add_children: bool = True,
    ) -> h5py.Group:
        """
        Save a :obj:`~geoh5py.shared.entity.Entity` to geoh5 with its
        :obj:`~geoh5py.shared.entity.Entity.children` recursively.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        :param compression: Compression level for the data.
        :param add_children: Add :obj:`~geoh5py.shared.entity.Entity.children`.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            new_entity = H5Writer.write_entity(h5file, entity, compression)

            if (
                add_children
                and not isinstance(entity, Concatenator)
                and hasattr(entity, "children")
            ):
                # Write children entities and add to current parent
                for child in entity.children:
                    if not isinstance(child, PropertyGroup):
                        H5Writer.save_entity(h5file, child, compression)

            H5Writer.write_to_parent(
                h5file, entity, compression=compression, recursively=False
            )

        return new_entity

    @staticmethod
    def update_concatenated_field(
        file: str | h5py.File, entity, attribute: str, channel: str
    ) -> None:
        """
        Update the attributes of a concatenated :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        :param attribute: Name of the attribute to get updated.
        :param channel: Name of the data or index to be modified.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            if entity_handle is None:
                return

            attr_handle = entity_handle["Concatenated Data"].get(attribute.capitalize())

            if attr_handle is None:
                attr_handle = entity_handle["Concatenated Data"].create_group(
                    attribute.capitalize()
                )
            name = channel.replace("/", "\u2044")
            if name in attr_handle:
                del attr_handle[name]
                entity.workspace.repack = True

            values = getattr(entity, attribute).get(channel, None)

            if values is None:
                return

            if isinstance(values, np.ndarray):
                if np.issubdtype(values.dtype, np.floating):
                    values = values.astype(np.float32)

                    if len(values) > 0:
                        values[np.isnan(values)] = FLOAT_NDV

                if np.issubdtype(values.dtype, np.str_):
                    values = values.astype(h5py.special_dtype(vlen=str))

            attr_handle.create_dataset(
                name,
                data=values,
                compression="gzip",
                compression_opts=9,
            )

    @staticmethod
    def update_field(
        file: str | h5py.File,
        entity,
        attribute: str,
        compression: int = 5,
        **kwargs,
    ) -> None:
        """
        Update the attributes of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        :param attribute: Name of the attribute to get updated.
        :param compression: Compression level for the data.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            if entity_handle is None:
                return
            if attribute == "concatenated_attributes":
                H5Writer.write_group_values(
                    h5file, entity, attribute, compression, **kwargs
                )
            elif attribute in [
                "values",
            ]:
                H5Writer.write_data_values(
                    h5file, entity, attribute, compression, **kwargs
                )
            elif attribute in ["metadata", "options"]:
                H5Writer.write_metadata(h5file, entity, attribute, **kwargs)
            elif attribute in [
                "cells",
                "concatenated_object_ids",
                "layers",
                "octree_cells",
                "property_group_ids",
                "prisms",
                "surveys",
                "trace",
                "trace_depth",
                "u_cell_delimiters",
                "v_cell_delimiters",
                "vertices",
                "z_cell_delimiters",
            ]:
                H5Writer.write_array_attribute(h5file, entity, attribute, **kwargs)
            elif attribute == "property_groups":
                H5Writer.write_property_groups(h5file, entity)
            elif attribute == "color_map":
                H5Writer.write_color_map(h5file, entity)
            elif attribute == "value_map":
                H5Writer.write_value_map(h5file, entity)
            elif attribute == "data_map":
                H5Writer.write_data_map(h5file, entity)
            elif attribute == "entity_type":
                del entity_handle["Type"]
                entity.workspace.repack = True
                new_type = H5Writer.write_entity_type(h5file, entity.entity_type)
                entity_handle["Type"] = new_type
            else:
                H5Writer.write_attributes(h5file, entity)

    @staticmethod
    def write_attributes(
        file: str | h5py.File,
        entity,
    ) -> None:
        """
        Write attributes of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Entity with attributes to be added to the geoh5 file.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            if entity_handle is None:
                return

            for key, attr in entity.attribute_map.items():
                try:
                    value = getattr(entity, attr)
                except AttributeError:
                    continue

                value = as_str_if_uuid(value)

                if (
                    key
                    in [
                        "PropertyGroups",
                        "Attributes",
                        "Attributes Jsons",
                        "Property Groups IDs",
                        "Concatenated object IDs",
                    ]
                    or value is None
                ):  # or key in Concatenator._attribute_map:
                    continue

                if key in ["Association", "Primitive type"]:
                    value = KEY_MAP[value.name]

                if isinstance(value, (np.int8, bool)):
                    entity_handle.attrs.create(key, int(value), dtype="int8")
                elif isinstance(value, str):
                    entity_handle.attrs.create(key, value, dtype=H5Writer.str_type)
                else:
                    entity_handle.attrs.create(
                        key, value, dtype=np.asarray(value).dtype
                    )

    @staticmethod
    def write_color_map(
        file: str | h5py.File,
        entity_type: shared.EntityType,
    ) -> None:
        """
        Add :obj:`~geoh5py.data.color_map.ColorMap` to a
        :obj:`~geoh5py.data.data_type.DataType`.

        :param file: Name or handle to a geoh5 file
        :param entity_type: Target entity_type with color_map
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            color_map = getattr(entity_type, "color_map", None)
            entity_type_handle = H5Writer.fetch_handle(h5file, entity_type)

            if entity_type_handle is None:
                return

            if "Color map" in entity_type_handle:
                del entity_type_handle["Color map"]
                entity_type.workspace.repack = True

            if color_map is not None and color_map.values is not None:
                H5Writer.create_dataset(
                    entity_type_handle,
                    color_map._values,  # pylint: disable=protected-access
                    "Color map",
                )
                entity_type_handle["Color map"].attrs.create(
                    "File name", color_map.name, dtype=H5Writer.str_type
                )

    @staticmethod
    def write_value_map(
        file: str | h5py.File,
        entity_type: ReferenceDataType,
        name="Value map",
        value_map: ReferenceValueMap | None = None,
    ) -> None:
        """
        Add :obj:`~geoh5py.data.reference_value_map.ReferenceValueMap` to a
        :obj:`~geoh5py.data.data_type.DataType`.

        :param file: Name or handle to a geoh5 file
        :param entity_type: Target entity_type with value_map
        :param name: Name of the value map
        :param value_map: Value map to be written
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            if isinstance(entity_type, GeometricDataValueMapType):
                return

            entity_type_handle = H5Writer.fetch_handle(h5file, entity_type)

            if entity_type_handle is None:
                return

            dtype = np.dtype([("Key", "<u4"), ("Value", H5Writer.str_type)])

            if value_map is None:
                value_map = entity_type.value_map

            if value_map is None:
                return

            if not isinstance(value_map, ReferenceValueMap):
                raise TypeError("Value map must be a ReferenceValueMap object.")

            if name in entity_type_handle:
                del entity_type_handle[name]
                entity_type.workspace.repack = True

            H5Writer.create_dataset(
                entity_type_handle,
                value_map.map.astype(dtype),
                name,
            )

            if name != "Value map":
                entity_type_handle[name].attrs.create(
                    "Name", value_map.name, dtype=H5Writer.str_type
                )
                entity_type_handle[name].attrs.create(
                    "Allow delete", True, dtype="int8"
                )

    @staticmethod
    def write_data_map(file: str | h5py.File, data: ReferencedData):
        """
        Write the value map of geometric data.

        :param file: Name or handle to a geoh5 file
        :param data: Target referenced data with value map
        """
        if data.data_maps is None:
            return

        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_type_handle = H5Writer.fetch_handle(h5file, data.entity_type)

            if entity_type_handle is None:
                return

            for name in entity_type_handle:
                if re.match("Value map [0-9]", name):
                    del entity_type_handle[name]

            for ii, child in enumerate(data.data_maps.values()):
                H5Writer.write_value_map(
                    h5file,
                    data.entity_type,
                    f"Value map {ii + 1}",
                    child.entity_type.value_map,
                )

    @staticmethod
    def write_visible(
        file: str | h5py.File,
        entity,
    ) -> None:
        """
        Needs revision once Visualization is implemented

        :param file: Name or handle to a geoh5 file
        :param entity: Target entity
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            if entity_handle is None:
                return

            dtype = np.dtype(
                [("ViewID", h5py.special_dtype(vlen=str)), ("Visible", "int8")]
            )

            if entity.visible:
                visible = entity_handle.create_dataset(
                    "Visible", shape=(1,), dtype=dtype
                )
                visible["Visible"] = 1

    @staticmethod
    def write_array_attribute(
        file: str | h5py.File, entity, attribute, values=None, **kwargs
    ) -> None:
        """
        Add :obj:`~geoh5py.objects.object_base.ObjectBase.surveys` of an object.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target entity.
        :param attribute: Name of the attribute to be written to geoh5
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            if entity_handle is None:
                return

            if values is None and getattr(entity, f"{attribute}", None) is not None:
                values = getattr(entity, f"_{attribute}", None)

            if (
                isinstance(entity, Concatenator)
                and attribute != "concatenated_object_ids"
            ):
                entity_handle = entity_handle["Concatenated Data"]

            if KEY_MAP[attribute] in entity_handle:
                del entity_handle[KEY_MAP[attribute]]
                entity.workspace.repack = True

            if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.str_):
                values = values.astype(h5py.special_dtype(vlen=str))

            if values is not None:
                entity_handle.create_dataset(
                    KEY_MAP[attribute],
                    data=values,
                    compression="gzip",
                    compression_opts=9,
                    **kwargs,
                )

    @staticmethod
    def prepare_data_to_write(
        h5file: h5py.File, entity: Data | Group | ObjectBase, attribute: str
    ) -> tuple[h5py.Group, str] | tuple[None, None]:
        """
        Prepare data to be written to a geoh5 file.

        :param h5file: Name or handle to a geoh5 file.
        :param entity: Entity with attributes to be added to the geoh5 file.
        :param attribute: The attribute to be written to the geoh5 file.

        :return: The entity handle, the name map, and the values to be written to the geoh5 file.
        """
        entity_handle = H5Writer.fetch_handle(h5file, entity)

        if entity_handle is None:
            return None, None

        name_map = KEY_MAP[attribute]

        if isinstance(entity, Concatenator) and name_map != "Metadata":
            entity_handle = entity_handle["Concatenated Data"]

            if (
                attribute == "concatenated_attributes"
                and entity.concat_attr_str == "Attributes Jsons"
            ):
                name_map = entity.concat_attr_str

        if name_map in entity_handle:
            del entity_handle[name_map]
            entity.workspace.repack = True

        return entity_handle, name_map

    @staticmethod
    def write_group_values(
        file: str | h5py.File,
        entity: Group,
        attribute,
        compression: int,
        values=None,
        **kwargs,
    ):
        """
        Add Concatenator values.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target entity.
        :param attribute: Name of the attribute to be written to geoh5
        :param compression: Compression level for the data.
        :param values: Data values.
        """
        if not isinstance(entity, Group):
            raise TypeError(
                "Entity must be a Group object; " f"got '{type(entity)}' instead."
            )
        # check type here
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle, name_map = H5Writer.prepare_data_to_write(
                h5file, entity, attribute
            )

            # Give the chance to fetch from file
            values = getattr(entity, attribute, None) if values is None else values

            if values is None or entity_handle is None:
                return

            if (
                attribute == "concatenated_attributes"
                and isinstance(entity, Concatenator)
                and entity.concat_attr_str == "Attributes Jsons"
            ):
                values = [
                    json.dumps(val).encode("utf-8") for val in values["Attributes"]
                ]

            if name_map == "Attributes Jsons":
                entity_handle.create_dataset(
                    name_map,
                    data=values,
                    compression="gzip",
                    compression_opts=compression,
                    **kwargs,
                )
            elif isinstance(values, dict):
                values = deepcopy(values)
                values = dict_mapper(values, [as_str_if_uuid])
                entity_handle.create_dataset(
                    name_map,
                    data=json.dumps(values, indent=4),
                    dtype=h5py.special_dtype(vlen=str),
                    shape=(1,),
                    **kwargs,
                )
            else:
                warn(f"Writing '{values}' on '{entity.name}' failed.")

    @staticmethod
    def write_data_values(  # pylint: disable=too-many-branches
        file: str | h5py.File,
        entity: Data,
        attribute,
        compression: int,
        **kwargs,
    ) -> None:
        """
        Add data :obj:`~geoh5py.data.data.Data.values`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target entity.
        :param attribute: Name of the attribute to be written to geoh5
        :param compression: Compression level for the data.
        """
        if not isinstance(entity, Data):
            raise TypeError(
                "Entity must be a Data object; " f"got '{type(entity)}' instead."
            )

        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle, name_map = H5Writer.prepare_data_to_write(
                h5file, entity, attribute
            )
            if entity.formatted_values is None or entity_handle is None:
                return
            if isinstance(entity, CommentsData):
                entity_handle.create_dataset(
                    name_map,
                    data=entity.formatted_values,
                    dtype=h5py.special_dtype(vlen=str),
                    shape=(1,),
                    **kwargs,
                )
            elif isinstance(entity, FilenameData):
                H5Writer.write_file_name_data(entity_handle, entity)
            elif isinstance(entity.values, str):
                entity_handle.create_dataset(
                    name_map,
                    data=entity.values,
                    dtype=h5py.special_dtype(vlen=str),
                    shape=(1,),
                    **kwargs,
                )
            else:
                entity_handle.create_dataset(
                    name_map,
                    data=entity.formatted_values,
                    compression="gzip",
                    compression_opts=compression,
                    **kwargs,
                )

    @staticmethod
    def write_metadata(  # pylint: disable=too-many-branches
        file: str | h5py.File,
        entity: Group | ObjectBase,
        attribute,
        values=None,
        **kwargs,
    ) -> None:
        """
        Add data :obj:`~geoh5py.data.data.Data.values`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target entity.
        :param attribute: Name of the attribute to be written to geoh5
        :param values: Data values.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle, name_map = H5Writer.prepare_data_to_write(
                h5file, entity, attribute
            )

            # Give the chance to fetch from file
            values = getattr(entity, attribute, None) if values is None else values

            if values is None or entity_handle is None:
                return

            values = deepcopy(values)
            values = dict_mapper(values, [as_str_if_uuid])
            entity_handle.create_dataset(
                name_map,
                data=json.dumps(values, indent=4),
                dtype=h5py.special_dtype(vlen=str),
                shape=(1,),
                **kwargs,
            )

    @staticmethod
    def clear_stats_cache(
        file: str | h5py.File,
        entity: Data,
    ) -> None:
        """
        Clear the StatsCache dataset.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target entity.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            if not isinstance(entity, Data):
                return

            entity_type_handle = H5Writer.fetch_handle(h5file, entity.entity_type)
            if entity_type_handle is None:
                return

            stats_cache = entity_type_handle.get("StatsCache")
            if stats_cache is not None:
                del entity_type_handle["StatsCache"]
                entity.workspace.repack = True

    @staticmethod
    def write_entity(
        file: str | h5py.File,
        entity,
        compression: int,
    ) -> h5py.Group:
        """
        Add an :obj:`~geoh5py.shared.entity.Entity` and its attributes to geoh5.
        The function returns a pointer to the entity if already present on file.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        :param compression: Compression level for data.

        :return entity: Pointer to the written entity. Active link if "close_file" is False.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            base = list(h5file)[0]
            entity_type = str_from_type(entity)
            uid = entity.uid

            if entity_type not in h5file[base]:
                h5file[base].create_group(entity_type)

            # Check if already in the project
            if as_str_if_uuid(uid) in h5file[base][entity_type]:
                entity.on_file = True

                return h5file[base][entity_type][as_str_if_uuid(uid)]

            entity_handle = h5file[base][entity_type].create_group(as_str_if_uuid(uid))
            if isinstance(entity, Concatenator):
                concat_group = entity_handle.create_group("Concatenated Data")
                entity_handle.create_group("Data")
                concat_group.create_group("Index")
                concat_group.create_group("Data")
                entity_handle.create_group("Groups")
            elif entity_type == "Groups":
                entity_handle.create_group("Data")
                entity_handle.create_group("Groups")
                entity_handle.create_group("Objects")
            elif entity_type == "Objects":
                entity_handle.create_group("Data")

            # Add the type
            new_type = H5Writer.write_entity_type(h5file, entity.entity_type)
            entity_handle["Type"] = new_type
            entity.entity_type.on_file = True

            H5Writer.write_properties(h5file, entity, compression)
            entity.on_file = True

            if isinstance(entity, RootGroup):
                if "Root" in h5file[base]:
                    del h5file[base]["Root"]

                h5file[base]["Root"] = entity_handle

        return entity_handle

    @staticmethod
    def write_entity_type(
        file: str | h5py.File,
        entity_type: shared.EntityType | ReferenceDataType,
    ) -> h5py.Group:
        """
        Add an :obj:`~geoh5py.shared.entity_type.EntityType` to geoh5.

        :param file: Name or handle to a geoh5 file.
        :param entity_type: Entity with type to be added.

        :return type: Pointer to :obj:`~geoh5py.shared.entity_type.EntityType` in geoh5.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            base = list(h5file)[0]
            uid = entity_type.uid
            entity_type_str = str_from_subtype(entity_type)

            if "Types" not in h5file[base]:
                h5file[base].create_group("Types")

            # Check if already in the project
            if entity_type_str not in h5file[base]["Types"]:
                h5file[base]["Types"].create_group(entity_type_str)

            if as_str_if_uuid(uid) in h5file[base]["Types"][entity_type_str]:
                entity_type.on_file = True

                return h5file[base]["Types"][entity_type_str][as_str_if_uuid(uid)]

            new_type = h5file[base]["Types"][entity_type_str].create_group(
                as_str_if_uuid(uid)
            )
            H5Writer.write_attributes(h5file, entity_type)

            if hasattr(entity_type, "color_map"):
                H5Writer.write_color_map(h5file, entity_type)

            if isinstance(entity_type, ReferenceDataType):
                H5Writer.write_value_map(h5file, entity_type)

            entity_type.on_file = True

        return new_type

    @staticmethod
    def write_file_name_data(entity_handle: h5py.Group, entity: FilenameData) -> None:
        """
        Write a dataset for the file name and file blob.

        :param entity_handle: Pointer to the geoh5 Group.
        :param entity: Target :obj:`~geoh5py.data.filename_data.FilenameData` entity.
        :param values: Bytes data
        """
        if entity.file_bytes is None or entity.values is None:
            raise AttributeError("FilenameData requires the 'file_bytes' to be set.")

        entity_handle.create_dataset(
            "Data",
            data=entity.values,
            dtype=h5py.special_dtype(vlen=str),
            shape=(1,),
        )

        if entity.values in entity_handle:
            del entity_handle[entity.values]
            entity.workspace.repack = True

        entity_handle.create_dataset(
            entity.values,
            data=np.asarray(np.void(entity.file_bytes[:])),
            shape=(1,),
        )

    @staticmethod
    def write_properties(
        file: str | h5py.File,
        entity: Entity,
        compression: int,
    ) -> None:
        """
        Add properties of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        :param compression: Compression level for data.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            H5Writer.update_field(h5file, entity, "attributes", compression)

            for attribute in KEY_MAP:
                if getattr(entity, attribute, None) is not None:
                    H5Writer.update_field(h5file, entity, attribute, compression)

    @staticmethod
    def write_property_groups(
        file: str | h5py.File,
        entity,
    ) -> None:
        """
        Write :obj:`~geoh5py.groups.property_group.PropertyGroup` associated with
        an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)
            if entity_handle is None:
                return

            if "PropertyGroups" in entity_handle:
                del entity_handle["PropertyGroups"]
                entity.workspace.repack = True

            if hasattr(entity, "property_groups") and isinstance(
                entity.property_groups, list
            ):
                for p_g in entity.property_groups:
                    H5Writer.add_or_update_property_group(h5file, p_g)

    @staticmethod
    def add_or_update_property_group(file, property_group, remove=False):
        """
        Update a :obj:`~geoh5py.groups.property_group.PropertyGroup` associated with
        an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param property_group: Target PropertyGroup
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, property_group.parent)

            if entity_handle is None:
                return

            if "PropertyGroups" not in entity_handle:
                entity_handle.create_group("PropertyGroups")

            uid = as_str_if_uuid(property_group.uid)
            if uid in entity_handle["PropertyGroups"]:
                del entity_handle["PropertyGroups"][uid]
                property_group.parent.workspace.repack = True

            if remove:
                return

            entity_handle["PropertyGroups"].create_group(uid)
            group_handle = entity_handle["PropertyGroups"][uid]

            for key, attr in property_group.attribute_map.items():
                try:
                    value = getattr(property_group, attr)
                except AttributeError:
                    continue

                if key == "Association":
                    value = value.name.capitalize()

                elif key == "Properties":
                    if value is None:
                        continue

                    value = np.asarray([as_str_if_uuid(val) for val in value])

                elif key == "ID":
                    value = as_str_if_uuid(value)
                elif key == "Property Group Type":
                    value = value.value

                group_handle.attrs.create(
                    key, value, dtype=h5py.special_dtype(vlen=str)
                )

            property_group.on_file = True

    @staticmethod
    def write_to_parent(
        file: str | h5py.File,
        entity: Entity,
        compression: int,
        recursively: bool = False,
    ) -> None:
        """
        Add/create an :obj:`~geoh5py.shared.entity.Entity` and add it to its parent.

        :param file: Name or handle to a geoh5 file.
        :param entity: Entity to be added or linked to a parent in geoh5.
        :param compression: Compression level for data.
        :param recursively: Add parents recursively until reaching the
            :obj:`~geoh5py.groups.root_group.RootGroup`.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            if isinstance(entity, RootGroup):
                return

            uid = entity.uid
            entity_handle = H5Writer.write_entity(h5file, entity, compression)
            parent_handle = H5Writer.write_entity(h5file, entity.parent, compression)
            entity_type = str_from_type(entity)

            # Check if child h5py.Group already exists
            if entity_type not in parent_handle:
                parent_handle.create_group(entity_type)

            # Check if child uuid not already in h5
            if as_str_if_uuid(uid) not in parent_handle[entity_type]:
                parent_handle[entity_type][as_str_if_uuid(uid)] = entity_handle

            if recursively:
                H5Writer.write_to_parent(
                    h5file, entity.parent, compression=compression, recursively=True
                )
