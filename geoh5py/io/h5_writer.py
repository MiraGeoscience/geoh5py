#  Copyright (c) 2023 Mira Geoscience Ltd.
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

# pylint: disable=R0904

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING

import h5py
import numpy as np

from ..data import CommentsData, Data, DataType, FilenameData, IntegerData, TextData
from ..groups import Group, GroupType, RootGroup
from ..objects import ObjectBase, ObjectType
from ..shared import FLOAT_NDV, Entity, EntityType, fetch_h5_handle
from ..shared.concatenation import Concatenator
from ..shared.utils import KEY_MAP, as_str_if_uuid, dict_mapper

if TYPE_CHECKING:
    from .. import shared, workspace


class H5Writer:
    """
    Writing class to a geoh5 file.
    """

    str_type = h5py.special_dtype(vlen=str)

    @classmethod
    def create_geoh5(
        cls,
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
            cls.write_attributes(h5file, workspace)
            project.create_group("Data")
            project.create_group("Groups")
            project.create_group("Objects")
            types = project.create_group("Types")
            types.create_group("Data types")
            types.create_group("Group types")
            types.create_group("Object types")

    @classmethod
    def create_dataset(cls, entity_handle, dataset: np.ndarray, label: str) -> None:
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

    @classmethod
    def fetch_handle(
        cls,
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
                try:
                    base_handle = base_handle["Types"]
                except KeyError:
                    base_handle = base_handle.create_group("Types")

            for key, value in hierarchy.items():
                if isinstance(entity, key):
                    try:
                        base_handle = base_handle[value]
                    except KeyError:
                        base_handle = base_handle.create_group(value)
                    break

            # Check if already in the project
            if as_str_if_uuid(uid) in base_handle:
                if return_parent:
                    return base_handle

                return base_handle[as_str_if_uuid(uid)]

        return None

    @classmethod
    def save_entity(
        cls,
        file: str | h5py.File,
        entity,
        add_children: bool = True,
    ) -> h5py.Group:
        """
        Write an :obj:`~geoh5py.shared.entity.Entity` to geoh5 with its
        :obj:`~geoh5py.shared.entity.Entity.children`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        :param add_children: Add :obj:`~geoh5py.shared.entity.Entity.children`.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            new_entity = H5Writer.write_entity(h5file, entity)

            if add_children:
                # Write children entities and add to current parent
                for child in entity.children:
                    H5Writer.write_entity(h5file, child)
                    H5Writer.write_to_parent(h5file, child, recursively=False)

            H5Writer.write_to_parent(h5file, entity)

        return new_entity

    @classmethod
    def update_concatenated_field(
        cls, file: str | h5py.File, entity, attribute: str, channel: str
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
            try:
                del attr_handle[name]
                entity.workspace.repack = True
            except KeyError:
                pass

            dict_values = getattr(entity, attribute)

            if channel in dict_values:
                values = dict_values[channel]
                if isinstance(values, np.ndarray) and values.dtype == np.float64:
                    values[np.isnan(values)] = FLOAT_NDV
                    values = values.astype(np.float32)

                if len(values) > 0:
                    attr_handle.create_dataset(
                        name,
                        data=values,
                        compression="gzip",
                        compression_opts=9,
                    )

    @classmethod
    def update_field(
        cls, file: str | h5py.File, entity, attribute: str, **kwargs
    ) -> None:
        """
        Update the attributes of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        :param attribute: Name of the attribute to get updated.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            if entity_handle is None:
                return

            if attribute in [
                "concatenated_attributes",
                "metadata",
                "options",
                "trace_depth",
                "values",
            ]:
                cls.write_data_values(h5file, entity, attribute, **kwargs)
            elif attribute in [
                "cells",
                "concatenated_object_ids",
                "layers",
                "octree_cells",
                "property_group_ids",
                "prisms",
                "surveys",
                "trace",
                "u_cell_delimiters",
                "v_cell_delimiters",
                "vertices",
                "z_cell_delimiters",
            ]:
                cls.write_array_attribute(h5file, entity, attribute, **kwargs)
            elif attribute == "property_groups":
                cls.write_property_groups(h5file, entity)
            elif attribute == "color_map":
                cls.write_color_map(h5file, entity)
            elif attribute == "entity_type":
                del entity_handle["Type"]
                entity.workspace.repack = True
                new_type = H5Writer.write_entity_type(h5file, entity.entity_type)
                entity_handle["Type"] = new_type
            else:
                cls.write_attributes(h5file, entity)

    @classmethod
    def write_attributes(
        cls,
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
                    entity_handle.attrs.create(key, value, dtype=cls.str_type)
                else:
                    entity_handle.attrs.create(
                        key, value, dtype=np.asarray(value).dtype
                    )

    @classmethod
    def write_color_map(
        cls,
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

            try:
                del entity_type_handle["Color map"]
                entity_type.workspace.repack = True
            except KeyError:
                pass

            if color_map is not None and color_map.values is not None:
                cls.create_dataset(
                    entity_type_handle,
                    getattr(color_map, "_values"),
                    "Color map",
                )
                entity_type_handle["Color map"].attrs.create(
                    "File name", color_map.name, dtype=cls.str_type
                )

    @classmethod
    def write_value_map(
        cls,
        file: str | h5py.File,
        entity_type: shared.EntityType,
    ) -> None:
        """
        Add :obj:`~geoh5py.data.reference_value_map.ReferenceValueMap` to a
        :obj:`~geoh5py.data.data_type.DataType`.

        :param file: Name or handle to a geoh5 file
        :param entity_type: Target entity_type with value_map
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            reference_value_map = getattr(entity_type, "value_map", None)
            names = ["Key", "Value"]
            formats = ["<u4", h5py.special_dtype(vlen=str)]

            entity_type_handle = H5Writer.fetch_handle(h5file, entity_type)

            if entity_type_handle is None:
                return

            try:
                del entity_type_handle["Value map"]
                entity_type.workspace.repack = True
            except KeyError:
                pass

            if reference_value_map is not None and reference_value_map.map is not None:
                dtype = list(zip(names, formats))
                array = np.array(list(reference_value_map.map.items()), dtype=dtype)
                cls.create_dataset(entity_type_handle, array, "Value map")

    @classmethod
    def write_visible(
        cls,
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

    @classmethod
    def write_array_attribute(
        cls, file: str | h5py.File, entity, attribute, values=None, **kwargs
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

            try:
                del entity_handle[KEY_MAP[attribute]]
                entity.workspace.repack = True
            except KeyError:
                pass

            if values is not None:
                entity_handle.create_dataset(
                    KEY_MAP[attribute],
                    data=values,
                    compression="gzip",
                    compression_opts=9,
                    **kwargs,
                )

    @classmethod
    def write_data_values(  # pylint: disable=too-many-branches
        cls, file: str | h5py.File, entity, attribute, values=None, **kwargs
    ) -> None:
        """
        Add data :obj:`~geoh5py.data.data.Data.values`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target entity.
        :param attribute: Name of the attribute to be written to geoh5
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            if entity_handle is None:
                return

            name_map = KEY_MAP[attribute]
            if isinstance(entity, Concatenator):
                entity_handle = entity_handle["Concatenated Data"]

                if (
                    attribute == "concatenated_attributes"
                    and entity.concat_attr_str == "Attributes Jsons"
                ):
                    name_map = entity.concat_attr_str

            if name_map in entity_handle:
                del entity_handle[name_map]
                entity.workspace.repack = True

            if values is None:
                if getattr(entity, attribute, None) is None:
                    return

                values = getattr(entity, "_" + attribute)
                if (
                    attribute == "concatenated_attributes"
                    and entity.concat_attr_str == "Attributes Jsons"
                ):
                    values = [
                        json.dumps(val).encode("utf-8") for val in values["Attributes"]
                    ]

            # Adding an array of values
            if isinstance(values, dict) or isinstance(entity, CommentsData):
                values = deepcopy(values)
                if isinstance(entity, CommentsData):
                    values = {"Comments": values}

                values = dict_mapper(values, [as_str_if_uuid])

                entity_handle.create_dataset(
                    name_map,
                    data=json.dumps(values, indent=4),
                    dtype=h5py.special_dtype(vlen=str),
                    shape=(1,),
                    **kwargs,
                )

            elif isinstance(entity, FilenameData):
                cls.write_file_name_data(entity_handle, entity, values)

            elif isinstance(values, str):
                entity_handle.create_dataset(
                    name_map,
                    data=values,
                    dtype=h5py.special_dtype(vlen=str),
                    shape=(1,),
                    **kwargs,
                )

            else:
                out_values = deepcopy(values)
                if isinstance(entity, IntegerData):
                    out_values = np.round(out_values).astype("int32")

                elif isinstance(entity, TextData) and not isinstance(values[0], bytes):
                    out_values = [val.encode() for val in values]

                if getattr(entity, "ndv", None) is not None:
                    out_values[np.isnan(out_values)] = entity.ndv

                entity_handle.create_dataset(
                    name_map,
                    data=out_values,
                    compression="gzip",
                    compression_opts=9,
                    **kwargs,
                )

    @classmethod
    def clear_stats_cache(
        cls,
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

    @classmethod
    def write_entity(
        cls,
        file: str | h5py.File,
        entity,
    ) -> h5py.Group:
        """
        Add an :obj:`~geoh5py.shared.entity.Entity` and its attributes to geoh5.
        The function returns a pointer to the entity if already present on file.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.

        :return entity: Pointer to the written entity. Active link if "close_file" is False.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            base = list(h5file)[0]

            if isinstance(entity, Data):
                entity_type = "Data"
            elif isinstance(entity, ObjectBase):
                entity_type = "Objects"
            else:
                entity_type = "Groups"

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

            cls.write_properties(h5file, entity)
            entity.on_file = True

            if isinstance(entity, RootGroup):
                if "Root" in h5file[base]:
                    del h5file[base]["Root"]

                h5file[base]["Root"] = entity_handle

        return entity_handle

    @classmethod
    def write_entity_type(
        cls,
        file: str | h5py.File,
        entity_type: shared.EntityType,
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

            if isinstance(entity_type, DataType):
                entity_type_str = "Data types"
            elif isinstance(entity_type, ObjectType):
                entity_type_str = "Object types"
            elif isinstance(entity_type, GroupType):
                entity_type_str = "Group types"
            else:
                return None

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

            if hasattr(entity_type, "value_map"):
                H5Writer.write_value_map(h5file, entity_type)

            entity_type.on_file = True

        return new_type

    @classmethod
    def write_file_name_data(
        cls, entity_handle: h5py.Group, entity: FilenameData, values: bytes
    ) -> None:
        """
        Write a dataset for the file name and file blob.

        :param entity_handle: Pointer to the geoh5 Group.
        :param entity: Target :obj:`~geoh5py.data.filename_data.FilenameData` entity.
        :param values: Bytes data
        """
        if entity.file_name is None:
            raise AttributeError("FilenameData requires the 'file_name' to be set.")

        entity_handle.create_dataset(
            "Data",
            data=entity.file_name,
            dtype=h5py.special_dtype(vlen=str),
            shape=(1,),
        )

        if entity.file_name in entity_handle:
            del entity_handle[entity.file_name]
            entity.workspace.repack = True

        entity_handle.create_dataset(
            entity.file_name,
            data=np.asarray(np.void(values[:])),
            shape=(1,),
        )

    @classmethod
    def write_properties(
        cls,
        file: str | h5py.File,
        entity: Entity,
    ) -> None:
        """
        Add properties of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            H5Writer.update_field(h5file, entity, "attributes")

            for attribute in KEY_MAP:
                if getattr(entity, attribute, None) is not None:
                    H5Writer.update_field(h5file, entity, attribute)

    @classmethod
    def write_property_groups(
        cls,
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

            try:
                del entity_handle["PropertyGroups"]
                entity.workspace.repack = True
            except KeyError:
                pass

            if hasattr(entity, "property_groups") and isinstance(
                entity.property_groups, list
            ):
                entity_handle.create_group("PropertyGroups")
                for p_g in entity.property_groups:
                    uid = as_str_if_uuid(p_g.uid)
                    if uid in entity_handle["PropertyGroups"]:
                        del entity_handle["PropertyGroups"][uid]
                        entity.workspace.repack = True

                    entity_handle["PropertyGroups"].create_group(uid)

                    group_handle = entity_handle["PropertyGroups"][uid]

                    for key, attr in p_g.attribute_map.items():
                        try:
                            value = getattr(p_g, attr)
                        except AttributeError:
                            continue

                        if key == "Association":
                            value = value.name.capitalize()

                        elif key == "Properties":
                            value = np.asarray([as_str_if_uuid(val) for val in value])

                        elif key == "ID":
                            value = as_str_if_uuid(value)

                        group_handle.attrs.create(
                            key, value, dtype=h5py.special_dtype(vlen=str)
                        )

    @classmethod
    def write_to_parent(
        cls,
        file: str | h5py.File,
        entity: Entity,
        recursively=False,
    ) -> None:
        """
        Add/create an :obj:`~geoh5py.shared.entity.Entity` and add it to its parent.

        :param file: Name or handle to a geoh5 file.
        :param entity: Entity to be added or linked to a parent in geoh5.
        :param recursively: Add parents recursively until reaching the
            :obj:`~geoh5py.groups.root_group.RootGroup`.
        """
        with fetch_h5_handle(file, mode="r+") as h5file:
            if isinstance(entity, RootGroup):
                return

            uid = entity.uid
            entity_handle = H5Writer.write_entity(h5file, entity)
            parent_handle = H5Writer.write_entity(h5file, entity.parent)

            if isinstance(entity, Data):
                entity_type = "Data"
            elif isinstance(entity, ObjectBase):
                entity_type = "Objects"
            elif isinstance(entity, Group):
                entity_type = "Groups"
            else:
                return

            # Check if child h5py.Group already exists
            if entity_type not in parent_handle:
                parent_handle.create_group(entity_type)

            # Check if child uuid not already in h5
            if as_str_if_uuid(uid) not in parent_handle[entity_type]:
                parent_handle[entity_type][as_str_if_uuid(uid)] = entity_handle

            if recursively:
                H5Writer.write_to_parent(h5file, entity.parent, recursively=True)
