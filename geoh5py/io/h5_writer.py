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

# pylint: disable=R0904

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import TYPE_CHECKING

import h5py
import numpy as np

from ..data import CommentsData, Data, DataType, IntegerData
from ..groups import Group, GroupType, RootGroup
from ..objects import ObjectBase, ObjectType
from ..shared import Entity, EntityType, fetch_h5_handle
from .utils import as_str_if_uuid

if TYPE_CHECKING:
    from .. import shared, workspace


class H5Writer:
    """
    Writing class to a geoh5 file.
    """

    str_type = h5py.special_dtype(vlen=str)

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
        "metadata": "Metadata",
    }

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
        with fetch_h5_handle(file) as h5file:
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
    def create_dataset(cls, entity_handle, dataset: np.ndarray, label: str):
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
    ):
        """
        Remove a child from a parent.

        :param file: Name or handle to a geoh5 file
        :param uid: uuid of the target :obj:`~geoh5py.shared.entity.Entity`
        :param ref_type: Input type from: 'Types', 'Groups', 'Objects' or 'Data
        :param parent: Remove entity from parent.
        """
        with fetch_h5_handle(file) as h5file:
            uid_str = as_str_if_uuid(uid)
            parent_handle = H5Writer.fetch_handle(h5file, parent)
            if parent_handle is not None and uid_str in parent_handle[ref_type].keys():
                del parent_handle[ref_type][uid_str]

    @staticmethod
    def remove_entity(
        file: str | h5py.File,
        uid: uuid.UUID,
        ref_type: str,
        parent: Entity = None,
    ):
        """
        Remove an entity and its type from the target geoh5 file.

        :param file: Name or handle to a geoh5 file
        :param uid: uuid of the target :obj:`~geoh5py.shared.entity.Entity`
        :param ref_type: Input type from: 'Types', 'Groups', 'Objects' or 'Data
        :param parent: Remove entity from parent.

        """
        with fetch_h5_handle(file) as h5file:
            base = list(h5file.keys())[0]
            base_type_handle = h5file[base][ref_type]
            uid_str = as_str_if_uuid(uid)

            if ref_type == "Types":
                for e_type in ["Data types", "Group types", "Object types"]:
                    if uid_str in base_type_handle[e_type].keys():
                        del base_type_handle[e_type][uid_str]
            else:
                if uid_str in base_type_handle.keys():
                    del base_type_handle[uid_str]

                if parent is not None:
                    H5Writer.remove_child(h5file, uid, ref_type, parent)

    @classmethod
    def fetch_handle(
        cls,
        file: str | h5py.File,
        entity,
        return_parent: bool = False,
    ):
        """
        Get a pointer to an :obj:`~geoh5py.shared.entity.Entity` in geoh5.

        :param file: Name or handle to a geoh5 file
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`
        :param return_parent: Option to return the handle to the parent entity.

        :return entity_handle: HDF5 pointer to an existing entity, parent or None if not found.
        """
        with fetch_h5_handle(file) as h5file:
            base = list(h5file.keys())[0]
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
            if as_str_if_uuid(uid) in base_handle.keys():

                if return_parent:
                    return base_handle

                return base_handle[as_str_if_uuid(uid)]

        return None

    @classmethod
    def finalize(cls, file: str | h5py.File, workspace: workspace.Workspace):
        """
        Add/replace the :obj:`~geoh5py.groups.root_group.RootGroup` in geoh5.

        :param file: Name or handle to a geoh5 file
        :param workspace: Workspace object defining the project structure.
        """
        with fetch_h5_handle(file) as h5file:
            root_handle = cls.save_entity(h5file, workspace.root)

            if "Root" in h5file[workspace.name].keys():
                del h5file[workspace.name]["Root"]
            else:
                h5file[workspace.name].create_group = "Root"

            h5file[workspace.name]["Root"] = root_handle

    @classmethod
    def save_entity(
        cls,
        file: str | h5py.File,
        entity,
        add_children: bool = True,
    ):
        """
        Write an :obj:`~geoh5py.shared.entity.Entity` to geoh5 with its
        :obj:`~geoh5py.shared.entity.Entity.children`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        :param add_children: Add :obj:`~geoh5py.shared.entity.Entity.children`.
        """
        with fetch_h5_handle(file) as h5file:
            new_entity = H5Writer.write_entity(h5file, entity)

            if add_children:
                # Write children entities and add to current parent
                for child in entity.children:
                    H5Writer.write_entity(h5file, child)
                    H5Writer.write_to_parent(h5file, child, recursively=False)

            H5Writer.write_to_parent(h5file, entity)

        return new_entity

    @classmethod
    def update_attributes(
        cls,
        file: str | h5py.File,
        entity,
    ):
        """
        Update the attributes of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        """
        with fetch_h5_handle(file) as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            for attr in entity.modified_attributes:

                try:
                    del entity_handle[cls.key_map[attr]]
                    if getattr(entity, attr, None) is None:
                        continue
                except KeyError:
                    pass

                if attr in ["values", "trace_depth", "metadata"]:
                    cls.write_data_values(h5file, entity, attr)
                elif attr == "cells":
                    cls.write_cells(h5file, entity)
                elif attr in ["surveys", "trace", "vertices"]:
                    cls.write_coordinates(h5file, entity, attr)
                elif attr == "octree_cells":
                    cls.write_octree_cells(h5file, entity)
                elif attr == "property_groups":
                    cls.write_property_groups(h5file, entity)
                elif attr == "cell_delimiters":
                    cls.write_cell_delimiters(h5file, entity)
                elif attr == "color_map":
                    cls.write_color_map(h5file, entity)
                else:
                    cls.write_attributes(h5file, entity)

    @classmethod
    def write_attributes(
        cls,
        file: str | h5py.File,
        entity,
    ):
        """
        Write attributes of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Entity with attributes to be added to the geoh5 file.
        """
        with fetch_h5_handle(file) as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            for key, attr in entity.attribute_map.items():

                try:
                    value = getattr(entity, attr)
                except AttributeError:
                    continue

                value = as_str_if_uuid(value)

                if key == "PropertyGroups" or (key == "Metadata" and value is None):
                    continue

                if key in ["Association", "Primitive type"]:
                    value = value.name.lower().capitalize()

                if isinstance(value, (np.int8, bool)):
                    entity_handle.attrs.create(key, int(value), dtype="int8")
                elif isinstance(value, str):
                    entity_handle.attrs.create(key, value, dtype=cls.str_type)
                elif value is None:
                    entity_handle.attrs.create(key, "None", dtype=cls.str_type)
                else:
                    entity_handle.attrs.create(
                        key, value, dtype=np.asarray(value).dtype
                    )

    @classmethod
    def write_cell_delimiters(
        cls,
        file: str | h5py.File,
        entity,
    ):
        """
        Add cell delimiters (u, v, z)  to a :obj:`~geoh5py.objects.block_model.BlockModel`.

        :param file: Name or handle to a geoh5 file
        :param entity: Target entity
        """
        with fetch_h5_handle(file) as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            if hasattr(entity, "u_cell_delimiters") and (
                entity.u_cell_delimiters is not None
            ):
                cls.create_dataset(
                    entity_handle, entity.u_cell_delimiters, "U cell delimiters"
                )

            if hasattr(entity, "v_cell_delimiters") and (
                entity.v_cell_delimiters is not None
            ):
                cls.create_dataset(
                    entity_handle, entity.v_cell_delimiters, "V cell delimiters"
                )

            if hasattr(entity, "z_cell_delimiters") and (
                entity.z_cell_delimiters is not None
            ):
                cls.create_dataset(
                    entity_handle, entity.z_cell_delimiters, "Z cell delimiters"
                )

    @classmethod
    def write_cells(
        cls,
        file: str | h5py.File,
        entity,
    ):
        """
        Add :obj:`~geoh5py.objects.object_base.ObjectBase.cells`.

        :param file: Name or handle to a geoh5 file
        :param entity: Target entity
        """
        with fetch_h5_handle(file) as h5file:

            if hasattr(entity, "cells") and (entity.cells is not None):
                entity_handle = H5Writer.fetch_handle(h5file, entity)
                cls.create_dataset(entity_handle, entity.cells, "Cells")

    @classmethod
    def write_color_map(
        cls,
        file: str | h5py.File,
        entity_type: shared.EntityType,
    ):
        """
        Add :obj:`~geoh5py.data.color_map.ColorMap` to a
        :obj:`~geoh5py.data.data_type.DataType`.

        :param file: Name or handle to a geoh5 file
        :param entity_type: Target entity_type with color_map
        """
        with fetch_h5_handle(file) as h5file:
            color_map = getattr(entity_type, "color_map", None)

            if color_map is not None and color_map.values is not None:
                entity_type_handle = H5Writer.fetch_handle(h5file, entity_type)
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
    ):
        """
        Add :obj:`~geoh5py.data.reference_value_map.ReferenceValueMap` to a
        :obj:`~geoh5py.data.data_type.DataType`.

        :param file: Name or handle to a geoh5 file
        :param entity_type: Target entity_type with value_map
        """
        with fetch_h5_handle(file) as h5file:
            reference_value_map = getattr(entity_type, "value_map", None)
            names = ["Key", "Value"]
            formats = ["<u4", h5py.special_dtype(vlen=str)]

            if reference_value_map is not None and reference_value_map.map is not None:
                entity_type_handle = H5Writer.fetch_handle(h5file, entity_type)

                dtype = list(zip(names, formats))
                array = np.array(list(reference_value_map.map.items()), dtype=dtype)
                cls.create_dataset(entity_type_handle, array, "Value map")

    @classmethod
    def write_visible(
        cls,
        file: str | h5py.File,
        entity,
    ):
        """
        Needs revision once Visualization is implemented

        :param file: Name or handle to a geoh5 file
        :param entity: Target entity
        """
        with fetch_h5_handle(file) as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)
            dtype = np.dtype(
                [("ViewID", h5py.special_dtype(vlen=str)), ("Visible", "int8")]
            )

            if entity.visible:
                visible = entity_handle.create_dataset(
                    "Visible", shape=(1,), dtype=dtype
                )
                visible["Visible"] = 1

    @classmethod
    def write_coordinates(
        cls,
        file: str | h5py.File,
        entity,
        attribute,
    ):
        """
        Add :obj:`~geoh5py.objects.object_base.ObjectBase.surveys` of an object.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target entity.
        :param attribute: Name of the attribute to be written to geoh5
        """
        with fetch_h5_handle(file) as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)

            if getattr(entity, attribute, None) is not None:
                entity_handle.create_dataset(
                    cls.key_map[attribute],
                    data=getattr(entity, "_" + attribute),
                    compression="gzip",
                    compression_opts=9,
                )

            elif attribute in entity_handle.keys():
                del entity_handle[attribute]

    @classmethod
    def write_data_values(
        cls,
        file: str | h5py.File,
        entity,
        attribute,
    ):
        """
        Add data :obj:`~geoh5py.data.data.Data.values`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target entity.
        :param attribute: Name of the attribute to be written to geoh5
        """
        with fetch_h5_handle(file) as h5file:
            entity_handle = H5Writer.fetch_handle(h5file, entity)
            if getattr(entity, attribute, None) is not None:
                values = getattr(entity, attribute)

            # Adding an array of values
            if isinstance(values, dict) or isinstance(entity, CommentsData):
                values = deepcopy(values)
                if isinstance(entity, CommentsData):
                    values = {"Comments": values}

                for key, val in values.items():
                    if isinstance(val, dict):
                        for sub_key, sub_val in val.items():
                            values[key][sub_key] = as_str_if_uuid(sub_val)
                    else:
                        values[key] = as_str_if_uuid(val)

                entity_handle.create_dataset(
                    cls.key_map[attribute],
                    data=json.dumps(values, indent=4),
                    dtype=h5py.special_dtype(vlen=str),
                    shape=(1,),
                )
            elif isinstance(values, str):
                entity_handle.create_dataset(
                    cls.key_map[attribute],
                    data=values,
                    dtype=h5py.special_dtype(vlen=str),
                    shape=(1,),
                )
            else:
                out_values = deepcopy(values)
                if isinstance(entity, IntegerData):
                    out_values = np.round(out_values).astype("int32")
                else:
                    out_values[np.isnan(out_values)] = entity.ndv()

                entity_handle.create_dataset(
                    cls.key_map[attribute],
                    data=out_values,
                    compression="gzip",
                    compression_opts=9,
                )
                entity_type_handle = H5Writer.fetch_handle(h5file, entity.entity_type)
                stats_cache = entity_type_handle.get("StatsCache")
                if stats_cache is not None:
                    del entity_type_handle["StatsCache"]

    @classmethod
    def write_entity(
        cls,
        file: str | h5py.File,
        entity,
    ):
        """
        Add an :obj:`~geoh5py.shared.entity.Entity` and its attributes to geoh5.
        The function returns a pointer to the entity if already present on file.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.

        :return entity: Pointer to the written entity. Active link if "close_file" is False.
        """
        with fetch_h5_handle(file) as h5file:

            base = list(h5file.keys())[0]

            if isinstance(entity, Data):
                entity_type = "Data"
            elif isinstance(entity, ObjectBase):
                entity_type = "Objects"
            else:
                entity_type = "Groups"

            uid = entity.uid

            if entity_type not in h5file[base].keys():
                h5file[base].create_group(entity_type)

            # Check if already in the project
            if as_str_if_uuid(uid) in h5file[base][entity_type].keys():

                if any([entity.modified_attributes]):

                    if "entity_type" in entity.modified_attributes:
                        entity_handle = cls.fetch_handle(h5file, entity)
                        del entity_handle["Type"]
                        new_type = H5Writer.write_entity_type(
                            h5file, entity.entity_type
                        )
                        entity_handle["Type"] = new_type

                    cls.update_attributes(h5file, entity)
                    entity.modified_attributes = []

                entity.existing_h5_entity = True

                return h5file[base][entity_type][as_str_if_uuid(uid)]

            entity_handle = h5file[base][entity_type].create_group(as_str_if_uuid(uid))

            if entity_type == "Groups":
                entity_handle.create_group("Data")
                entity_handle.create_group("Groups")
                entity_handle.create_group("Objects")
            elif entity_type == "Objects":
                entity_handle.create_group("Data")

            H5Writer.write_attributes(h5file, entity)

            # Add the type and return a pointer
            new_type = H5Writer.write_entity_type(h5file, entity.entity_type)
            entity_handle["Type"] = new_type
            entity.entity_type.modified_attributes = []
            entity.entity_type.existing_h5_entity = True

            cls.write_properties(h5file, entity)

            entity.modified_attributes = []
            entity.existing_h5_entity = True

        return entity_handle

    @classmethod
    def write_entity_type(
        cls,
        file: str | h5py.File,
        entity_type: shared.EntityType,
    ):
        """
        Add an :obj:`~geoh5py.shared.entity_type.EntityType` to geoh5.

        :param file: Name or handle to a geoh5 file.
        :param entity_type: Entity with type to be added.

        :return type: Pointer to :obj:`~geoh5py.shared.entity_type.EntityType` in geoh5.
        """
        with fetch_h5_handle(file) as h5file:
            base = list(h5file.keys())[0]
            uid = entity_type.uid

            if isinstance(entity_type, DataType):
                entity_type_str = "Data types"
            elif isinstance(entity_type, ObjectType):
                entity_type_str = "Object types"
            elif isinstance(entity_type, GroupType):
                entity_type_str = "Group types"
            else:
                return None

            # Check if already in the project
            if entity_type_str not in h5file[base]["Types"].keys():
                h5file[base]["Types"].create_group(entity_type_str)

            if as_str_if_uuid(uid) in h5file[base]["Types"][entity_type_str].keys():

                if any([entity_type.modified_attributes]):
                    cls.update_attributes(h5file, entity_type)
                    entity_type.modified_attributes = []

                entity_type.existing_h5_entity = True

                return h5file[base]["Types"][entity_type_str][as_str_if_uuid(uid)]

            new_type = h5file[base]["Types"][entity_type_str].create_group(
                as_str_if_uuid(uid)
            )
            H5Writer.write_attributes(h5file, entity_type)

            if hasattr(entity_type, "color_map"):
                H5Writer.write_color_map(h5file, entity_type)

            if hasattr(entity_type, "value_map"):
                H5Writer.write_value_map(h5file, entity_type)

            entity_type.modified_attributes = False
            entity_type.existing_h5_entity = True

        return new_type

    @classmethod
    def write_octree_cells(
        cls,
        file: str | h5py.File,
        entity,
    ):
        """
        Add :obj:`~geoh5py.object.object_base.ObjectBase.cells` of an
        :obj:`~geoh5py.object.octree.Octree` object to geoh5.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target entity_type with color_map.
        """
        with fetch_h5_handle(file) as h5file:

            if hasattr(entity, "octree_cells") and (entity.octree_cells is not None):
                entity_handle = H5Writer.fetch_handle(h5file, entity)
                cls.create_dataset(entity_handle, entity.octree_cells, "Octree Cells")

    @classmethod
    def write_properties(
        cls,
        file: str | h5py.File,
        entity: Entity,
    ):
        """
        Add properties of an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        """
        with fetch_h5_handle(file) as h5file:

            for attribute in ["values", "trace_depth", "metadata"]:
                if getattr(entity, attribute, None) is not None:
                    H5Writer.write_data_values(h5file, entity, attribute)

            if isinstance(entity, ObjectBase) and isinstance(
                entity.property_groups, list
            ):
                H5Writer.write_property_groups(h5file, entity)

            for attribute in ["surveys", "trace", "vertices"]:
                if getattr(entity, attribute, None) is not None:
                    H5Writer.write_coordinates(h5file, entity, attribute)

            if getattr(entity, "u_cell_delimiters", None) is not None:
                H5Writer.write_cell_delimiters(h5file, entity)

            if getattr(entity, "cells", None) is not None:
                H5Writer.write_cells(h5file, entity)

            if getattr(entity, "octree_cells", None) is not None:
                H5Writer.write_octree_cells(h5file, entity)

    @classmethod
    def write_property_groups(
        cls,
        file: str | h5py.File,
        entity,
    ):
        """
        Write :obj:`~geoh5py.groups.property_group.PropertyGroup` associated with
        an :obj:`~geoh5py.shared.entity.Entity`.

        :param file: Name or handle to a geoh5 file.
        :param entity: Target :obj:`~geoh5py.shared.entity.Entity`.
        """
        with fetch_h5_handle(file) as h5file:

            if hasattr(entity, "property_groups") and isinstance(
                entity.property_groups, list
            ):

                entity_handle = H5Writer.fetch_handle(h5file, entity)

                # Check if a group already exists, then remove and write
                if "PropertyGroups" in entity_handle.keys():
                    del entity_handle["PropertyGroups"]

                entity_handle.create_group("PropertyGroups")
                for p_g in entity.property_groups:

                    uid = as_str_if_uuid(p_g.uid)
                    if uid in entity_handle["PropertyGroups"].keys():
                        del entity_handle["PropertyGroups"][uid]

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
    ):
        """
        Add/create an :obj:`~geoh5py.shared.entity.Entity` and add it to its parent.

        :param file: Name or handle to a geoh5 file.
        :param entity: Entity to be added or linked to a parent in geoh5.
        :param recursively: Add parents recursively until reaching the
            :obj:`~geoh5py.groups.root_group.RootGroup`.
        """
        with fetch_h5_handle(file) as h5file:

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
            if entity_type not in parent_handle.keys():
                parent_handle.create_group(entity_type)

            # Check if child uuid not already in h5
            if as_str_if_uuid(uid) not in parent_handle[entity_type].keys():
                parent_handle[entity_type][as_str_if_uuid(uid)] = entity_handle

            if recursively:
                H5Writer.write_to_parent(h5file, entity.parent, recursively=True)
