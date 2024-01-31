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

# pylint: disable=too-many-lines

from __future__ import annotations

import uuid
import warnings
from typing import TYPE_CHECKING

import numpy as np
from h5py import special_dtype

from ...data import Data, DataAssociationEnum, DataType
from ...groups import Group
from ..entity import Entity
from ..utils import INV_KEY_MAP, KEY_MAP, as_str_if_utf8_bytes, as_str_if_uuid
from .concatenated import Concatenated
from .data import ConcatenatedData
from .object import ConcatenatedObject
from .property_group import ConcatenatedPropertyGroup

if TYPE_CHECKING:
    from ...groups import GroupType


PROPERTY_KWARGS = {
    "trace": {"maxshape": (None,)},
    "trace_depth": {"maxshape": (None,)},
    "property_group_ids": {
        "dtype": special_dtype(vlen=str),
        "maxshape": (None,),
    },
    "surveys": {"maxshape": (None,)},
}


class Concatenator(Group):  # pylint: disable=too-many-public-methods
    """
    Class modifier for concatenation of objects and data.
    """

    _concatenated_attributes: dict | None = None
    _attributes_keys: list[uuid.UUID] | None = None
    _concatenated_object_ids: list[bytes] | None = None
    _concat_attr_str: str | None = None
    _data: dict
    _index: dict
    _property_group_ids: np.ndarray | None = None
    _unique_property_groups_names: dict = {}
    _property_group_by_data_name: dict = {}

    def __init__(self, group_type: GroupType, **kwargs):
        super().__init__(group_type, **kwargs)

        getattr(self, "_attribute_map").update(
            {
                self.concat_attr_str: "concatenated_attributes",
                "Property Groups IDs": "property_group_ids",
                "Concatenated object IDs": "concatenated_object_ids",
            }
        )

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

    def add_children(self, children: list[ConcatenatedObject] | list[Entity]) -> None:
        """
        :param children: Add a list of entities as
            :obj:`~geoh5py.shared.entity.Entity.children`
        """
        for child in children:
            if not (
                isinstance(child, Concatenated)
                or (
                    isinstance(child, Data)
                    and child.association
                    in (DataAssociationEnum.OBJECT, DataAssociationEnum.GROUP)
                )
            ):
                warnings.warn(
                    f"Expected a Concatenated object, not {type(child).__name__}"
                )
                continue

            if child not in self._children:
                self._children.append(child)

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
                if uid not in self._concatenated_object_ids:  # type: ignore
                    concat_object_ids = (
                        self._concatenated_object_ids + concat_object_ids  # type: ignore
                    )
                else:
                    concat_object_ids = self._concatenated_object_ids  # type: ignore

            self.concatenated_object_ids = concat_object_ids
            self.update_array_attribute(child, "surveys")
            self.update_array_attribute(child, "trace")

        child.on_file = True

    @property
    def concat_attr_str(self) -> str:
        """String identifier for the concatenated attributes."""
        if self._concat_attr_str is None:
            self._concat_attr_str = "Attributes"
            if self.workspace.version is not None and self.workspace.version > 2.0:
                self._concat_attr_str = "Attributes Jsons"
        return self._concat_attr_str

    @property
    def concatenated_attributes(self) -> dict | None:
        """Dictionary of concatenated objects and data attributes."""
        if self._concatenated_attributes is None:
            concatenated_attributes = self.workspace.fetch_concatenated_attributes(self)

            if concatenated_attributes is None:
                concatenated_attributes = {"Attributes": []}

            self._concatenated_attributes = concatenated_attributes

        return self._concatenated_attributes

    @concatenated_attributes.setter
    def concatenated_attributes(self, concatenated_attributes: dict):
        if not isinstance(concatenated_attributes, (dict, type(None))):
            raise ValueError(
                "Input 'concatenated_attributes' must be a dictionary or None"
            )

        self._concatenated_attributes = concatenated_attributes

    @property
    def concatenated_object_ids(self) -> list[bytes] | None:
        """Dictionary of concatenated objects and data concatenated_object_ids."""
        if getattr(self, "_concatenated_object_ids", None) is None:
            concatenated_object_ids = self.workspace.fetch_array_attribute(
                self, "concatenated_object_ids"
            )
            if isinstance(concatenated_object_ids, np.ndarray):
                concatenated_object_ids = concatenated_object_ids.tolist()

            self._concatenated_object_ids = concatenated_object_ids  # type: ignore

        return self._concatenated_object_ids

    @concatenated_object_ids.setter
    def concatenated_object_ids(self, object_ids: list[bytes] | None):
        if isinstance(object_ids, np.ndarray):
            object_ids = object_ids.tolist()

        if not isinstance(object_ids, (list, type(None))):
            raise AttributeError(
                "Input value for 'concatenated_object_ids' must be of type list."
            )

        self._concatenated_object_ids = object_ids
        self.workspace.update_attribute(self, "concatenated_object_ids")

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Function to copy an entity to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: Create copies of all children entities along with it.
        :param mask: Array of indices to sub-sample the input entity.
        :param clear_cache: Clear array attributes after copy.

        :return entity: Registered Entity to the workspace.
        """
        if mask is not None:
            warnings.warn("Masking is not supported for Concatenated objects.")

        new_entity: Concatenator = super().copy(  # mypy: ignore-errors
            parent=parent,
            copy_children=False,
            clear_cache=clear_cache,
            omit_list=[
                "_concatenated_object_ids",
                "_concatenated_attributes",
                "_data",
                "_index",
                "_property_group_ids",
            ],
            **kwargs,
        )

        if not copy_children or self.concatenated_attributes is None:
            return new_entity

        if (
            mask is None and new_entity.workspace != self.workspace
        ):  # Fast copy to new workspace
            new_entity.concatenated_attributes = self.concatenated_attributes
            new_entity.concatenated_object_ids = self.concatenated_object_ids

            for field in self.index:
                values = self.workspace.fetch_concatenated_values(self, field)
                if isinstance(values, tuple):
                    new_entity.data[field], new_entity.index[field] = values

                new_entity.save_attribute(field)

                # Copy over the data type
            for elem in self.concatenated_attributes["Attributes"]:
                if "Name" in elem and "Type ID" in elem:
                    attr_type = self.workspace.fetch_type(
                        uuid.UUID(elem["Type ID"]), "Data"
                    )
                    data_type = DataType.find_or_create(
                        new_entity.workspace, **attr_type
                    )
                    new_entity.workspace.save_entity_type(data_type)

            new_entity.workspace.fetch_children(new_entity)
        else:
            for child in self.children:
                child.copy(
                    parent=new_entity, clear_cache=clear_cache, omit_list=["_uid"]
                )

        return new_entity

    @property
    def data(self) -> dict:
        """
        Concatenated data values stored as a dictionary.
        """
        if getattr(self, "_data", None) is None:
            self._data, self._index = self.fetch_concatenated_data_index()

        return self._data

    @data.setter
    def data(self, data: dict):
        if not isinstance(data, dict):
            raise ValueError("Input 'data' must be a dictionary")

        self._data = data

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

    def fetch_concatenated_data_index(self):
        """Extract concatenation arrays."""
        data, index = {}, {}
        data_list = self.workspace.fetch_concatenated_list(self, "Index")

        if data_list is not None:
            for field in data_list:
                name = field.replace("\u2044", "/")
                values = self.workspace.fetch_concatenated_values(self, field)
                if isinstance(values, tuple):
                    data[name], index[name] = values

        return data, index

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
                for attr, val in self.get_concatenated_attributes(key).items()
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

        uid = as_str_if_uuid(entity.uid).encode()

        if isinstance(entity, ConcatenatedData):
            ind = np.where(self.index[field]["Data ID"] == uid)[0]
            if len(ind) == 1:
                return ind[0]
        else:
            ind = np.where(self.index[field]["Object ID"] == uid)[0]
            if len(ind) == 1:
                return ind[0]

        return None

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

    def get_concatenated_attributes(self, uid: bytes | str | uuid.UUID) -> dict:
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
            if self.attributes_keys is not None:
                self.attributes_keys.append(uid)

            if self.concatenated_attributes is not None:
                self.concatenated_attributes["Attributes"].append({})

            index = -1

        return self.concatenated_attributes["Attributes"][index]

    @property
    def index(self) -> dict:
        """
        Concatenated index stored as a dictionary.
        """
        if getattr(self, "_index", None) is None:
            self._data, self._index = self.fetch_concatenated_data_index()

        return self._index

    @index.setter
    def index(self, index: dict):
        if not isinstance(index, dict):
            raise ValueError("Input 'index' must be a dictionary")

        self._index = index

    @property
    def property_group_ids(self) -> list | None:
        """Dictionary of concatenated objects and data property_group_ids."""
        if not self._property_group_ids:
            property_groups_ids = self.workspace.fetch_concatenated_values(
                self, "property_group_ids"
            )

            if property_groups_ids is not None:
                self._property_group_ids = property_groups_ids[0].tolist()

        return self._property_group_ids

    def remove_entity(self, entity: Concatenated):
        """Remove a concatenated entity."""

        if isinstance(entity, ConcatenatedData):
            # Remove the rows of data and index
            self.update_array_attribute(entity, entity.name, remove=True)
            # Remove from the concatenated Attributes
            parent_attr = self.get_concatenated_attributes(entity.parent.uid)
            name = entity.name
            del parent_attr[f"Property:{name}"]
        elif isinstance(entity, ConcatenatedObject):
            if entity.property_groups is not None:  # type: ignore
                self.update_array_attribute(entity, "property_groups", remove=True)

            object_ids = self.concatenated_object_ids

            if object_ids is not None:
                object_ids.remove(as_str_if_uuid(entity.uid).encode())
                self.concatenated_object_ids = object_ids

        if self.concatenated_attributes is not None:
            attr_handle = self.get_concatenated_attributes(entity.uid)
            self.concatenated_attributes["Attributes"].remove(attr_handle)
            self.workspace.repack = True

        entity.parent._children.remove(entity)  # pylint: disable=protected-access

    def save_attribute(self, field: str):
        """
        Save a concatenated attribute.

        :param field: Name of the attribute
        """
        field = INV_KEY_MAP.get(field, field)
        alias = KEY_MAP.get(field, field)
        self.workspace.update_attribute(self, "index", alias)

        if field in PROPERTY_KWARGS:  # For group property
            if field == "property_groups":
                field = "property_group_ids"

            self.workspace.update_attribute(
                self,
                field,
                values=self.data.get(alias),
                **PROPERTY_KWARGS.get(field, {}),
            )
        else:  # For data values
            self.workspace.update_attribute(self, "data", field)

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
        target_attributes = self.get_concatenated_attributes(entity.uid)

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

    def update_array_attribute(
        self, entity: Concatenated, field: str, remove=False
    ) -> None:
        """
        Update values stored as data.        Row data and indices are first remove then appended.

        :param entity: Concatenated entity with array values.
        :param field: Name of the valued field.
        :param remove: Remove the data from the concatenated array.
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
            field = "property_group_ids"
            values = [as_str_if_uuid(val.uid).encode() for val in values]

        alias = KEY_MAP.get(field, field)

        start = self.fetch_start_index(entity, alias)

        if values is not None and not remove:
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
                indices = np.hstack([self.index[alias], indices]).astype(
                    self.index[alias].dtype
                )

            self.index[alias] = indices

            if alias in self.data:
                values = np.hstack([self.data[alias], values])

            self.data[alias] = values

        self.save_attribute(field)

    @staticmethod
    def get_depth_association(
        property_group: ConcatenatedPropertyGroup,
    ) -> tuple[str] | tuple[str, str] | tuple[None]:
        """
        Based on a PropertyGroup, it gets the name of the depth (or from to)
            associated with the data.

        :param property_group: The property group to extract.

        return: the name of the values to used for association.
        """
        if getattr(property_group, "property_group_type", None) == "Interval table":
            return property_group.from_.name, property_group.to_.name
        if getattr(property_group, "property_group_type", None) == "Depth table":
            return (property_group.depth_.name,)

        return (None,)

    @staticmethod
    def get_properties_names(property_group: ConcatenatedPropertyGroup) -> tuple:
        """
        Get the names of the properties in the property group.

        :param property_group: The property group to extract.
        """
        if property_group.properties is None:
            return (None,)
        return tuple(
            property_group.parent.get_data(property_)[0].name
            for property_ in property_group.properties
        )

    @property
    def unique_property_group_names(self) -> dict:
        """
        Returns the unique property groups of the parent and there type.
        The structure is a dictionary with the name of the property group as key,
        and the name of the association and the names of the data as values.
        """
        if (
            not self._unique_property_groups_names
            and self.property_group_ids is not None
        ):
            unique_property_groups: dict = {}
            for property_group_uid in self.property_group_ids:
                property_group = self.workspace.get_entity(
                    uuid.UUID(property_group_uid.decode("utf-8"))
                )[0]
                if isinstance(property_group, ConcatenatedPropertyGroup):
                    # get the association and the names
                    association = self.get_depth_association(property_group)
                    data_names = self.get_properties_names(property_group)
                    data_names = tuple(
                        name for name in data_names if name not in association
                    )

                    if property_group.name in unique_property_groups:
                        unique_property_groups[property_group.name] = (
                            tuple(
                                sorted(
                                    set(
                                        unique_property_groups[property_group.name][0]
                                        + association
                                    )
                                )
                            ),
                            tuple(
                                sorted(
                                    set(
                                        unique_property_groups[property_group.name][1]
                                        + data_names
                                    )
                                )
                            ),
                        )
                    else:
                        unique_property_groups[property_group.name] = (
                            association,
                            data_names,
                        )

            self._unique_property_groups_names = unique_property_groups

        return self._unique_property_groups_names

    @property
    def property_group_by_data_name(self) -> dict:
        """
        Returns a dictionary with the data names as keys and the property group names as values.
        """
        # New dictionary to store the transformed data
        if not self._property_group_by_data_name:
            new_dict = {}

            # Iterate over the original dictionary
            for key, (
                association,
                data_names,
            ) in self.unique_property_group_names.items():
                for name in association + data_names:
                    # Assign Names (the original key) as the value for each new key
                    new_dict[name] = key

            self._property_group_by_data_name = new_dict

        return self._property_group_by_data_name

    def get_data_from_name(self, name: str) -> ConcatenatedData:
        """
        Get the data from a given name.

        :param name: The name of the data to extract.

        :return: The first data found with the given name in index.
        """

        # ensure data_name is in the data
        if name not in self.data:
            raise KeyError(f"Data '{name}' not found in concatenated data.")

        # get the data of the know association
        data: ConcatenatedData = self.workspace.get_entity(  # type: ignore
            uuid.UUID(self.index[name][0][-2].decode("utf-8").strip("{}"))
        )[0].get_data(uuid.UUID(self.index[name][0][-1].decode("utf-8").strip("{}")))[0]

        return data
