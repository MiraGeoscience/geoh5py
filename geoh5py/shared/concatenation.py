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

# pylint: disable=too-many-lines

from __future__ import annotations

import uuid
import warnings
from typing import TYPE_CHECKING

import numpy as np
from h5py import special_dtype

from geoh5py.data import Data, DataAssociationEnum, DataType
from geoh5py.groups import Group, PropertyGroup
from geoh5py.objects import ObjectBase
from geoh5py.shared.entity import Entity
from geoh5py.shared.utils import (
    INV_KEY_MAP,
    KEY_MAP,
    as_str_if_utf8_bytes,
    as_str_if_uuid,
)

if TYPE_CHECKING:
    from ..groups import GroupType

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

            self._concatenated_object_ids = concatenated_object_ids

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
        if self._property_group_ids is None:
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
            if entity.property_groups is not None:
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


class Concatenated(Entity):
    """
    Base class modifier for concatenated objects and data.
    """

    _parent: Concatenated | Concatenator
    _concat_attr_str: str = "Attributes"

    def __init__(self, entity_type, **kwargs):
        attribute_map = getattr(self, "_attribute_map", {})
        attr = {"name": "Entity", "parent": None}
        for key, value in kwargs.items():
            attr[attribute_map.get(key, key)] = value

        super().__init__(entity_type, **attr)

    @property
    def concat_attr_str(self) -> str:
        """String identifier for the concatenated attributes."""
        return self._concat_attr_str

    @property
    def concatenator(self) -> Concatenator:
        """
        Parental Concatenator entity.
        """
        if isinstance(self._parent, Concatenated):
            return self._parent.concatenator

        return self._parent


class ConcatenatedData(Concatenated):
    _parent: ConcatenatedObject

    def __init__(self, entity_type, **kwargs):
        if kwargs.get("parent") is None or not isinstance(
            kwargs.get("parent"), ConcatenatedObject
        ):
            raise UserWarning(
                "Creating a concatenated data must have a parent "
                "of type Concatenated."
            )

        super().__init__(entity_type, **kwargs)

    @property
    def property_group(self):
        """Get the property group containing the data interval."""
        if self.parent.property_groups is None:
            return None

        for prop_group in self.parent.property_groups:
            if prop_group.properties is None:
                continue

            if self.uid in prop_group.properties:
                return prop_group

        return None

    @property
    def parent(self) -> ConcatenatedObject:
        return self._parent

    @parent.setter
    def parent(self, parent):
        if not isinstance(parent, ConcatenatedObject):
            raise AttributeError(
                "The 'parent' of a concatenated Data must be of type 'Concatenated'."
            )
        self._parent = parent
        self._parent.add_children([self])  # type: ignore

        parental_attr = self.concatenator.get_concatenated_attributes(self.parent.uid)

        if f"Property:{self.name}" not in parental_attr:
            parental_attr[f"Property:{self.name}"] = as_str_if_uuid(self.uid)

    @property
    def n_values(self) -> np.ndarray:
        """Number of values in the data."""

        n_values = None
        depths = getattr(self.property_group, "depth_", None)
        if depths and depths is not self:
            n_values = len(depths.values)
        intervals = getattr(self.property_group, "from_", None)
        if intervals and intervals is not self:
            n_values = len(intervals.values)

        return n_values


class ConcatenatedPropertyGroup(PropertyGroup):
    _parent: ConcatenatedObject

    def __init__(self, parent: ConcatenatedObject, **kwargs):
        if not isinstance(parent, ConcatenatedObject):
            raise UserWarning(
                "Creating a concatenated data must have a parent "
                "of type Concatenated."
            )

        super().__init__(parent, **kwargs)

    @property
    def depth_(self):
        if self.properties is None or len(self.properties) < 1:
            return None

        data = self.parent.get_data(  # pylint: disable=no-value-for-parameter
            self.properties[0]
        )[0]

        if isinstance(data, Data) and "depth" in data.name.lower():
            return data

        return None

    @property
    def from_(self):
        """Return the data entities defined the 'from' depth intervals."""
        if self.properties is None or len(self.properties) < 1:
            return None

        data = self.parent.get_data(  # pylint: disable=no-value-for-parameter
            self.properties[0]
        )[0]

        if isinstance(data, Data) and "from" in data.name.lower():
            return data

        return None

    @property
    def to_(self):
        """Return the data entities defined the 'to' depth intervals."""
        if self.properties is None or len(self.properties) < 2:
            return None

        data = self.parent.get_data(  # pylint: disable=no-value-for-parameter
            self.properties[1]
        )[0]

        if isinstance(data, Data) and "to" in data.name.lower():
            return data

        return None

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent):
        if self._parent is not None:
            raise AttributeError("Cannot change parent of a property group.")

        if not isinstance(parent, ConcatenatedObject):
            raise AttributeError(
                "The 'parent' of a concatenated Data must be of type 'Concatenated'."
            )
        parent.add_children([self])
        self._parent = parent
        parent.workspace.add_or_update_property_group(self)


class ConcatenatedObject(Concatenated, ObjectBase):
    _parent: Concatenator
    _property_groups: list | None = None

    def __init__(self, entity_type, **kwargs):
        if kwargs.get("parent") is None or not isinstance(
            kwargs.get("parent"), Concatenator
        ):
            raise UserWarning(
                "Creating a concatenated object must have a parent "
                "of type Concatenator."
            )

        super().__init__(entity_type, **kwargs)

    def create_property_group(
        self, name=None, on_file=False, **kwargs
    ) -> ConcatenatedPropertyGroup:
        """
        Create a new :obj:`~geoh5py.groups.property_group.PropertyGroup`.

        :param kwargs: Any arguments taken by the
            :obj:`~geoh5py.groups.property_group.PropertyGroup` class.

        :return: A new :obj:`~geoh5py.groups.property_group.PropertyGroup`
        """
        if self._property_groups is not None and name in [
            pg.name for pg in self._property_groups
        ]:
            raise KeyError(f"A Property Group with name '{name}' already exists.")

        if "property_group_type" not in kwargs and "Property Group Type" not in kwargs:
            kwargs["property_group_type"] = "Interval table"

        prop_group = ConcatenatedPropertyGroup(
            self, name=name, on_file=on_file, **kwargs
        )

        return prop_group

    def get_data(self, name: str | uuid.UUID) -> list[Data]:
        """
        Generic function to get data values from object.
        """
        entity_list = []
        attr = self.concatenator.get_concatenated_attributes(
            getattr(self, "uid")
        ).copy()

        for key, value in attr.items():
            if "Property:" in key:
                child_data = self.workspace.get_entity(uuid.UUID(value))[0]
                if child_data is None:
                    attributes: dict = self.concatenator.get_concatenated_attributes(
                        value
                    ).copy()
                    attributes["parent"] = self
                    self.workspace.create_from_concatenation(attributes)
                elif not isinstance(child_data, PropertyGroup):
                    self.add_children([child_data])
                else:
                    warnings.warn(f"Failed: '{name}' is a property group, not a Data.")

        for child in getattr(self, "children"):
            if (
                isinstance(name, str) and hasattr(child, "name") and child.name == name
            ) or (
                isinstance(name, uuid.UUID)
                and hasattr(child, "uid")
                and child.uid == name
            ):
                entity_list.append(child)

        return entity_list

    def get_data_list(self, attribute="name"):
        """
        Get list of data names.
        """
        data_list = [
            attr.replace("Property:", "").replace("\u2044", "/")
            for attr in self.concatenator.get_concatenated_attributes(self.uid)
            if "Property:" in attr
        ]

        return data_list

    @property
    def parent(self) -> Concatenator:
        return self._parent

    @parent.setter
    def parent(self, parent):
        if not isinstance(parent, Concatenator):
            raise AttributeError(
                "The 'parent' of a concatenated Object must be of type "
                "'Concatenator'."
            )
        self._parent = parent
        self._parent.add_children([self])

    @property
    def property_groups(self) -> list | None:
        if self._property_groups is None:
            property_groups = self.concatenator.fetch_values(self, "property_group_ids")

            if property_groups is None or isinstance(self, Data):
                property_groups = []

            for key in property_groups:
                self.find_or_create_property_group(
                    **self.concatenator.get_concatenated_attributes(key), on_file=True
                )

            property_groups = [
                child
                for child in self.children
                if isinstance(child, ConcatenatedPropertyGroup)
            ]

            if len(property_groups) > 0:
                self._property_groups = property_groups

        return self._property_groups


class ConcatenatedDrillhole(ConcatenatedObject):
    @property
    def depth_(self) -> list[Data]:
        obj_list = []
        for prop_group in (
            self.property_groups if self.property_groups is not None else []
        ):
            properties = [] if prop_group.properties is None else prop_group.properties
            data = [self.get_data(child)[0] for child in properties]
            if data and "depth" in data[0].name.lower():
                obj_list.append(data[0])

        return obj_list

    @property
    def from_(self) -> list[Data]:
        """
        Depth data corresponding to the tops of the interval values.
        """
        obj_list = []
        for prop_group in (
            self.property_groups if self.property_groups is not None else []
        ):
            properties = [] if prop_group.properties is None else prop_group.properties
            data = [self.get_data(child)[0] for child in properties]
            if len(data) > 0 and "from" in data[0].name.lower():
                obj_list.append(data[0])
        return obj_list

    @property
    def to_(self) -> list[Data]:
        """
        Depth data corresponding to the bottoms of the interval values.
        """
        obj_list = []
        for prop_group in (
            self.property_groups if self.property_groups is not None else []
        ):
            data = [self.get_data(child)[0] for child in prop_group.properties]
            if len(data) > 1 and "to" in data[1].name.lower():
                obj_list.append(data[1])
        return obj_list

    def validate_data(
        self, attributes: dict, property_group=None, collocation_distance=None
    ) -> tuple:
        """
        Validate input drillhole data attributes.

        :param attributes: Dictionary of data attributes.
        :param property_group: Input property group to validate against.
        """
        if collocation_distance is None:
            collocation_distance = attributes.get(
                "collocation_distance", getattr(self, "default_collocation_distance")
            )
        if collocation_distance < 0:
            raise UserWarning("Input depth 'collocation_distance' must be >0.")

        if (
            "depth" not in attributes
            and "from-to" not in attributes
            and "association" not in attributes
        ):
            if property_group is None:
                raise AttributeError(
                    "Input data dictionary must contain {key:values} "
                    + "{'from-to':numpy.ndarray} "
                    + "or {'association': 'OBJECT'}."
                )
            attributes["from-to"] = None

        if "depth" in attributes.keys():
            values = attributes.get("values")
            attributes["association"] = "DEPTH"
            property_group = self.validate_depth_data(
                attributes.get("name"),
                attributes.get("depth"),
                values,
                property_group=property_group,
                collocation_distance=collocation_distance,
            )

            if (
                isinstance(values, np.ndarray)
                and values.shape[0] < property_group.depth_.values.shape[0]
            ):
                attributes["values"] = np.pad(
                    values,
                    (0, property_group.depth_.values.shape[0] - len(values)),
                    constant_values=np.nan,
                )

            del attributes["depth"]

        if "from-to" in attributes.keys():
            values = attributes.get("values")
            attributes["association"] = "DEPTH"
            property_group = self.validate_interval_data(
                attributes.get("name"),
                attributes.get("from-to"),
                attributes.get("values"),
                property_group=property_group,
                collocation_distance=collocation_distance,
            )
            if (
                isinstance(values, np.ndarray)
                and values.shape[0] < property_group.from_.values.shape[0]
            ):
                attributes["values"] = np.pad(
                    values,
                    (0, property_group.from_.values.shape[0] - len(values)),
                    constant_values=np.nan,
                )

            del attributes["from-to"]

        return attributes, property_group

    def validate_depth_data(
        self,
        name: str | None,
        depth: list | np.ndarray | None,
        values: np.ndarray,
        property_group: str | ConcatenatedPropertyGroup | None = None,
        collocation_distance: float | None = None,
    ) -> ConcatenatedPropertyGroup:
        """
        :param name: Data name.
        :param depth: Sampling depths.
        :param values: Data samples to depths.
        :param property_group: Group for possibly collocated data.
        :param collocation_distance: Tolerance to determine collocated data for
            property group assignment

        :return: Augmented property group with name/values added for collocated data
            otherwise newly created property group with name/depth/values added.
        """
        if depth is not None:
            if isinstance(depth, list):
                depth = np.vstack(depth)

            if len(depth) < len(values):
                msg = f"Mismatch between input 'depth' shape{depth.shape} "
                msg += f"and 'values' shape{values.shape}"
                raise ValueError(msg)

        if (
            depth is not None
            and property_group is None
            and self.property_groups is not None
        ):
            for group in self.property_groups:
                if (
                    group.depth_ is not None
                    and group.depth_.values.shape[0] == depth.shape[0]
                    and np.allclose(
                        group.depth_.values, depth, atol=collocation_distance
                    )
                ):
                    return group

        ind = 0
        label = ""
        if len(self.depth_) > 0:
            ind = len(self.depth_)
            label = f"({ind})"

        if property_group is None:
            property_group = f"depth_{ind}"

        if isinstance(property_group, str):
            out_group: ConcatenatedPropertyGroup = (
                self.find_or_create_property_group(  # type: ignore
                    name=property_group,
                    association="DEPTH",
                    property_group_type="Depth table",
                )
            )
        else:
            out_group = property_group

        if out_group.depth_ is not None:
            if out_group.depth_.values.shape[0] != values.shape[0]:
                raise ValueError(
                    f"Input values for '{name}' with shape({values.shape[0]}) "
                    f"do not match the depths of the group '{out_group}' "
                    f"with shape({out_group.depth_.values.shape[0]}). Check values or "
                    "assign to a new property group"
                )
            return out_group

        depth = getattr(self, "add_data")(
            {
                f"DEPTH{label}": {
                    "association": "DEPTH",
                    "values": depth,
                    "entity_type": {"primitive_type": "FLOAT"},
                    "parent": self,
                    "allow_move": False,
                    "allow_delete": False,
                },
            },
            out_group,
        )

        return out_group

    def validate_interval_data(
        self,
        name: str | None,
        from_to: list | np.ndarray | None,
        values: np.ndarray,
        property_group: str | ConcatenatedPropertyGroup | None = None,
        collocation_distance=1e-4,
    ) -> ConcatenatedPropertyGroup:
        """
        Compare new and current depth values and re-use the property group if possible.
        Otherwise a new property group is added.

        :param from_to: Array of from-to values.
        :param values: Data values to be added on the from-to intervals.
        :param property_group: Property group name
        :collocation_distance: Threshold on the comparison between existing depth values.
        """
        if from_to is not None:
            if isinstance(from_to, list):
                from_to = np.vstack(from_to)
                if from_to.shape[0] == 2:
                    from_to = from_to.T

            assert from_to.shape[0] >= len(values), (
                f"Mismatch between input 'from_to' shape{from_to.shape} "
                + f"and 'values' shape{values.shape}"
            )
            assert from_to.shape[1] == 2, "The `from-to` values must have shape(*, 2)"

        if (
            from_to is not None
            and property_group is None
            and self.property_groups is not None
        ):
            for p_g in self.property_groups:
                if (
                    p_g.from_ is not None
                    and p_g.from_.values.shape[0] == from_to.shape[0]
                    and np.allclose(
                        np.c_[p_g.from_.values, p_g.to_.values],
                        from_to,
                        atol=collocation_distance,
                    )
                ):
                    return p_g

        ind = 0
        label = ""
        if len(self.from_) > 0:
            ind = len(
                list(set(self.from_))
            )  # todo: from_ return the same value x time why?
            label = f"({ind})"

        if property_group is None:
            property_group = f"Interval_{ind}"

        if isinstance(property_group, str):
            out_group: ConcatenatedPropertyGroup = getattr(
                self, "find_or_create_property_group"
            )(name=property_group, association="DEPTH")
        else:
            out_group = property_group

        if out_group.from_ is not None:
            if out_group.from_.values.shape[0] != values.shape[0]:
                raise ValueError(
                    f"Input values for '{name}' with shape({values.shape[0]}) "
                    f"do not match the from-to intervals of the group '{out_group}' "
                    f"with shape({out_group.from_.values.shape[0]}). Check values or "
                    f"assign to a new property group."
                )
            return out_group

        from_to = getattr(self, "add_data")(
            {
                f"FROM{label}": {
                    "association": "DEPTH",
                    "values": from_to[:, 0],
                    "entity_type": {"primitive_type": "FLOAT"},
                    "parent": self,
                    "allow_move": False,
                    "allow_delete": False,
                },
                f"TO{label}": {
                    "association": "DEPTH",
                    "values": from_to[:, 1],
                    "entity_type": {"primitive_type": "FLOAT"},
                    "parent": self,
                    "allow_move": False,
                    "allow_delete": False,
                },
            },
            out_group,
        )

        return out_group

    def sort_depths(self):
        """Bypass sort_depths from previous version."""
