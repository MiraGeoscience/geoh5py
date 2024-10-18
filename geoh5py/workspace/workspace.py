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

# pylint: disable=too-many-lines, too-many-public-methods, too-many-arguments, too-many-locals

from __future__ import annotations

import inspect
import subprocess
import tempfile
import uuid
import warnings
import weakref
from contextlib import AbstractContextManager, contextmanager
from gc import collect
from getpass import getuser
from io import BytesIO
from pathlib import Path
from shutil import copy, move
from subprocess import CalledProcessError
from typing import Any, ClassVar, cast
from weakref import ReferenceType

import h5py
import numpy as np

from .. import data, groups, objects
from ..data import CommentsData, Data, DataType, PrimitiveTypeEnum
from ..data.text_data import TextData
from ..data.visual_parameters import VisualParameters
from ..groups import (
    CustomGroup,
    DrillholeGroup,
    Group,
    IntegratorDrillholeGroup,
    PropertyGroup,
    RootGroup,
)
from ..io import H5Reader, H5Writer
from ..io.utils import str_from_subtype, str_from_type
from ..objects import Drillhole, ObjectBase
from ..shared import weakref_utils
from ..shared.concatenation import (
    Concatenated,
    ConcatenatedData,
    ConcatenatedDrillhole,
    ConcatenatedObject,
    ConcatenatedPropertyGroup,
    Concatenator,
)
from ..shared.entity import Entity
from ..shared.entity_type import EntityType
from ..shared.exceptions import Geoh5FileClosedError
from ..shared.utils import (
    as_str_if_utf8_bytes,
    clear_array_attributes,
    get_attributes,
    str2uuid,
)


# pylint: disable=too-many-instance-attributes
class Workspace(AbstractContextManager):
    """
    The Workspace class manages all Entities created or imported from the
    *geoh5* structure.

    The basic requirements needed to create a Workspace are:

    :param h5file: Path to the *geoh5* file or :obj:`oi.BytesIO` representation
        of a geoh5 structure.
    :param contributors: List of contributors to the project.
    :param distance_unit: Distance unit used in the project.
    :param ga_version: Version of the *geoh5* file format.
    :param mode: Mode in which the *geoh5* file is opened.
    :param name: Name of the project.
    :param repack: Repack the *geoh5* file after closing.
    :param version: Version of the project.
    """

    _active_ref: ClassVar[ReferenceType[Workspace]] | type(None) = type(None)  # type: ignore
    _attribute_map = {
        "Contributors": "contributors",
        "Distance unit": "distance_unit",
        "GA Version": "ga_version",
        "Version": "version",
    }

    def __init__(
        self,
        h5file: str | Path | BytesIO | None = None,
        *,
        contributors: tuple[str] = (getuser(),),
        distance_unit: str = "meter",
        ga_version: str = "1",
        mode="r+",
        name: str = "GEOSCIENCE",
        repack: bool = False,
        version: float = 2.1,
    ):
        self._root: RootGroup
        self._data: dict[uuid.UUID, ReferenceType[data.Data]] = {}
        self._distance_unit: str = distance_unit
        self._contributors: np.ndarray = np.asarray(
            contributors, dtype=h5py.special_dtype(vlen=str)
        )
        self._ga_version: str = ga_version
        self._geoh5: h5py.File | bool = False
        self._groups: dict[uuid.UUID, ReferenceType[Group]] = {}
        self._property_groups: dict[uuid.UUID, ReferenceType[PropertyGroup]] = {}
        self._h5file: str | Path | BytesIO | None = None
        self._mode: str = mode
        self._name: str = name
        self._objects: dict[uuid.UUID, ReferenceType[ObjectBase]] = {}
        self._repack: bool = repack
        self._types: dict[uuid.UUID, ReferenceType[EntityType]] = {}
        self._version: float = version

        self.h5file = h5file
        self.open()

    def activate(self):
        """Makes this workspace the active one.

        In case the workspace gets deleted, Workspace.active() safely returns None.
        """
        if Workspace._active_ref() is not self:
            Workspace._active_ref = weakref.ref(self)

    @staticmethod
    def active() -> Workspace:
        """Get the active workspace."""
        active_one = Workspace._active_ref()
        if active_one is None:
            raise RuntimeError("No active workspace.")

        # so that type check does not complain of possible returned None
        return cast(Workspace, active_one)

    def _all_data(self) -> list[data.Data]:
        """Get all active Data entities registered in the workspace."""
        self.remove_none_referents(self._data, "Data")
        return [cast("Data", v()) for v in self._data.values()]

    def _all_groups(self) -> list[groups.Group]:
        """Get all active Group entities registered in the workspace."""
        self.remove_none_referents(self._groups, "Groups")
        return [cast("Group", v()) for v in self._groups.values()]

    def _all_property_groups(self) -> list[PropertyGroup]:
        """Get all active PropertyGroup entities registered in the workspace."""
        self.remove_none_referents(self._property_groups, "PropertyGroups")
        return [cast("PropertyGroup", v()) for v in self._property_groups.values()]

    def _all_objects(self) -> list[objects.ObjectBase]:
        """Get all active Object entities registered in the workspace."""
        self.remove_none_referents(self._objects, "Objects")
        return [cast("ObjectBase", v()) for v in self._objects.values()]

    def _all_types(self) -> list[EntityType]:
        """Get all active entity types registered in the workspace."""
        self.remove_none_referents(self._types, "Types")
        return [cast("EntityType", v()) for v in self._types.values()]

    @property
    def attribute_map(self) -> dict:
        """
        Mapping between names used in the geoh5 database.
        """
        return self._attribute_map

    def close(self):
        """
        Close the file and clear properties for future open.
        """
        if not self._geoh5:
            return

        if self.geoh5.mode in ["r+", "a"]:
            for entity in self.groups:
                if isinstance(entity, Concatenator) and self.repack:
                    self.update_attribute(entity, "concatenated_attributes")

            self._io_call(H5Writer.save_entity, self.root, add_children=True, mode="r+")

        self.geoh5.close()

        if (
            self.repack
            and not isinstance(self._h5file, BytesIO)
            and self._h5file is not None
        ):
            temp_file = Path(tempfile.gettempdir()) / Path(self._h5file).name
            try:
                subprocess.run(
                    f'h5repack --native "{self._h5file}" "{temp_file}"',
                    check=True,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                )
                Path(self._h5file).unlink()
                move(temp_file, self._h5file, copy)
            except CalledProcessError:
                pass

            self.repack = False

    @property
    def contributors(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`str` List of contributors name.
        """
        return self._contributors

    @contributors.setter
    def contributors(self, value: list[str]):
        self._contributors = np.asarray(value, dtype=h5py.special_dtype(vlen=str))

    def copy_to_parent(
        self,
        entity,
        parent,
        omit_list: tuple = (),
        clear_cache: bool = False,
        **kwargs,
    ):
        """
        Copy an entity to a different parent with copies of children.

        :param entity: Entity to be copied.
        :param parent: Target parent to copy the entity under.
        :param omit_list: List of property names to omit on copy
        :param clear_cache: Clear array attributes after copy.
        :param kwargs: Additional keyword arguments passed to the copy constructor.

        :return: The Entity registered to the workspace.
        """
        entity_kwargs = get_attributes(
            entity,
            omit_list=[
                "_uid",
                "_entity_type",
                "_on_file",
                "_centroids",
                "_parts",
                "_extent",
                "_visual_parameters",
            ]
            + list(omit_list),
            attributes={"uid": None, "parent": None},
        )

        if entity_kwargs is None:
            return None

        entity_type_kwargs = get_attributes(
            entity.entity_type,
            omit_list=["_workspace", "_on_file"] + list(omit_list),
        )

        # overwrite kwargs
        entity_kwargs.update(
            (k, kwargs[k]) for k in entity_kwargs.keys() & kwargs.keys()
        )
        entity_type_kwargs.update(
            (k, kwargs[k]) for k in entity_type_kwargs.keys() & kwargs.keys()
        )

        if not isinstance(parent, (ObjectBase, Group, Workspace)):
            raise ValueError(
                "Input 'parent' should be of type (ObjectBase, Group, Workspace)"
            )

        if isinstance(parent, Workspace):
            parent = parent.root

        # Assign the same uid if possible
        if parent.workspace.get_entity(entity.uid)[0] is None:
            entity_kwargs["uid"] = entity.uid

        entity_kwargs["parent"] = parent

        entity_type = type(entity)
        if isinstance(entity, Data):
            entity_type = Data

        entity_kwargs.pop("property_groups", None)

        new_object = parent.workspace.create_entity(
            entity_type, entity=entity_kwargs, entity_type=entity_type_kwargs
        )

        if clear_cache:
            clear_array_attributes(entity)
            clear_array_attributes(new_object)

        return new_object

    @classmethod
    def copy_property_groups(
        cls, entity: ObjectBase, property_groups: list[PropertyGroup], data_map: dict
    ):
        """
        Copy property groups to a new entity.
        Keep the same uid if it's not present on the new workspace.

        :param entity: The entity associated to the property groups.
        :param property_groups: The property groups to copy.
        :param data_map: the data map to use for the property groups.
        """
        for prop_group in property_groups:
            properties = None
            if prop_group.properties is not None:
                properties = [data_map[uid] for uid in prop_group.properties]

            # prepare the kwargs
            property_group_kwargs = {
                "association": prop_group.association,
                "name": prop_group.name,
                "property_group_type": prop_group.property_group_type,
                "properties": properties,
            }

            # Assign the same uid if possible
            if entity.workspace.find_property_group(prop_group.uid) is None:
                property_group_kwargs["uid"] = prop_group.uid

            entity.fetch_property_group(**property_group_kwargs)

    @classmethod
    def create(cls, path: str | Path, **kwargs) -> Workspace:
        """Create a named blank workspace and save to disk."""
        return cls(**kwargs).save_as(path)

    def create_from_concatenation(self, attributes):
        if "Name" in attributes:
            attributes["Name"] = attributes["Name"].replace("\u2044", "/")

        recovered_entity = None

        if "Object Type ID" in attributes:
            recovered_entity = self.create_entity(
                ObjectBase,
                save_on_creation=False,
                entity=attributes,
                entity_type={"uid": attributes.pop("Object Type ID")},
            )

        elif "Type ID" in attributes:
            recovered_entity = self.create_entity(
                Data,
                save_on_creation=False,
                entity=attributes,
                entity_type=self.fetch_type(
                    uuid.UUID(attributes.pop("Type ID")), "Data"
                ),
            )

        if recovered_entity is not None:
            recovered_entity.on_file = True
            recovered_entity.entity_type.on_file = True

        return recovered_entity

    def create_data(
        self,
        entity_class,
        entity: dict,
        entity_type: dict | EntityType,
    ) -> Data:
        """
        Create a new Data entity with attributes.

        :param entity_class: :obj:`~geoh5py.data.data.Data` class.
        :param entity: Properties of the entity.
        :param entity_type: Properties of the entity_type.

        :return: The newly created entity.
        """
        if isinstance(entity_type, dict):
            entity_type = DataType.find_or_create_type(
                self,
                entity_type.pop("primitive_type"),
                parent=entity["parent"],
                **entity_type,
            )
        elif not isinstance(entity_type, DataType):
            raise TypeError(
                f"Expected `entity_type` to be of type `dict` or `DataType`, "
                f"got {type(entity_type)}."
            )

        for name, member in inspect.getmembers(data):
            if (
                inspect.isclass(member)
                and issubclass(member, entity_class)
                and member is not entity_class
                and hasattr(member, "primitive_type")
                and inspect.ismethod(member.primitive_type)
                and entity_type.primitive_type is member.primitive_type()
            ):
                if member is CommentsData and not any(
                    isinstance(val, str) and val == "UserComments"
                    for val in entity.values()
                ):
                    continue

                if self.version > 1.0 and isinstance(
                    entity["parent"], ConcatenatedObject
                ):
                    member = type("Concatenated" + name, (ConcatenatedData, member), {})

                if member is TextData and any(
                    isinstance(val, str) and "Visual Parameters" == val
                    for val in entity.values()
                ):
                    member = VisualParameters

                created_entity = member(entity_type=entity_type, **entity)

                return created_entity

        raise TypeError(
            f"Data type {entity_class} not found in {entity_type.primitive_type}."
        )

    def create_entity(
        self,
        entity_class,
        *,
        compression: int = 5,
        entity: dict | None = None,
        entity_type: EntityType | dict | None = None,
        save_on_creation: bool = True,
    ):
        """
        Function to create and register a new entity and its entity_type.

        :param entity_class: Type of entity to be created
        :param compression: Compression level for data.
        :param entity: Attributes of the entity.
        :param entity_type: Attributes of the entity_type.
        :param save_on_creation: Save the entity to geoh5 immediately

        :return entity: Newly created entity registered to the workspace
        """

        entity = entity or {}
        entity_type = entity_type or {}

        if entity_class is not RootGroup and (
            "parent" not in entity or entity["parent"] is None
        ):
            entity["parent"] = self.root

        if entity_class is None or issubclass(entity_class, Data):
            created_data = self.create_data(Data, entity, entity_type)
            if save_on_creation and self.h5file is not None:
                self.save_entity(created_data, compression=compression)
            return created_data

        created_entity = self.create_object_or_group(entity_class, entity, entity_type)
        if save_on_creation and self.h5file is not None:
            self.save_entity(created_entity, compression=compression)
        return created_entity

    def add_or_update_property_group(
        self, property_group: PropertyGroup, remove: bool = False
    ):
        """
        Add or update a property group to the workspace.
        """
        if isinstance(property_group, ConcatenatedPropertyGroup):
            parent = property_group.parent
            parent.concatenator.update_attributes(parent, "property_groups")
        else:
            self._io_call(
                H5Writer.add_or_update_property_group,
                property_group,
                remove=remove,
                mode="r+",
            )

    def create_object_or_group(
        self, entity_class, entity: dict, entity_type: dict | EntityType
    ) -> Group | ObjectBase:
        """
        Create an object or a group with attributes.

        :param entity_class: :obj:`~geoh5py.objects.object_base.ObjectBase` or
            :obj:`~geoh5py.groups.group.Group` class.
        :param entity: Attributes of the entity.
        :param entity_type: Attributes of the entity_type.

        :return: A new Object or Group.
        """
        entity_type_uid = None

        if isinstance(entity_type, EntityType):
            entity_type = get_attributes(entity_type)

        for key, val in entity_type.items():
            if key.lower() in ["id", "uid"]:
                entity_type_uid = uuid.UUID(str(val))

        if entity_type_uid is None:
            if hasattr(entity_class, "default_type_uid"):
                entity_type_uid = entity_class.default_type_uid()
            else:
                entity_type_uid = uuid.uuid4()

        for name, member in inspect.getmembers(groups) + inspect.getmembers(objects):
            if (
                inspect.isclass(member)
                and issubclass(member, entity_class.__bases__)
                and member is not entity_class.__bases__
                and not member == CustomGroup
                and member.default_type_uid() == entity_type_uid
            ):
                if self.version > 1.0:
                    if member in (DrillholeGroup, IntegratorDrillholeGroup):
                        member = type("Concatenator" + name, (Concatenator, member), {})
                    elif member is Drillhole and isinstance(
                        entity.get("parent"),
                        (DrillholeGroup, IntegratorDrillholeGroup),
                    ):
                        member = ConcatenatedDrillhole

                entity_type = member.find_or_create_type(self, **entity_type)

                created_entity = member(entity_type=entity_type, **entity)

                return created_entity

        # Special case for CustomGroup without uuid
        if entity_class == Group:
            entity_type = groups.custom.CustomGroup.find_or_create_type(
                self, **entity_type
            )
            created_entity = groups.custom.CustomGroup(
                entity_type=entity_type, **entity
            )

            return created_entity

        raise TypeError(f"Entity class type {entity_class} not recognized.")

    def create_root(
        self, entity_attributes: dict | None = None, type_attributes: dict | None = None
    ) -> RootGroup:
        """
        Create a RootGroup entity.

        :param entity_attributes: Attributes of the entity.
        :param type_attributes: Attributes of the entity_type.

        :return: The newly created RootGroup entity.
        """
        type_attributes = type_attributes or {}
        group_type = RootGroup.find_or_create_type(self, **type_attributes)

        entity_attributes = entity_attributes or {}
        root = RootGroup(entity_type=group_type, **entity_attributes)

        return root

    @property
    def data(self) -> list[data.Data]:
        """Get all active Data entities registered in the workspace."""
        return self._all_data()

    def fetch_or_create_root(self):
        """
        Fetch the root group or create a new one if it does not exist.
        """
        attrs, type_attrs, _ = self._io_call(
            H5Reader.fetch_attributes, uuid.uuid4(), "root", mode="r"
        )
        self._root = self.create_root(
            entity_attributes=attrs, type_attributes=type_attrs
        )

        if attrs is not None:
            self._root.on_file = True
            self._root.entity_type.on_file = True
            self.fetch_children(self._root, recursively=True)

            return

        # Fetch all entities and build the family tree with RootGroup at the base
        for entity_type in ["group", "object"]:
            uuids = self._io_call(H5Reader.fetch_uuids, entity_type, mode="r")

            for uid in uuids:
                if isinstance(self.get_entity(uid)[0], Entity):
                    continue

                recovered_object = self.load_entity(uid, entity_type)

                if isinstance(recovered_object, (Group, ObjectBase)):
                    self.fetch_children(recovered_object, recursively=True)

    def remove_children(self, parent, children: list):
        """
        Remove a list of entities from a parent. The target entities remain
        present on file.
        """
        for child in children:
            if isinstance(child, PropertyGroup):
                self._io_call(
                    H5Writer.add_or_update_property_group, child, remove=True, mode="r+"
                )
            else:
                ref_type = str_from_type(child)
                self._io_call(
                    H5Writer.remove_child, child.uid, ref_type, parent, mode="r+"
                )

    def remove_entity(self, entity: Entity | PropertyGroup | EntityType, force=False):
        """
        Function to remove an entity and its children from the workspace.
        """
        if not getattr(entity, "allow_delete", True) and not force:
            raise UserWarning(
                f"The 'allow_delete' property of entity {entity} prevents it from "
                "being removed. Please revise."
            )

        if isinstance(entity, (Concatenated, ConcatenatedPropertyGroup)):
            entity.concatenator.remove_entity(entity)
            return

        if isinstance(entity, Entity | PropertyGroup):
            self.workspace.remove_recursively(entity)
            entity.parent.remove_children([entity])

        if not isinstance(entity, PropertyGroup):
            ref_type = str_from_type(entity)
            self._io_call(
                H5Writer.remove_entity,
                entity.uid,
                ref_type,
                mode="r+",
            )

        del entity
        collect()
        self.remove_none_referents(self._types, "Types")

    def remove_none_referents(
        self,
        referents: dict[uuid.UUID, ReferenceType],
        rtype: str,
    ):
        """
        Search and remove deleted entities
        """
        rem_list: list = []
        for key, value in referents.items():
            if value() is None:
                rem_list += [key]
                self._io_call(
                    H5Writer.remove_entity, key, rtype, parent=self, mode="r+"
                )

        for key in rem_list:
            del referents[key]

    def remove_recursively(self, entity: Entity | PropertyGroup):
        """Delete an entity and its children from the workspace and geoh5 recursively"""
        if hasattr(entity, "children"):
            for child in entity.children:
                self.remove_entity(child, force=True)

    def deactivate(self):
        """Deactivate this workspace if it was the active one, else does nothing."""
        if Workspace._active_ref() is self:
            Workspace._active_ref = type(None)

    @property
    def distance_unit(self) -> str:
        """
        :obj:`str` Distance unit used in the project.
        """
        return self._distance_unit

    @distance_unit.setter
    def distance_unit(self, value: str):
        self._distance_unit = value

    def fetch_array_attribute(
        self, entity: Entity | EntityType, key: str = "cells"
    ) -> np.ndarray:
        """
        Fetch attribute stored as structured array from the source geoh5.

        :param entity: Unique identifier of target entity.
        :param key: Field array name

        :return: Structured array.
        """
        if isinstance(entity, Concatenated):
            return entity.concatenator.fetch_values(entity, key)  # type: ignore

        if isinstance(entity, EntityType):
            entity_type = str_from_subtype(entity)
        else:
            entity_type = str_from_type(entity)

        return self._io_call(
            H5Reader.fetch_array_attribute,
            entity.uid,
            entity_type,
            key,
            mode="r",
        )

    def fetch_children(
        self,
        entity: Entity | PropertyGroup | None,
        recursively: bool = False,
    ) -> list:
        """
        Recover and register children entities from the geoh5.

        :param entity: Parental entity.
        :param recursively: Recover all children down the project tree.
        :return list: List of children entities.
        """
        if entity is None or isinstance(entity, ConcatenatedData):
            return []

        entity_type = str_from_type(entity)

        if isinstance(entity, RootGroup) and not entity.on_file:
            children_list = {child.uid: "" for child in entity.children}

        else:
            children_list = self._io_call(
                H5Reader.fetch_children, entity.uid, entity_type, mode="r"
            )

            if isinstance(entity, Concatenator):
                if any(entity.children):
                    return entity.children

                cat_children = entity.fetch_concatenated_objects()
                children_list.update(
                    {
                        str2uuid(as_str_if_utf8_bytes(uid)): attr
                        for uid, attr in cat_children.items()
                    }
                )

        family_tree: list[Any] = []
        for uid, child_type in children_list.items():
            recovered_object = self.get_entity(uid)[0]
            if recovered_object is None and not isinstance(entity, PropertyGroup):
                recovered_object = self.load_entity(uid, child_type, parent=entity)

            if recovered_object is None or isinstance(recovered_object, PropertyGroup):
                continue

            recovered_object.on_file = True
            recovered_object.entity_type.on_file = True
            family_tree += [recovered_object]

            if recursively and isinstance(recovered_object, (Group, ObjectBase)):
                family_tree += self.fetch_children(recovered_object, recursively=True)
                if (
                    isinstance(recovered_object, ObjectBase)
                    and recovered_object.property_groups is not None
                ):
                    family_tree += recovered_object.property_groups

        if isinstance(entity, ObjectBase) and entity.property_groups is not None:
            family_tree += entity.property_groups

        return family_tree

    def fetch_concatenated_attributes(
        self, entity: Concatenator | ConcatenatedObject
    ) -> dict | None:
        """
        Fetch attributes of Concatenated entity.
        :param entity: Concatenator group or ConcatenateObject.
        :return: Dictionary of attributes.
        """
        if isinstance(entity, Group):
            entity_type = "Group"
        else:
            raise NotImplementedError(
                "Method 'fetch_concatenated_attributes' currently only implemented "
                "for 'Group' entities."
            )

        return self._io_call(
            H5Reader.fetch_concatenated_attributes,
            entity.uid,
            entity_type,
            entity.concat_attr_str,
            mode="r",
        )

    def fetch_concatenated_list(
        self, entity: Group | ObjectBase, label: str
    ) -> list | None:
        """
        Fetch list of data or indices of ConcatenatedData entities.
        :param entity: Concatenator group.
        :param label: Label name of the h5py.Group.
        :return: List of concatenated Data names.
        """
        if isinstance(entity, Group):
            entity_type = "Group"
        else:
            raise NotImplementedError(
                "Method 'fetch_concatenated_list' currently only implemented "
                "for 'Group' entities."
            )

        return self._io_call(
            H5Reader.fetch_concatenated_attributes,
            entity.uid,
            entity_type,
            label,
            mode="r",
        )

    def fetch_concatenated_values(
        self, entity: Group | ObjectBase, label: str
    ) -> tuple | None:
        """
        Fetch data under the ConcatenatedData Data group of an entity.

        :param entity: Concatenator group.
        :param label: Name of the target data.

        :return: Index array and data values for the target label.
        """
        if isinstance(entity, Group):
            entity_type = "Group"
        else:
            raise NotImplementedError(
                "Method 'fetch_concatenated_values' currently only implemented "
                "for 'Group' entities."
            )

        return self._io_call(
            H5Reader.fetch_concatenated_values,
            entity.uid,
            entity_type,
            label,
            mode="r",
        )

    def fetch_metadata(self, entity: Entity, argument="Metadata") -> dict | None:
        """
        Fetch the metadata of an entity from the source geoh5.

        :param entity: Entity uid containing the metadata.
        :param argument: Optional argument for other json-like attributes.

        :return: Dictionary of values.
        """
        entity_type = str_from_type(entity)
        return self._io_call(
            H5Reader.fetch_metadata,
            entity.uid,
            argument=argument,
            entity_type=entity_type,
            mode="r",
        )

    def fetch_property_groups(self, entity: Entity) -> list[PropertyGroup]:
        """
        Fetch all property_groups on an object from the source geoh5

        :param entity: Target object

        :return: List of PropertyGroups
        """
        raise DeprecationWarning(
            f"Method 'fetch_property_groups' of {self} as been removed. "
            "Use `entity.property_groups` instead."
        )

    def fetch_type(self, uid: uuid.UUID, entity_type: str) -> dict:
        """
        Fetch attributes of a specific entity type.
        :param uid: Unique identifier of the entity type.
        :param entity_type: One of 'Data', 'Object' or 'Group'
        """
        return self._io_call(H5Reader.fetch_type, uid, entity_type)

    def fetch_values(self, entity: Entity) -> np.ndarray | str | float | None:
        """
        Fetch the data values from the source geoh5.

        :param entity: Entity with 'values'.

        :return: Array of values.
        """
        if isinstance(entity, (ConcatenatedObject | ConcatenatedData)):
            return entity.concatenator.fetch_values(entity, entity.name)

        return self._io_call(H5Reader.fetch_values, entity.uid)

    def fetch_file_object(self, uid: uuid.UUID, file_name: str) -> bytes | None:
        """
        Fetch an image from file name.
        :param uid: Unique identifier of target data object.
        :param file_name: Name of the file to fetch.
        :return: Array of values.
        """
        return self._io_call(H5Reader.fetch_file_object, uid, file_name)

    def finalize(self) -> None:
        """
        Deprecate method finalize.
        """
        warnings.warn(
            "The 'finalize' method will be deprecated in future versions of geoh5py in"
            " favor of `workspace.close()`. "
            "Please update your code to suppress this warning.",
            DeprecationWarning,
        )
        self.close()

    def find_data(self, data_uid: uuid.UUID) -> Entity | None:
        """
        Find an existing and active Data entity.
        """
        return weakref_utils.get_clean_ref(self._data, data_uid)

    def find_entity(self, entity_uid: uuid.UUID) -> Entity | PropertyGroup | None:
        """Get all active entities registered in the workspace."""
        return (
            self.find_group(entity_uid)
            or self.find_data(entity_uid)
            or self.find_object(entity_uid)
            or self.find_property_group(entity_uid)
        )

    def find_group(self, group_uid: uuid.UUID) -> Group | None:
        """
        Find an existing and active Group object.
        """
        return weakref_utils.get_clean_ref(self._groups, group_uid)

    def find_property_group(
        self, property_group_uid: uuid.UUID
    ) -> PropertyGroup | None:
        """
        Find an existing and active PropertyGroup object.
        """
        return weakref_utils.get_clean_ref(self._property_groups, property_group_uid)

    def find_object(self, object_uid: uuid.UUID) -> ObjectBase | None:
        """
        Find an existing and active Object.
        """
        return weakref_utils.get_clean_ref(self._objects, object_uid)

    def find_type(
        self, type_uid: uuid.UUID, type_class: type[EntityType]
    ) -> EntityType | None:
        """
        Find an existing and active EntityType.
        :param type_uid: Unique identifier of target type.
        :param type_class: The type of entity to find.
        """
        found_type = weakref_utils.get_clean_ref(self._types, type_uid)
        return found_type if isinstance(found_type, type_class) else None

    @property
    def ga_version(self) -> str:
        """
        :obj:`str` Version of Geoscience Analyst software.
        """
        return self._ga_version

    @ga_version.setter
    def ga_version(self, value: str):
        self._ga_version = value

    def get_entity(self, name: str | uuid.UUID) -> list[Entity | PropertyGroup | None]:
        """
        Retrieve an entity from one of its identifier, either by name or :obj:`uuid.UUID`.
        :param name: Object identifier, either name or uuid.
        :return: List of entities with the same given name.
        """
        if isinstance(name, uuid.UUID):
            list_entity_uid = [name]

        else:  # Extract all objects uuid with matching name
            list_entity_uid = [
                key for key, val in self.list_entities_name.items() if val == name
            ]

        if not list_entity_uid:
            return [None]

        entity_list: list[Entity | None | PropertyGroup] = []
        for uid in list_entity_uid:
            entity_list.append(self.find_entity(uid))

        return entity_list

    @property
    def groups(self) -> list[groups.Group]:
        """Get all active Group entities registered in the workspace."""
        return self._all_groups()

    @property
    def property_groups(self) -> list[PropertyGroup]:
        """Get all active PropertyGroup entities registered in the workspace."""
        return self._all_property_groups()

    @property
    def geoh5(self) -> h5py.File:
        """
        Instance of h5py.File.
        """
        if not self._geoh5:
            raise Geoh5FileClosedError

        return self._geoh5

    @property
    def h5file(self) -> str | Path | BytesIO | None:
        """
        Target *geoh5* file name with path or BytesIO object representation.

        On :func:`geoh5py.workspace.Workspace.save`, the BytesIO representation
        gets replaced by a Path to a file on disk.
        """
        return self._h5file

    @h5file.setter
    def h5file(self, file: str | Path | BytesIO | None):
        if self._h5file is not None:
            raise ValueError(
                "The 'h5file' attribute cannot be changed once it has been set."
            )

        if not isinstance(file, (str, Path, BytesIO, type(None))):
            raise ValueError(
                "The 'h5file' attribute must be a str, "
                "pathlib.Path to the target geoh5 file or BytesIO. "
                f"Provided {file} of type({type(file)})"
            )

        if isinstance(file, type(None)) or (
            isinstance(file, (str, Path)) and not Path(file).is_file()
        ):
            self._h5file = BytesIO()
            self._geoh5 = h5py.File(self.h5file, "a")

            with self._geoh5:
                self._root = self.create_root()
                H5Writer.init_geoh5(self.geoh5, self)

        elif isinstance(file, BytesIO):
            self._h5file = file

        if isinstance(file, (str, Path)):
            if Path(file).suffix != ".geoh5":
                raise ValueError("Input 'h5file' file must have a 'geoh5' extension.")

            if not Path(file).is_file():
                warnings.warn(
                    "From version 0.8.0, the 'h5file' attribute must be a string "
                    "or path to an existing file, or user must call the 'create' "
                    "method. We will attempt to `save` the file for you, but this "
                    "behaviour will be removed in future releases.",
                )
                self.save_as(file)
                self.close()
            else:
                self._h5file = Path(file)

    @property
    def list_data_name(self) -> dict[uuid.UUID, str]:
        """
        :obj:`dict` of :obj:`uuid.UUID` keys and name values for all registered Data.
        """
        data_name = {}
        for key, val in self._data.items():
            entity = val()
            if entity is not None:
                data_name[key] = entity.name
        return data_name

    @property
    def list_entities_name(self) -> dict[uuid.UUID, str]:
        """
        :return: :obj:`dict` of :obj:`uuid.UUID` keys and name values for all registered Entities.
        """
        entities_name = self.list_groups_name
        entities_name.update(self.list_objects_name)
        entities_name.update(self.list_data_name)
        entities_name.update(self.list_property_groups_name)

        return entities_name

    @property
    def list_groups_name(self) -> dict[uuid.UUID, str]:
        """
        :obj:`dict` of :obj:`uuid.UUID` keys and name values for all registered Groups.
        """
        groups_name = {}
        for key, val in self._groups.items():
            entity = val()
            if entity is not None:
                groups_name[key] = entity.name
        return groups_name

    @property
    def list_property_groups_name(self) -> dict[uuid.UUID, str]:
        """
        :obj:`dict` of :obj:`uuid.UUID` keys and name values for all registered Groups.
        """
        property_groups_name = {}
        for key, val in self._property_groups.items():
            entity = val()
            if entity is not None:
                property_groups_name[key] = entity.name
        return property_groups_name

    @property
    def list_objects_name(self) -> dict[uuid.UUID, str]:
        """
        :obj:`dict` of :obj:`uuid.UUID` keys and name values for all registered Objects.
        """
        objects_name = {}
        for key, val in self._objects.items():
            entity = val()
            if entity is not None:
                objects_name[key] = entity.name
                objects_name[key] = entity.name
        return objects_name

    def load_entity(
        self,
        uid: uuid.UUID,
        entity_type: str,
        parent: Entity | None = None,
    ) -> Entity | PropertyGroup | None:
        """
        Recover an entity from geoh5.
        :param uid: Unique identifier of entity
        :param entity_type: One of entity type 'group', 'object', 'data' or 'root'
        :param parent: Parent entity.
        :return entity: Entity loaded from geoh5
        """
        child = self.get_entity(uid)[0]
        if isinstance(child, (Entity, PropertyGroup)):
            return child

        base_classes = {
            "group": Group,
            "object": ObjectBase,
            "data": Data,
            "root": RootGroup,
        }

        entity_attrs, type_attrs, prop_groups = self._io_call(
            H5Reader.fetch_attributes, uid, entity_type, mode="r"
        )

        if entity_attrs is None:
            return None

        if parent is not None:
            entity_attrs["parent"] = parent
        try:
            entity = self.create_entity(
                base_classes[entity_type],
                save_on_creation=False,
                entity=entity_attrs,
                entity_type=type_attrs,
            )
        except TypeError as error:
            warnings.warn(
                f"Could not create an entity from the given attributes {type_attrs}. Skipping over."
                f"Error: {error}"
            )
            return None

        if isinstance(entity, ObjectBase) and len(prop_groups) > 0:
            for kwargs in prop_groups.values():
                entity.create_property_group(on_file=True, **kwargs)

        if entity is not None:
            entity.on_file = True

        return entity

    @property
    def name(self) -> str:
        """
        :obj:`str` Name of the project.
        """
        return self._name

    @property
    def objects(self) -> list[objects.ObjectBase]:
        """Get all active Object entities registered in the workspace."""
        return self._all_objects()

    def open(self, mode: str | None = None) -> Workspace:
        """
        Open a geoh5 file and load the tree structure.

        :param mode: Optional mode of h5py.File. Defaults to 'r+'.
        :return: `self`
        """
        if isinstance(self._geoh5, h5py.File) and self._geoh5:
            warnings.warn(f"Workspace already opened in mode {self._geoh5.mode}.")
            return self

        if mode is None:
            mode = self._mode

        try:
            self._geoh5 = h5py.File(self.h5file, mode)
        except OSError:
            self._geoh5 = h5py.File(self.h5file, "r")

        self._data = {}
        self._objects = {}
        self._groups = {}
        self._types = {}
        self._property_groups = {}

        proj_attributes = self._io_call(H5Reader.fetch_project_attributes, mode="r")

        for key, attr in proj_attributes.items():
            setattr(self, self._attribute_map[key], attr)

        self.fetch_or_create_root()

        return self

    def register(self, entity: Entity | EntityType | PropertyGroup):
        """
        Register an entity to the workspace based on its type.

        :param entity: The entity to be registered.
        """
        if isinstance(entity, EntityType):
            weakref_utils.insert_once(self._types, entity.uid, entity)
        elif isinstance(entity, Group):
            weakref_utils.insert_once(self._groups, entity.uid, entity)
        elif isinstance(entity, Data):
            weakref_utils.insert_once(self._data, entity.uid, entity)
        elif isinstance(entity, ObjectBase):
            weakref_utils.insert_once(self._objects, entity.uid, entity)
        elif isinstance(entity, PropertyGroup):
            weakref_utils.insert_once(self._property_groups, entity.uid, entity)
            if not entity.on_file:
                self.add_or_update_property_group(entity)
        else:
            raise ValueError(f"Entity of type {type(entity)} is not supported.")

    @property
    def root(self) -> RootGroup:
        """
        :obj:`~geoh5py.groups.root_group.RootGroup` entity.
        """
        return self._root

    @property
    def repack(self) -> bool:
        """
        Flag to repack the file after data deletion.
        """
        return self._repack

    @repack.setter
    def repack(self, value: bool):
        """
        :type value: bool.
        """
        self._repack = value

    def save(self, filepath: str | Path) -> Workspace:
        warnings.warn(
            "Workspace.save is deprecated and will be remove in future versions,"
            "use Workspace.save_as instead.",
            DeprecationWarning,
        )

        return self.save_as(filepath)

    def save_as(self, filepath: str | Path) -> Workspace:
        """
        Save the workspace to disk.
        """
        if self._geoh5 is not None:
            self.close()

        filepath = Path(filepath)

        if filepath.suffix == "":
            filepath = filepath.with_suffix(".geoh5")

        if filepath.suffix != ".geoh5":
            raise ValueError("Input 'h5file' file must have a 'geoh5' extension.")

        if filepath.exists():
            raise FileExistsError(f"File {filepath} already exists.")

        if isinstance(self.h5file, BytesIO):
            with open(filepath, "wb") as file:
                file.write(self.h5file.getbuffer())
        elif self.h5file is None:
            raise ValueError("Input 'h5file' file must be specified.")
        else:
            move(self.h5file, filepath, copy)

        self._h5file = filepath

        self.open()

        return self

    def save_entity(
        self,
        entity: Entity,
        add_children: bool = True,
        compression: int = 5,
    ) -> None:
        """
        Save or update an entity to geoh5.
        :param entity: Entity to be written to geoh5.
        :param add_children: Add children entities to geoh5.
        :param compression: Compression level for data.
        """
        if isinstance(entity, Concatenated):
            active_parent = self.get_entity(entity.concatenator.uid)[0]
            if not isinstance(active_parent, Concatenator):
                raise ValueError(
                    f"DrillholeGroup {entity.concatenator.name} is not registered in the "
                    "workspace. Please add it first."
                )
            active_parent.add_save_concatenated(entity)

            if hasattr(entity, "entity_type"):
                self.save_entity_type(entity.entity_type)

        else:
            self._io_call(
                H5Writer.save_entity,
                entity,
                add_children=add_children,
                mode="r+",
                compression=compression,
            )

    def save_entity_type(self, entity_type: EntityType) -> None:
        """
        Save or update an entity_type to geoh5.

        :param entity_type: Entity to be written to geoh5.
        """
        self._io_call(H5Writer.write_entity_type, entity_type, mode="r+")

    @property
    def types(self) -> list[EntityType]:
        """Get all active entity types registered in the workspace."""
        return self._all_types()

    def update_attribute(
        self,
        entity: Entity | EntityType | DataType,
        attribute: str,
        channel: str | None = None,
        **kwargs,
    ) -> None:
        """
        Save or update an entity to geoh5.

        :param entity: Entity to be written to geoh5.
        :param attribute: Name of the attribute to get updated to geoh5.
        :param channel: Optional channel argument for concatenated data and index.
        """
        if entity.on_file:
            if isinstance(entity, (ConcatenatedObject | ConcatenatedData)):
                entity.concatenator.update_attributes(entity, attribute)
            elif channel is not None:
                self._io_call(
                    H5Writer.update_concatenated_field,
                    entity,
                    attribute,
                    channel,
                    mode="r+",
                )
            else:
                self._io_call(
                    H5Writer.update_field, entity, attribute, mode="r+", **kwargs
                )

            self._io_call(H5Writer.clear_stats_cache, entity, mode="r+")

    def validate_data_type(self, attributes: dict, values) -> DataType:
        """
        Find or create a data type from input dictionary.

        :param attributes: Dictionary of attributes.
        :param values: Values to be stored as data.
        """
        entity_type = attributes.pop("entity_type", {})
        if isinstance(entity_type, DataType):
            if (entity_type.uid not in self._types) or (entity_type.workspace != self):
                entity_type = entity_type.copy(workspace=self)
        else:
            primitive_type = attributes.pop(
                "type", attributes.pop("primitive_type", None)
            )

            if primitive_type is None:
                primitive_type = DataType.primitive_type_from_values(values)

            if isinstance(primitive_type, str):
                primitive_type = DataType.validate_primitive_type(
                    primitive_type.upper()
                )

            # Generate a value map based on type of values
            if (
                primitive_type is PrimitiveTypeEnum.REFERENCED
                and "value_map" not in attributes
            ):
                attributes["value_map"] = values

            entity_type = DataType.find_or_create_type(
                self, primitive_type, **attributes
            )

        return entity_type

    @property
    def version(self) -> float:
        """
        :obj:`float` Version of the geoh5 file format.
        """
        return self._version

    @version.setter
    def version(self, value: float):
        self._version = value

    @property
    def workspace(self) -> Workspace:
        """
        This workspace instance itself.
        """
        return self

    def _io_call(self, fun, *args, mode="r", **kwargs):
        """
        Run a H5Writer or H5Reader function with validation of target geoh5
        """
        try:
            if self._geoh5 is None:
                return None

            if mode in ["r+", "a"] and self.geoh5.mode == "r":
                raise UserWarning(
                    f"Error performing {fun}. "
                    "Attempting to write to a geoh5 file in read-only mode. "
                    "Consider closing the workspace (Geoscience ANALYST) and "
                    "re-opening in mode='r+'."
                )

            return fun(self.geoh5, *args, **kwargs)

        except Geoh5FileClosedError as error:
            if not Path(str(self.h5file)).is_file() and not isinstance(
                self.h5file, BytesIO
            ):
                raise FileNotFoundError(
                    f"Error performing {fun}. "
                    "The geoh5 file does not exist."
                    r"Consider creating the geoh5 with Workspace().save('PATH\*.geoh5)'"
                ) from error

            raise Geoh5FileClosedError(
                f"Error executing {fun}. "
                + "Consider re-opening with `Workspace.open()' "
                "or used within a context manager."
            ) from error

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@contextmanager
def active_workspace(workspace: Workspace):
    previous_active_ref = Workspace._active_ref  # pylint: disable=protected-access
    workspace.activate()
    yield workspace

    workspace.deactivate()
    # restore previous active workspace when leaving the context
    previous_active = previous_active_ref()
    if previous_active is not None:
        previous_active.activate()  # pylint: disable=no-member
