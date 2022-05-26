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
# pylint: disable=C0302

from __future__ import annotations

import inspect
import uuid
import warnings
import weakref
from contextlib import AbstractContextManager, contextmanager
from gc import collect
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast
from weakref import ReferenceType

import h5py
import numpy as np

from .. import data, groups, objects
from ..data import CommentsData, Data, DataType
from ..groups import CustomGroup, Group, PropertyGroup, RootGroup
from ..io import H5Reader, H5Writer
from ..objects import ObjectBase
from ..shared import weakref_utils
from ..shared.concatenator import Concatenator
from ..shared.entity import Entity
from ..shared.exceptions import Geoh5FileClosedError

if TYPE_CHECKING:
    from ..groups import group
    from ..objects import object_base
    from ..shared import EntityType


class Workspace(AbstractContextManager):
    """
    The Workspace class manages all Entities created or imported from the *geoh5* structure.

    The basic requirements needed to create a Workspace are:

    :param geoh5: File name of the target *geoh5* file.
        A new project is created if the target file cannot by found on disk.
    """

    _active_ref: ClassVar[ReferenceType[Workspace]] = type(None)  # type: ignore

    _attribute_map = {
        "Contributors": "contributors",
        "Distance unit": "distance_unit",
        "GA Version": "ga_version",
        "Version": "version",
    }

    def __init__(self, h5file: str | Path = "Analyst.geoh5", mode="a", **kwargs):

        self._contributors = np.asarray(
            ["UserName"], dtype=h5py.special_dtype(vlen=str)
        )
        self._root: Entity | None = None
        self._mode = mode
        self._distance_unit = "meter"
        self._ga_version = "1"
        self._version = 2.0
        self._name = "GEOSCIENCE"
        self._types: dict[uuid.UUID, ReferenceType[EntityType]] = {}
        self._groups: dict[uuid.UUID, ReferenceType[group.Group]] = {}
        self._objects: dict[uuid.UUID, ReferenceType[object_base.ObjectBase]] = {}
        self._data: dict[uuid.UUID, ReferenceType[data.Data]] = {}
        self._geoh5: h5py.File | None = None
        self.h5file: str | Path = h5file

        for attr, item in kwargs.items():
            if attr in self._attribute_map:
                attr = self._attribute_map[attr]

            if getattr(self, attr, None) is None:
                warnings.warn(
                    f"Argument {attr} with value {item} is not a valid attribute of workspace. "
                    f"Argument ignored.",
                    UserWarning,
                )
            else:
                setattr(self, attr, item)

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
        return [cast("data.Data", v()) for v in self._data.values()]

    def _all_groups(self) -> list[groups.Group]:
        """Get all active Group entities registered in the workspace."""
        self.remove_none_referents(self._groups, "Groups")
        return [cast("group.Group", v()) for v in self._groups.values()]

    def _all_objects(self) -> list[objects.ObjectBase]:
        """Get all active Object entities registered in the workspace."""
        self.remove_none_referents(self._objects, "Objects")
        return [cast("object_base.ObjectBase", v()) for v in self._objects.values()]

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
        if self._geoh5 is None:
            return

        if self.geoh5.mode in ["r+", "a"]:
            self._io_call(
                H5Writer.save_entity, self.root, add_children=False, mode="r+"
            )
        self.geoh5.close()
        self._geoh5 = None

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
        self, entity, parent, copy_children: bool = True, omit_list: tuple = ()
    ):
        """
        Copy an entity to a different parent with copies of children.

        :param entity: Entity to be copied.
        :param parent: Target parent to copy the entity under.
        :param copy_children: Copy all children of the entity.
        :param omit_list: List of property names to omit on copy

        :return: The Entity registered to the workspace.
        """

        entity_kwargs: dict = {"entity": {"uid": None, "parent": None}}
        for key in entity.__dict__:
            if key not in ["_uid", "_entity_type", "_on_file"] + list(omit_list):
                if key[0] == "_":
                    key = key[1:]

                entity_kwargs["entity"][key] = getattr(entity, key)

        entity_type_kwargs: dict = {"entity_type": {}}
        for key in entity.entity_type.__dict__:
            if key not in ["_workspace", "_on_file"] + list(omit_list):
                if key[0] == "_":
                    key = key[1:]

                entity_type_kwargs["entity_type"][key] = getattr(
                    entity.entity_type, key
                )

        if not isinstance(parent, (ObjectBase, Group, Workspace)):
            raise ValueError(
                "Input 'parent' should be of type (ObjectBase, Group, Workspace)"
            )

        if isinstance(parent, Workspace):
            parent = parent.root

        # Assign the same uid if possible
        if parent.workspace.get_entity(entity.uid)[0] is None:
            entity_kwargs["entity"]["uid"] = entity.uid

        entity_kwargs["entity"]["parent"] = parent

        entity_type = type(entity)
        if isinstance(entity, Data):
            entity_type = Data

        if not copy_children and "property_groups" in entity_kwargs["entity"]:
            del entity_kwargs["entity"]["property_groups"]

        new_object = parent.workspace.create_entity(
            entity_type, **{**entity_kwargs, **entity_type_kwargs}
        )

        if copy_children:
            for child in entity.children:
                new_object.add_children(
                    [self.copy_to_parent(child, new_object, copy_children=True)]
                )

        return new_object

    def create_data(
        self,
        entity_class,
        entity_kwargs: dict,
        entity_type_kwargs: dict | DataType,
    ) -> Entity | None:
        """
        Create a new Data entity with attributes.

        :param entity_class: :obj:`~geoh5py.data.data.Data` class.
        :param entity_kwargs: Properties of the entity.
        :param entity_type_kwargs: Properties of the entity_type.

        :return: The newly created entity.
        """
        if isinstance(entity_type_kwargs, DataType):
            data_type = entity_type_kwargs
        else:
            data_type = data.data_type.DataType.find_or_create(
                self, **entity_type_kwargs
            )

        for _, member in inspect.getmembers(data):
            if (
                inspect.isclass(member)
                and issubclass(member, entity_class)
                and member is not entity_class
                and hasattr(member, "primitive_type")
                and inspect.ismethod(member.primitive_type)
                and data_type.primitive_type is member.primitive_type()
            ):
                if (
                    member is CommentsData
                    and "UserComments" not in entity_kwargs.values()
                ):
                    continue

                created_entity = member(data_type, **entity_kwargs)

                return created_entity

        return None

    def create_entity(
        self,
        entity_class,
        save_on_creation: bool = True,
        **kwargs,
    ) -> Entity | None:
        """
        Function to create and register a new entity and its entity_type.

        :param entity_class: Type of entity to be created
        :param save_on_creation: Save the entity to geoh5 immediately

        :return entity: Newly created entity registered to the workspace
        """
        entity_kwargs: dict = kwargs.get("entity", {})
        entity_type_kwargs: dict = kwargs.get("entity_type", {})

        if entity_class is not RootGroup and (
            "parent" not in entity_kwargs or entity_kwargs["parent"] is None
        ):
            entity_kwargs["parent"] = self.root

        if entity_class is Data or entity_class is None:
            created_entity = self.create_data(Data, entity_kwargs, entity_type_kwargs)
        elif entity_class is RootGroup:
            created_entity = RootGroup(
                RootGroup.find_or_create_type(self, **entity_type_kwargs),
                **entity_kwargs,
            )
        else:
            created_entity = self.create_object_or_group(
                entity_class, entity_kwargs, entity_type_kwargs
            )

        if created_entity is not None and save_on_creation:
            self.save_entity(created_entity)

        return created_entity

    def create_object_or_group(
        self, entity_class, entity_kwargs: dict, entity_type_kwargs: dict
    ) -> Entity | None:
        """
        Create an object or a group with attributes.

        :param entity_class: :obj:`~geoh5py.objects.object_base.ObjectBase` or
            :obj:`~geoh5py.groups.group.Group` class.
        :param entity_kwargs: Attributes of the entity.
        :param entity_type_kwargs: Attributes of the entity_type.

        :return: A new Object or Group.
        """
        entity_type_uid = None
        for key, val in entity_type_kwargs.items():
            if key.lower() in ["id", "uid"]:
                entity_type_uid = uuid.UUID(str(val))

        if entity_type_uid is None:
            if hasattr(entity_class, "default_type_uid"):
                entity_type_uid = entity_class.default_type_uid()
            else:
                entity_type_uid = uuid.uuid4()

        entity_class = entity_class.__bases__
        for _, member in inspect.getmembers(groups) + inspect.getmembers(objects):
            if (
                inspect.isclass(member)
                and issubclass(member, entity_class)
                and member is not entity_class
                and hasattr(member, "default_type_uid")
                and not member == CustomGroup
                and member.default_type_uid() == entity_type_uid
            ):
                entity_type = member.find_or_create_type(self, **entity_type_kwargs)
                created_entity = member(entity_type, **entity_kwargs)

                return created_entity

        # Special case for CustomGroup without uuid
        if entity_class == Group:
            entity_type = groups.custom_group.CustomGroup.find_or_create_type(
                self, **entity_type_kwargs
            )
            created_entity = groups.custom_group.CustomGroup(
                entity_type, **entity_kwargs
            )

            return created_entity

        return None

    @property
    def data(self) -> list[data.Data]:
        """Get all active Data entities registered in the workspace."""
        return self._all_data()

    def fetch_or_create_root(self):
        try:
            self._root = self.load_entity(uuid.uuid4(), "root")
            self._root.on_file = True
            self._root.entity_type.on_file = True
            self.fetch_children(self._root, recursively=True)

        except KeyError:
            self._root = self.create_entity(RootGroup, save_on_creation=False)

            for entity_type in ["group", "object"]:
                uuids = self._io_call(H5Reader.fetch_uuids, entity_type, mode="r")

                for uid in uuids:
                    recovered_object = self.load_entity(uid, entity_type)

                    if isinstance(recovered_object, Entity):
                        self.fetch_children(recovered_object, recursively=True)
                        getattr(recovered_object, "parent", None)

    def remove_children(self, parent, children: list):
        """
        Remove a list of entities from a parent.
        """
        for child in children:
            if isinstance(child, Data):
                ref_type = "Data"
                parent.remove_data_from_group(child)
            elif isinstance(child, Group):
                ref_type = "Groups"
            elif isinstance(child, ObjectBase):
                ref_type = "Objects"

            self._io_call(H5Writer.remove_child, child.uid, ref_type, parent, mode="r+")

    def remove_entity(self, entity: Entity):
        """
        Function to remove an entity and its children from the workspace
        """
        parent = entity.parent

        self.workspace.remove_recursively(entity)

        parent.children.remove(entity)

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
                self._io_call(H5Writer.remove_entity, key, rtype, mode="r+")

        for key in rem_list:
            del referents[key]

    def remove_recursively(self, entity: Entity):
        """Delete an entity and its children from the workspace and geoh5 recursively"""
        parent = entity.parent
        for child in entity.children:
            self.remove_recursively(child)

        entity.remove_children(entity.children)  # Remove link to children

        if isinstance(entity, Data):
            ref_type = "Data"
            parent.remove_data_from_group(entity)
        elif isinstance(entity, Group):
            ref_type = "Groups"
        elif isinstance(entity, ObjectBase):
            ref_type = "Objects"
        else:
            raise NotImplementedError(
                "Method 'remove_recursively only implemented for classes of type"
                f" {Data}, {Group} or {ObjectBase}"
            )

        self._io_call(
            H5Writer.remove_entity,
            entity.uid,
            ref_type,
            parent=parent,
            mode="r+",
        )

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

    def fetch_array_attribute(self, uid: uuid.UUID, key: str = "cells") -> np.ndarray:
        """
        Fetch attribute stored as structured array from the source geoh5.

        :param uid: Unique identifier of target entity.
        :param file: :obj:`h5py.File` or name of the target geoh5 file
        :param key: Field array name

        :return: Structured array.
        """
        return self._io_call(H5Reader.fetch_array_attribute, uid, key, mode="r")

    def fetch_children(
        self,
        entity: Entity | Concatenator | None,
        recursively: bool = False,
    ) -> list:
        """
        Recover and register children entities from the geoh5

        :param entity: Parental entity
        :param recursively: Recover all children down the project tree
        :param file: :obj:`h5py.File` or name of the target geoh5 file
        """
        if entity is None:
            return []

        if isinstance(entity, Group):
            entity_type = "group"
        elif isinstance(entity, ObjectBase):
            entity_type = "object"
        else:
            entity_type = "data"

        children_list = self._io_call(
            H5Reader.fetch_children, entity.uid, entity_type, mode="r"
        )

        family_tree = []
        for uid, child_type in children_list.items():
            if uid == "Attributes" and isinstance(entity, Concatenator):
                family_tree += [self.fetch_concatenated_children(entity, child_type)]
                continue

            if self.get_entity(uid)[0] is not None:
                recovered_object = self.get_entity(uid)[0]
            else:
                recovered_object = self.load_entity(uid, child_type, parent=entity)

            if recovered_object is not None:

                # Assumes the object was pulled from h5
                recovered_object.on_file = True
                recovered_object.entity_type.on_file = True
                family_tree += [recovered_object]

                if recursively:
                    family_tree += self.fetch_children(
                        recovered_object, recursively=True
                    )

                    if getattr(recovered_object, "property_groups", None) is not None:
                        family_tree += getattr(recovered_object, "property_groups")

        if getattr(entity, "property_groups", None) is not None:
            family_tree += getattr(entity, "property_groups")

        return family_tree

    def fetch_concatenated_children(self, group: Concatenator, attributes: list):
        """
        Load all concatenated children.
        :param group: Concatenator group
        :param attributes: Entities stored as list of dictionaries.
        """
        group.attributes = {uuid.UUID(elem["ID"]): elem for elem in attributes}
        prop_groups = {}
        for key in group.property_group_ids:
            if key not in group.attributes:
                continue

            attrs = group.attributes[key]
            prop_groups[attrs["ID"]] = PropertyGroup(**attrs)

            for uid in prop_groups[attrs["ID"]].properties:
                data_attrs = group.attributes[uid]
                data_attrs["association"] = prop_groups[attrs["ID"]].association

                entity_type = self.fetch_type(uuid.UUID(data_attrs["Type ID"]), "Data")
                data_entity = self.create_entity(
                    Data,
                    save_on_creation=False,
                    **{
                        "entity": data_attrs,
                        "entity_type": entity_type,
                    },
                )
                if data_entity is not None:
                    setattr(data_entity, "_concatenated", True)

        for key in group.concatenated_object_ids:
            attrs = group.attributes[key]
            attrs["parent"] = group
            attrs["concatenated"] = True
            class_type = ObjectBase
            obj = self.create_entity(
                class_type,
                save_on_creation=False,
                **{"entity": attrs, "entity_type": {"uid": attrs["Object Type ID"]}},
            )
            if obj is not None:
                setattr(obj, "_concatenated", True)
                for attr, val in attrs.items():
                    if "Property" in attr and isinstance(
                        self.get_entity(uuid.UUID(val))[0], Data
                    ):
                        data_entity = self.get_entity(uuid.UUID(val))[0]
                        setattr(data_entity, "_parent", obj)

    def fetch_concatenated_data(self, entity: Group | ObjectBase, *arguments: str):
        """Fetch data under the Concatenated Data group of an entity."""
        if isinstance(entity, Group):
            entity_type = "Group"
        else:
            raise NotImplementedError(
                "Method 'fetch_concatenated_data' currently only implemented for 'Group' entities."
            )

        return self._io_call(
            H5Reader.fetch_concatenated_data,
            entity.uid,
            entity_type,
            *arguments,
            mode="r",
        )

    def fetch_metadata(self, uid: uuid.UUID, argument="Metadata"):
        """
        Fetch the metadata of an entity from the source geoh5.


        :return:
        """
        return self._io_call(
            H5Reader.fetch_metadata,
            uid,
            argument=argument,
            entity_type="Groups"
            if isinstance(self.get_entity(uid)[0], Group)
            else "Objects",
            mode="r",
        )

    def fetch_property_groups(self, entity: Entity) -> list[PropertyGroup]:
        """
        Fetch all property_groups on an object from the source geoh5

        :param entity: Target object
        """

        group_dict = self._io_call(H5Reader.fetch_property_groups, entity.uid)

        property_groups = []
        for pg_id, attrs in group_dict.items():

            group = PropertyGroup(uid=uuid.UUID(pg_id), parent=entity)

            for attr, val in attrs.items():

                try:
                    setattr(group, group.attribute_map[attr], val)
                except AttributeError:
                    continue

            property_groups.append(group)

        return property_groups

    def fetch_type(self, uid: uuid.UUID, entity_type: str):
        """
        Fetch attributes of a specific entity type.
        :param uid: Unique identifier of the entity type.
        :param entity_type: One of 'Data', 'Object' or 'Group'
        """
        return self._io_call(H5Reader.fetch_type, uid, entity_type)

    def fetch_values(self, uid: uuid.UUID) -> float | None:
        """
        Fetch the data values from the source geoh5.

        :param uid: Unique identifier of target data object.

        :return: Array of values.
        """
        return self._io_call(H5Reader.fetch_values, uid)

    def fetch_file_object(self, uid: uuid.UUID, file_name: str) -> bytes | None:
        """
        Fetch an image from file name.

        :param uid: Unique identifier of target data object.

        :return: Array of values.
        """
        return self._io_call(H5Reader.fetch_file_object, uid, file_name)

    def finalize(self):
        """
        Deprecate method finalize

        :param file: :obj:`h5py.File` or name of the target geoh5 file
        """
        warnings.warn(
            "The 'finalize' method will be deprecated in future versions of geoh5py in"
            " favor of `workspace.close()`. "
            "Please update your code to suppress this warning."
        )
        self.close()

    def find_data(self, data_uid: uuid.UUID) -> Entity | None:
        """
        Find an existing and active Data entity.
        """
        return weakref_utils.get_clean_ref(self._data, data_uid)

    def find_entity(self, entity_uid: uuid.UUID) -> Entity | None:
        """Get all active entities registered in the workspace."""
        return (
            self.find_group(entity_uid)
            or self.find_data(entity_uid)
            or self.find_object(entity_uid)
        )

    def find_group(self, group_uid: uuid.UUID) -> group.Group | None:
        """
        Find an existing and active Group object.
        """
        return weakref_utils.get_clean_ref(self._groups, group_uid)

    def find_object(self, object_uid: uuid.UUID) -> object_base.ObjectBase | None:
        """
        Find an existing and active Object.
        """
        return weakref_utils.get_clean_ref(self._objects, object_uid)

    def find_type(
        self, type_uid: uuid.UUID, type_class: type[EntityType]
    ) -> EntityType | None:
        """
        Find an existing and active EntityType

        :param type_uid: Unique identifier of target type
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

    def get_entity(self, name: str | uuid.UUID) -> list[Entity | None]:
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

        entity_list: list[Entity | None] = []
        for uid in list_entity_uid:
            entity_list.append(self.find_entity(uid))

        return entity_list

    @property
    def groups(self) -> list[groups.Group]:
        """Get all active Group entities registered in the workspace."""
        return self._all_groups()

    @property
    def geoh5(self) -> h5py.File:
        """
        Instance of h5py.File.
        """
        if self._geoh5 is None:
            raise Geoh5FileClosedError

        return self._geoh5

    @property
    def h5file(self) -> str | Path:
        """
        :str: Target *geoh5* file name with path.
        """
        return self._h5file

    @h5file.setter
    def h5file(self, file: str | Path):

        if not isinstance(file, (str, Path)):
            raise ValueError(
                "The 'h5file' attribute must be a str or pathlib.Path to the target geoh5 file. "
                f"Provided {file} of type({type(file)})"
            )

        if not str(file).endswith("geoh5"):
            raise ValueError("Input 'h5file' file must have a 'geoh5' extension.")

        self._h5file = file

    @property
    def list_data_name(self) -> dict[uuid.UUID, str]:
        """
        :obj:`dict` of :obj:`uuid.UUID` keys and name values for all registered Data.
        """
        data_name = {}
        for key, val in self._data.items():
            entity = val.__call__()
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
        return entities_name

    @property
    def list_groups_name(self) -> dict[uuid.UUID, str]:
        """
        :obj:`dict` of :obj:`uuid.UUID` keys and name values for all registered Groups.
        """
        groups_name = {}
        for key, val in self._groups.items():
            entity = val.__call__()
            if entity is not None:
                groups_name[key] = entity.name
        return groups_name

    @property
    def list_objects_name(self) -> dict[uuid.UUID, str]:
        """
        :obj:`dict` of :obj:`uuid.UUID` keys and name values for all registered Objects.
        """
        objects_name = {}
        for key, val in self._objects.items():
            entity = val.__call__()
            if entity is not None:
                objects_name[key] = entity.name
        return objects_name

    def load_entity(
        self,
        uid: uuid.UUID,
        entity_type: str,
        parent: Entity = None,
    ) -> Entity | None:
        """
        Recover an entity from geoh5.

        :param uid: Unique identifier of entity
        :param entity_type: One of entity type 'group', 'object', 'data' or 'root'

        :return entity: Entity loaded from geoh5
        """
        base_classes = {
            "group": Group,
            "object": ObjectBase,
            "data": Data,
            "root": RootGroup,
        }
        (
            attributes,
            type_attributes,
            property_groups,
        ) = self._io_call(H5Reader.fetch_attributes, uid, entity_type, mode="r")

        if parent is not None:
            attributes["entity"]["parent"] = parent

        entity = self.create_entity(
            base_classes[entity_type],
            save_on_creation=False,
            **{**attributes, **type_attributes},
        )

        if isinstance(entity, ObjectBase) and len(property_groups) > 0:
            for kwargs in property_groups.values():
                entity.find_or_create_property_group(**kwargs)

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

    def open(self, mode: str | None = None):
        if isinstance(self._geoh5, h5py.File):
            warnings.warn(f"Workspace already opened in mode {self._geoh5.mode}.")
            return

        if mode is None:
            mode = self._mode

        try:
            self._geoh5 = h5py.File(self._h5file, mode)
        except OSError:
            mode = "r"
            self._geoh5 = h5py.File(self._h5file, mode)

        self._data = {}
        self._objects = {}
        self._groups = {}
        self._types = {}

        try:
            proj_attributes = self._io_call(H5Reader.fetch_project_attributes, mode="r")

            for key, attr in proj_attributes.items():
                setattr(self, self._attribute_map[key], attr)

        except FileNotFoundError:
            self._io_call(H5Writer.create_geoh5, self, mode="a")

        self.fetch_or_create_root()

    def _register_type(self, entity_type: EntityType):
        weakref_utils.insert_once(self._types, entity_type.uid, entity_type)

    def _register_group(self, group: group.Group):
        weakref_utils.insert_once(self._groups, group.uid, group)

    def _register_data(self, data_obj: Entity):
        weakref_utils.insert_once(self._data, data_obj.uid, data_obj)

    def _register_object(self, obj: object_base.ObjectBase):
        weakref_utils.insert_once(self._objects, obj.uid, obj)

    @property
    def root(self) -> Entity | None:
        """
        :obj:`~geoh5py.groups.root_group.RootGroup` entity.
        """
        return self._root

    def save_entity(
        self,
        entity: Entity,
        add_children: bool = True,
    ):
        """
        Save or update an entity to geoh5.

        :param entity: Entity to be written to geoh5.
        :param add_children: Add children entities to geoh5.
        :param file: :obj:`h5py.File` or name of the target geoh5
        """
        self._io_call(
            H5Writer.save_entity, entity, add_children=add_children, mode="r+"
        )

    @property
    def types(self) -> list[EntityType]:
        """Get all active entity types registered in the workspace."""
        return self._all_types()

    def update_attribute(
        self,
        entity: Entity | EntityType,
        attribute: str,
    ):
        """
        Save or update an entity to geoh5.

        :param entity: Entity to be written to geoh5.
        :param attribute: Name of the attribute to get updated to geoh5.
        """
        if entity.on_file:
            self._io_call(H5Writer.update_field, entity, attribute, mode="r+")

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
    def workspace(self):
        """
        This workspace instance itself.
        """
        return self

    def _io_call(self, fun, *args, mode="r", **kwargs):
        """
        Run a H5Writer or H5Reader function with validation of target geoh5
        """
        try:
            if mode in ["r+", "a"] and self.geoh5.mode == "r":
                raise UserWarning(
                    f"Error performing {fun}. "
                    "Attempting to write to a geoh5 file in read-only mode. "
                    "Consider closing the workspace (Geoscience ANALYST) and "
                    "re-opening in mode='r+'."
                )

            return fun(self.geoh5, *args, **kwargs)

        except Geoh5FileClosedError as error:
            raise Geoh5FileClosedError(
                f"Error executing {fun}. "
                + "Consider re-opening with `Workspace.open()' or used within a context manager."
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
