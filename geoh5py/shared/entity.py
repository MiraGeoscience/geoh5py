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

import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from geoh5py.shared.utils import str2uuid

if TYPE_CHECKING:
    from .. import shared
    from ..workspace import Workspace


class Entity(ABC):
    """
    Base Entity class
    """

    _attribute_map: dict = {
        "Allow delete": "allow_delete",
        "Allow move": "allow_move",
        "Allow rename": "allow_rename",
        "Clipping IDs": "clipping_ids: list | None",
        "ID": "uid",
        "Name": "name",
        "Partially hidden": "partially_hidden",
        "Public": "public",
        "Visible": "visible",
    }
    _visible = True

    def __init__(self, uid: uuid.UUID | None = None, name="Entity", **kwargs):
        self._uid = (
            str2uuid(uid) if isinstance(str2uuid(uid), uuid.UUID) else uuid.uuid4()
        )
        self._name = name
        self._parent: Entity | None = None
        self._children: list = []
        self._allow_delete = True
        self._allow_move = True
        self._allow_rename = True
        self._partially_hidden = False
        self._clipping_ids: list[uuid.UUID] | None = None
        self._public = True
        self._on_file = False
        self._metadata: dict | None = None

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map:
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue

    def add_children(self, children: list[shared.Entity]):
        """
        :param children: Add a list of entities as
            :obj:`~geoh5py.shared.entity.Entity.children`
        """
        for child in children:
            if child not in self._children:
                self._children.append(child)

    def add_file(self, file: str):
        """
        Add a file to the object or group stored as bytes on a FilenameData

        :param file: File name with path to import.
        """
        if not os.path.exists(file):
            raise ValueError(f"Input file '{file}' does not exist.")

        with open(file, "rb") as raw_binary:
            blob = raw_binary.read()

        _, name = os.path.split(file)
        attributes = {
            "name": name,
            "file_name": name,
            "association": "OBJECT",
            "parent": self,
            "values": blob,
        }
        entity_type = {"name": "UserFiles", "primitive_type": "FILENAME"}

        file_data = self.workspace.create_entity(
            None, entity=attributes, entity_type=entity_type
        )

        return file_data

    @property
    def allow_delete(self) -> bool:
        """
        :obj:`bool` Entity can be deleted from the workspace.
        """
        return self._allow_delete

    @allow_delete.setter
    def allow_delete(self, value: bool):
        self._allow_delete = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def allow_move(self) -> bool:
        """
        :obj:`bool` Entity can change :obj:`~geoh5py.shared.entity.Entity.parent`
        """
        return self._allow_move

    @allow_move.setter
    def allow_move(self, value: bool):
        self._allow_move = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def allow_rename(self) -> bool:
        """
        :obj:`bool` Entity can change name
        """
        return self._allow_rename

    @allow_rename.setter
    def allow_rename(self, value: bool):
        self._allow_rename = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def attribute_map(self) -> dict:
        """
        :obj:`dict` Correspondence map between property names used in geoh5py and
        geoh5.
        """
        return self._attribute_map

    @property
    def children(self):
        """
        :obj:`list` Children entities in the workspace tree
        """
        return self._children

    @property
    def clipping_ids(self) -> list[uuid.UUID] | None:
        """
        List of clipping uuids
        """
        return self._clipping_ids

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        **kwargs,
    ):
        """
        Function to copy an entity to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: (Optional) Create copies of all children entities along with it.
        :param kwargs: Additional keyword arguments to pass to the copy constructor.
        :return entity: Registered Entity to the workspace.
        """

        if parent is None:
            parent = self.parent

        new_entity = parent.workspace.copy_to_parent(
            self, parent, copy_children=copy_children, clear_cache=clear_cache, **kwargs
        )

        return new_entity

    @classmethod
    def create(cls, workspace, **kwargs):
        """
        Function to create an entity.

        :param workspace: Workspace to be added to.
        :param kwargs: List of keyword arguments defining the properties of a class.

        :return entity: Registered Entity to the workspace.
        """
        entity_type_kwargs = (
            {"entity_type": {"uid": kwargs["entity_type_uid"]}}
            if "entity_type_uid" in kwargs
            else {}
        )
        entity_kwargs = {"entity": kwargs}
        new_object = workspace.create_entity(
            cls,
            **{**entity_kwargs, **entity_type_kwargs},
        )
        return new_object

    @property
    @abstractmethod
    def entity_type(self) -> shared.EntityType:
        ...

    @classmethod
    def fix_up_name(cls, name: str) -> str:
        """If the given  name is not a valid one, transforms it to make it valid
        :return: a valid name built from the given name. It simply returns the given name
        if it was already valid.
        """
        # TODO: implement an actual fixup
        #  (possibly it has to be abstract with different implementations per Entity type)
        return name

    def get_entity(self, name: str | uuid.UUID) -> list[Entity]:
        """
        Get a child :obj:`~geoh5py.data.data.Data` by name.

        :param name: Name of the target child data
        :param entity_type: Sub-select entities based on type.
        :return: A list of children Data objects
        """

        if isinstance(name, uuid.UUID):
            entity_list = [child for child in self.children if child.uid == name]
        else:
            entity_list = [child for child in self.children if child.name == name]

        return entity_list

    def get_entity_list(self, entity_type=ABC) -> list[str]:
        """
        Get a list of names of all children :obj:`~geoh5py.data.data.Data`.

        :param entity_type: Option to sub-select based on type.
        :return: List of names of data associated with the object.
        """
        name_list = [
            child.name for child in self.children if isinstance(child, entity_type)
        ]
        return sorted(name_list)

    @property
    def metadata(self) -> dict | None:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            self._metadata = self.workspace.fetch_metadata(self.uid)

        return self._metadata

    @metadata.setter
    def metadata(self, value: dict | None):
        if value is not None:
            assert isinstance(
                value, (dict, str)
            ), f"Input metadata must be of type {dict} or None"
        self._metadata = value
        self.workspace.update_attribute(self, "metadata")

    @property
    def name(self) -> str:
        """
        :obj:`str` Name of the entity
        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = self.fix_up_name(new_name)
        self.workspace.update_attribute(self, "attributes")

    @property
    def on_file(self) -> bool:
        """
        Whether this Entity is already stored on
        :obj:`~geoh5py.workspace.workspace.Workspace.h5file`.
        """
        return self._on_file

    @on_file.setter
    def on_file(self, value: bool):
        self._on_file = value

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent: shared.Entity):
        current_parent = self._parent

        if parent is not None:
            self._parent = parent
            self._parent.add_children([self])

            if current_parent is not None and current_parent != self._parent:
                current_parent.remove_children([self])
                self.workspace.save_entity(self)

    @property
    def partially_hidden(self) -> bool:
        """
        Whether this Entity is partially hidden.
        """
        return self._partially_hidden

    @partially_hidden.setter
    def partially_hidden(self, value: bool):
        self._partially_hidden = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def public(self) -> bool:
        """
        Whether this Entity is accessible in the workspace tree and other parts
            of the the user interface in ANALYST.
        """
        return self._public

    @public.setter
    def public(self, value: bool):
        self._public = value
        self.workspace.update_attribute(self, "attributes")

    def reference_to_uid(self, value: Entity | str | uuid.UUID) -> list[uuid.UUID]:
        """
        General entity reference translation.

        :param value: Either an `Entity`, string or uuid

        :return: List of unique identifier associated with the input reference.
        """
        children_uid = [child.uid for child in self.children]
        if isinstance(value, Entity):
            uid = [value.uid]
        elif isinstance(value, str):
            uid = [
                obj.uid
                for obj in self.workspace.get_entity(value)
                if (obj is not None) and (obj.uid in children_uid)
            ]
        elif isinstance(value, uuid.UUID):
            uid = [value]
        return uid

    def remove_children(self, children: list[shared.Entity]):
        """
        Remove children from the list of children entities.

        :param children: List of entities

        .. warning::
            Removing a child entity without re-assigning it to a different
            parent may cause it to become inactive. Inactive entities are removed
            from the workspace by
            :func:`~geoh5py.shared.weakref_utils.remove_none_referents`.
        """
        self._children = [child for child in self._children if child not in children]
        self.workspace.remove_children(self, children)

    def remove_data_from_group(
        self, data: list | Entity | uuid.UUID | str, name: str | None = None
    ) -> None:
        """
        Remove data children to a :obj:`~geoh5py.groups.property_group.PropertyGroup`
        All given data must be children of the parent object.

        :param data: :obj:`~geoh5py.data.data.Data` object,
            :obj:`~geoh5py.shared.entity.Entity.uid` or
            :obj:`~geoh5py.shared.entity.Entity.name` of data.
        :param name: Name of a :obj:`~geoh5py.groups.property_group.PropertyGroup`.
            A new group is created if none exist with the given name.
        """
        if getattr(self, "property_groups", None) is not None:
            if isinstance(data, list):
                uids = []
                for datum in data:
                    uids += self.reference_to_uid(datum)
            else:
                uids = self.reference_to_uid(data)

            if name is not None:
                prop_groups = [
                    prop_group
                    for prop_group in getattr(self, "property_groups")
                    if prop_group.name == name
                ]
            else:
                prop_groups = getattr(self, "property_groups")

            for prop_group in prop_groups:
                for uid in uids:
                    if uid in prop_group.properties:
                        prop_group.properties.remove(uid)

            self.workspace.update_attribute(self, "property_groups")

    def save(self, add_children: bool = True):
        """
        Alias method of :func:`~geoh5py.workspace.Workspace.save_entity`.

        :param add_children: Option to also save the children.
        """
        return self.workspace.save_entity(self, add_children=add_children)

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @uid.setter
    def uid(self, uid: str | uuid.UUID):
        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        self._uid = uid

    @property
    def visible(self) -> bool:
        """
        Whether the Entity is visible in camera (checked in ANALYST object tree).
        """
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def workspace(self) -> Workspace:
        """
        :obj:`~geoh5py.workspace.workspace.Workspace` to which the Entity belongs to.
        """
        return self.entity_type.workspace
