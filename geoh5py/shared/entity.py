#  Copyright (c) 2021 Mira Geoscience Ltd.
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

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import shared


class Entity(ABC):
    """
    Base Entity class
    """

    _attribute_map = {
        "Allow delete": "allow_delete",
        "Allow move": "allow_move",
        "Allow rename": "allow_rename",
        "ID": "uid",
        "Name": "name",
        "Public": "public",
        "Visible": "visible",
    }

    def __init__(self, **kwargs):

        self._uid: uuid.UUID = uuid.uuid4()
        self._name = "Entity"
        self._parent = None
        self._children: list = []
        self._visible = True
        self._allow_delete = True
        self._allow_move = True
        self._allow_rename = True
        self._public = True
        self._existing_h5_entity = False
        self._metadata = None
        self._modified_attributes: list[str] = []

        if "parent" in kwargs.keys():
            setattr(self, "parent", kwargs["parent"])

        for attr, item in kwargs.items():
            try:
                if attr in self._attribute_map.keys():
                    attr = self._attribute_map[attr]
                setattr(self, attr, item)
            except AttributeError:
                continue
        self.modified_attributes = []

    def add_children(self, children: list[shared.Entity]):
        """
        :param children: Add a list of entities as
            :obj:`~geoh5py.shared.entity.Entity.children`
        """
        for child in children:
            if child not in self._children:
                self._children.append(child)

    @property
    def allow_delete(self) -> bool:
        """
        :obj:`bool` Entity can be deleted from the workspace.
        """
        return self._allow_delete

    @allow_delete.setter
    def allow_delete(self, value: bool):
        self._allow_delete = value
        self.modified_attributes = "attributes"

    @property
    def allow_move(self) -> bool:
        """
        :obj:`bool` Entity can change :obj:`~geoh5py.shared.entity.Entity.parent`
        """
        return self._allow_move

    @allow_move.setter
    def allow_move(self, value: bool):
        self._allow_move = value
        self.modified_attributes = "attributes"

    @property
    def allow_rename(self) -> bool:
        """
        :obj:`bool` Entity can change name
        """
        return self._allow_rename

    @allow_rename.setter
    def allow_rename(self, value: bool):
        self._allow_rename = value
        self.modified_attributes = "attributes"

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

    def copy(self, parent=None, copy_children: bool = True):
        """
        Function to copy an entity to a different parent entity.

        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: Create copies of all children entities along with it.

        :return entity: Registered Entity to the workspace.
        """

        if parent is None:
            parent = self.parent

        new_entity = parent.workspace.copy_to_parent(
            self, parent, copy_children=copy_children
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
        if "entity_type_uid" in kwargs.keys():
            entity_type_kwargs = {"entity_type": {"uid": kwargs["entity_type_uid"]}}
        else:
            entity_type_kwargs = {}

        entity_kwargs = {"entity": kwargs}
        new_object = workspace.create_entity(
            cls, **{**entity_kwargs, **entity_type_kwargs}
        )

        # Add to root if parent is not set
        if new_object.parent is None:
            new_object.parent = workspace.root

        workspace.finalize()

        return new_object

    @property
    @abstractmethod
    def entity_type(self) -> shared.EntityType:
        ...

    @property
    def existing_h5_entity(self) -> bool:
        """
        :obj:`bool` Entity already present in
        :obj:`~geoh5py.workspace.workspace.Workspace.h5file`.
        """
        return self._existing_h5_entity

    @existing_h5_entity.setter
    def existing_h5_entity(self, value: bool):
        self._existing_h5_entity = value

    @classmethod
    def fix_up_name(cls, name: str) -> str:
        """If the given  name is not a valid one, transforms it to make it valid
        :return: a valid name built from the given name. It simply returns the given name
        if it was already valid.
        """
        # TODO: implement an actual fixup
        #  (possibly it has to be abstract with different implementations per Entity type)
        return name

    @property
    def metadata(self) -> str | dict | None:
        """
        Metadata attached to the entity.
        """
        if getattr(self, "_metadata", None) is None:
            self._metadata = self.workspace.fetch_metadata(self.uid)

        return self._metadata

    @metadata.setter
    def metadata(self, value: dict | str | None):
        if value is not None:
            assert isinstance(
                value, (dict, str)
            ), f"Input metadata must be of type {dict}, {str} or None"
        self._metadata = value
        self.modified_attributes = "metadata"

    @property
    def modified_attributes(self):
        """
        :obj:`list[str]` List of attributes to be updated in associated workspace
        :obj:`~geoh5py.workspace.workspace.Workspace.h5file`.
        """
        return self._modified_attributes

    @modified_attributes.setter
    def modified_attributes(self, values: list | str):
        if self.existing_h5_entity:
            if not isinstance(values, list):
                values = [values]

            # Check if re-setting the list or appending
            if len(values) == 0:
                self._modified_attributes = []
            else:
                for value in values:
                    if value not in self._modified_attributes:
                        self._modified_attributes.append(value)

    @property
    def name(self) -> str:
        """
        :obj:`str` Name of the entity
        """
        return self._name

    @name.setter
    def name(self, new_name: str):
        self._name = self.fix_up_name(new_name)
        self.modified_attributes = "attributes"

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, parent: shared.Entity | uuid.UUID):

        if parent is not None:
            if isinstance(parent, uuid.UUID):
                uid = parent
            else:
                uid = parent.uid

            # Remove as child of previous parent
            if self.parent is not None:
                self._parent.remove_children([self])

            self._parent = self.workspace.get_entity(uid)[0]
            self._parent.add_children([self])

    @parent.getter
    def parent(self):
        """
        Parental :obj:`~geoh5py.shared.entity.Entity` in the workspace tree. The
        workspace :obj:`~geoh5py.groups.root_group.RootGroup` is used by default.
        """
        if self._parent is None:
            self._parent = self.workspace.root
            self._parent.add_children([self])

        return self._parent

    @property
    def public(self) -> bool:
        """
        :obj:`bool` Entity is accessible in the workspace tree and other parts
            of the the user interface in ANALYST.
        """
        return self._public

    @public.setter
    def public(self, value: bool):
        self._public = value
        self.modified_attributes = "attributes"

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
                if obj.uid in children_uid
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

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @uid.setter
    def uid(self, uid: str | uuid.UUID):
        if isinstance(uid, str):
            uid = uuid.UUID(uid)

        self._uid = uid

    @uid.getter
    def uid(self):
        """
        :obj:`uuid.UUID` The unique identifier of an entity, either as stored
        in geoh5 or generated in :func:`~uuid.UUID.uuid4` format.
        """
        if self._uid is None:
            self._uid = uuid.uuid4()

        return self._uid

    @property
    def visible(self) -> bool:
        """
        :obj:`bool` Entity visible in camera (checked in ANALYST object tree).
        """
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = value
        self.modified_attributes = "attributes"

    @property
    def workspace(self):
        """
        :obj:`~geoh5py.workspace.workspace.Workspace` to which the Entity belongs to.
        """
        return self.entity_type.workspace
