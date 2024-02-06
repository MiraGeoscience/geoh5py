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

# pylint: disable=R0904

from __future__ import annotations

import uuid
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from geoh5py.shared.utils import str2uuid

if TYPE_CHECKING:
    from numpy import ndarray

    from .. import shared
    from ..groups import PropertyGroup
    from ..workspace import Workspace

DEFAULT_CRS = {"Code": "Unknown", "Name": "Unknown"}


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
        if not Path(file).is_file():
            raise ValueError(f"Input file '{file}' does not exist.")

        with open(file, "rb") as raw_binary:
            blob = raw_binary.read()

        name = Path(file).name
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

    @abstractmethod
    def mask_by_extent(
        self, extent: np.ndarray, inverse: bool = False
    ) -> np.ndarray | None:
        """
        Get a mask array from coordinate extent.

        :param extent: Bounding box extent coordinates defined by either:
            - obj:`numpy.ndarray` of shape (2, 3)
            3D coordinate: [[west, south, bottom], [east, north, top]]
            - obj:`numpy.ndarray` of shape (2, 2)
            Horizontal coordinates: [[west, south], [east, north]].
        :param inverse: Return the complement of the mask extent. Default to False

        :return: Array of bool defining the vertices or cell centers
            within the mask extent, or None if no intersection.
        """

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
    def coordinate_reference_system(self) -> dict:
        """
        Coordinate reference system attached to the entity.
        """
        coordinate_reference_system = DEFAULT_CRS

        if self.metadata is not None and "Coordinate Reference System" in self.metadata:
            coordinate_reference_system = self.metadata[
                "Coordinate Reference System"
            ].get("Current", DEFAULT_CRS)

        return coordinate_reference_system

    @coordinate_reference_system.setter
    def coordinate_reference_system(self, value: dict):
        # assert value is a dictionary containing "Code" and "Name" keys
        if not isinstance(value, dict):
            raise TypeError("Input coordinate reference system must be a dictionary")

        if value.keys() != {"Code", "Name"}:
            raise KeyError(
                "Input coordinate reference system must only contain a 'Code' and 'Name' keys"
            )

        # get the actual coordinate reference system
        coordinate_reference_system = {
            "Current": value,
            "Previous": self.coordinate_reference_system,
        }

        # update the metadata
        metadata = self.metadata
        if isinstance(metadata, dict):
            metadata["Coordinate Reference System"] = coordinate_reference_system
        else:
            metadata = {"Coordinate Reference System": coordinate_reference_system}

        self.metadata = metadata

    @abstractmethod
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
        :param copy_children: (Optional) Create copies of all children entities along with it.
        :param clear_cache: Clear array attributes after copy to minimize the
            memory footprint of the workspace.
        :param mask: Array of indices to sub-sample the input entity.
        :param kwargs: Additional keyword arguments to pass to the copy constructor.

        :return entity: Registered Entity to the workspace.
        """

    def copy_from_extent(
        self,
        extent: ndarray,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        inverse: bool = False,
        **kwargs,
    ) -> Entity | None:
        """
        Function to copy an entity to a different parent entity.

        :param extent: Bounding box extent requested for the input entity, as supplied for
            :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        :param parent: Target parent to copy the entity under. Copied to current
            :obj:`~geoh5py.shared.entity.Entity.parent` if None.
        :param copy_children: (Optional) Create copies of all children entities along with it.
        :param clear_cache: Clear array attributes after copy.
        :param inverse: Keep the inverse (clip) of the extent selection.
        :param kwargs: Additional keyword arguments to pass to the copy constructor.

        :return entity: Registered Entity to the workspace.
        """
        indices = self.mask_by_extent(extent, inverse=inverse)
        if indices is None:
            return None

        return self.copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=indices,
            **kwargs,
        )

    @property
    @abstractmethod
    def entity_type(self) -> shared.EntityType:
        """Abstract property to get the entity type of the entity."""

    @classmethod
    def fix_up_name(cls, name: str) -> str:
        """If the given  name is not a valid one, transforms it to make it valid
        :return: a valid name built from the given name. It simply returns the given name
        if it was already valid.
        """
        # TODO: implement an actual fixup
        #  (possibly it has to be abstract with different implementations per Entity type)
        return name

    def get_entity(self, name: str | uuid.UUID) -> list[Entity | None]:
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

        if not entity_list:
            return [None]

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
            parent.add_children([self])
            self._parent = parent

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

    def reference_to_uid(
        self, value: Entity | PropertyGroup | str | uuid.UUID
    ) -> list[uuid.UUID]:
        """
        General entity reference translation.

        :param value: Either an `Entity`, string or uuid

        :return: List of unique identifier associated with the input reference.
        """
        children_uid = [child.uid for child in self.children]
        if hasattr(value, "uid"):
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

    def remove_children(self, children: list[shared.Entity] | list[PropertyGroup]):
        """
        Remove children from the list of children entities.

        :param children: List of entities

        .. warning::
            Removing a child entity without re-assigning it to a different
            parent may cause it to become inactive. Inactive entities are removed
            from the workspace by
            :func:`~geoh5py.shared.weakref_utils.remove_none_referents`.
        """
        if not isinstance(children, list):
            children = [children]

        self._children = [child for child in self._children if child not in children]
        self.workspace.remove_children(self, children)

    def save(self, add_children: bool = True):
        """
        Alias method of :func:`~geoh5py.workspace.Workspace.save_entity`.
        WILL BE DEPRECATED AS ENTITIES ARE ALWAYS AUTOMATICALLY UPDATED.
        :param add_children: Option to also save the children.
        """
        warnings.warn(
            "Entity.save() is deprecated and will be removed in next versions.",
            DeprecationWarning,
        )
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
