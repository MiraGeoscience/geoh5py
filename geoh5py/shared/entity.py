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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..shared.utils import map_attributes, str2uuid, str_json_to_dict

if TYPE_CHECKING:
    from .. import shared
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
    _default_name: str = "Entity"

    def __init__(self, uid: uuid.UUID | None = None, **kwargs):
        self._uid = (
            str2uuid(uid) if isinstance(str2uuid(uid), uuid.UUID) else uuid.uuid4()
        )

        self._allow_delete = True
        self._allow_move = True
        self._allow_rename = True
        self._clipping_ids: list[uuid.UUID] | None = None
        self._metadata: dict | None = None
        self._name = self._default_name
        self._on_file = False
        self._parent: Entity | None = None
        self._partially_hidden = False
        self._public = True

        map_attributes(self, **kwargs)

        self.workspace.register(self)

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
    def clipping_ids(self) -> list[uuid.UUID] | None:
        """
        List of clipping uuids
        """
        return self._clipping_ids

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
        self.metadata = {"Coordinate Reference System": coordinate_reference_system}

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

    @property
    def metadata(self) -> dict | None:
        """
        Metadata attached to the entity.
        To update the metadata, use the setter method.
        To remove the metadata, set it to None.
        """
        if getattr(self, "_metadata", None) is None:
            value = self.workspace.fetch_metadata(self.uid)
            self._metadata = self.validate_metadata(value)

        return self._metadata

    @metadata.setter
    def metadata(self, value: dict | np.ndarray | bytes | None):
        self._metadata = self.validate_metadata(value)

        if self.on_file:
            self.workspace.update_attribute(self, "metadata")

    @staticmethod
    def validate_metadata(value):
        if isinstance(value, np.ndarray):
            value = value[0]

        if isinstance(value, bytes):
            value = str_json_to_dict(value)

        if not isinstance(value, (dict, type(None))):  # remove the metadata
            raise TypeError(
                "Input metadata must be of type dict or None. "
                f"Provided value of type '{type(value)}'."
            )

        return value

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

        if hasattr(parent, "add_children") and hasattr(parent, "remove_children"):
            parent.add_children([self])
            self._parent = parent

            if (
                current_parent is not None
                and current_parent != self._parent
                and hasattr(current_parent, "remove_children")
            ):
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
