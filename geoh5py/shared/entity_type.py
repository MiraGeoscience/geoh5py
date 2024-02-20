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

from __future__ import annotations

import uuid
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from ..workspace import Workspace

EntityTypeT = TypeVar("EntityTypeT", bound="EntityType")


class EntityType(ABC):
    """
    The base class for all entity types.

    :param workspace: The workspace to which the entity type belongs.
    :param uid: The unique identifier of the entity type.
    :param description: The description of the entity type.
    :param name: The name of the entity type.
    :param kwargs: Additional keyword arguments to set as attributes.
    """

    # todo: This mechanic feels quite hacky
    _attribute_map = {"Description": "description", "ID": "uid", "Name": "name"}

    def __init__(
        self,
        workspace: Workspace,
        uid: uuid.UUID | None = None,
        description: str | None = "Entity",
        name: str | None = "Entity",
        **kwargs,
    ):

        self._workspace: Workspace = self._set_workspace(workspace)
        self.uid = self._modify_attribute("ID", uid, **kwargs)
        self.description = self._modify_attribute("Description", description, **kwargs)
        self.name = self._modify_attribute("Name", name, **kwargs)
        self._on_file = False

        self.workspace.register(self)

    @classmethod
    def _modify_attribute(cls, attribute_name: str, attribute: Any, **kwargs):
        """
        Modify the attribute based on the name and kwargs.

        :param attribute_name: The name of the attribute to modify.
        :param attribute: The attribute to modify.
        :param kwargs: The kwargs to modify the attribute.

        :return: The modified attribute.
        """
        if cls._attribute_map.get(attribute_name, None):
            if attribute_name in kwargs:
                return kwargs[attribute_name]
        return attribute

    def _set_workspace(self, workspace: Workspace) -> Workspace:
        """
        Set the workspace for the entity type.
        It is private as workspace should not be changed

        :param workspace: The workspace to set.
        """
        if not hasattr(workspace, "create_entity"):
            raise TypeError(f"Workspace must be a Workspace, not {type(workspace)}")

        workspace_ = weakref.ref(workspace)()

        if workspace_ is None:
            raise ValueError("Workspace is not available.")

        return workspace_

    @property
    def attribute_map(self) -> dict[str, str]:
        """
        Correspondence map between property names used in geoh5py and
        geoh5.
        """
        return self._attribute_map

    def copy(self, **kwargs) -> EntityType:
        """
        Copy this entity type to another workspace.
        """

        attributes = {
            prop: getattr(self, prop)
            for prop in dir(self)
            if isinstance(getattr(self.__class__, prop, None), property)
            and getattr(self, prop) is not None
        }

        attributes.update(kwargs)

        if attributes.get("uid") in getattr(
            attributes.get("workspace", self.workspace), "_types"
        ):
            del attributes["uid"]

        return self.__class__(**attributes)

    @property
    def description(self) -> str | None:
        """
        The description of the entity type.
        """
        return self._description

    @description.setter
    def description(self, description: str | None):
        if not isinstance(description, (str | type(None))):
            raise TypeError(
                f"Description must be a string or None, find {type(description)}"
            )

        self._description = description

        if hasattr(self, "_on_file"):
            self.workspace.update_attribute(self, "attributes")

    @classmethod
    def find(
        cls: type[EntityTypeT], workspace: Workspace, type_uid: uuid.UUID
    ) -> EntityTypeT | None:
        """
        Finds in the given Workspace the EntityType with the given UUID for
        this specific EntityType implementation class.

        :return: EntityType of None
        """
        return cast(EntityTypeT, workspace.find_type(type_uid, cls))

    @classmethod
    @abstractmethod
    def find_or_create(cls, workspace: Workspace, **kwargs) -> EntityType:
        """
        Find or creates an EntityType with given UUID that matches the given
        Entity implementation class.

        :param workspace: An active Workspace class

        :return: EntityType
        """

    @property
    def name(self) -> str | None:
        """
        The name of the entity type.
        """
        return self._name

    @name.setter
    def name(self, name: str | None):
        if not isinstance(name, (str | type(None))):
            raise TypeError(f"name must be a string or None, not {type(name)}")

        self._name = name

        if hasattr(self, "_on_file"):
            self.workspace.update_attribute(self, "attributes")

    @property
    def on_file(self) -> bool:
        """
        Return True if Entity already present in
        the workspace.
        """
        return self._on_file

    @on_file.setter
    def on_file(self, value: bool):
        if not isinstance(value, bool) and value != 1 and value != 0:
            raise TypeError(f"on_file must be a bool, not {type(value)}")
        self._on_file = bool(value)

    @property
    def uid(self) -> uuid.UUID:
        """
        The unique identifier of an entity, either as stored
        in geoh5 or generated in :func:`~uuid.UUID.uuid4` format.
        """
        return self._uid

    @uid.setter
    def uid(self, uid: str | uuid.UUID | None):
        if uid is None:
            uid = uuid.uuid4()
        if isinstance(uid, str):
            uid = uuid.UUID(uid)
        if not isinstance(uid, uuid.UUID):
            raise TypeError(f"uid must be a string or uuid.UUID, not {type(uid)}")

        self._uid = uid

        if hasattr(self, "_on_file"):
            self.workspace.update_attribute(self, "attributes")

    @property
    def workspace(self) -> Workspace:
        """
        The Workspace associated to the object.
        """
        if not hasattr(self._workspace, "create_entity"):
            raise AssertionError("Cannot access the workspace, ensure it is open.")

        return self._workspace
