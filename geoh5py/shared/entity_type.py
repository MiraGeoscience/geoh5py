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
from abc import ABC
from typing import TYPE_CHECKING, Any, TypeVar, cast
from warnings import warn

from ..shared.utils import ensure_uuid


if TYPE_CHECKING:
    from ..workspace import Workspace
    from .entity import Entity

EntityTypeT = TypeVar("EntityTypeT", bound="EntityType")


class EntityType(ABC):
    # pylint: disable=too-many-arguments
    """
    The base class for all entity types.

    :param workspace: The workspace to associate the entity type with.
    :param uid: The unique identifier of the entity type.
    :param description: The description of the entity type.
    :param name: The name of the entity type.
    :param on_file: Return True if the entity is on file.
    """

    _attribute_map = {"Description": "description", "ID": "uid", "Name": "name"}

    def __init__(
        self,
        workspace: Workspace,
        *,
        uid: uuid.UUID | None = None,
        description: str | None = "Entity",
        name: str = "Entity",
        on_file: bool = False,
        **_,
    ):
        self._uid: uuid.UUID = ensure_uuid(uid) if uid is not None else uuid.uuid4()

        self.description = description
        self.name = name
        self.workspace = workspace
        self.on_file = on_file

    @property
    def attribute_map(self) -> dict[str, str]:
        """
        Correspondence map between property names used in geoh5py and
        geoh5.
        """
        return self._attribute_map

    @classmethod
    def convert_kwargs(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Convert the kwargs to the geoh5py attribute names.

        :param kwargs: The kwargs to convert.

        :return: The converted kwargs.
        """
        return {
            cls._attribute_map.get(key, key): value for key, value in kwargs.items()
        }

    def copy(self, **kwargs):
        """
        Copy this entity type to another workspace.
        """

        attributes = {
            prop: getattr(self, prop)
            for prop in dir(self)
            if isinstance(getattr(self.__class__, prop, None), property)
            and getattr(self, prop) is not None
            and prop != "attribute_map"
        }

        attributes.update(kwargs)

        if attributes.get("uid") in attributes.get("workspace", self.workspace)._types:  # pylint: disable=protected-access
            del attributes["uid"]

        return self.__class__(**attributes)

    @classmethod
    def create_custom(cls, workspace: Workspace, **kwargs):
        """
        WILL BE  DEPRECATED IN 10.0.0

        Creates a new instance of GroupType for an unlisted custom Group type with a
        new auto-generated UUID.
        """
        warn("This method will be deprecated in 10.0.0. Use the class constructor")
        return cls(workspace, **kwargs)

    @property
    def description(self) -> str | None:
        """
        The description of the entity type.
        """
        return self._description

    @description.setter
    def description(self, description: str | None):
        if not isinstance(description, (str, type(None))):
            raise TypeError(
                f"Description must be a string or None, find {type(description)}"
            )

        self._description = description

        if self.workspace:
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
        return cast(EntityTypeT, workspace.find_type(ensure_uuid(type_uid), cls))

    @classmethod
    def find_or_create(
        cls,
        workspace: Workspace,
        uid: uuid.UUID | None = None,
        entity_class: type[Entity] | None = None,
        **kwargs,
    ):
        """
        Find or creates an EntityType with given uid that matches the given
        Group implementation class.

        It is expected to have a single instance of EntityType in the Workspace
        for each concrete Entity class.

        To find an object, the kwargs must contain an existing 'uid' keyword,
        or a 'entity_class' keyword, containing an object class.

        :param workspace: An active Workspace class
        :param uid: The unique identifier of the entity type.
        :param entity_class: The class of the entity.
        :param kwargs: The attributes of the entity type.

        :return: EntityType
        """
        kwargs = cls.convert_kwargs(kwargs)
        uid = kwargs.pop("uid", uid)

        if entity_class is not None and uid is None:
            uid = entity_class.default_type_uid()

        if uid is not None:
            entity_type = cls.find(workspace, ensure_uuid(uid))
            if entity_type is not None:
                return entity_type

        return cls(workspace, uid=uid, **kwargs)

    @property
    def name(self) -> str:
        """
        The name of the entity type.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, not {type(name)}")

        self._name = name

        if self.workspace:
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

    @property
    def workspace(self) -> Workspace:
        """
        The Workspace associated to the object.
        """
        if not hasattr(self, "_workspace"):
            return None  # type: ignore

        _workspace = self._workspace()
        if _workspace is None:
            raise AssertionError("Cannot access the workspace, ensure it is open.")

        return _workspace

    @workspace.setter
    def workspace(self, workspace: Workspace):
        if hasattr(self, "_workspace"):
            raise AssertionError("Cannot change the workspace of an entity type.")
        if not hasattr(workspace, "create_entity"):
            raise TypeError(f"Workspace must be a Workspace, not {type(workspace)}")

        self._workspace = weakref.ref(workspace)
        self.workspace.register(self)
