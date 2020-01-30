from __future__ import annotations

import uuid
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type, TypeVar, cast

if TYPE_CHECKING:
    from geoh5io import workspace as ws

TEntityType = TypeVar("TEntityType", bound="EntityType")


class EntityType(ABC):

    attribute_map = {"Description": "description", "ID": "uid", "Name": "name"}

    def __init__(
        self,
        workspace: "ws.Workspace",
        uid: uuid.UUID,
        name: str = "None",
        description: str = "None",
    ):
        assert workspace is not None
        assert uid is not None
        assert uid.int != 0

        self._workspace = weakref.ref(workspace)
        self._uid = uid
        self._name = name
        self._description = description
        self._existing_h5_entity = False
        self._update_h5 = False

        workspace._register_type(self)

    @property
    def existing_h5_entity(self) -> bool:
        return self._existing_h5_entity

    @existing_h5_entity.setter
    def existing_h5_entity(self, value: bool):
        self._existing_h5_entity = value

    @property
    def update_h5(self) -> bool:
        return self._update_h5

    @update_h5.setter
    def update_h5(self, value: bool):
        self._update_h5 = value

    @staticmethod
    @abstractmethod
    def _is_abstract() -> bool:
        """ Trick to prevent from instantiating abstract base class. """
        return True

    @property
    def workspace(self) -> "ws.Workspace":
        """ Return the workspace which owns this type. """
        workspace = self._workspace()

        # Workspace should never be null, unless this is a dangling type object,
        # which workspace has been deleted.
        assert workspace is not None
        return workspace

    @property
    def uid(self) -> uuid.UUID:
        return self._uid

    @property
    def name(self) -> Optional[str]:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def description(self) -> Optional[str]:
        return self._description

    @description.setter
    def description(self, description: str):
        self._description = description

    @classmethod
    def find(
        cls: Type[TEntityType], workspace: "ws.Workspace", type_uid: uuid.UUID
    ) -> Optional[TEntityType]:
        """ Finds in the given Workspace the EntityType with the given UUID for
        this specific EntityType implementation class.

        Returns None if not found.
        """
        return cast(TEntityType, workspace.find_type(type_uid, cls))
