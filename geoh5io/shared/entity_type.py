from __future__ import annotations

import uuid
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Type, TypeVar, cast

if TYPE_CHECKING:
    from geoh5io import workspace as ws

TEntityType = TypeVar("TEntityType", bound="EntityType")


class EntityType(ABC):
    def __init__(
        self,
        workspace: "ws.Workspace",
        uid: uuid.UUID,
        name: str = None,
        description: str = None,
    ):
        assert workspace is not None
        assert uid is not None
        assert uid.int != 0

        self._workspace = weakref.ref(workspace)
        self._uid = uid
        self._name = name
        self._description = description
        workspace.register_type(self)

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

    @property
    def description(self) -> Optional[str]:
        return self._description

    @classmethod
    def find(
        cls: Type[TEntityType], workspace: "ws.Workspace", type_uid: uuid.UUID
    ) -> Optional[TEntityType]:
        """ Finds in the given Workspace the EntityType with the given UUID for
        this specific EntityType implementation class.

        Returns None if not found.
        """
        return cast(TEntityType, workspace.find_type(type_uid, cls))
