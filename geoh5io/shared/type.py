from __future__ import annotations

import weakref
import uuid
from abc import ABC, abstractmethod
from typing import Optional
from geoh5io import workspace


class EntityType(ABC):
    def __init__(self, workspace: 'workspace.Workspace',
                 uid: uuid.UUID, name: str = None, description: str = None):
        assert workspace is not None
        assert uid is not None and uid.int != 0
        self._workspace = weakref.ref(workspace)
        self._uid = uid
        self._name = name
        self._description = description
        workspace.register_type(self)

    @property
    def workspace(self) -> 'workspace.Workspace':
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
    @abstractmethod
    def find(cls, workspace: 'workspace.Workspace', type_uid: uuid.UUID) -> Optional[EntityType]:
        """ Finds in the given Workspace the EntityType with the given UUID for
        this specific EntityType implementation class.

        Returns None if not found.
        """
        ...
