from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Optional


class EntityType(ABC):
    def __init__(self, uid: uuid.UUID, name: str = None, description: str = None):
        self._uid = uid
        self._name = name
        self._description = description

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
    def find(cls, type_uid: uuid.UUID) -> Optional[EntityType]:
        """ Finds and returns the EntityType for the given UUID for a specific EntityType
        implementation class.

        Returns None if not found.
        """
        ...
