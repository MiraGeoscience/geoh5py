from __future__ import annotations

import uuid
from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Type

from geoh5io.shared import Entity


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
    def _create(cls, entity_class: Type[Entity]) -> EntityType:
        """ Creates a new instance of the proper type for the given entity type.
        """
        ...
