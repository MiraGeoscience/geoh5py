import uuid
from abc import ABC
from abc import abstractmethod
from typing import Optional


class Type(ABC):
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
    def create(cls):
        """ Creates a new instance of a concrete Type.

        Note: the abstract declaration also makes it possible to prevent instantiation of
        intermediate partial Type classes.
        """
        ...
