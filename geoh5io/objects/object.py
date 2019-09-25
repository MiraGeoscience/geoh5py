import uuid
from abc import abstractmethod
from typing import List, Optional

from geoh5io.shared import Entity

from .object_type import ObjectType


class Object(Entity):
    def __init__(
        self, object_type: ObjectType, name: str, uid: uuid.UUID = None
    ):
        assert object_type is not None
        super().__init__(name, uid)

        self._type = object_type
        self._allow_move = 1
        self._clipping_ids: List[uuid.UUID] = []

    def get_type(self) -> ObjectType:
        return self._type

    @classmethod
    @abstractmethod
    def static_class_id(cls) -> Optional[uuid.UUID]:
        ...
