import uuid
from abc import abstractmethod
from typing import List
from typing import Optional

from .object_type import ObjectType
from geoh5io.shared import Entity


class Object(Entity):
    def __init__(
        self, name: str, uid: uuid.UUID = None, object_type: ObjectType = None
    ):
        super().__init__(name, uid)
        self._type = (
            object_type
            if object_type is not None
            else ObjectType.find_or_create(self.__class__)
        )
        self._allow_move = 1
        self._clipping_ids: List[uuid.UUID] = []

    def get_type(self) -> ObjectType:
        return self._type

    @classmethod
    @abstractmethod
    def static_class_id(cls) -> Optional[uuid.UUID]:
        ...
