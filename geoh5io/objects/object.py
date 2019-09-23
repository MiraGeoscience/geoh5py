import uuid
from abc import abstractmethod
from typing import List
from typing import Optional

from geoh5io.objects import ObjectType
from geoh5io.shared import Entity


class Object(Entity):
    def __init__(self, object_type: ObjectType, uid: uuid.UUID, name: str):
        super().__init__(uid, name)
        self._type = object_type
        self._allow_move = 1
        self._clipping_ids: List[uuid.UUID] = []

    def get_type(self) -> ObjectType:
        return self._type

    @abstractmethod
    @classmethod
    def static_class_id(cls) -> Optional[uuid.UUID]:
        ...
