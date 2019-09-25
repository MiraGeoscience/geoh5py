import uuid
from abc import abstractmethod
from typing import List
from typing import Optional

from .group_type import GroupType
from geoh5io.shared import Entity


class Group(Entity):
    def __init__(self, name: str, uid: uuid.UUID = None, group_type: GroupType = None):
        super().__init__(name, uid)
        self._type = (
            group_type
            if group_type is not None
            else GroupType.find_or_create(self.__class__)
        )
        self._allow_move = True
        self._clipping_ids: List[uuid.UUID] = []

    def get_type(self) -> GroupType:
        return self._type

    @classmethod
    @abstractmethod
    def static_class_id(cls) -> Optional[uuid.UUID]:
        ...

    @classmethod
    @abstractmethod
    def static_type_name(cls) -> Optional[str]:
        ...

    @classmethod
    @abstractmethod
    def static_type_description(cls) -> Optional[str]:
        ...
