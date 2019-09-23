import uuid
from abc import abstractmethod
from typing import List
from typing import Optional

from geoh5io.groups import GroupType
from geoh5io.shared import Entity


class Group(Entity):
    def __init__(self, group_type: GroupType):
        super().__init__()
        self._allow_move = True
        self._clipping_ids: List[uuid.UUID] = []
        self._type = group_type

    def get_type(self) -> GroupType:
        return self._type

    @abstractmethod
    @classmethod
    def static_class_id(cls) -> Optional[uuid.UUID]:
        ...

    @abstractmethod
    @classmethod
    def static_type_name(cls) -> Optional[str]:
        ...

    @abstractmethod
    @classmethod
    def static_type_description(cls) -> Optional[str]:
        ...
