import uuid
from abc import abstractmethod
from typing import List, Optional, TYPE_CHECKING

from geoh5io.shared import Entity

if TYPE_CHECKING:
    from geoh5io import workspace

from .group_type import GroupType


class Group(Entity):
    def __init__(self, group_type: GroupType, name: str, uid: uuid.UUID = None):
        assert group_type is not None
        super().__init__(name, uid)

        self._type = group_type
        self._allow_move = True
        self._clipping_ids: List[uuid.UUID] = []

    def get_type(self) -> GroupType:
        return self._type

    @classmethod
    def find_or_create_type(cls, workspace: 'workspace.Workspace') -> Optional[GroupType]:
        return GroupType.find_or_create(workspace, cls)

    @property
    def allow_move(self) -> bool:
        return self._allow_move

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
