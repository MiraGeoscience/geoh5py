import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional

from geoh5io.shared import Entity

from .object_type import ObjectType

if TYPE_CHECKING:
    from geoh5io import workspace


class Object(Entity):
    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        assert object_type is not None
        super().__init__(name, uid)

        self._type = object_type
        self._allow_move = 1
        self._clipping_ids: List[uuid.UUID] = []

    @property
    def entity_type(self) -> ObjectType:
        return self._type

    @classmethod
    def find_or_create_type(
        cls, workspace: "workspace.Workspace"
    ) -> ObjectType:
        return ObjectType.find_or_create(workspace, cls)

    @classmethod
    @abstractmethod
    def static_type_uid(cls) -> uuid.UUID:
        ...

    @classmethod
    def static_class_id(cls) -> Optional[uuid.UUID]:
        return None
