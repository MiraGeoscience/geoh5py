import uuid
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from geoh5io.shared import Entity

from .group_type import GroupType

if TYPE_CHECKING:
    from geoh5io import workspace


class Group(Entity):
    def __init__(
        self, group_type: GroupType, name: str = "Group", uid: uuid.UUID = None
    ):
        assert group_type is not None
        super().__init__(name=name, uid=uid)

        self._type = group_type

        # self._clipping_ids: List[uuid.UUID] = []
        group_type.workspace._register_group(self)

    @property
    def entity_type(self) -> GroupType:
        return self._type

    @classmethod
    def find_or_create_type(
        cls, workspace: "workspace.Workspace", uid: Optional[uuid.UUID] = None
    ) -> GroupType:

        return GroupType.find_or_create(workspace, cls, uid=uid)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> Optional[uuid.UUID]:
        ...

    @classmethod
    @abstractmethod
    def default_type_name(cls) -> Optional[str]:
        ...

    @classmethod
    def default_type_description(cls) -> Optional[str]:
        return cls.default_type_name()
