from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional, Type

from geoh5io.shared import EntityType
from geoh5io.workspace import Workspace

if TYPE_CHECKING:
    from . import group


class GroupType(EntityType):
    def __init__(self, uid, name=None, description=None, class_id: uuid.UUID = None):
        super().__init__(uid, name, description)

        self._class_id = class_id
        self._allow_move_content = True
        self._allow_delete_content = True

    @property
    def allow_move_content(self) -> bool:
        return self._allow_move_content

    @allow_move_content.setter
    def allow_move_content(self, allow: bool):
        self._allow_move_content = bool(allow)

    @property
    def allow_delete_content(self) -> bool:
        return self._allow_delete_content

    @allow_delete_content.setter
    def allow_delete_content(self, allow: bool):
        self._allow_delete_content = bool(allow)

    @property
    def class_id(self) -> Optional[uuid.UUID]:
        return self._class_id

    @classmethod
    def find(cls, type_uid: uuid.UUID) -> Optional[GroupType]:
        return Workspace.active().find_type(type_uid, cls)

    @classmethod
    def find_or_create(cls, group_class: Type["group.Group"]) -> GroupType:
        """ Find or creates the GroupType with the class_id from the given Group
        implementation class.

        The class_id is also used as the UUID for the newly created GroupType.
        It is expected to have a single instance of GroupType in the Workspace
        for each concrete Group class.

        :param group_class: An Group implementation class.
        :return: A new instance of GroupType.
        """
        class_id = group_class.static_class_id()
        if class_id is None:
            raise RuntimeError(
                f"Cannot create GroupType with null UUID from {group_class.__name__}."
            )

        group_type = cls.find(class_id)
        if group_type is not None:
            return group_type

        return cls(
            class_id,
            group_class.static_type_name(),
            group_class.static_type_description(),
            class_id,
        )

    @staticmethod
    def create_custom() -> GroupType:
        """ Creates a new instance of GroupType for an unlisted custom Group type with a
        new auto-generated UUID.

        The same UUID is used for class_id.
        """
        class_id = uuid.uuid4()
        return GroupType(class_id, None, None, class_id)
