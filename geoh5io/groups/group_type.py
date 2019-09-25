from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional, Type, cast

from geoh5io.shared import EntityType

if TYPE_CHECKING:
    from geoh5io import workspace
    from . import group


class GroupType(EntityType):

    def __init__(self, workspace: 'workspace.Workspace', uid: uuid.UUID, name=None, description=None,
                 class_id: uuid.UUID = None):
        super().__init__(workspace, uid, name, description)
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
    def find(cls, workspace: 'workspace.Workspace', type_uid: uuid.UUID) -> Optional[GroupType]:
        return cast(GroupType, workspace.find_type(type_uid, cls))

    @classmethod
    def find_or_create(cls, workspace: 'workspace.Workspace', group_class: Type["group.Group"]) -> \
            GroupType:
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

        group_type = cls.find(workspace, class_id)
        if group_type is not None:
            return group_type

        return cls(
            workspace,
            class_id,
            group_class.static_type_name(),
            group_class.static_type_description(),
            class_id,
        )

    @staticmethod
    def create_custom(workspace: 'workspace.Workspace', name=None, description=None) -> GroupType:
        """ Creates a new instance of GroupType for an unlisted custom Group type with a
        new auto-generated UUID.

        The same UUID is used for class_id.
        """
        class_id = uuid.uuid4()
        return GroupType(workspace, class_id, name, description, class_id)
