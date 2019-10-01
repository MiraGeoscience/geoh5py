from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Type

from geoh5io.shared import EntityType

if TYPE_CHECKING:
    from geoh5io import workspace
    from . import group  # noqa: F401


class GroupType(EntityType):
    def __init__(
        self,
        workspace: "workspace.Workspace",
        uid: uuid.UUID,
        name=None,
        description=None,
    ):
        super().__init__(workspace, uid, name, description)
        self._allow_move_content = True
        self._allow_delete_content = True

    @staticmethod
    def _is_abstract() -> bool:
        return False

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

    @classmethod
    def find_or_create(
        cls, workspace: "workspace.Workspace", group_class: Type["group.Group"]
    ) -> GroupType:
        """ Find or creates the GroupType with the pre-defined type UUID that matches the given
        Group implementation class.

        It is expected to have a single instance of GroupType in the Workspace
        for each concrete Group class.

        :param group_class: An Group implementation class.
        :return: A new instance of GroupType.
        """
        type_uid = group_class.default_type_uid()
        if type_uid is None or type_uid.int == 0:
            raise RuntimeError(
                f"Cannot create GroupType with null UUID from {group_class.__name__}."
            )

        group_type = cls.find(workspace, type_uid)
        if group_type is not None:
            return group_type

        return cls(
            workspace,
            type_uid,
            group_class.default_type_name(),
            group_class.default_type_description(),
        )

    @staticmethod
    def create_custom(
        workspace: "workspace.Workspace", name=None, description=None
    ) -> GroupType:
        """ Creates a new instance of GroupType for an unlisted custom Group type with a
        new auto-generated UUID.
        """
        type_id = uuid.uuid4()
        return GroupType(workspace, type_id, name, description)
