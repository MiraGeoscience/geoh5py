from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from geoh5io.groups.group import Group, GroupType

if TYPE_CHECKING:
    from geoh5io import workspace


class RootGroup(Group):
    """ The type for the workspace root group."""

    __TYPE_UID = uuid.UUID("{dd99b610-be92-48c0-873c-5b5946ea2840}")
    __TYPE_NAME = "NoType"
    __ROOT_NAME = "Workspace"
    __TYPE_DESCRIPTION = "<Unknown>"

    def __init__(
        self,
        workspace: "workspace.Workspace",
        group_type: GroupType = None,
        uid: uuid.UUID = None,
    ):
        if group_type is None:
            group_type = self.find_or_create_type(workspace)
        super().__init__(group_type, self.__ROOT_NAME, uid)
        self._allow_move = False
        self._allow_delete = False
        self._allow_rename = False

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_type_name(cls) -> str:
        return cls.__TYPE_NAME

    @classmethod
    def default_type_description(cls) -> str:
        return cls.__TYPE_DESCRIPTION
