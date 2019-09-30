from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from geoh5io.groups import GroupType, NoTypeGroup

if TYPE_CHECKING:
    from geoh5io import workspace


class RootGroup(NoTypeGroup):
    """ The type for the workspace root group."""

    __ROOT_NAME = "Workspace"

    def __init__(
        self,
        workspace: "workspace.Workspace",
        group_type: GroupType = None,
        uid: uuid.UUID = None,
    ):
        if group_type is None:
            group_type = NoTypeGroup.find_or_create_type(workspace)
        super().__init__(group_type, self.__ROOT_NAME, uid)
        self._allow_move = False
        self._allow_delete = False
        self._allow_rename = False
