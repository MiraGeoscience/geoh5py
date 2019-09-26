from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from geoh5io.groups.group import Group

if TYPE_CHECKING:
    from geoh5io import workspace


class RootGroup(Group):
    """ The type for the workspace root group."""

    __type_uid = uuid.UUID("{dd99b610-be92-48c0-873c-5b5946ea2840}")
    __type_name = "NoType"
    __type_description = "<Unknown>"

    def __init__(self, workspace: "workspace.Workspace", uid: uuid.UUID = None):
        group_type = self.find_or_create_type(workspace)
        super().__init__(group_type, "Workspace", uid)
        self._allow_move = False
        self._allow_delete = False
        self._allow_rename = False

    @classmethod
    def static_type_uid(cls) -> uuid.UUID:
        return cls.__type_uid

    @classmethod
    def static_type_name(cls) -> str:
        return cls.__type_name

    @classmethod
    def static_type_description(cls) -> str:
        return cls.__type_description
