from __future__ import annotations

import uuid
from typing import Dict
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING

from geoh5io.shared import EntityType

if TYPE_CHECKING:
    from geoh5io.groups import group
    from geoh5io.objects import object
    from geoh5io.shared import type


class Workspace:
    _active = None  # type: Optional[Workspace]

    def __init__(self):
        if Workspace._active is None:
            Workspace._active = self

        self.version = None
        self._distance_unit = None
        self._contributors = []
        self._groups: Dict[uuid.UUID, group.Group] = {}
        self._objects: Dict[uuid.UUID, object.Object] = {}
        self._types: Dict[uuid.UUID, type.EntityType] = {}

        # TODO: must always have a root group (cannot be None)
        # self._root: group.Group

    @staticmethod
    def active() -> Workspace:
        """ Get the active workspace. """
        if Workspace._active is None:
            raise RuntimeError("No active workspace.")

        return Workspace._active

    def find_type(
        self, type_uid: uuid.UUID, type_class: Type["type.EntityType"]
    ) -> Optional[EntityType]:
        found_type = self._types.get(type_uid, None)
        if found_type is not None and isinstance(found_type, type_class):
            return found_type

        return None
