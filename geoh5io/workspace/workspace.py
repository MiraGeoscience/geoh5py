from __future__ import annotations

import uuid
from typing import Optional

from geoh5io.data.data_type import DataType
from geoh5io.groups.group_type import GroupType
from geoh5io.objects.object_type import ObjectType


class Workspace:
    _active = None  # type: Optional[Workspace]

    def __init__(self):
        self.version = None
        self.distance_unit = None
        self.contributors = []

    @staticmethod
    def active() -> Workspace:
        """ Get the active workspace. """
        if Workspace._active is None:
            raise RuntimeError("No active workspace.")

        return Workspace._active

    # pylint: disable=unused-argument, no-self-use
    def find_group_type(self, type_uid: uuid.UUID) -> Optional[GroupType]:
        # TODO
        ...

    # pylint: disable=unused-argument, no-self-use
    def find_object_type(self, type_uid: uuid.UUID) -> Optional[ObjectType]:
        # TODO
        ...

    # pylint: disable=unused-argument, no-self-use
    def find_data_type(self, type_uid: uuid.UUID) -> Optional[DataType]:
        # TODO
        ...
