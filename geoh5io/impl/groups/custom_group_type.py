from __future__ import annotations

import uuid
from typing import Optional

from . import GroupType


class CustomGroupType(GroupType):
    """ A type to deal with custom group types possibly present in geoh5 but unknown from Analyst."""

    def __init__(self, uid, name=None, description=None, class_id=None):
        super().__init__(uid, name, description, class_id)

    @property
    def name(self) -> Optional[str]:
        return super().name

    @name.setter
    def name(self, name: Optional[str]):
        self._name = name

    @property
    def description(self) -> Optional[str]:
        return super().description

    @description.setter
    def description(self, description: Optional[str]):
        self._description = description

    @classmethod
    def create(cls) -> CustomGroupType:
        """ Creates a new instance of CustomGroupType with a new auto-generated UUID.

        The same  UUID is used for class_id.
        """
        class_id = uuid.uuid4()
        return CustomGroupType(class_id, None, None, class_id)
