from __future__ import annotations

import uuid

from .group import Group
from .group_type import GroupType


class ContainerGroup(Group):
    """ The type for the basic Container group."""

    __class_id = uuid.UUID("{61FBB4E8-A480-11E3-8D5A-2776BDF4F982}")
    __type_name = "Container"

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id

    @classmethod
    def static_type_name(cls) -> str:
        return cls.__type_name

    @classmethod
    def static_type_description(cls) -> str:
        return cls.static_type_name()
