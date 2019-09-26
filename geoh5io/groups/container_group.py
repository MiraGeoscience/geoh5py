from __future__ import annotations

import uuid

from .group import Group


class ContainerGroup(Group):
    """ The type for the basic Container group."""

    __type_uid = uuid.UUID("{61FBB4E8-A480-11E3-8D5A-2776BDF4F982}")

    @classmethod
    def static_type_uid(cls) -> uuid.UUID:
        return cls.__type_uid

    @classmethod
    def static_type_name(cls) -> str:
        return "Container"
