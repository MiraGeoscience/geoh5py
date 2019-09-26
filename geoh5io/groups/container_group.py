from __future__ import annotations

import uuid

from .group import Group


class ContainerGroup(Group):
    """ The type for the basic Container group."""

    __TYPE_UID = uuid.UUID("{61FBB4E8-A480-11E3-8D5A-2776BDF4F982}")

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_type_name(cls) -> str:
        return "Container"
