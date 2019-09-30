import uuid
from typing import Optional

from .group import Group


class ContainerGroup(Group):
    """ The type for the basic Container group."""

    __TYPE_UID = uuid.UUID(
        fields=(0x61FBB4E8, 0xA480, 0x11E3, 0x8D, 0x5A, 0x2776BDF4F982)
    )
    __CLASS_UID = uuid.UUID(
        fields=(0x0B2A043F, 0x11C6, 0x11D7, 0xB3, 0x2E, 0x00B0D03E31EF)
    )

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_class_id(cls) -> Optional[uuid.UUID]:
        return cls.__CLASS_UID

    @classmethod
    def default_type_name(cls) -> str:
        return "Container"
