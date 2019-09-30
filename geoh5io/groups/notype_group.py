from typing import Optional

import uuid

from .group import Group

class NoTypeGroup(Group):
    """ A group with no type."""

    __TYPE_UID = uuid.UUID(fields=(0xDD99B610, 0xBE92, 0x48C0, 0x87, 0x3C, 0x5B5946EA2840))
    __CLASS_UID = uuid.UUID(fields=(0xC15ADBDE, 0xEC23, 0x4A11, 0x93, 0xB5, 0x2AEF597643DB))

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_class_id(cls) -> Optional[uuid.UUID]:
        return cls.__CLASS_UID

    @classmethod
    def default_type_name(cls) -> str:
        return "NoType"

    @classmethod
    def default_type_description(cls) -> Optional[str]:
        return "<Unknown>"
