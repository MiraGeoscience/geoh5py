import uuid
from typing import Optional

from .object_base import ObjectBase
from .object_type import ObjectType


class NoTypeObject(ObjectBase):

    __TYPE_UID = uuid.UUID(
        fields=(0x849D2F3E, 0xA46E, 0x11E3, 0xB4, 0x01, 0x2776BDF4F982)
    )
    __CLASS_UID = uuid.UUID(
        fields=(0xF060E15F, 0x7ACC, 0x408B, 0x8F, 0x23, 0xAFC893EE3B42)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_class_id(cls) -> Optional[uuid.UUID]:
        return cls.__CLASS_UID
