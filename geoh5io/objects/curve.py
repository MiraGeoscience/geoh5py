import uuid
from typing import Optional

from .object_base import ObjectBase, ObjectType


class Curve(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )
    __CLASS_UID = uuid.UUID(
        fields=(0x7808367B, 0xF514, 0x429D, 0xB8, 0xF5, 0xB6984831E0CC)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        # self._vertices = []
        # self._cells = []

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_class_id(cls) -> Optional[uuid.UUID]:
        return cls.__CLASS_UID
