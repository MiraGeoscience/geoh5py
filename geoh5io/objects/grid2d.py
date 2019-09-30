import uuid
from typing import Optional

from .object_base import ObjectBase, ObjectType


class Grid2D(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0x48F5054A, 0x1C5C, 0x4CA4, 0x90, 0x48, 0x80F36DC60A06)
    )
    __CLASS_UID = uuid.UUID(
        fields=(0x6859F026, 0x9F21, 0x4096, 0xB3, 0x6C, 0x087D03D0DA21)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        self.origin = None
        self.u_size = None
        self.v_size = None
        self.u_count = None
        self.v_count = None
        self.rotation = 0
        self.is_vertical = 0

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_class_id(cls) -> Optional[uuid.UUID]:
        return cls.__CLASS_UID
