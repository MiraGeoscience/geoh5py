import uuid
from typing import Optional

from .object_base import ObjectBase, ObjectType


class GeoImage(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0x77AC043C, 0xFE8D, 0x4D14, 0x81, 0x67, 0x75E300FB835A)
    )
    __CLASS_UID = uuid.UUID(
        fields=(0x848A7105, 0xA227, 0x46B4, 0xBB, 0x19, 0x59AAD747C351)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        self.vertices = None

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_class_id(cls) -> Optional[uuid.UUID]:
        return cls.__CLASS_UID
