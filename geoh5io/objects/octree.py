import uuid

from .object_base import ObjectBase, ObjectType


class Octree(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0x4EA87376, 0x3ECE, 0x438B, 0xBF, 0x12, 0x3479733DED46)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
