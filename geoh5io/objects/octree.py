import uuid

from .object_base import ObjectBase, ObjectType


class Octree(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0x4EA87376, 0x3ECE, 0x438B, 0xBF, 0x12, 0x3479733DED46)
    )
    __CLASS_ID = uuid.UUID(
        fields=(0xD23BFBF5, 0x6A64, 0x4138, 0x8B, 0xE4, 0x088BD60E35C2)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_class_id(cls) -> uuid.UUID:
        return cls.__CLASS_ID
