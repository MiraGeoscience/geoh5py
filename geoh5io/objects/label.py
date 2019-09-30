import uuid

from .object_base import ObjectBase, ObjectType


class Label(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0xE79F449D, 0x74E3, 0x4598, 0x9C, 0x9C, 0x351A28B8B69E)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        self.target_position = None
        self.label_position = None

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
