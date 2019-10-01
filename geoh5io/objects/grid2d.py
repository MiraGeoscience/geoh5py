import uuid

from .object_base import ObjectBase, ObjectType


class Grid2D(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0x48F5054A, 0x1C5C, 0x4CA4, 0x90, 0x48, 0x80F36DC60A06)
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
