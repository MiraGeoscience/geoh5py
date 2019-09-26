import uuid

from .object import Object, ObjectType


class Grid2D(Object):
    __type_uid = uuid.UUID("{48f5054a-1c5c-4ca4-9048-80f36dc60a06}")

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
    def static_type_uid(cls) -> uuid.UUID:
        return cls.__type_uid
