import uuid

from .object import Object, ObjectType


class Label(Object):
    __type_uid = uuid.UUID("{E79F449D-74E3-4598-9C9C-351A28B8B69E}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        self.target_position = None
        self.label_position = None

    @classmethod
    def static_type_uid(cls) -> uuid.UUID:
        return cls.__type_uid
