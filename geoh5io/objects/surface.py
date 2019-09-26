import uuid

from .object_base import ObjectBase, ObjectType


class Surface(ObjectBase):
    __TYPE_UID = uuid.UUID("{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        # self._vertices = []
        # self._cells = []

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
