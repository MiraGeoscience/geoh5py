import uuid

from .object_base import ObjectBase, ObjectType


class Curve(ObjectBase):
    __TYPE_UID = uuid.UUID("{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        # self._vertices = []
        # self._cells = []

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
