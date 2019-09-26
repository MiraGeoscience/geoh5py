import uuid

from .object_base import ObjectBase, ObjectType


class Octree(ObjectBase):
    # TODO: put the right UUID. Beware class_id differs from type uid (!!!)
    __type_uid = uuid.UUID("{10000000-0000-0000-0000-000000000000}")
    __class_id = uuid.UUID("{20000000-0000-0000-0000-000000000000}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO

    @classmethod
    def static_type_uid(cls) -> uuid.UUID:
        return cls.__type_uid

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
