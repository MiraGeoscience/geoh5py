import uuid

from .object import Object


class Octree(Object):
    # TODO: put the right UUID
    __class_id = uuid.UUID("{00000000-0000-0000-0000-000000000000}")

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
