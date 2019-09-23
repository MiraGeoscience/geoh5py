import uuid

from geoh5io.objects import Object


class Octree(Object):
    __class_id = uuid.UUID("{???}")  # TODO: put thr right UUID

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
