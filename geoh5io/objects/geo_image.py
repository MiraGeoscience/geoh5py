import uuid

from .object import Object, ObjectType


class GeoImage(Object):
    __type_uid = uuid.UUID("{77AC043C-FE8D-4D14-8167-75E300FB835A}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        self.vertices = None

    @classmethod
    def static_type_uid(cls) -> uuid.UUID:
        return cls.__type_uid
