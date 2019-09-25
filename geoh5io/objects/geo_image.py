import uuid

from .object import Object


class GeoImage(Object):
    __class_id = uuid.UUID("{77AC043C-FE8D-4D14-8167-75E300FB835A}")

    def __init__(self):
        super().__init__()
        self.vertices = None

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
