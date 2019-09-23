import uuid

from geoh5io.objects import Object


class Surface(Object):
    __class_id = uuid.UUID("{F26FEBA3-ADED-494B-B9E9-B2BBCBE298E1}")

    def __init__(self):
        super().__init__()
        self.vertices = []
        self.cells = []

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
