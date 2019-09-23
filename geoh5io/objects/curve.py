import uuid

from geoh5io.objects import Object


class Curve(Object):
    __class_id = uuid.UUID("{6A057FDC-B355-11E3-95BE-FD84A7FFCB88}")

    def __init__(self):
        super().__init__()
        self.vertices = []
        self.cells = []

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
