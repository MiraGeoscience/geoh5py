import uuid

from geoh5io.objects import Object


class Points(Object):
    __class_id = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self):
        super().__init__()
        self._vertices = []

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
