import uuid

from geoh5io.objects import Object


class Grid2D(Object):
    __class_id = uuid.UUID("{48f5054a-1c5c-4ca4-9048-80f36dc60a06}")

    def __init__(self):
        super().__init__()
        self.origin = None
        self.u_size = None
        self.v_size = None
        self.u_count = None
        self.v_count = None
        self.rotation = 0
        self.is_vertical = 0

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
