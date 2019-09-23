import uuid

from geoh5io.objects import Object


class Drillhole(Object):
    __class_id = uuid.UUID("{7CAEBF0E-D16E-11E3-BC69-E4632694AA37}")

    def __init__(self):
        super().__init__()
        self.vertices = []
        self.cells = []
        self.collar = None
        self.surveys = []
        self.trace = []

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
