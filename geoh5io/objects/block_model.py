import uuid

from .object import Object


class BlockModel(Object):
    __class_id = uuid.UUID("{B020A277-90E2-4CD7-84D6-612EE3F25051}")

    def __init__(self):
        super().__init__()
        self.origin = None
        self.rotation = 0
        self.u_cell_delimiters = []
        self.v_cell_delimiters = []
        self.z_cell_delimiters = []

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
