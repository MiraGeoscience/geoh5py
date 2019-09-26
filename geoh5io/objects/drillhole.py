import uuid

from .object import Object, ObjectType


class Drillhole(Object):
    __type_uid = uuid.UUID("{7CAEBF0E-D16E-11E3-BC69-E4632694AA37}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        self.vertices = []
        self.cells = []
        self.collar = None
        self.surveys = []
        self.trace = []

    @classmethod
    def static_type_uid(cls) -> uuid.UUID:
        return cls.__type_uid
