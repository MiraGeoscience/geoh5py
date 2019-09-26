import uuid
from typing import List

from geoh5io.shared import Coord3D

from .object import Object, ObjectType


class Points(Object):
    __type_uid = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        self._vertices: List[Coord3D] = []

    @classmethod
    def static_type_uid(cls) -> uuid.UUID:
        return cls.__type_uid
