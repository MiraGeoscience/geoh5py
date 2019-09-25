import uuid
from typing import List

from .object import Object
from geoh5io.shared import Coord3D


class Points(Object):
    __class_id = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self, name: str, uid: uuid.UUID = None):
        super().__init__(name, uid)
        self._vertices: List[Coord3D] = []

    @classmethod
    def static_class_id(cls) -> uuid.UUID:
        return cls.__class_id
