import uuid
from typing import Optional

from geoh5io.shared import Coord3D

from .object_base import ObjectBase, ObjectType


class BlockModel(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0xB020A277, 0x90E2, 0x4CD7, 0x84, 0xD6, 0x612EE3F25051)
    )
    __CLASS_UID = uuid.UUID(
        fields=(0x68BFBE67, 0xE557, 0x44C8, 0x80, 0x0A, 0x10DD67F5166C)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        # TODO
        self._origin = Coord3D()
        self._rotation = 0
        # self._u_cell_delimiters = []
        # self._v_cell_delimiters = []
        # self._z_cell_delimiters = []

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @classmethod
    def default_class_id(cls) -> Optional[uuid.UUID]:
        return cls.__CLASS_UID
