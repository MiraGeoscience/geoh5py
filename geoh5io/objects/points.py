import uuid

from geoh5io.shared import Coord3D

from .object_base import ObjectBase, ObjectType


class Points(ObjectBase):
    __TYPE_UID = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)

        self._vertices: Coord3D = Coord3D()

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def vertices(self) -> Coord3D:
        """

        :return:
        """
        if getattr(self, "_vertices", None) is None:
            self._vertices = self.entity_type.workspace.get_vertices(self.uid)

        return self._vertices
