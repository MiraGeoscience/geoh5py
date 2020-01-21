import uuid
from typing import Optional

from numpy import ndarray

from geoh5io.shared import Coord3D

from .object_base import ObjectBase, ObjectType


class Points(ObjectBase):
    __TYPE_UID = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)
        self._vertices: Optional[Coord3D] = None

        if object_type.name is None:
            self.entity_type.name = "Points"
        else:
            self.entity_type.name = object_type.name

        if object_type.description is None:
            self.entity_type.description = "Points"
        else:
            self.entity_type.description = object_type.description

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def vertices(self) -> Optional[Coord3D]:
        """
        @property
        vertices(xyz)

        Function to return the object vertices coordinates.

        Returns
        -------
        vertices: geoh5io.Coord3D
            Coord3D object holding the vertices coordinates
        """
        if (getattr(self, "_vertices", None) is None) and self.existing_h5_entity:
            self._vertices = self.workspace.fetch_vertices(self.uid)

        return self._vertices

    @vertices.setter
    def vertices(self, xyz: ndarray):
        """
        @property.setter

        vertices(xyz)

        Parameters
        ----------
        xyz: numpy.array
            Coordinate xyz locations [n x 3]

        Returns
        -------
        vertices: geoh5io.Coord3D
            Coord3D object holding the vertices coordinates
        """
        if self.existing_h5_entity:
            self.update_h5 = ["vertices"]
        self._vertices = Coord3D(xyz)

    @property
    def locations(self):
        """
        @property
        location

        Returns
        -------
        locations: numpy.array
            Rapid access to the (x, y, z) coordinates from the Coord3D
        """
        return self.vertices.locations

    @property
    def size(self):
        """
        @property
        size

        Returns
        -------
        size: int
            Number of vertices
        """
        if self._vertices is not None:
            return self.locations.shape[0]

        return None
