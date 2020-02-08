import uuid
from typing import Optional

from numpy import ndarray

from geoh5io.shared import Coord3D

from .object_base import ObjectBase, ObjectType


class Points(ObjectBase):
    """
    Points object.

    Functions
    ---------

    """

    __TYPE_UID = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self, object_type: ObjectType, **kwargs):
        self._vertices: Optional[Coord3D] = None

        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Points"

        # if object_type.description is None:
        #     self.entity_type.description = "Points"
        #

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def vertices(self) -> Optional[Coord3D]:
        """
        vertices

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
        self.update_h5 = "vertices"
        self._vertices = Coord3D(xyz)

    @property
    def locations(self):
        """
        locations

        Returns
        -------
        locations: numpy.array
            [x, y, z] array of coordinates from the Coord3D
        """
        return self.vertices.locations

    @property
    def size(self):
        """
        size

        Returns
        -------
        size: int
            Number of vertices
        """
        if self._vertices is not None:
            return self.locations.shape[0]

        return None
