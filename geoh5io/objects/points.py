import uuid
from typing import Optional

import numpy as np

from geoh5io.data import Data
from geoh5io.shared import Coord3D

from .object_base import ObjectBase, ObjectType


class Points(ObjectBase):
    __TYPE_UID = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(
        self,
        object_type: ObjectType,
        name: str,
        uid: uuid.UUID = None,
        locations: np.ndarray = None,
    ):

        super().__init__(object_type, name, uid)

        if object_type.name is None:
            self.entity_type.name = "Points"
        else:
            self.entity_type.name = object_type.name

        if object_type.description is None:
            self.entity_type.description = "Points"
        else:
            self.entity_type.description = object_type.description

        if locations is not None:
            assert (
                locations.shape[1] == 3
            ), "Locations should be an an array of shape N x 3"
            self.vertices = locations

        else:
            self._vertices: Optional[Coord3D] = None

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

        if getattr(self, "_vertices", None) is None:
            self._vertices = self.entity_type.workspace.fetch_vertices(self.uid)

        return self._vertices

    @vertices.setter
    def vertices(self, xyz):
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

    def add_data(self, data: dict):

        data_list = []
        for key, value in data.items():

            if isinstance(value, np.ndarray):
                float_data = self.entity_type.workspace.create_entity(
                    Data,
                    key,
                    uuid.uuid4(),
                    entity_type_uid=uuid.uuid4(),
                    attributes={"association": "VERTEX"},
                    type_attributes={"primitive_type": "FLOAT"},
                    parent=self,
                )

                float_data.values = value

                # Add the new object and type to tree
                # self.entity_type.workspace.add_to_tree(float_data)

                # float_data.set_parent(self)

                data_list.append(float_data)

        return data_list
