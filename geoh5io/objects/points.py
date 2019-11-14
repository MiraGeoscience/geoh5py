import uuid
from typing import Optional

import numpy as np

from geoh5io import workspace
from geoh5io.data import Data
from geoh5io.shared import Coord3D

from .object_base import ObjectBase, ObjectType


class Points(ObjectBase):
    __TYPE_UID = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)

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

    @classmethod
    def create(
        cls,
        locations=None,
        work_space=None,
        name: str = "NewPoints",
        uid: uuid.UUID = uuid.uuid4(),
        parent=None,
        data: Optional[dict] = None,
    ):
        """
        create(
            locations, workspace=None, name="NewPoints",
            uid=uuid.uuid4(), parent=None, data=None
        )

        Function to create a point object from xyz locations and data

        Parameters
        ----------
        locations: numpy.array()
            Coordinate xyz locations [n x 3]

        work_space: geoh5io.Workspace
            Workspace or active worskapce if [None]

        name: str optional
            Name of the point object [NewPoints]

        uid: uuid.UUID optional
            Unique identifier, or randomly generated using uuid.uuid4 if None

        parent: uuid.UUID | Entity | None optional
            Parental Entity or reference uuid to be linked to.
            If None, the object is added to the base Workspace.

        data: Dict{'name': values} optional
            Dictionary of data values to be added to the point object

        Returns
        -------
        entity: geoh5io.Points
            Point object registered to the workspace.
        """

        object_type = cls.find_or_create_type(
            workspace.Workspace.active() if work_space is None else work_space
        )

        if object_type.name is None:
            object_type.name = "Points"

        if object_type.description is None:
            object_type.description = "Points"

        point_object = Points(object_type, name, uid)

        # Add the new object and type to tree
        object_type.workspace.add_to_tree(point_object)

        point_object.set_parent(parent)

        if isinstance(locations, np.ndarray):
            assert (
                locations.shape[1] == 3
            ), "Locations should be an an array of shape N x 3"
            point_object.vertices = locations

        elif isinstance(locations, list):
            assert (
                len(locations) == 3
            ), "List of coordinates [x, y, z] must be of length 3"
            point_object.vertices = np.c_[locations]

        if data is not None:
            for key, value in data.items():

                if isinstance(value, np.ndarray):

                    float_data = object_type.workspace.create_entity(
                        Data,
                        key,
                        uuid.uuid4(),
                        uuid.uuid4(),
                        attributes={"association": "VERTEX"},
                        type_attributes={"primitive_type": "FLOAT"},
                    )

                    float_data.values = value

                    # Add the new object and type to tree
                    object_type.workspace.add_to_tree(float_data)

                    float_data.set_parent(point_object)

        return point_object

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
        return self._vertices.locations
