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

        :return:
        """
        if getattr(self, "_vertices", None) is None:
            self._vertices = self.entity_type.workspace.get_vertices(self.uid)

        return self._vertices

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
            point_object.set_vertices(locations)

        elif isinstance(locations, list):
            assert (
                len(locations) == 3
            ), "List of coordinates [x, y, z] must be of length 3"
            point_object.set_vertices(np.c_[locations])

        if data is not None:
            for key, value in data.items():

                if isinstance(value, np.ndarray):

                    float_data = object_type.workspace.create_entity(
                        Data,
                        uuid.uuid4(),
                        key,
                        uuid.uuid4(),
                        attributes={"association": "VERTEX"},
                        type_attributes={"primitive_type": "FLOAT"},
                    )

                    float_data.values = value

                    # Add the new object and type to tree
                    object_type.workspace.add_to_tree(float_data)

                    float_data.set_parent(point_object)

        return point_object

    def set_vertices(self, xyz):
        """
        set_vertices(xyz)

        Function to assign coordinate locations to a point object.

        :param xyz: numpy.ndarray of node locations [nDx3] (x,y,z)

        """

        self._vertices = Coord3D(xyz)

    @property
    def locations(self):
        """
        Property locations
        :return: Coordinates (x, y, z) from the shared.Coord3D class
        """
        return self._vertices.locations
