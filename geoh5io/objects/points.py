import uuid
from typing import TYPE_CHECKING, Optional

from numpy import ndarray

from geoh5io.data import Data
from geoh5io.shared import Coord3D

from .object_base import ObjectBase, ObjectType

if TYPE_CHECKING:
    from geoh5io import workspace


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

        if getattr(self, "_vertices", None) is None:
            self._vertices = self.workspace.fetch_vertices(self.uid)

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

    @classmethod
    def create(
        cls,
        workspace: "workspace.Workspace",
        vertices: ndarray,
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
        workspace: geoh5io.Workspace
            Workspace to be added to

        vertices: numpy.ndarray("float64")
            Coordinate xyz vertices [n x 3]

        cells: numpy.ndarray("uint32")
            Array on integers defining the connection between vertices

        name: str optional
            Name of the point object [NewPoints]

        uid: uuid.UUID optional
            Unique identifier, or randomly generated using uuid.uuid4 if None

        parent: uuid.UUID | Entity | None optional
            Parental Entity or reference uuid to be linked to.
            If None, the object is added to the base Workspace.

        data: Dict{'name': ['association', values]} optional
            Dictionary of data values associated to "CELL" or "VERTEX" values

        Returns
        -------
        entity: geoh5io.Points
            Point object registered to the workspace.
        """

        object_type = cls.find_or_create_type(workspace)

        if object_type.name is None:
            object_type.name = "Points"

        if object_type.description is None:
            object_type.description = "Points"

        point_object = Points(object_type, name, uid)

        # Add the new object and type to tree
        object_type.workspace.add_to_tree(point_object)

        point_object.parent = parent

        if isinstance(vertices, ndarray):
            assert (
                vertices.shape[1] == 3
            ), "Locations should be an an array of shape N x 3"
            point_object.vertices = vertices

        # elif isinstance(vertices, list):
        #     assert (
        #             len(vertices) == 3
        #     ), "List of coordinates [x, y, z] must be of length 3"
        #     point_object.vertices = c_[vertices]
        #
        # # If segments are not provided, connect sequentially
        # if cells is None:
        #     vertices = point_object.vertices
        #     n_segments = vertices().shape[0]
        #     cells = c_[arange(0, n_segments - 1), arange(1, n_segments)].astype(
        #         "uint32"
        #     )
        #
        # point_object.cells = cells

        if data is not None:
            data_objects = []
            for key, value in data.items():

                if isinstance(value, ndarray):
                    data_object = object_type.workspace.create_entity(
                        Data,
                        key,
                        uuid.uuid4(),
                        entity_type_uid=uuid.uuid4(),
                        attributes={"association": "VERTEX"},
                        type_attributes={"primitive_type": "FLOAT"},
                    )

                    data_object.values = value

                    # Add the new object and type to tree
                    object_type.workspace.add_to_tree(data_object)

                    data_object.parent = point_object

                data_objects.append(data_object)

            return tuple([point_object] + data_objects)

        return point_object
