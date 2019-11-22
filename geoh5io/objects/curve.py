import uuid
from typing import Optional

from numpy import ndarray

from geoh5io.data import Data
from geoh5io.shared import Coord3D

from .cell import Cell
from .object_base import ObjectBase, ObjectType


class Curve(ObjectBase):
    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)

        self._vertices: Optional[Coord3D] = None
        self._cells: Optional[Cell] = None

    @property
    def cells(self) -> Optional[ndarray]:
        """
        @property
        cells(xyz)

        Function to return the object cells coordinates.

        Returns
        -------
        cells: geoh5io.objects.Cell
            Cell object holding vertices index
        """

        if getattr(self, "_cells", None) is None:
            self._cells = self.entity_type.workspace.fetch_cells(self.uid)

        return self._cells

    @cells.setter
    def cells(self, indices):
        """
        @property.setter

        cells(id1, id2)

        Parameters
        ----------
        indices: numpy.array
            Integer values [n x 2]

        Returns
        -------
        cells: geoh5io.objects.Cell
            Cell object holding vertices index
        """

        assert indices.dtype == "uint32", "Indices array must be of type 'uint32'"

        self._cells = indices

    @property
    def vertices(self) -> ndarray:
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
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    def add_data(self, data: dict):

        data_list = []
        for key, value in data.items():

            if isinstance(value[1], ndarray):

                float_data = self.entity_type.workspace.create_entity(
                    Data,
                    key,
                    uuid.uuid4(),
                    entity_type_uid=uuid.uuid4(),
                    attributes={"association": value[0].upper()},
                    type_attributes={"primitive_type": "FLOAT"},
                )

                float_data.values = value[1]

                # Add the new object and type to tree
                # self.entity_type.workspace.add_to_tree(float_data)

                # float_data.set_parent(curve_object)
                data_list.append(float_data)

        return data_list

    # @classmethod
    # def create(
    #     cls,
    #     locations: ndarray,
    #     cells=None,
    #     work_space=None,
    #     name: str = "NewPoints",
    #     uid: uuid.UUID = uuid.uuid4(),
    #     parent=None,
    #     data: Optional[dict] = None,
    # ):
    #     """
    #     create(
    #         locations, workspace=None, name="NewPoints",
    #         uid=uuid.uuid4(), parent=None, data=None
    #     )
    #
    #     Function to create a point object from xyz locations and data
    #
    #     Parameters
    #     ----------
    #     locations: numpy.ndarray("float64")
    #         Coordinate xyz locations [n x 3]
    #
    #     cells: numpy.ndarray("uint32")
    #         Array on integers defining the connection between vertices
    #
    #     work_space: geoh5io.Workspace
    #         Workspace or active workspace if [None]
    #
    #     name: str optional
    #         Name of the point object [NewPoints]
    #
    #     uid: uuid.UUID optional
    #         Unique identifier, or randomly generated using uuid.uuid4 if None
    #
    #     parent: uuid.UUID | Entity | None optional
    #         Parental Entity or reference uuid to be linked to.
    #         If None, the object is added to the base Workspace.
    #
    #     data: Dict{'name': ['association', values]} optional
    #         Dictionary of data values associated to "CELL" or "VERTEX" values
    #
    #     Returns
    #     -------
    #     entity: geoh5io.Points
    #         Point object registered to the workspace.
    #     """
    #
    #     object_type = cls.find_or_create_type(
    #         workspace.Workspace.active() if work_space is None else work_space
    #     )
    #
    #     if object_type.name is None:
    #         object_type.name = "Curve"
    #
    #     if object_type.description is None:
    #         object_type.description = "Curve"
    #
    #     curve_object = Curve(object_type, name, uid)
    #
    #     # Add the new object and type to tree
    #     object_type.workspace.add_to_tree(curve_object)
    #
    #     # curve_object.set_parent(parent)
    #
    #     if isinstance(locations, ndarray):
    #         assert (
    #             locations.shape[1] == 3
    #         ), "Locations should be an an array of shape N x 3"
    #         curve_object.vertices = locations
    #
    #     elif isinstance(locations, list):
    #         assert (
    #             len(locations) == 3
    #         ), "List of coordinates [x, y, z] must be of length 3"
    #         curve_object.vertices = c_[locations]
    #
    #     # If segments are not provided, connect sequentially
    #     if cells is None:
    #         vertices = curve_object.vertices
    #         n_segments = vertices().shape[0]
    #         cells = c_[arange(0, n_segments - 1), arange(1, n_segments)].astype(
    #             "uint32"
    #         )
    #
    #     curve_object.cells = cells
