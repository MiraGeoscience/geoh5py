import uuid
from typing import Optional

from numpy import arange, c_, ndarray

from .cell import Cell
from .object_base import ObjectType
from .points import Points


class Curve(Points):
    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )

    def __init__(self, object_type: ObjectType, name: str, uid: uuid.UUID = None):
        super().__init__(object_type, name, uid)

        self._cells: Optional[Cell] = None

        if object_type.name is None:
            self.entity_type.name = "Curve"
        else:
            self.entity_type.name = object_type.name

        if object_type.description is None:
            self.entity_type.description = "Curve"
        else:
            self.entity_type.description = object_type.description

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
            if self._existing_h5_entity:
                self._cells = self.entity_type.workspace.fetch_cells(self.uid)
            else:
                if self.vertices is not None:
                    n_segments = self.vertices.locations.shape[0]
                    self._cells = c_[
                        arange(0, n_segments - 1), arange(1, n_segments)
                    ].astype("uint32")

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

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    # @classmethod
    # def create(
    #     cls,
    #     workspace: "workspace.Workspace",
    #     vertices: ndarray,
    #     name: str = "NewCurve",
    #     uid: uuid.UUID = uuid.uuid4(),
    #     parent=None,
    #     data: Optional[dict] = None,
    #     *args
    # ):
    #     """
    #     create(
    #         workspace, vertices, cells, name=["NewCurve"],
    #         uid=[uuid.uuid4()], parent=None, data=None
    #     )
    #
    #     Function to create a curve object from xyz vertices and data
    #
    #     Parameters
    #     ----------
    #     workspace: geoh5io.Workspace
    #         Workspace to be added to
    #
    #     vertices: numpy.ndarray("float64")
    #         Coordinate xyz vertices [n x 3]
    #
    #     name: str optional
    #         Name of the curve object ["NewCurve"]
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
    #     *cells: numpy.ndarray("uint32")
    #         Array on integers defining the connection between vertices
    #
    #     Returns
    #     -------
    #     entity: geoh5io.Points
    #         Point object registered to the workspace.
    #     """
    #
    #     object_type = cls.find_or_create_type(workspace)
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
    #     curve_object.parent = parent
    #
    #     if isinstance(vertices, ndarray):
    #         assert (
    #             vertices.shape[1] == 3
    #         ), "Locations should be an an array of shape N x 3"
    #         curve_object.vertices = vertices
    #
    #     elif isinstance(vertices, list):
    #         assert (
    #             len(vertices) == 3
    #         ), "List of coordinates [x, y, z] must be of length 3"
    #         curve_object.vertices = c_[vertices]
    #
    #     # If segments are not provided, connect sequentially
    #     if "cells" not in args:
    #         vertices = curve_object.vertices
    #         n_segments = vertices().shape[0]
    #         cells = c_[arange(0, n_segments - 1), arange(1, n_segments)].astype(
    #             "uint32"
    #         )
    #     else:
    #         cells = args["cells"]
    #
    #     curve_object.cells = cells
    #
    #     if data is not None:
    #
    #         data_objects = []
    #         for key, value in data.items():
    #
    #             if isinstance(value[1], ndarray):
    #
    #                 ######## NEED TO ADD A CHECK ON n_data and n_cell #################
    #                 # if (value[0].upper() == "VERTEX") and (
    #                 #     curve_object.vertices is not None
    #                 # ):
    #                 #     assert (
    #                 #         value[1].shape[0] == curve_object.vertices().shape[0]
    #                 #     ), "VERTEX data values must be of shape ({n_data}, )".format(
    #                 #         n_data=curve_object.vertices().shape[0]
    #                 #     )
    #                 #
    #                 # if (value[0].upper() == "CELL") and (
    #                 #     curve_object.vertices is not None
    #                 # ):
    #                 #     assert (
    #                 #         value[1].shape[0] == curve_object.cells.shape[0]
    #                 #     ), "CELL data values must be of shape ({n_data}, )".format(
    #                 #         n_data=curve_object.cells.shape[0]
    #                 #     )
    #
    #                 data_object = object_type.workspace.create_entity(
    #                     Data,
    #                     key,
    #                     uuid.uuid4(),
    #                     entity_type_uid=uuid.uuid4(),
    #                     attributes={"association": value[0].upper()},
    #                     type_attributes={"primitive_type": "FLOAT"},
    #                 )
    #
    #                 data_object.values = value[1]
    #
    #                 # Add the new object and type to tree
    #                 object_type.workspace.add_to_tree(data_object)
    #
    #                 data_object.parent = curve_object
    #
    #                 data_objects.append(data_object)
    #
    #         return tuple([curve_object] + data_objects)
    #
    #     return curve_object
