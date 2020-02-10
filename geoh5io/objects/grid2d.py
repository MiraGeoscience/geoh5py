import uuid
from typing import Optional

import numpy as np

from .object_base import ObjectBase, ObjectType


class Grid2D(ObjectBase):
    """
    The ``Grid2D`` is a rectilinear array uniform cell size. The grid can
    be oriented in 3D space through ``rotation`` and ``dip`` parameters.

    The basic requirements needed to create an Octree mesh are:
        u, v_count = Number of cells along each axis
        u, v_cell_size = Cell size along each axis

                origin      origin
               V            V
     .____.____.            .____.____.
    -2.  -1.   O            O    1.   2.
    (-) cell_size         (+) cell_size

    """

    __TYPE_UID = uuid.UUID(
        fields=(0x48F5054A, 0x1C5C, 0x4CA4, 0x90, 0x48, 0x80F36DC60A06)
    )

    _attribute_map = ObjectBase._attribute_map.copy()
    _attribute_map.update(
        {
            "Dip": "dip",
            "U Count": "u_count",
            "V Count": "v_count",
            "Origin": "origin",
            "Rotation": "rotation",
            "U Size": "u_cell_size",
            "V Size": "v_cell_size",
            "Vertical": "vertical",
        }
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._origin = [0, 0, 0]
        self._u_size = None
        self._v_size = None
        self._u_count = None
        self._v_count = None
        self._rotation = 0.0
        self._vertical = False
        self._dip = 0.0
        self._centroids = None

        super().__init__(object_type, **kwargs)

        if object_type.name == "None":
            self.entity_type.name = "Grid"
        #
        # if object_type.description is None:
        #     self.entity_type.description = "Grid"

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def origin(self):
        """
        Coordinates of the origin: array of floats, shape (3,)
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        if value is not None:

            if isinstance(value, np.ndarray):
                value = value.tolist()

            assert len(value) == 3, "Origin must be a list or numpy array of shape (3,)"

            self.modified_entity = "attributes"
            self._centroids = None

            value = np.asarray(
                tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
            )
            self._origin = value

    @property
    def dip(self):
        """"
        Dip angle (positive down) in degree: float
        """
        return self._dip

    @dip.setter
    def dip(self, value):
        if value is not None:
            assert isinstance(value, float), "Dip angle must be a float"

            self.modified_entity = "attributes"
            self._centroids = None
            self._dip = value

    @property
    def u_cell_size(self) -> Optional[float]:
        """
        Cell size along the u-axis: float
        """
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "u_cell_size must be a float of shape (1,)"
            self.modified_entity = "attributes"
            self._centroids = None

            self._u_cell_size = value.astype(float)

    @property
    def v_cell_size(self) -> Optional[float]:
        """
        Cell size along the v-axis: float
        """
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "v_cell_size must be a float of shape (1,)"
            self.modified_entity = "attributes"
            self._centroids = None

            self._v_cell_size = value.astype(float)

    @property
    def u_count(self) -> Optional[int]:
        """
        Number of cells along u-axis: int
        """
        return self._u_count

    @u_count.setter
    def u_count(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "u_count must be an integer of shape (1,)"
            self.modified_entity = "attributes"
            self._centroids = None

            self._u_count = value.astype(int)

    @property
    def v_count(self) -> Optional[int]:
        """
        Number of cells along v-axis: int
        """
        return self._v_count

    @v_count.setter
    def v_count(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "v_count must be an integer of shape (1,)"
            self.modified_entity = "attributes"
            self._centroids = None

            self._v_count = value.astype(int)

    @property
    def rotation(self) -> Optional[float]:
        """
        Clockwise rotation angle about the vertical axis in degree: float
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "Rotation angle must be a float of shape (1,)"
            self.modified_entity = "attributes"
            self._centroids = None

            self._rotation = value.astype(float)

    @property
    def vertical(self) -> Optional[bool]:
        """
        Set the grid to be vertical: bool
        """
        return self._vertical

    @vertical.setter
    def vertical(self, value: bool):
        if value is not None:
            assert isinstance(value, bool) or value in [
                0,
                1,
            ], "vertical must be of type 'bool'"
            self.modified_entity = "attributes"
            self._centroids = None

            self._vertical = value

    @property
    def n_cells(self) -> Optional[int]:
        """
        Total number of cells in the grid: int
        """
        assert (self.u_count is not None) and (
            self.v_count is not None
        ), "'u_count' and 'v_count' must be set"

        return int(self.u_count * self.v_count)

    @property
    def cell_center_u(self):
        """
        The cell center location along u-axis: array of floats, shape(u_count,)
        """
        return (
            np.cumsum(np.ones(self.u_count) * self.u_cell_size) - self.u_cell_size / 2.0
        )

    @property
    def cell_center_v(self):
        """
        The cell center location along v-axis: array of floats, shape(u_count,)
        """
        return (
            np.cumsum(np.ones(self.v_count) * self.v_cell_size) - self.v_cell_size / 2.0
        )

    @property
    def centroids(self):
        """
        Cell center locations in world coordinates [x_i, y_i, z_i]:
        array of floats, shape(n_cells, 3)
        """
        if getattr(self, "_centroids", None) is None:

            angle = np.deg2rad(self.rotation)
            rot = np.r_[
                np.c_[np.cos(angle), -np.sin(angle), 0],
                np.c_[np.sin(angle), np.cos(angle), 0],
                np.c_[0, 0, 1],
            ]

            u_grid, v_grid = np.meshgrid(self.cell_center_u, self.cell_center_v)
            if self.vertical:
                xyz = np.c_[np.ravel(u_grid), np.zeros(self.n_cells), np.ravel(v_grid)]

            else:
                xyz = np.c_[np.ravel(u_grid), np.ravel(v_grid), np.zeros(self.n_cells)]

            self._centroids = np.dot(rot, xyz.T).T

            for ind, axis in enumerate(["x", "y", "z"]):
                self._centroids[:, ind] += self.origin[axis]

        return self._centroids
