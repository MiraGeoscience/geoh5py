#  Copyright (c) 2020 Mira Geoscience Ltd.
#
#  This file is part of geoh5py.
#
#  geoh5py is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  geoh5py is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with geoh5py.  If not, see <https://www.gnu.org/licenses/>.

import uuid
from typing import Optional, Tuple

import numpy as np

from .object_base import ObjectBase, ObjectType


class Grid2D(ObjectBase):
    """
    A Grid2D is a rectilinear array uniform cell size. The grid can
    be oriented in 3D space through rotation and dip parameters.
    Nodal coordinates are determined relative to the origin and the sign
    of cell delimiters. Negative and positive cell delimiters are accepted
    to denote relative "left/right" offsets.

    Coordinates
    2  3  4  5  6
    origin   V
    .__.__.__.__.__
    -1 -1 -1  1  1
    Delimiters
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
        self._u_cell_size = None
        self._v_cell_size = None
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
    def origin(self) -> np.ndarray:
        """
        Coordinates of the origin: shape (3,)
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        if value is not None:

            if isinstance(value, np.ndarray):
                value = value.tolist()

            assert len(value) == 3, "Origin must be a list or numpy array of shape (3,)"

            self.modified_attributes = "attributes"
            self._centroids = None

            value = np.asarray(
                tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
            )
            self._origin = value

    @property
    def dip(self) -> float:
        """"
        Dip angle (positive down) in degree
        """
        return self._dip

    @dip.setter
    def dip(self, value):
        if value is not None:
            assert isinstance(value, float), "Dip angle must be a float"
            self.modified_attributes = "attributes"
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

            self.modified_attributes = "attributes"
            self._centroids = None

            self._u_cell_size = value.astype(float)

    @property
    def v_cell_size(self) -> Optional[float]:
        """
        Cell size along the v-axis
        """
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "v_cell_size must be a float of shape (1,)"
            self.modified_attributes = "attributes"
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
            self.modified_attributes = "attributes"
            self._centroids = None

            self._u_count = int(value)

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
            self.modified_attributes = "attributes"
            self._centroids = None

            self._v_count = int(value)

    @property
    def rotation(self) -> float:
        """
        Clockwise rotation angle about the vertical axis in degree: float
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "Rotation angle must be a float of shape (1,)"
            self.modified_attributes = "attributes"
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
            self.modified_attributes = "attributes"
            self._centroids = None

            self._vertical = value

    @property
    def cell_center_u(self) -> np.ndarray:
        """
        The cell center location along u-axis: shape(u_count,)
        """
        if self.u_count is not None and self.u_cell_size is not None:
            return (
                np.cumsum(np.ones(self.u_count) * self.u_cell_size)
                - self.u_cell_size / 2.0
            )
        return None

    @property
    def cell_center_v(self) -> np.ndarray:
        """
        The cell center location along v-axis: shape(u_count,)
        """
        if self.v_count is not None and self.v_cell_size is not None:
            return (
                np.cumsum(np.ones(self.v_count) * self.v_cell_size)
                - self.v_cell_size / 2.0
            )
        return None

    @property
    def centroids(self) -> np.ndarray:
        """
        Cell center locations in world coordinates [x_i, y_i, z_i]: shape(n_cells, 3)
        """
        if (
            getattr(self, "_centroids", None) is None
            and self.cell_center_u is not None
            and self.cell_center_v is not None
            and self.n_cells is not None
            and self.origin is not None
        ):

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

            centroids = np.asarray(np.dot(rot, xyz.T).T)

            for ind, axis in enumerate(["x", "y", "z"]):
                centroids[:, ind] += self.origin[axis]

            self._centroids = centroids

        return self._centroids

    @property
    def shape(self) -> Optional[Tuple]:
        """
        Number of cells along the u, v and z-axis: list[int], length (3,)
        """
        if self.u_count is not None and self.v_count is not None:
            return self.u_count, self.v_count
        return None

    @property
    def n_cells(self) -> Optional[int]:
        """
        Number of cells
        """
        if self.shape is not None:
            return np.prod(self.shape)
        return None
