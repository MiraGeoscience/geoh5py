#  Copyright (c) 2024 Mira Geoscience Ltd.
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

from __future__ import annotations

import uuid

import numpy as np

from ..shared.utils import xy_rotation_matrix
from .grid_object import GridObject


class BlockModel(GridObject):
    """
    Rectilinear 3D tensor mesh defined by three perpendicular axes.

    Each axis is divided into discrete intervals that define the cell dimensions.
    Nodal coordinates are determined relative to the origin and the sign of cell delimiters.
    Negative and positive cell delimiters
    are accepted to denote relative offsets from the origin.

    :param object_type: Type of object registered in the workspace.
    :param u_cell_delimiters: Nodal offsets along the u-axis relative to the origin.
    :param v_cell_delimiters: Nodal offsets along the v-axis relative to the origin.
    :param z_cell_delimiters: Nodal offsets along the z-axis relative to the origin.
    :param kwargs: Additional attributes defined by the
        :obj:`~geoh5py.objects.grid_object.GridObject` class.
    """

    _TYPE_UID = uuid.UUID(
        fields=(0xB020A277, 0x90E2, 0x4CD7, 0x84, 0xD6, 0x612EE3F25051)
    )
    _attribute_map = GridObject._attribute_map.copy()
    _attribute_map.update({"Origin": "origin", "Rotation": "rotation"})

    def __init__(
        self,
        u_cell_delimiters: np.ndarray = np.array([0.0, 1.0]),
        v_cell_delimiters: np.ndarray = np.array([0.0, 1.0]),
        z_cell_delimiters: np.ndarray = np.array([0.0, 1.0]),
        **kwargs,
    ):
        self._u_cell_delimiters = self.validate_cell_delimiters(u_cell_delimiters, "u")
        self._v_cell_delimiters = self.validate_cell_delimiters(v_cell_delimiters, "v")
        self._z_cell_delimiters = self.validate_cell_delimiters(z_cell_delimiters, "z")

        super().__init__(
            **kwargs,
        )

    def local_axis_centers(self, axis: str) -> np.ndarray:
        """
        Get the local axis centers for the block model.

        :param axis: Axis to get the centers for.
        """
        out = np.cumsum(getattr(self, f"{axis}_cell_delimiters"))
        out[2:] = out[2:] - out[:-2]
        return out[1:] / 2

    @property
    def centroids(self) -> np.ndarray:
        """
        Cell center locations in world coordinates of shape (n_cells, 3).

        .. code-block:: python

            centroids = [
                [x_1, y_1, z_1],
                ...,
                [x_N, y_N, z_N]
            ]
        """
        if getattr(self, "_centroids", None) is None:
            cell_center_u = self.local_axis_centers("u")
            cell_center_v = self.local_axis_centers("v")
            cell_center_z = self.local_axis_centers("z")
            angle = np.deg2rad(self.rotation)
            rot = xy_rotation_matrix(angle)
            u_grid, v_grid, z_grid = np.meshgrid(
                cell_center_u, cell_center_v, cell_center_z
            )
            xyz = np.c_[np.ravel(u_grid), np.ravel(v_grid), np.ravel(z_grid)]
            self._centroids = np.dot(rot, xyz.T).T

            for ind, axis in enumerate(["x", "y", "z"]):
                self._centroids[:, ind] += self.origin[axis]

        return self._centroids

    @property
    def cell_delimiters(self) -> list[np.ndarray]:
        """
        List of cell delimiters along the u, v and z-axis.
        """
        return [
            self._u_cell_delimiters,
            self._v_cell_delimiters,
            self._z_cell_delimiters,
        ]

    @property
    def shape(self) -> tuple[int, int, int]:
        """
        Number of cells along the u, v and z-axis
        """
        return self.u_cells.shape[0], self.v_cells.shape[0], self.z_cells.shape[0]

    @property
    def u_cell_delimiters(self) -> np.ndarray:
        """
        Nodal offsets along the u-axis relative to the origin.
        """
        return self._u_cell_delimiters

    @property
    def u_cells(self) -> np.ndarray:
        """
        Cell size along the u-axis.
        """
        return self.u_cell_delimiters[1:] - self.u_cell_delimiters[:-1]

    @property
    def v_cell_delimiters(self) -> np.ndarray:
        """
        Nodal offsets along the v-axis relative to the origin.
        """
        return self._v_cell_delimiters

    @property
    def v_cells(self) -> np.ndarray:
        """
        Cell size along the v-axis.
        """
        return self.v_cell_delimiters[1:] - self.v_cell_delimiters[:-1]

    @property
    def z_cell_delimiters(self) -> np.ndarray:
        """
        Nodal offsets along the u-axis relative to the origin.
        """
        return self._z_cell_delimiters

    @property
    def z_cells(self) -> np.ndarray:
        """
        Cell size along the z-axis
        """
        return self.z_cell_delimiters[1:] - self.z_cell_delimiters[:-1]

    @staticmethod
    def validate_cell_delimiters(value: np.ndarray, axis: str) -> np.ndarray:
        """
        Validate the cell delimiters for the block model.

        :param value: Cell delimiters along the axis.
        :param axis: Axis to validate the cell delimiters for.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"Attribute '{axis}_cell_delimiters' must be a numpy array."
            )

        if not np.issubdtype(value.dtype, np.number) or value.ndim != 1:
            raise ValueError(
                f"Attribute '{axis}_cell_delimiters' must be a 1D array of floats. "
                f"Provided {value}"
            )

        return value.astype(float)
