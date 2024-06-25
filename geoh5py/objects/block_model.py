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
from typing import TYPE_CHECKING

import numpy as np

from .grid_object import GridObject

if TYPE_CHECKING:
    from geoh5py.objects import ObjectType


class BlockModel(GridObject):
    """
    Rectilinear 3D tensor mesh defined by three perpendicular axes.

    Each axis is divided into discrete intervals that define the cell dimensions.
    Nodal coordinates are determined relative to the origin and the sign of cell delimiters.
    Negative and positive cell delimiters
    are accepted to denote relative offsets from the origin.

    :param object_type: Type of object registered in the workspace.
    :param rotation: Clockwise rotation angle (degree) about the vertical axis.
    :param u_cell_delimiters: Nodal offsets along the u-axis relative to the origin.
    :param v_cell_delimiters: Nodal offsets along the v-axis relative to the origin.
    :param z_cell_delimiters: Nodal offsets along the z-axis relative to the origin.
    :param kwargs: Additional attributes to assign to the object as defined by the base class.
    """

    __TYPE_UID = uuid.UUID(
        fields=(0xB020A277, 0x90E2, 0x4CD7, 0x84, 0xD6, 0x612EE3F25051)
    )
    _attribute_map = GridObject._attribute_map.copy()
    _attribute_map.update({"Origin": "origin", "Rotation": "rotation"})

    def __init__(
        self,
        object_type: ObjectType,
        rotation: float = 0.0,
        u_cell_delimiters: np.ndarray = np.array([0.0, 1.0]),
        v_cell_delimiters: np.ndarray = np.array([0.0, 1.0]),
        z_cell_delimiters: np.ndarray = np.array([0.0, 1.0]),
        **kwargs,
    ):

        self._rotation: float
        self._u_cell_delimiters: np.ndarray
        self._v_cell_delimiters: np.ndarray
        self._z_cell_delimiters: np.ndarray

        super().__init__(
            object_type,
            rotation=rotation,
            u_cell_delimiters=u_cell_delimiters,
            v_cell_delimiters=v_cell_delimiters,
            z_cell_delimiters=z_cell_delimiters,
            **kwargs,
        )

    @property
    def centroids(self) -> np.ndarray | None:
        """
        :obj:`numpy.array`,
        shape (:obj:`~geoh5py.objects.block_model.BlockModel.n_cells`, 3):
        Cell center locations in world coordinates.

        .. code-block:: python

            centroids = [
                [x_1, y_1, z_1],
                ...,
                [x_N, y_N, z_N]
            ]
        """
        if getattr(self, "_centroids", None) is None:
            cell_center_u = np.cumsum(self.u_cells) - self.u_cells / 2.0
            cell_center_v = np.cumsum(self.v_cells) - self.v_cells / 2.0
            cell_center_z = np.cumsum(self.z_cells) - self.z_cells / 2.0

            angle = np.deg2rad(self.rotation)
            rot = np.r_[
                np.c_[np.cos(angle), -np.sin(angle), 0],
                np.c_[np.sin(angle), np.cos(angle), 0],
                np.c_[0, 0, 1],
            ]

            u_grid, v_grid, z_grid = np.meshgrid(
                cell_center_u, cell_center_v, cell_center_z
            )

            xyz = np.c_[np.ravel(u_grid), np.ravel(v_grid), np.ravel(z_grid)]

            self._centroids = np.dot(rot, xyz.T).T

            for ind, axis in enumerate(["x", "y", "z"]):
                self._centroids[:, ind] += self.origin[axis]

        return self._centroids

    @property
    def cell_delimiters(self):
        return [
            self._u_cell_delimiters,
            self._v_cell_delimiters,
            self._z_cell_delimiters,
        ]

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def n_cells(self) -> int:
        """
        :obj:`int`: Total number of cells
        """
        return int(np.prod(self.shape))

    @property
    def rotation(self) -> float:
        """
        :obj:`float`: Clockwise rotation angle (degree) about the vertical axis.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value: np.ndarray | float):
        if isinstance(value, float):
            value = np.r_[value]

        if not isinstance(value, np.ndarray) or value.shape != (1,):
            raise TypeError("Rotation angle must be a float of shape (1,)")

        self._centroids = None
        self._rotation = value.astype(float).item()

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def shape(self) -> tuple | None:
        """
        :obj:`list` of :obj:`int`, len (3, ): Number of cells along the u, v and z-axis
        """
        return tuple(
            [self.u_cells.shape[0], self.v_cells.shape[0], self.z_cells.shape[0]]
        )

    @property
    def u_cell_delimiters(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`float`:
        Nodal offsets along the u-axis relative to the origin.
        """
        return self._u_cell_delimiters

    @u_cell_delimiters.setter
    def u_cell_delimiters(self, value: np.ndarray):
        if getattr(self, "_u_cell_delimiters", None) is not None:
            raise ValueError(
                "The 'u_cell_delimiters' is a read-only property. "
                "Consider creating a new BlockModel."
            )

        if not isinstance(value, np.ndarray):
            raise TypeError("u_cell_delimiters must be a numpy array.")

        self._centroids = None
        self._u_cell_delimiters = value.astype(float)

    @property
    def u_cells(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`float`,
        shape (:obj:`~geoh5py.objects.block_model.BlockModel.shape` [0], ):
        Cell size along the u-axis.
        """
        return self.u_cell_delimiters[1:] - self.u_cell_delimiters[:-1]

    @property
    def v_cell_delimiters(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`float`:
        Nodal offsets along the v-axis relative to the origin.
        """
        return self._v_cell_delimiters

    @v_cell_delimiters.setter
    def v_cell_delimiters(self, value: np.ndarray):
        if getattr(self, "_v_cell_delimiters", None) is not None:
            raise ValueError(
                "The 'v_cell_delimiters' is a read-only property. "
                "Consider creating a new BlockModel."
            )

        if not isinstance(value, np.ndarray):
            raise TypeError("v_cell_delimiters must be a numpy array.")

        self._centroids = None
        self._v_cell_delimiters = value.astype(float)

    @property
    def v_cells(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`float`,
        shape (:obj:`~geoh5py.objects.block_model.BlockModel.shape` [1], ):
        Cell size along the v-axis.
        """
        return self.v_cell_delimiters[1:] - self.v_cell_delimiters[:-1]

    @property
    def z_cell_delimiters(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`float`:
        Nodal offsets along the u-axis relative to the origin.
        """
        return self._z_cell_delimiters

    @z_cell_delimiters.setter
    def z_cell_delimiters(self, value: np.ndarray):
        if getattr(self, "_z_cell_delimiters", None) is not None:
            raise ValueError(
                "The 'z_cell_delimiters' is a read-only property. "
                "Consider creating a new BlockModel."
            )

        if not isinstance(value, np.ndarray):
            raise TypeError("z_cell_delimiters must be a numpy array.")

        self._centroids = None
        self._z_cell_delimiters = value.astype(float)

    @property
    def z_cells(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`float`,
        shape (:obj:`~geoh5py.objects.block_model.BlockModel.shape` [2], ):
        Cell size along the z-axis
        """
        return self.z_cell_delimiters[1:] - self.z_cell_delimiters[:-1]
