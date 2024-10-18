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
from numbers import Real

import numpy as np

from ..shared.utils import xy_rotation_matrix
from .grid_object import GridObject


class Octree(GridObject):
    """
    Octree mesh class that uses a tree structure such that cells
    can be subdivided it into eight octants.

    :param u_count: Number of cells along the u-axis.
    :param v_count: Number of cells along the v-axis.
    :param w_count: Number of cells along the w-axis.
    :param u_cell_size: Base cell size along the u-axis.
    :param v_cell_size: Base cell size along the v-axis.
    :param w_cell_size: Base cell size along the w-axis.
    :param octree_cells: Array defining the i, j, k position and size of each cell.
    """

    _TYPE_UID = uuid.UUID(
        fields=(0x4EA87376, 0x3ECE, 0x438B, 0xBF, 0x12, 0x3479733DED46)
    )
    __OCTREE_DTYPE = np.dtype(
        [("I", "<i4"), ("J", "<i4"), ("K", "<i4"), ("NCells", "<i4")]
    )
    _attribute_map: dict = GridObject._attribute_map.copy()
    _attribute_map.update(
        {
            "NU": "u_count",
            "NV": "v_count",
            "NW": "w_count",
            "Origin": "origin",
            "Rotation": "rotation",
            "U Cell Size": "u_cell_size",
            "V Cell Size": "v_cell_size",
            "W Cell Size": "w_cell_size",
        }
    )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        u_count: int = 1,
        v_count: int = 1,
        w_count: int = 1,
        u_cell_size: float = 1.0,
        v_cell_size: float = 1.0,
        w_cell_size: float = 1.0,
        octree_cells: np.ndarray | list | tuple | None = None,
        **kwargs,
    ):
        self._u_count = self.validate_octree_count(u_count, "u")
        self._v_count = self.validate_octree_count(v_count, "v")
        self._w_count = self.validate_octree_count(w_count, "w")

        if octree_cells is None:
            octree_cells = self.base_refine()

        self._octree_cells: np.ndarray = self.validate_octree_cells(octree_cells)

        super().__init__(
            **kwargs,
        )

        self.u_cell_size = u_cell_size
        self.v_cell_size = v_cell_size
        self.w_cell_size = w_cell_size

    def base_refine(self) -> np.ndarray:
        """
        Refine the mesh to its base octree level resulting in a
        single cell along the shortest dimension.
        """
        # Number of octree levels allowed on each dimension
        level_u = np.log2(self.u_count)
        level_v = np.log2(self.v_count)
        level_w = np.log2(self.w_count)

        min_level = np.min([level_u, level_v, level_w])

        # Check that the refine level doesn't exceed the shortest dimension
        level = np.min([0, min_level])

        # Number of additional break to account for variable dimensions
        add_u = int(level_u - min_level)
        add_v = int(level_v - min_level)
        add_w = int(level_w - min_level)

        j, k, i = np.meshgrid(
            np.arange(0, self.v_count, 2 ** (level_v - add_v - level)),
            np.arange(0, self.w_count, 2 ** (level_w - add_w - level)),
            np.arange(0, self.u_count, 2 ** (level_u - add_u - level)),
        )

        octree_cells = np.c_[
            i.flatten(),
            j.flatten(),
            k.flatten(),
            np.ones_like(i.flatten()) * 2 ** (min_level - level),
        ]

        return octree_cells

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
            angle = np.deg2rad(self.rotation)
            rot = xy_rotation_matrix(angle)
            u_grid = (
                self.octree_cells["I"] + self.octree_cells["NCells"] / 2.0
            ) * self.u_cell_size
            v_grid = (
                self.octree_cells["J"] + self.octree_cells["NCells"] / 2.0
            ) * self.v_cell_size
            w_grid = (
                self.octree_cells["K"] + self.octree_cells["NCells"] / 2.0
            ) * self.w_cell_size
            xyz = np.c_[u_grid, v_grid, w_grid]
            self._centroids = np.dot(rot, xyz.T).T

            for ind, axis in enumerate(["x", "y", "z"]):
                self._centroids[:, ind] += self.origin[axis]

        return self._centroids

    @property
    def n_cells(self) -> int:
        """
        Total number of cells in the mesh
        """
        return self.octree_cells.shape[0]

    @property
    def octree_cells(self) -> np.ndarray:
        """
        Array defining the i, j, k position and size of each cell.
        The size defines the width of a cell in number of base cells,
        shape (:obj:`~geoh5py.objects.octree.Octree.n_cells`, 4).

         .. code-block:: python

            cells = [
                [i_1, j_1, k_1, size_1],
                ...,
                [i_N, j_N, k_N, size_N]
            ]
        """
        if self._octree_cells is None and self.on_file:
            self._octree_cells = self.workspace.fetch_array_attribute(
                self, "octree_cells"
            )

        return self._octree_cells

    @octree_cells.setter
    def octree_cells(self, octree_cells: np.ndarray | list | tuple):
        octree_cells = self.validate_octree_cells(octree_cells)
        if (
            self._octree_cells is not None
            and self._octree_cells.shape != octree_cells.shape
        ):
            raise ValueError(
                "New octree_cells array must have the same shape as the current octree_cells array."
            )
        self._octree_cells = octree_cells

        self.workspace.update_attribute(self, "octree_cells")

    @property
    def shape(self) -> tuple[np.int32, np.int32, np.int32]:
        """
        Number of cells along the u, v and w-axis.
        """
        return self.u_count, self.v_count, self.w_count

    @property
    def u_cell_size(self) -> float:
        """
        Base cell size along the u-axis.
        """
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, value: Real):
        if not isinstance(value, Real):
            raise TypeError("Attribute 'u_cell_size' must be type(float).")

        self._u_cell_size = float(value)

    @property
    def u_count(self) -> int:
        """
        Number of cells along u-axis.
        """
        return self._u_count

    @property
    def v_cell_size(self) -> float:
        """
        Base cell size along the v-axis.
        """
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, value: Real):
        if not isinstance(value, Real):
            raise TypeError("Attribute 'v_cell_size' must be type(float).")

        self._v_cell_size = float(value)

    @property
    def v_count(self) -> int:
        """
        Number of cells along v-axis.
        """
        return self._v_count

    @staticmethod
    def validate_octree_count(value: int, axis: str) -> np.int32:
        """
        Validate the number of cells along an axis are in power of 2.

        :param value: Number of cells along the axis.
        :param axis: Axis name.

        :return: Number of cells along the axis.
        """
        if not isinstance(value, (np.integer, int)):
            raise TypeError(f"Attribute '{axis}_count' must be type(int).")

        if np.log2(value) % 1.0 != 0:
            raise ValueError(
                f"Attribute '{axis}_count' must be type(int) in power of 2."
            )

        return np.int32(value)

    @classmethod
    def validate_octree_cells(
        cls, array: np.ndarray | list | tuple | None
    ) -> np.ndarray:
        """
        Validate the octree cell array.

        :param array: An array or list defining the i, j, k position
            and size of each cell.

        :return: A formatted recarray
        """
        if isinstance(array, (list, tuple)):
            array = np.array(array, ndmin=2)

        if not isinstance(array, np.ndarray):
            raise TypeError(
                "Attribute 'octree_cells' must be a list, tuple or numpy array. "
                f"Object of type {type(array)} provided."
            )

        if np.issubdtype(array.dtype, np.number):
            assert (
                array.shape[1] == 4
            ), "'octree_cells' requires an ndarray of shape (*, 4)"
            array = np.asarray(
                np.core.records.fromarrays(
                    array.T.tolist(),
                    dtype=cls.__OCTREE_DTYPE,
                )
            )
        if array.dtype != cls.__OCTREE_DTYPE:
            raise ValueError(
                f"Array of 'octree_cells' must be of dtype = {cls.__OCTREE_DTYPE}"
            )

        return array

    @property
    def w_cell_size(self) -> float:
        """
        Base cell size along the w-axis.
        """
        return self._w_cell_size

    @w_cell_size.setter
    def w_cell_size(self, value: Real):
        if not isinstance(value, Real):
            raise TypeError("Attribute 'w_cell_size' must be type(float).")

        self._w_cell_size = float(value)

    @property
    def w_count(self) -> int:
        """
        Number of cells along w-axis.
        """
        return self._w_count
