#  Copyright (c) 2023 Mira Geoscience Ltd.
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

from .object_base import ObjectBase, ObjectType


class Octree(ObjectBase):
    """
    Octree mesh class that uses a tree structure such that cells
    can be subdivided it into eight octants.
    """

    __TYPE_UID = uuid.UUID(
        fields=(0x4EA87376, 0x3ECE, 0x438B, 0xBF, 0x12, 0x3479733DED46)
    )

    _attribute_map: dict = ObjectBase._attribute_map.copy()
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

    def __init__(self, object_type: ObjectType, **kwargs):

        self._origin = [0, 0, 0]
        self._rotation = 0.0
        self._u_count = None
        self._v_count = None
        self._w_count = None
        self._u_cell_size = None
        self._v_cell_size = None
        self._w_cell_size = None
        self._octree_cells = None
        self._centroids: np.ndarray | None = None
        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    def base_refine(self):
        """
        Refine the mesh to its base octree level resulting in a
        single cell along the shortest dimension.
        """
        assert (
            self._octree_cells is None
        ), "'base_refine' function only implemented if 'octree_cells' is None "

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

        self._octree_cells = np.rec.fromarrays(
            octree_cells.T,
            names=["I", "J", "K", "NCells"],
            formats=["<i4", "<i4", "<i4", "<i4"],
        )

    @property
    def centroids(self):
        """
        :obj:`numpy.array` of :obj:`float`,
        shape (:obj:`~geoh5py.objects.octree.Octree.n_cells`, 3):
        Cell center locations in world coordinates.

        .. code-block:: python

            centroids = [
                [x_1, y_1, z_1],
                ...,
                [x_N, y_N, z_N]
            ]
        """
        if getattr(self, "_centroids", None) is None:
            assert self.octree_cells is not None, "octree_cells must be set"
            assert self.u_cell_size is not None, "u_cell_size must be set"
            assert self.v_cell_size is not None, "v_cell_size must be set"
            assert self.w_cell_size is not None, "w_cell_size must be set"

            angle = np.deg2rad(self.rotation)
            rot = np.r_[
                np.c_[np.cos(angle), -np.sin(angle), 0],
                np.c_[np.sin(angle), np.cos(angle), 0],
                np.c_[0, 0, 1],
            ]

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

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    @property
    def n_cells(self) -> int | None:
        """
        :obj:`int`: Total number of cells in the mesh
        """
        if self.octree_cells is not None:
            return self.octree_cells.shape[0]
        return None

    @property
    def octree_cells(self) -> np.ndarray | None:
        """
        :obj:`numpy.ndarray` of :obj:`int`,
        shape (:obj:`~geoh5py.objects.octree.Octree.n_cells`, 4):
        Array defining the i, j, k position and size of each cell.
        The size defines the width of a cell in number of base cells.

         .. code-block:: python

            cells = [
                [i_1, j_1, k_1, size_1],
                ...,
                [i_N, j_N, k_N, size_N]
            ]
        """
        if getattr(self, "_octree_cells", None) is None:
            if self.on_file:
                octree_cells = self.workspace.fetch_array_attribute(
                    self, "octree_cells"
                )
                self._octree_cells = octree_cells

            else:
                self.base_refine()

        return self._octree_cells

    @octree_cells.setter
    def octree_cells(self, value):

        if value is not None:
            dtypes = [("I", "<i4"), ("J", "<i4"), ("K", "<i4"), ("NCells", "<i4")]
            if len(value.dtype) > 1:
                dtype = np.dtype(dtypes)
                assert (
                    value.dtype == dtype
                ), f"Input of type {np.ndarray} must be of {dtype}"
                self._octree_cells = value
            else:
                value = np.vstack(value)
                assert (
                    value.shape[1] == 4
                ), "'octree_cells' requires an ndarray of shape (*, 4)"
                self._centroids = None
                self._octree_cells = np.asarray(
                    np.core.records.fromarrays(
                        value.T, names="I, J, K, NCells", formats="<i4, <i4, <i4, <i4"
                    )
                )
                self.workspace.update_attribute(self, "octree_cells")

    @property
    def origin(self):
        """
        :obj:`numpy.array` of :obj:`float`, shape (3, ): Coordinates of the origin
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                value = value.tolist()

            assert len(value) == 3, "Origin must be a list or numpy array of shape (3,)"

            self.workspace.update_attribute(self, "attributes")
            self._centroids = None

            value = np.asarray(
                tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
            )
            self._origin = value

    @property
    def rotation(self) -> float:
        """
        :obj:`float`: Clockwise rotation angle (degree) about the vertical axis.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "Rotation angle must be a float of shape (1,)"
            self._centroids = None
            self._rotation = value.astype(float)
            self.workspace.update_attribute(self, "attributes")

    @property
    def shape(self) -> tuple | None:
        """
        :obj:`list` of :obj:`int`, len (3, ): Number of cells along the u, v and w-axis.
        """
        if (
            self.u_count is not None
            and self.v_count is not None
            and self.w_count is not None
        ):
            return self.u_count, self.v_count, self.w_count
        return None

    @property
    def u_cell_size(self) -> float | None:
        """
        :obj:`float`: Base cell size along the u-axis.
        """
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, value: float):
        if not isinstance(value, (float, np.ndarray)):
            raise TypeError("Attribute 'u_cell_size' must be type(float).")

        self.workspace.update_attribute(self, "attributes")
        self._centroids = None

        self._u_cell_size = np.r_[value].astype(float)

    @property
    def u_count(self) -> int | None:
        """
        :obj:`int`: Number of cells along u-axis.
        """
        return self._u_count

    @u_count.setter
    def u_count(self, value: int):
        if not isinstance(value, (float, np.int32, int)) or np.log2(value) % 1.0 != 0:
            raise TypeError("Attribute 'u_count' must be type(int) in power of 2.")

        self._centroids = None

        self._u_count = np.int32(value)
        self.workspace.update_attribute(self, "attributes")

    @property
    def v_cell_size(self) -> float | None:
        """
        :obj:`float`: Base cell size along the v-axis.
        """
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, value: float):
        if not isinstance(value, (float, np.ndarray)):
            raise TypeError("Attribute 'v_cell_size' must be type(float).")
        self.workspace.update_attribute(self, "attributes")
        self._centroids = None

        self._v_cell_size = np.r_[value].astype(float)

    @property
    def v_count(self) -> int | None:
        """
        :obj:`int`: Number of cells along v-axis.
        """
        return self._v_count

    @v_count.setter
    def v_count(self, value: int):
        if not isinstance(value, (float, np.int32, int)) or np.log2(value) % 1.0 != 0:
            raise TypeError("Attribute 'v_count' must be type(int) in power of 2.")
        self._centroids = None
        self._v_count = np.int32(value)
        self.workspace.update_attribute(self, "attributes")

    @property
    def w_cell_size(self) -> float | None:
        """
        :obj:`float`: Base cell size along the w-axis.
        """
        return self._w_cell_size

    @w_cell_size.setter
    def w_cell_size(self, value: float):
        if not isinstance(value, (float, np.ndarray)):
            raise TypeError("Attribute 'w_cell_size' must be type(float).")
        self.workspace.update_attribute(self, "attributes")
        self._centroids = None

        self._w_cell_size = np.r_[value].astype(float)

    @property
    def w_count(self) -> int | None:
        """
        :obj:`int`: Number of cells along w-axis.
        """
        return self._w_count

    @w_count.setter
    def w_count(self, value: int):
        if not isinstance(value, (float, np.int32, int)) or np.log2(value) % 1.0 != 0:
            raise TypeError("Attribute 'w_count' must be type(int) in power of 2.")
        self._centroids = None
        self._w_count = np.int32(value)
        self.workspace.update_attribute(self, "attributes")
