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

from ..objects import GeoImage
from ..shared.conversion import Grid2DConversion
from ..shared.utils import mask_by_extent, xy_rotation_matrix, yz_rotation_matrix
from .grid_object import GridObject


class Grid2D(GridObject):
    """
    Rectilinear 2D grid of uniform cell size. The grid can
    be oriented in 3D space through horizontal :obj:`~geoh5py.objects.grid2d.Grid2D.rotation`
    and :obj:`~geoh5py.objects.grid2d.Grid2D.dip` parameters.
    Nodal coordinates are determined relative to the origin and the sign
    of cell delimiters.

    :param u_cell_size: Cell size along the u-axis.
    :param v_cell_size: Cell size along the v-axis.
    :param u_count: Number of cells along the u-axis.
    :param v_count: Number of cells along the v-axis.
    :param vertical: Set the grid to be vertical.
    :param dip: Dip angle from horizontal (positive down) in degrees.
        Defaults to an horizontal grid (dip=0).
    """

    _TYPE_UID = uuid.UUID(
        fields=(0x48F5054A, 0x1C5C, 0x4CA4, 0x90, 0x48, 0x80F36DC60A06)
    )
    _attribute_map = GridObject._attribute_map.copy()
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
    _converter: type[Grid2DConversion] = Grid2DConversion

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        u_cell_size: float = 1.0,
        v_cell_size: float = 1.0,
        u_count: int = 1,
        v_count: int = 1,
        vertical: bool = False,
        dip: float = 0.0,
        **kwargs,
    ):
        self._u_count: np.int32 = self.validate_count(u_count, "u")
        self._v_count: np.int32 = self.validate_count(v_count, "v")

        super().__init__(
            **kwargs,
        )

        self.u_cell_size = u_cell_size
        self.v_cell_size = v_cell_size
        self.dip = dip
        self.vertical = vertical

    @property
    def cell_center_u(self) -> np.ndarray:
        """
        Cell center local coordinate along the u-axis,
        shape(:obj:`~geoh5py.objects.grid2d.Grid2D.u_count`, )
        """
        return (
            np.cumsum(np.ones(self.u_count) * self.u_cell_size) - self.u_cell_size / 2.0
        )

    @property
    def cell_center_v(self) -> np.ndarray:
        """
        The cell center local coordinate along the v-axis,
        shape(:obj:`~geoh5py.objects.grid2d.Grid2D.u_count`, )
        """
        return (
            np.cumsum(np.ones(self.v_count) * self.v_cell_size) - self.v_cell_size / 2.0
        )

    @property
    def centroids(self) -> np.ndarray:
        """
        Cell center locations in world coordinates,
        shape(:obj:`~geoh5py.objects.grid2d.Grid2D.n_cells`, 3).

        .. code-block:: python

            centroids = [
                [x_1, y_1, z_1],
                ...,
                [x_N, y_N, z_N]
            ]
        """
        if getattr(self, "_centroids", None) is None:
            rotation_matrix = xy_rotation_matrix(np.deg2rad(self.rotation))
            dip_matrix = yz_rotation_matrix(np.deg2rad(self.dip))

            u_grid, v_grid = np.meshgrid(self.cell_center_u, self.cell_center_v)
            xyz = np.c_[np.ravel(u_grid), np.ravel(v_grid), np.zeros(self.n_cells)]

            xyz_dipped = dip_matrix @ xyz.T
            centroids = (rotation_matrix @ xyz_dipped).T

            for ind, axis in enumerate(["x", "y", "z"]):
                centroids[:, ind] += self.origin[axis]

            self._centroids = centroids

        return self._centroids

    def copy_from_extent(  # pylint: disable=too-many-locals
        self,
        extent: np.ndarray,
        parent=None,
        *,
        copy_children: bool = True,
        clear_cache: bool = False,
        inverse: bool = False,
        **kwargs,
    ) -> Grid2D | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.copy_from_extent`.
        """
        if not isinstance(extent, np.ndarray):
            raise TypeError("Expected a numpy array of extent values.")

        if not extent.ndim == 2 or extent.shape not in [(2, 3), (2, 2)]:
            raise TypeError("Expected a 2D numpy array with 2 or 3 columns")

        # get the centroids
        selected_centroids = mask_by_extent(
            self.centroids, extent, inverse=inverse
        ).reshape((self.v_count, self.u_count))

        u_ind = np.any(selected_centroids, axis=0)
        v_ind = np.any(selected_centroids, axis=1)

        indices = np.kron(v_ind, u_ind).flatten()

        if not np.any(indices):
            return None

        if not inverse:
            delta_orig = np.c_[
                np.argmax(u_ind) * self.u_cell_size,
                np.argmax(v_ind) * self.v_cell_size,
                0.0,
            ].T

            dip_matrix = yz_rotation_matrix(np.deg2rad(self.dip))
            delta_orig = dip_matrix @ delta_orig

            rotation_matrix = xy_rotation_matrix(np.deg2rad(self.rotation))
            delta_orig = (rotation_matrix @ delta_orig).T

            kwargs.update(
                {
                    "origin": np.r_[
                        self.origin["x"] + delta_orig[0, 0],
                        self.origin["y"] + delta_orig[0, 1],
                        self.origin["z"] + delta_orig[0, 2],
                    ],
                    "u_count": np.sum(u_ind),
                    "v_count": np.sum(v_ind),
                }
            )

        else:
            indices = GridObject.mask_by_extent(self, extent, inverse=inverse)

        copy = super(GridObject, self).copy(
            parent=parent,
            copy_children=copy_children,
            clear_cache=clear_cache,
            mask=indices,
            **kwargs,
        )

        if not inverse:
            for child in copy.children:
                if isinstance(getattr(child, "values", None), np.ndarray):
                    indices = child.mask_by_extent(extent, inverse=inverse)
                    values = child.values
                    values[~indices] = child.nan_value
                    child.values = values

        return copy

    @property
    def dip(self) -> float:
        """
        Dip angle from horizontal (positive down) in degrees.
        """
        if self.vertical:
            self._dip = 90.0
            return self._dip

        return self._dip

    @dip.setter
    def dip(self, value: Real):
        if not isinstance(value, Real):
            raise TypeError("Dip angle must be a float.")

        self._centroids = None

        self._dip = float(value)
        if self._dip == 90:
            self._vertical = True

        self.workspace.update_attribute(self, "attributes")

    @property
    def shape(self) -> tuple[np.int32, np.int32]:
        """
        Number of cells along the u and v-axis.
        """
        return self.u_count, self.v_count

    @property
    def u_cell_size(self) -> float:
        """
        Cell size along the u-axis.
        """
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, value: Real | np.ndarray):
        self._u_cell_size = self.validate_cell_size(value, "u")
        self._centroids = None

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def u_count(self) -> np.int32:
        """
        Number of cells along v-axis.
        """
        return self._u_count

    @property
    def v_cell_size(self) -> float:
        """
        Cell size along the v-axis
        """
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, value: Real | np.ndarray):
        self._v_cell_size = self.validate_cell_size(value, "v")
        self._centroids = None

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    @property
    def v_count(self) -> np.int32:
        """
        Number of cells along v-axis.
        """
        return self._v_count

    @staticmethod
    def validate_cell_size(value: Real | np.ndarray, axis: str) -> float:
        """
        Validate and format type of cell size value.
        """
        if not isinstance(value, (Real, np.ndarray)):
            raise TypeError(f"Attribute '{axis}_cell_size' must be type(float).")

        if isinstance(value, np.ndarray):
            if not len(value) == 1:
                raise ValueError(
                    "Attribute 'v_cell_size' must be a float of shape (1,)"
                )

            return np.r_[value].astype(float).item()

        return float(value)

    @property
    def vertical(self) -> bool:
        """
        Set the grid to be vertical.
        """
        return self._vertical

    @vertical.setter
    def vertical(self, value: bool):
        if not isinstance(value, bool) is not None or value not in [
            0,
            1,
        ]:
            raise TypeError("Attribute 'vertical' must be type(bool).")

        self._centroids = None
        self._vertical = bool(value)

        if self.dip != 90 and value is True:
            self._dip = 90.0

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    def to_geoimage(
        self, keys: list | str, mode: str | None = None, **geoimage_kwargs
    ) -> GeoImage:
        """
        Create a :obj:geoh5py.objects.geo_image.GeoImage object from the current Grid2D.

        :param keys: the list of the data name to pass as band in the image.
            Warning: The len of the list can only be 1, 3, 4 (Pillow restrictions).
        :param mode: The mode of the image. One of 'GRAY', 'RGB', 'RGBA' or 'CMYK'.

        :return: a new georeferenced :obj:`geoh5py.objects.geo_image.GeoImage`.
        """
        return self.converter.to_geoimage(self, keys, mode=mode, **geoimage_kwargs)

    @staticmethod
    def validate_count(value: int, axis: str) -> np.int32:
        """
        Validate and format type of count value.
        """
        if not isinstance(value, (np.integer, int)) or value < 1:
            raise TypeError(
                f"Attribute '{axis}_count' must be a type(int32) greater than 1."
            )

        return np.int32(value)
