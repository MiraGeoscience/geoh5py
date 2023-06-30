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
from typing import TYPE_CHECKING

import numpy as np

from ..objects import GeoImage
from ..shared.conversion import Grid2DConversion
from ..shared.utils import xy_rotation_matrix
from .grid_object import GridObject

if TYPE_CHECKING:
    from geoh5py.objects import ObjectType


class Grid2D(GridObject):
    """
    Rectilinear 2D grid of uniform cell size. The grid can
    be oriented in 3D space through horizontal :obj:`~geoh5py.objects.grid2d.Grid2D.rotation`
    and :obj:`~geoh5py.objects.grid2d.Grid2D.dip` parameters.
    Nodal coordinates are determined relative to the origin and the sign
    of cell delimiters.
    """

    __TYPE_UID = uuid.UUID(
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

    def __init__(self, object_type: ObjectType, **kwargs):
        self._origin: np.ndarray = np.asarray(
            tuple(np.zeros(3)), dtype=[("x", float), ("y", float), ("z", float)]
        )
        self._u_cell_size: float | None = None
        self._v_cell_size: float | None = None
        self._u_count: int | None = None
        self._v_count: int | None = None
        self._rotation: float = 0.0
        self._vertical: bool = False
        self._dip: float = 0.0

        super().__init__(object_type, **kwargs)

        object_type.workspace._register_object(self)

    @property
    def cell_center_u(self) -> np.ndarray | None:
        """
        :obj:`numpy.array` of :obj:`float`, shape(:obj:`~geoh5py.objects.grid2d.Grid2D.u_count`, ):
        Cell center local coordinate along the u-axis.
        """
        if self.u_count is not None and self.u_cell_size is not None:
            return (
                np.cumsum(np.ones(self.u_count) * self.u_cell_size)
                - self.u_cell_size / 2.0
            )
        return None

    @property
    def cell_center_v(self) -> np.ndarray | None:
        """
        :obj:`numpy.array` of :obj:`float` shape(:obj:`~geoh5py.objects.grid2d.Grid2D.u_count`, ):
        The cell center local coordinate along the v-axis.
        """
        if self.v_count is not None and self.v_cell_size is not None:
            return (
                np.cumsum(np.ones(self.v_count) * self.v_cell_size)
                - self.v_cell_size / 2.0
            )
        return None

    @property
    def centroids(self) -> np.ndarray | None:
        """
        :obj:`numpy.array` of :obj:`float`,
        shape (:obj:`~geoh5py.objects.grid2d.Grid2D.n_cells`, 3):
        Cell center locations in world coordinates.

        .. code-block:: python

            centroids = [
                [x_1, y_1, z_1],
                ...,
                [x_N, y_N, z_N]
            ]
        """
        if (
            getattr(self, "_centroids", None) is None
            and self.cell_center_u is not None
            and self.cell_center_v is not None
            and self.n_cells is not None
            and self.origin is not None
        ):
            rot = xy_rotation_matrix(np.deg2rad(self.rotation))
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

    def copy_from_extent(  # pylint: disable=too-many-locals disable=too-many-arguments
        self,
        extent: np.ndarray,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        inverse: bool = False,
        from_image: bool = False,
        **kwargs,
    ) -> Grid2D | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.copy_from_extent`.
        """
        if not isinstance(extent, np.ndarray):
            raise TypeError("Expected a numpy array of extent values.")

        if self.u_cell_size is None or self.v_cell_size is None:
            raise AttributeError("Cell sizes are not defined.")

        if extent.shape[1] == 2:
            extent = np.c_[extent, np.r_[-np.inf, np.inf]]

        local_extent = extent.astype(float)
        local_extent = np.vstack(
            [
                local_extent[0, :],
                np.c_[local_extent[0, 0], local_extent[1, 1], local_extent[0, 2]],
                local_extent[1, :],
                np.c_[local_extent[1, 0], local_extent[0, 1], local_extent[1, 2]],
            ]
        )
        z_extent = local_extent[:, 2]
        origin = np.r_[self.origin["x"], self.origin["y"], self.origin["z"]].astype(
            float
        )
        local_extent[:, :2] -= origin[:2]

        if self.rotation != 0.0:
            local_extent[:, 2] = 0
            rot = xy_rotation_matrix(-np.deg2rad(self.rotation))
            local_extent = np.dot(rot, local_extent.T).T
            local_extent[:, 2] = z_extent

        u_ind = (self.cell_center_u <= np.max(local_extent[:, 0])) & (
            self.cell_center_u >= np.min(local_extent[:, 0])
        )
        v_ind = (self.cell_center_v <= np.max(local_extent[:, 1])) & (
            self.cell_center_v >= np.min(local_extent[:, 1])
        )

        indices = np.kron(v_ind, u_ind).flatten()

        if not np.any(indices):
            return None

        if not inverse:
            delta_orig = np.c_[
                np.argmax(u_ind) * self.u_cell_size,
                np.argmax(v_ind) * self.v_cell_size,
                0.0,
            ]
            rot = xy_rotation_matrix(np.deg2rad(self.rotation))
            delta_orig = np.dot(rot, delta_orig.T).T
            kwargs.update(
                {
                    "origin": np.r_[
                        origin[0] + delta_orig[0, 0],
                        origin[1] + delta_orig[0, 1],
                        origin[2],
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
                nan_value = 0 if from_image else np.nan

                if isinstance(child.values, np.ndarray):
                    indices = child.mask_by_extent(extent, inverse=inverse)
                    values = child.values.astype(float)
                    values[~indices] = nan_value
                    child.values = values

        return copy

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def dip(self) -> float:
        """
        :obj:`float`: Dip angle from horizontal (positive down) in degrees.
        """
        return self._dip

    @dip.setter
    def dip(self, value):
        if value is not None:
            assert isinstance(value, float), "Dip angle must be a float"
            self._centroids = None
            self._dip = value
            self.workspace.update_attribute(self, "attributes")

    @property
    def n_cells(self) -> int | None:
        """
        :obj:`int`: Total number of cells.
        """
        if self.shape is not None:
            return np.prod(self.shape)
        return None

    @property
    def origin(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`float`, shape (3, ): Coordinates of the origin.
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                value = value.tolist()

            assert len(value) == 3, "Origin must be a list or numpy array of shape (3,)"

            self._centroids = None

            value = np.asarray(
                tuple(value), dtype=[("x", float), ("y", float), ("z", float)]
            )
            self._origin = value
            self.workspace.update_attribute(self, "attributes")

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
            self._rotation = value.astype(float).item()
            self.workspace.update_attribute(self, "attributes")

    @property
    def shape(self) -> tuple | None:
        """
        :obj:`list` of :obj:`int`, len (2, ): Number of cells along the u and v-axis.
        """
        if self.u_count is not None and self.v_count is not None:
            return self.u_count, self.v_count
        return None

    @property
    def u_cell_size(self) -> float | None:
        """
        :obj:`np.ndarray`: Cell size along the u-axis.
        """
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, value: float | np.ndarray):
        if not isinstance(value, (float, np.ndarray)):
            raise TypeError("Attribute 'u_cell_size' must be type(float).")

        self._centroids = None
        if isinstance(value, np.ndarray):
            assert len(value) == 1, "u_cell_size must be a float of shape (1,)"
            self._u_cell_size = np.r_[value].astype(float).item()
        else:
            self._u_cell_size = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def u_count(self) -> int | None:
        """
        :obj:`int`: Number of cells along u-axis
        """
        return self._u_count

    @u_count.setter
    def u_count(self, value):
        if value is not None:
            value = np.r_[value]
            self._centroids = None
            self._u_count = int(value)
            assert len(value) == 1, "u_count must be an integer of shape (1,)"
            self.workspace.update_attribute(self, "attributes")

    @property
    def v_cell_size(self) -> float | None:
        """
        :obj:`np.ndarray`: Cell size along the v-axis
        """
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, value: float | np.ndarray):
        if not isinstance(value, (float, np.ndarray)):
            raise TypeError("Attribute 'v_cell_size' must be type(float).")

        self._centroids = None
        if isinstance(value, np.ndarray):
            assert len(value) == 1, "v_cell_size must be a float of shape (1,)"
            self._v_cell_size = np.r_[value].astype(float).item()
        else:
            self._v_cell_size = value
        self.workspace.update_attribute(self, "attributes")

    @property
    def v_count(self) -> int | None:
        """
        :obj:`int`: Number of cells along v-axis
        """
        return self._v_count

    @v_count.setter
    def v_count(self, value):
        if value is not None:
            value = np.r_[value]
            assert len(value) == 1, "v_count must be an integer of shape (1,)"
            self._centroids = None
            self._v_count = int(value)
            self.workspace.update_attribute(self, "attributes")

    @property
    def vertical(self) -> bool | None:
        """
        :obj:`bool`: Set the grid to be vertical.
        """
        return self._vertical

    @vertical.setter
    def vertical(self, value: bool):
        if value is not None:
            assert isinstance(value, bool) or value in [
                0,
                1,
            ], "vertical must be of type 'bool'"
            self._centroids = None
            self._vertical = value
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
