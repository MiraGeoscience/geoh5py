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
import warnings

import numpy as np

from ..shared.utils import mask_by_extent
from .object_base import ObjectBase, ObjectType


class Points(ObjectBase):
    """
    Points object made up of vertices.
    """

    __TYPE_UID = uuid.UUID("{202C5DB1-A56D-4004-9CAD-BAAFD8899406}")

    def __init__(self, object_type: ObjectType, name="Points", **kwargs):
        self._vertices: np.ndarray | None = None

        super().__init__(object_type, name=name, **kwargs)

        object_type.workspace._register_object(self)

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID

    def clip_by_extent(
        self, bounds: np.ndarray, clear_cache: bool = False
    ) -> Points | None:
        """
        Find indices of vertices within a rectangular bounds.

        :param bounds: shape(2, 2) Bounding box defined by the South-West and
            North-East coordinates. Extents can also be provided as 3D coordinates
            with shape(2, 3) defining the top and bottom limits.
        """
        indices = mask_by_extent(self.vertices, bounds)
        self.remove_vertices(~indices, clear_cache=clear_cache)
        return self

    @property
    def extent(self) -> np.ndarray | None:
        """
        Geography bounding box of the object.

        :return: shape(2, 3) Bounding box defined by the bottom South-West and
            top North-East coordinates.
        """
        if self._extent is None and self.vertices is not None:
            self._extent = np.c_[self.vertices.min(axis=0), self.vertices.max(axis=0)].T

        return self._extent

    def mask_by_extent(
        self,
        extent: np.ndarray,
    ) -> np.ndarray | None:
        """
        Find indices of vertices or centroids within a rectangular extent.

        :param extent: shape(2, 2) Bounding box defined by the South-West and
            North-East coordinates. Extents can also be provided as 3D coordinates
            with shape(2, 3) defining the top and bottom limits.
        """
        if not any(mask_by_extent(extent, self.extent)) and not any(
            mask_by_extent(self.extent, extent)
        ):
            return None

        if self.vertices is not None:
            return mask_by_extent(self.vertices, extent)

        return None

    @property
    def vertices(self) -> np.ndarray | None:
        """
        :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`
        """
        if self._vertices is None and self.on_file:
            self._vertices = self.workspace.fetch_array_attribute(self, "vertices")

        if self._vertices is not None:
            return self._vertices.view("<f8").reshape((-1, 3))

        return None

    @vertices.setter
    def vertices(self, xyz: np.ndarray):
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(
                f"Array of vertices must be of shape (*, 3). Array of shape {xyz.shape} provided."
            )

        if self._vertices is not None and xyz.shape[0] < self._vertices.shape[0]:
            raise ValueError(
                "Attempting to assign 'vertices' with fewer values. "
                "Use the `remove_vertices` method instead."
            )

        self._vertices = np.asarray(
            np.core.records.fromarrays(
                xyz.T.tolist(),
                dtype=[("x", "<f8"), ("y", "<f8"), ("z", "<f8")],
            )
        )
        self._extent = None
        self.workspace.update_attribute(self, "vertices")

    def remove_vertices(self, indices: list[int], clear_cache: bool = False):
        """Safely remove vertices and corresponding data entries."""

        if self._vertices is None:
            warnings.warn("No vertices to be removed.", UserWarning)
            return

        if (
            isinstance(self.vertices, np.ndarray)
            and np.max(indices) > self.vertices.shape[0] - 1
        ):
            raise ValueError("Found indices larger than the number of vertices.")

        vertices = np.delete(self.vertices, indices, axis=0)
        self._vertices = None
        self.vertices = vertices
        self.remove_children_values(indices, "VERTEX", clear_cache=clear_cache)
