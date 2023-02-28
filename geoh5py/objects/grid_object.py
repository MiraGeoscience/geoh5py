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
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..shared.utils import mask_by_extent
from .object_base import ObjectBase

if TYPE_CHECKING:
    from geoh5py.objects import ObjectType


class GridObject(ObjectBase, ABC):
    """
    Base class for object with centroids.
    """

    _attribute_map = ObjectBase._attribute_map.copy()

    def __init__(self, object_type: ObjectType, **kwargs):
        self._centroids: np.ndarray | None = None

        super().__init__(object_type, **kwargs)

    @property
    @abstractmethod
    def centroids(self) -> np.ndarray | None:
        """
        Cell center locations in world coordinates.
        """

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        """Default type uid."""

    @property
    def extent(self):
        """
        Geography bounding box of the object.

        :return: shape(2, 3) Bounding box defined by the bottom South-West and
            top North-East coordinates.
        """
        if self._extent is None and self.centroids is not None:
            self._extent = np.c_[
                self.centroids.min(axis=0), self.centroids.max(axis=0)
            ].T

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

        if self.centroids is not None:
            return mask_by_extent(self.centroids, extent)

        return None
