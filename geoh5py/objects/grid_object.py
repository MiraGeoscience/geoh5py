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

    def copy(
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        cell_mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Function to copy an entity to a different parent entity.

        :param parent: New parent for the copied object.
        :param copy_children: Copy children entities.
        :param clear_cache: Clear cache of data values.
        :param mask: Array of indices to sub-sample the input entity.
        :param cell_mask: Array of indices to sub-sample the input entity cells.
        :param kwargs: Additional keyword arguments.

        :return: New copy of the input entity.
        """
        if cell_mask is not None:
            warnings.warn("Cell masking is not supported for Grid objects.")

        if parent is None:
            parent = self.parent

        if (
            mask is not None
            and self.centroids is not None
            and (
                not isinstance(mask, np.ndarray)
                or mask.shape != (self.centroids.shape[0],)
            )
        ):
            raise ValueError("Mask must be an array of shape (n_vertices,).")

        new_entity = parent.workspace.copy_to_parent(
            self,
            parent,
            clear_cache=clear_cache,
            **kwargs,
        )
        if copy_children:
            children_map = {}
            for child in self.children:
                if (
                    isinstance(mask, np.ndarray)
                    and isinstance(getattr(child, "values", None), np.ndarray)
                    and child.values.shape == mask.shape
                ):
                    values = np.ones_like(child.values) * np.nan
                    values[mask] = child.values[mask]
                else:
                    values = child.values

                child_copy = child.copy(
                    parent=new_entity, clear_cache=clear_cache, values=values
                )
                children_map[child.uid] = child_copy.uid

            if self.property_groups:
                self.workspace.copy_property_groups(
                    new_entity, self.property_groups, children_map
                )
                new_entity.workspace.update_attribute(new_entity, "property_groups")

        return new_entity

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
