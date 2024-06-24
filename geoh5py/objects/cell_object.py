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
import warnings
from abc import ABC, abstractmethod

import numpy as np

from ..data import Data, DataAssociationEnum
from ..groups import PropertyGroup
from ..shared.utils import box_intersect, mask_by_extent
from .points import Points


class CellObject(Points, ABC):
    """
    Base class for object with cells.
    """

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        """Default type uid."""

    @property
    @abstractmethod
    def cells(self) -> np.ndarray:
        """
        Definition of cells.
        """

    @cells.setter
    def cells(self, _) -> np.ndarray:
        """
        Definition of cells.
        """

    def mask_by_extent(
        self,
        extent: np.ndarray,
        inverse: bool = False,
    ) -> np.ndarray | None:
        """
        Sub-class extension of :func:`~geoh5py.shared.entity.Entity.mask_by_extent`.
        """
        if self.extent is None or not box_intersect(self.extent, extent):
            return None

        vert_mask = mask_by_extent(self.vertices, extent, inverse=inverse)

        # Check for orphan vertices
        cell_mask = np.all(vert_mask[self.cells], axis=1)
        orphan_mask = np.zeros_like(vert_mask, dtype=bool)
        orphan_mask[self.cells[cell_mask].flatten()] = True
        vert_mask &= orphan_mask

        if ~np.any(vert_mask):
            return None

        return vert_mask

    def remove_cells(self, indices: list[int] | np.ndarray, clear_cache: bool = False):
        """
        Safely remove cells and corresponding data entries.

        :param indices: Indices of cells to be removed.
        :param clear_cache: Clear cache of data values.
        """

        if self.cells is None:
            warnings.warn("No cells to be removed.", UserWarning)
            return

        if isinstance(indices, (list, tuple)):
            indices = np.array(indices)

        if not isinstance(indices, np.ndarray):
            raise TypeError("Indices must be a list or numpy array.")

        if (
            isinstance(self.cells, np.ndarray)
            and np.max(indices) > self.cells.shape[0] - 1
        ):
            raise ValueError("Found indices larger than the number of cells.")

        cells = np.delete(self.cells, indices, axis=0)
        setattr(self, "_cells", None)
        self.cells = cells
        self.remove_children_values(indices, "CELL", clear_cache=clear_cache)

    def remove_vertices(
        self, indices: list[int] | np.ndarray, clear_cache: bool = False
    ):
        """
        Safely remove vertices and cells and corresponding data entries.

        :param indices: Indices of vertices to be removed.
        :param clear_cache: Clear cache of data values.
        """

        if self.vertices is None:
            warnings.warn("No vertices to be removed.", UserWarning)
            return

        if isinstance(indices, list):
            indices = np.array(indices)

        if not isinstance(indices, np.ndarray):
            raise TypeError("Indices must be a list or numpy array.")

        if (
            isinstance(self.vertices, np.ndarray)
            and np.max(indices) > self.vertices.shape[0] - 1
        ):
            raise ValueError("Found indices larger than the number of vertices.")

        vert_index = np.ones(self.vertices.shape[0], dtype=bool)
        vert_index[indices] = False
        vertices = self.vertices[vert_index, :]

        setattr(self, "_vertices", None)
        setattr(self, "vertices", vertices)
        self.remove_children_values(indices, "VERTEX", clear_cache=clear_cache)

        new_index = np.ones_like(vert_index, dtype=int)
        new_index[vert_index] = np.arange(self.vertices.shape[0])
        self.remove_cells(np.where(~np.all(vert_index[self.cells], axis=1)))
        new_cells = new_index[self.cells]
        setattr(self, "_cells", None)
        setattr(self, "cells", new_cells)

    def copy(  # pylint: disable=too-many-branches
        self,
        parent=None,
        copy_children: bool = True,
        clear_cache: bool = False,
        mask: np.ndarray | None = None,
        cell_mask: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Sub-class extension of :func:`~geoh5py.objects.points.Points.copy`.

        Additions:
            cell_mask: Array of indices to sub-sample the input entity cells.
        """
        if mask is not None and self.vertices is not None:
            if not isinstance(mask, np.ndarray) or mask.shape != (
                self.vertices.shape[0],
            ):
                raise ValueError("Mask must be an array of shape (n_vertices,).")

            kwargs.update({"vertices": self.vertices[mask, :]})

        new_cells = getattr(self, "cells", None)
        if mask is not None:
            new_id = np.ones_like(mask, dtype=int)
            new_id[mask] = np.arange(np.sum(mask))

            if cell_mask is None:
                cell_mask = np.all(mask[self.cells], axis=1)

            new_cells = new_id[self.cells]

        if cell_mask is not None and new_cells is not None:
            new_cells = new_cells[cell_mask, :]
            kwargs.update(
                {
                    "cells": new_cells,
                }
            )

        new_object = super().copy(
            parent=parent,
            copy_children=False,
            clear_cache=clear_cache,
            mask=mask,
            **kwargs,
        )

        if copy_children:
            children_map = {}
            for child in self.children:
                if isinstance(child, PropertyGroup):
                    continue
                if isinstance(child, Data):
                    if child.name in ["A-B Cell ID", "Transmitter ID"]:
                        continue

                    child_mask = mask
                    if (
                        child.association is DataAssociationEnum.CELL
                        and cell_mask is not None
                    ):
                        child_mask = cell_mask
                    elif child.association is not DataAssociationEnum.VERTEX:
                        child_mask = None

                    child_copy = child.copy(
                        parent=new_object,
                        clear_cache=clear_cache,
                        mask=child_mask,
                    )
                else:
                    child_copy = self.workspace.copy_to_parent(
                        child, new_object, clear_cache=clear_cache
                    )
                children_map[child.uid] = child_copy.uid

            if self.property_groups:
                self.workspace.copy_property_groups(
                    new_object, self.property_groups, children_map
                )

        return new_object
