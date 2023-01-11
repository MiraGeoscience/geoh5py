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

from .points import Points

if TYPE_CHECKING:
    from geoh5py.objects import ObjectType


class CellObject(Points, ABC):
    """
    Base class for object with cells.
    """

    _attribute_map: dict = Points._attribute_map.copy()

    def __init__(self, object_type: ObjectType, **kwargs):

        self._cells: np.ndarray | None = None

        super().__init__(object_type, **kwargs)

    @classmethod
    @abstractmethod
    def default_type_uid(cls) -> uuid.UUID:
        """Default type uid."""

    def remove_cells(self, indices: list[int]):
        """Safely remove cells and corresponding data entries."""

        if self._cells is None:
            warnings.warn("No cells to be removed.", UserWarning)
            return

        if (
            isinstance(self.cells, np.ndarray)
            and np.max(indices) > self.cells.shape[0] - 1
        ):
            raise ValueError("Found indices larger than the number of cells.")

        cells = np.delete(self.cells, indices, axis=0)
        self._cells = None
        setattr(self, "cells", cells)

        self.remove_children_values(indices, "CELL")

    def remove_vertices(self, indices: list[int]):
        """Safely remove vertices and corresponding data entries."""

        if self.vertices is None:
            warnings.warn("No vertices to be removed.", UserWarning)
            return

        if (
            isinstance(self.vertices, np.ndarray)
            and np.max(indices) > self.vertices.shape[0] - 1
        ):
            raise ValueError("Found indices larger than the number of vertices.")

        vert_index = np.ones(self.vertices.shape[0], dtype=bool)
        vert_index[indices] = False
        vertices = self.vertices[vert_index, :]

        self._vertices = None
        setattr(self, "vertices", vertices)
        self.remove_children_values(indices, "VERTEX")

        new_index = np.ones_like(vert_index, dtype=int)
        new_index[vert_index] = np.arange(self.vertices.shape[0])
        self.remove_cells(np.where(~np.all(vert_index[self.cells], axis=1)))
        setattr(self, "cells", new_index[self.cells])
