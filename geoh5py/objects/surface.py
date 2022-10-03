#  Copyright (c) 2022 Mira Geoscience Ltd.
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

from .object_base import ObjectType
from .points import Points


class Surface(Points):
    """
    Surface object defined by vertices and cells
    """

    __TYPE_UID = uuid.UUID(
        fields=(0xF26FEBA3, 0xADED, 0x494B, 0xB9, 0xE9, 0xB2BBCBE298E1)
    )

    def __init__(self, object_type: ObjectType, **kwargs):

        self._cells: np.ndarray | None = None

        super().__init__(object_type, **kwargs)

    @property
    def cells(self) -> np.ndarray | None:
        """
        Array of vertices index forming triangles
        :return cells: :obj:`numpy.array` of :obj:`int`, shape ("*", 3)
        """
        if getattr(self, "_cells", None) is None:
            if self.on_file:
                self._cells = self.workspace.fetch_array_attribute(self)

        return self._cells

    @cells.setter
    def cells(self, indices: list | np.ndarray | None):
        if isinstance(indices, list):
            indices = np.vstack(indices)

        if self._cells is not None and (
            indices is None or indices.shape[0] < self._cells.shape[0]
        ):
            raise ValueError(
                "Attempting to assign 'cells' with fewer values. "
                "Use the `remove_cells` method instead."
            )

        if indices.shape[1] != 3:
            raise ValueError("Array of cells should be of shape (*, 3).")

        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Indices array must be of integer type")

        self._cells = indices.astype(np.int32)
        self.workspace.update_attribute(self, "cells")

    def remove_cells(self, indices: list[int]):
        """Safely remove cells and corresponding data entries."""

        if self._cells is None:
            warnings.warn("No cells to be removed.")

        if (
            isinstance(self.cells, np.ndarray)
            and np.max(indices) > self.cells.shape[0] - 1
        ):
            raise ValueError("Found indices larger than the number of cells.")

        cells = np.delete(self.cells, indices, axis=0)
        self._cells = None
        self.cells = cells

        self.remove_children_values(indices, "CELL")

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        return cls.__TYPE_UID
