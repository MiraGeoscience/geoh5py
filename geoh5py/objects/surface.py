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

import numpy as np

from .cell_object import CellObject


class Surface(CellObject):
    """
    Surface object defined by vertices and cells
    """

    _TYPE_UID = uuid.UUID(
        fields=(0xF26FEBA3, 0xADED, 0x494B, 0xB9, 0xE9, 0xB2BBCBE298E1)
    )
    _minimum_vertices = 3

    def validate_cells(self, indices: list | tuple | np.ndarray | None) -> np.ndarray:
        """
        Validate or generate cells made up of triplets of vertices making
            up triangles.

        :param indices: Array of indices, shape(*, 3). If None provided, the
            vertices are connected sequentially.

        :return: Array of indices defining connecting vertices.
        """
        if isinstance(indices, (tuple | list)):
            indices = np.array(indices, ndmin=2)

        if indices is None:
            n_vert = self.vertices.shape[0]
            indices = np.c_[
                np.arange(0, n_vert - 2), np.arange(1, n_vert - 1), np.arange(2, n_vert)
            ].astype("uint32")

        if not isinstance(indices, np.ndarray):
            raise AttributeError(
                "Attribute 'cells' must be provided as type numpy.ndarray, list or tuple."
            )

        if indices.ndim != 2 or indices.shape[-1] != 3:
            raise ValueError("Array of 'cells' should be of shape (*, 3).")

        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Indices array must be of integer type")

        return indices.astype(np.int32)
