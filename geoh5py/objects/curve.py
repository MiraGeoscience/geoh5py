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

from ..shared.utils import str2uuid
from .cell_object import CellObject


class Curve(CellObject):
    """
    Curve object defined by a series of line segments (:obj:`~geoh5py.objects.curve.Curve.cells`)
    connecting :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`.

    :param current_line_id: Unique identifier of the current line.
    :param parts: Group identifiers for vertices connected by line segments as defined by the
        :obj:`~geoh5py.objects.curve.Curve.cells` property.
    :param vertices: Array of vertices as defined by :obj:`~geoh5py.objects.points.Points.vertices`.
    """

    _attribute_map: dict = CellObject._attribute_map.copy()
    _attribute_map.update(
        {
            "Current line property ID": "current_line_id",
        }
    )
    _default_name = "Curve"
    _TYPE_UID: uuid.UUID | None = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )
    _minimum_vertices = 2

    def __init__(  # pylint: disable="too-many-arguments"
        self,
        current_line_id: uuid.UUID | None = None,
        parts: np.ndarray | None = None,
        vertices: np.ndarray | list | tuple | None = None,
        **kwargs,
    ):
        self._parts = self.validate_parts(parts, vertices)

        super().__init__(
            vertices=vertices,
            **kwargs,
        )
        self.current_line_id = current_line_id

    @property
    def current_line_id(self) -> uuid.UUID | None:
        """
        Unique identifier of the current line.
        """
        return self._current_line_id

    @current_line_id.setter
    def current_line_id(self, value: uuid.UUID | None):
        value = str2uuid(value)

        if not isinstance(value, (uuid.UUID, type(None))):
            raise TypeError(
                f"Input current_line_id value should be of type {uuid.UUID}."
                f" {type(value)} provided"
            )

        self._current_line_id = value

        if self.on_file:
            self.workspace.update_attribute(self, "attributes")

    def _make_cells_from_parts(self) -> np.ndarray:
        """
        Generate cells from parts.
        """
        cells = []
        for part_id in self.unique_parts:
            ind = np.where(self.parts == part_id)[0]
            cells.append(np.sort(np.c_[ind[:-1], ind[1:]], axis=0))
        return np.vstack(cells)

    @property
    def parts(self) -> np.ndarray:
        """
        Group identifiers for vertices connected by line segments as defined by the
        :obj:`~geoh5py.objects.curve.Curve.cells`
        property. The definition of the :obj:`~geoh5py.objects.curve.Curve.cells`
        property get modified by the setting of parts.
        """
        if self._parts is None:
            if len(self.cells) == 0:
                parts = np.arange(self.n_vertices, dtype=int)
            else:
                cell_parts = np.r_[
                    0, np.cumsum(self.cells[1:, 0] != self.cells[:-1, 1])
                ]
                parts = np.zeros(self.n_vertices, dtype=int)
                parts[self.cells.flatten()] = np.kron(cell_parts, np.ones(2))
            self._parts = parts

        return self._parts

    def remove_vertices(
        self, indices: list[int] | np.ndarray, clear_cache: bool = False
    ):
        """
        Safely remove vertices and cells and corresponding data entries.

        :param indices: Indices of vertices to be removed.
        :param clear_cache: Clear cache of data values.
        """
        super().remove_vertices(indices, clear_cache=clear_cache)
        self._parts = None

    @staticmethod
    def validate_parts(
        indices: list | tuple | np.ndarray | None, vertices: np.ndarray | None
    ):
        if indices is None:
            return None

        if isinstance(indices, (list | tuple)):
            indices = np.asarray(indices, dtype="int32")

        if not isinstance(indices, np.ndarray):
            raise TypeError("Parts must be a list or numpy array.")

        indices = indices.astype("int32")

        if vertices is not None and len(indices) != vertices.shape[0]:
            raise ValueError(f"Provided parts must be of shape {vertices.shape[0]}")

        return indices

    @property
    def unique_parts(self) -> list[int]:
        """
        Unique :obj:`~geoh5py.objects.curve.Curve.parts`
        identifiers.
        """
        return np.unique(self.parts).tolist()

    def validate_cells(self, indices: list | tuple | np.ndarray | None) -> np.ndarray:
        """
        Validate or generate cells made up of pairs of vertices making
            up line segments.

        :param indices: Array of indices, shape(*, 2). If None provided, the
            vertices are connected sequentially.

        :return: Array of indices defining connected vertices.
        """
        # Auto-create from parts or connect vertices sequentially
        if indices is None:
            if self._parts is not None:
                indices = self._make_cells_from_parts()
            else:
                n_segments = self.n_vertices
                indices = np.c_[
                    np.arange(0, n_segments - 1), np.arange(1, n_segments)
                ].astype("uint32")

        if isinstance(indices, (list, tuple)):
            indices = np.array(indices, ndmin=2)

        if not isinstance(indices, np.ndarray):
            raise TypeError(
                "Attribute 'cells' must be provided as type numpy.ndarray, list or tuple."
            )

        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError("Array of cells should be of shape (*, 2).")

        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Indices array must be of integer type")

        if len(indices) > 0 and np.max(indices) > (self.n_vertices - 1):
            raise ValueError("Found cell indices larger than the number of vertices.")

        return indices
