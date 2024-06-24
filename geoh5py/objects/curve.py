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
from .object_base import ObjectType


class Curve(CellObject):
    """
    Curve object defined by a series of line segments (:obj:`~geoh5py.objects.curve.Curve.cells`)
    connecting :obj:`~geoh5py.objects.object_base.ObjectBase.vertices`.
    """

    _attribute_map: dict = CellObject._attribute_map.copy()
    _attribute_map.update(
        {
            "Current line property ID": "current_line_id",
        }
    )

    __TYPE_UID = uuid.UUID(
        fields=(0x6A057FDC, 0xB355, 0x11E3, 0x95, 0xBE, 0xFD84A7FFCB88)
    )

    def __init__(
        self,
        object_type: ObjectType,
        cells: np.ndarray | None = None,
        current_line_id: uuid.UUID | None = None,
        parts: np.ndarray | None = None,
        name="Curve",
        **kwargs,
    ):
        self._current_line_id: uuid.UUID | None = None
        self._parts: np.ndarray | None = None

        super().__init__(object_type, name=name, **kwargs)

        if parts is not None:
            if cells is not None:
                raise ValueError("Parts can only be set if cells are not provided.")

            self.parts = parts

            cells = self.make_cells_from_parts()

        self.cells = cells
        self.current_line_id = current_line_id

    @property
    def cells(self) -> np.ndarray:
        r"""
        :obj:`numpy.ndarray` of :obj:`int`, shape (\*, 2):
        Array of indices defining segments connecting vertices. Defined based on
        :obj:`~geoh5py.objects.curve.Curve.parts` if set by the user.
        """
        return self._cells

    @cells.setter
    def cells(self, indices: list | np.ndarray | None):
        if getattr(self, "_cells", None) is not None:
            raise UserWarning(
                "Attempting to re-assign 'cells'. "
                "Consider using the `remove_cells` method or create a new entity."
            )

        if indices is None and self.on_file:
            indices = self.workspace.fetch_array_attribute(self, "cells")

        if indices is None:
            n_segments = self.vertices.shape[0]
            indices = np.c_[
                np.arange(0, n_segments - 1), np.arange(1, n_segments)
            ].astype("uint32")

        if isinstance(indices, list):
            indices = np.vstack(indices)

        if indices.ndim != 2 or indices.shape[1] != 2:
            raise ValueError("Array of cells should be of shape (*, 2).")

        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Indices array must be of integer type")

        self._cells = indices.astype(np.int32)
        self._parts = None

        if self.on_file:
            self.workspace.update_attribute(self, "cells")

    @property
    def current_line_id(self) -> uuid.UUID | None:
        if getattr(self, "_current_line_id", None) is None:
            self._current_line_id = uuid.uuid4()

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
        self.workspace.update_attribute(self, "attributes")

    @classmethod
    def default_type_uid(cls) -> uuid.UUID:
        """
        :return: Default unique identifier
        """
        return cls.__TYPE_UID

    @property
    def parts(self) -> np.ndarray:
        """
        :obj:`numpy.array` of :obj:`int`, shape
        (:obj:`~geoh5py.objects.object_base.ObjectBase.n_vertices`, 2):
        Group identifiers for vertices connected by line segments as defined by the
        :obj:`~geoh5py.objects.curve.Curve.cells`
        property. The definition of the :obj:`~geoh5py.objects.curve.Curve.cells`
        property get modified by the setting of parts.
        """
        if getattr(self, "_parts", None) is None:
            cells = self.cells
            parts = np.zeros(self.vertices.shape[0], dtype="int")
            count = 0
            for ind in range(1, cells.shape[0]):
                if cells[ind, 0] != cells[ind - 1, 1]:
                    count += 1

                parts[cells[ind, :]] = count

            self._parts = parts

        return self._parts

    @parts.setter
    def parts(self, indices: list | np.ndarray):
        if getattr(self, "_cells", None) is not None:
            raise UserWarning(
                "Attempting to re-assign 'parts'. "
                "Consider using the `remove_cells` method or create a new entity."
            )

        if isinstance(indices, list):
            indices = np.asarray(indices, dtype="int32")
        else:
            indices = indices.astype("int32")

        assert indices.shape == (
            self.vertices.shape[0],
        ), f"Provided parts must be of shape {self.vertices.shape[0]}"
        self._parts = indices

    @property
    def unique_parts(self):
        """
        :obj:`list` of :obj:`int`: Unique :obj:`~geoh5py.objects.curve.Curve.parts`
        identifiers.
        """
        return np.unique(self.parts).tolist()

    def make_cells_from_parts(self) -> np.ndarray | None:
        """
        Generate cells from parts.
        """
        if self.unique_parts is None:
            return None

        cells = []
        for part_id in self.unique_parts:
            ind = np.where(self.parts == part_id)[0]
            cells.append(np.sort(np.c_[ind[:-1], ind[1:]], axis=0))
        return np.vstack(cells)
